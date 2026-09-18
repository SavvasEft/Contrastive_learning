"""Step 1 tests for the config models (part A): defaults, strictness, validators."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cle.config import (
    Config,
    apply_overrides,
    config_hash,
    load_config,
    parse_overrides,
    run_id,
    run_slug,
)


def test_defaults_form_a_valid_config():
    cfg = Config()
    assert cfg.encoder.embed_dim == 2
    assert cfg.normalize.labels.method == "minmax"  # the lambda default
    assert cfg.normalize.features.method == "standard"


def test_unknown_top_level_key_raises():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Config(sede=1)


def test_unknown_nested_key_raises_with_its_path():
    with pytest.raises(ValidationError, match=r"encoder\.embed_dimm"):
        Config(encoder={"embed_dimm": 8})


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(ValidationError):
        cfg.seed = 3
    with pytest.raises(ValidationError):
        cfg.encoder.embed_dim = 8


def test_lists_become_tuples():
    cfg = Config(encoder={"hidden": [32, 16]})
    assert cfg.encoder.hidden == (32, 16)


def test_string_numbers_are_coerced():
    assert Config(encoder={"embed_dim": "8"}).encoder.embed_dim == 8


@pytest.mark.parametrize(
    "bad",
    [
        {"encoder": {"embed_dim": 0}},  # gt=0
        {"encoder": {"final_activation": "tanh"}},  # not in Literal
        {"pairs": {"types": []}},  # min_length=1
        {"split": {"val_size": 0.5, "test_size": 0.5}},  # model_validator
        {"pairs": {"binarize": {"rule": "quantile"}}},  # rule needs a value
        {"pairs": {"binarize": {"rule": "quantile", "value": 2.0}}},  # out of (0, 1)
    ],
)
def test_invalid_values_raise(bad):
    with pytest.raises(ValidationError):
        Config(**bad)


# ---------------------------------------------------------------- part B: loading

BASE_YAML = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def test_base_yaml_equals_code_defaults():
    # base.yaml documents the defaults; this catches the two drifting apart
    assert load_config(BASE_YAML) == Config()


def test_yaml_to_config_to_dict_round_trips():
    raw = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    assert load_config(BASE_YAML).model_dump(mode="json") == raw


def test_empty_yaml_gives_defaults(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config(empty) == Config()


def test_partial_yaml_fills_in_defaults(tmp_path):
    small = tmp_path / "small.yaml"
    small.write_text("encoder:\n  embed_dim: 8\n", encoding="utf-8")
    cfg = load_config(small)
    assert cfg.encoder.embed_dim == 8
    assert cfg.encoder.hidden == (64, 64)  # untouched default


def test_dotted_override_hits_the_right_field():
    overrides = {
        "encoder.embed_dim": 8,
        "pairs.binarize.rule": "threshold",  # three levels deep
        "pairs.binarize.value": 0.05,
    }
    cfg = load_config(BASE_YAML, overrides)
    assert cfg.encoder.embed_dim == 8
    assert cfg.pairs.binarize.rule == "threshold"
    assert cfg.pairs.binarize.value == 0.05
    assert cfg.encoder.hidden == (64, 64)  # neighbours unchanged


def test_override_does_not_modify_the_input_dict():
    raw = {"encoder": {"embed_dim": 2}}
    apply_overrides(raw, {"encoder.embed_dim": 8})
    assert raw == {"encoder": {"embed_dim": 2}}


def test_override_with_typo_raises():
    with pytest.raises(ValidationError, match=r"encoder\.embed_dimm"):
        load_config(BASE_YAML, {"encoder.embed_dimm": 8})


def test_override_into_a_non_section_raises():
    with pytest.raises(ValueError, match="is not a section"):
        apply_overrides({"seed": 0}, {"seed.x": 1})


def test_parse_overrides_converts_types():
    parsed = parse_overrides(
        [
            "encoder.embed_dim=8",
            "encoder.hidden=[32, 32]",
            "train.grad_clip=null",
            "encoder.append_domain_onehot=false",
        ]
    )
    assert parsed == {
        "encoder.embed_dim": 8,
        "encoder.hidden": [32, 32],
        "train.grad_clip": None,
        "encoder.append_domain_onehot": False,
    }


def test_parse_overrides_rejects_missing_equals():
    with pytest.raises(ValueError, match="must look like"):
        parse_overrides(["encoder.embed_dim"])




def test_column_exception_overrides_only_what_it_sets():
    cfg = Config(
        normalize={"features": {"columns": {"compressor_power": {"method": "minmax"}}}}
    )
    feats = cfg.normalize.features
    assert feats.spec_for("compressor_power") == ("minmax", "joint")  # scope inherited
    assert feats.spec_for("anything_else") == ("standard", "joint")  # default


def test_column_exception_via_dotted_override():
    key = "normalize.features.columns.compressor_power.scope"
    cfg = load_config(BASE_YAML, {key: "per_unit"})
    assert cfg.normalize.features.spec_for("compressor_power") == ("standard", "per_unit")


def test_empty_column_exception_raises():
    with pytest.raises(ValidationError, match="must set method and/or scope"):
        Config(normalize={"features": {"columns": {"compressor_power": {}}}})


def test_typo_inside_column_exception_raises():
    with pytest.raises(ValidationError, match=r"compressor_power\.methd"):
        Config(normalize={"features": {"columns": {"compressor_power": {"methd": "minmax"}}}})


# ---------------------------------------------------------------- part C: identity


def test_hash_is_8_hex_chars():
    h = config_hash(Config())
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_ignores_key_order_in_yaml(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("seed: 1\nencoder: {embed_dim: 8, hidden: [32]}\n", encoding="utf-8")
    b.write_text("encoder: {hidden: [32], embed_dim: 8}\nseed: 1\n", encoding="utf-8")
    assert config_hash(load_config(a)) == config_hash(load_config(b))


def test_hash_ignores_how_a_number_is_written():
    assert config_hash(Config(train={"lr": 1.0e-3})) == config_hash(
        Config(train={"lr": 0.001})
    )


def test_volatile_fields_do_not_change_the_id():
    moved = Config(
        logging={"dir": "/somewhere/else"}, run={"device": "cpu", "num_workers": 8}
    )
    assert run_id(moved) == run_id(Config())


@pytest.mark.parametrize(
    "change",
    [
        {"encoder": {"embed_dim": 8}},
        {"seed": 1},
        {"pairs": {"balance": "naive"}},
        {"normalize": {"features": {"columns": {"power": {"method": "minmax"}}}}},
    ],
)
def test_real_changes_change_the_hash(change):
    assert config_hash(Config(**change)) != config_hash(Config())


def test_slug_is_readable_and_filesystem_safe():
    slug = run_slug(Config(data={"name": "my data/v2"}, seed=3))
    assert "ds-my-data-v2" in slug  # space and slash replaced
    assert slug.endswith("__s3")
    assert all(c.isalnum() or c in "._-" for c in slug)


def test_run_id_is_slug_plus_hash():
    cfg = Config()
    assert run_id(cfg) == f"{run_slug(cfg)}__{config_hash(cfg)}"


# ---------------------------------------------------------------- the gate

_PRINT_ID = "from cle.config import Config, run_id; print(run_id(Config(seed=7)))"


def test_run_id_is_stable_across_processes():
    ids = []
    for hash_seed in ["1", "2"]:  # force Python's built-in hash() to differ
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", _PRINT_ID],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        ids.append(result.stdout.strip())
    assert ids[0] == ids[1] == run_id(Config(seed=7))