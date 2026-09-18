"""Experiment configuration: one pydantic model per section, composed into `Config`.

Defaults live here. A YAML file only needs to state what differs from them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    """Base for every config model: unknown keys raise, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- data & split


class DataConfig(_Section):
    name: str = "sine_freq_shift"  # registry key in category "dataset"
    params: dict[str, Any] = Field(default_factory=dict)  # checked by the adapter


class SplitConfig(_Section):
    scheme: Literal["group_holdout"] = "group_holdout"
    group_key: str = "u"
    val_size: float = Field(0.1, ge=0.0, lt=1.0)
    test_size: float = Field(0.2, ge=0.0, lt=1.0)
    target_labeled_n: int = Field(8, ge=0)

    @model_validator(mode="after")
    def _sizes_leave_room_for_train(self) -> SplitConfig:
        if self.val_size + self.test_size >= 1.0:
            raise ValueError(
                "val_size + test_size must be < 1 "
                f"(got {self.val_size} + {self.test_size})"
            )
        return self


# ---------------------------------------------------------------- normalization

NormMethod = Literal["none", "minmax", "standard"]
NormScope = Literal["none", "joint", "per_domain", "per_unit"]


class NormColumn(_Section):
    """Per-column exception. A field left as None inherits from the target's default."""

    method: NormMethod | None = None
    scope: NormScope | None = None

    @model_validator(mode="after")
    def _overrides_something(self) -> NormColumn:
        if self.method is None and self.scope is None:
            raise ValueError("a column override must set method and/or scope")
        return self


class NormTarget(_Section):
    method: NormMethod = "standard"  # default for every column of this target
    scope: NormScope = "joint"
    columns: dict[str, NormColumn] = Field(default_factory=dict)  # exceptions

    def spec_for(self, column: str) -> tuple[NormMethod, NormScope]:
        """Resolved (method, scope) for one column: its exception, else the default."""
        col = self.columns.get(column)
        if col is None:
            return self.method, self.scope
        return col.method or self.method, col.scope or self.scope


class NormalizeConfig(_Section):
    features: NormTarget = Field(default_factory=NormTarget)
    labels: NormTarget = Field(default_factory=lambda: NormTarget(method="minmax"))
    physics: NormTarget = Field(default_factory=NormTarget)


# ---------------------------------------------------------------- pairs


class BinarizeConfig(_Section):
    rule: Literal["none", "threshold", "quantile"] = "none"
    value: float | None = None

    @model_validator(mode="after")
    def _value_matches_rule(self) -> BinarizeConfig:
        if self.rule == "none" and self.value is not None:
            raise ValueError("binarize.value must be null when rule is 'none'")
        if self.rule != "none" and self.value is None:
            raise ValueError(f"binarize.rule={self.rule!r} needs a value")
        if self.rule == "quantile" and not 0.0 < self.value < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {self.value}")
        return self


PairType = Literal["within", "cross"]


class PairsConfig(_Section):
    similarity: Literal["label_gap", "physics_distance", "hybrid"] = "label_gap"
    hybrid_alpha: float = Field(0.5, ge=0.0, le=1.0)  # weight on label_gap in hybrid
    physics_weights: tuple[float, ...] | None = None  # None = all ones
    types: tuple[PairType, ...] = Field(("within", "cross"), min_length=1)
    budget: int = Field(20_000, gt=0)
    balance: Literal["cap_to_scarcer", "oversample_scarce", "naive"] = "cap_to_scarcer"
    binarize: BinarizeConfig = Field(default_factory=BinarizeConfig)


# ---------------------------------------------------------------- model & loss


class EncoderConfig(_Section):
    name: str = "mlp"  # registry key in category "encoder"
    stem: Literal["shared", "per_domain"] = "shared"
    hidden: tuple[int, ...] = (64, 64)
    embed_dim: int = Field(2, gt=0)
    final_activation: Literal["linear", "relu"] = "linear"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    append_domain_onehot: bool = True


class LossConfig(_Section):
    name: str = "psp"  # registry key in category "loss"
    params: dict[str, Any] = Field(default_factory=dict)


class SecondaryConfig(_Section):
    probes: tuple[Literal["ridge", "mlp"], ...] = ("ridge", "mlp")
    ridge_alpha: float = Field(1.0, ge=0.0)
    mlp_every_n_epochs: int = Field(10, gt=0)


# ---------------------------------------------------------------- training & infra


class TrainConfig(_Section):
    epochs: int = Field(200, gt=0)
    batch_size: int = Field(256, gt=0)
    optimizer: Literal["adam", "adamw"] = "adamw"
    lr: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(1e-4, ge=0.0)
    scheduler: Literal["none", "cosine"] = "cosine"
    grad_clip: float | None = 1.0
    checkpoint_every: int = Field(10, gt=0)


class LoggingConfig(_Section):
    dir: str = "runs"  # where run directories go -- volatile, not hashed


class RunConfig(_Section):
    device: Literal["auto", "cpu", "cuda"] = "auto"  # volatile, not hashed
    num_workers: int = Field(0, ge=0)  # volatile, not hashed


# ---------------------------------------------------------------- the whole thing


class Config(_Section):
    experiment: str = "default"
    seed: int = 0

    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    pairs: PairsConfig = Field(default_factory=PairsConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    secondary: SecondaryConfig = Field(default_factory=SecondaryConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    run: RunConfig = Field(default_factory=RunConfig)


# ---------------------------------------------------------------- loading (part B)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Turn CLI-style strings ["encoder.embed_dim=8", ...] into {dotted_key: value}.

    Values are parsed as YAML, so "8" -> 8, "[32, 32]" -> [32, 32], "null" -> None.
    """
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override {item!r} must look like 'section.field=value'")
        out[key.strip()] = yaml.safe_load(value)
    return out


def apply_overrides(
    raw: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a copy of `raw` with each dotted key set. `raw` itself is not modified."""
    tree = copy.deepcopy(raw)
    for dotted_key, value in (overrides or {}).items():
        *parents, leaf = dotted_key.split(".")
        node = tree
        for key in parents:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {dotted_key!r}: {key!r} is not a section")
        node[leaf] = value
    return tree


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """YAML file -> apply dotted overrides -> validate -> frozen Config."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}  # an empty file loads as None
    if not isinstance(raw, dict):
        kind = type(raw).__name__
        raise ValueError(f"{path}: top level must be a mapping, got {kind}")
    return Config.model_validate(apply_overrides(raw, overrides))


# ---------------------------------------------------------------- identity (part C)

# (section, field) pairs that describe WHERE a run executes, not WHAT it is.
VOLATILE_FIELDS: tuple[tuple[str, str], ...] = (
    ("logging", "dir"),
    ("run", "device"),
    ("run", "num_workers"),
) #Defines which config fields should NOT affect the experiment identity


def identity_dict(cfg: Config) -> dict[str, Any]:
    """
    The config as plain JSON-able data, minus the volatile fields.
    Converts a Config object to a dictionary, removing volatile fields
    """
    data = cfg.model_dump(mode="json")
    for section, field in VOLATILE_FIELDS:
        del data[section][field]
    return data


def config_hash(cfg: Config) -> str:
    """
    Creates a unique 8-character hash from the config
    First 8 hex chars of SHA-256 over canonical JSON of `identity_dict(cfg)`.
    """
    canonical = json.dumps(
        identity_dict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


# fmt: off
_ABBREV = {
    "standard": "std", "minmax": "mm", "none": "no",
    "joint": "J", "per_domain": "D", "per_unit": "U",
    "label_gap": "lbl", "physics_distance": "phys", "hybrid": "hyb",
    "cap_to_scarcer": "cap", "oversample_scarce": "over", "naive": "naive",
    "linear": "lin", "relu": "relu",
}
# fmt: on


def _ab(value: str) -> str:
    return _ABBREV.get(value, value)


def run_slug(cfg: Config) -> str:
    """
    Short, human-readable summary of the main axes. NOT unique -- the hash is.
    Generates a human-readable summary of the most important config settings
    """
    feats, pairs, enc = cfg.normalize.features, cfg.pairs, cfg.encoder
    hidden = "x".join(str(h) for h in enc.hidden) or "0"
    parts = [
        f"ds-{cfg.data.name}",
        f"nf-{_ab(feats.method)}{_ab(feats.scope)}",
        f"sim-{_ab(pairs.similarity)}",
        f"pr-{''.join(t[0] for t in pairs.types)}-{_ab(pairs.balance)}",
        f"enc-{enc.name}{hidden}d{enc.embed_dim}{_ab(enc.final_activation)}",
        f"loss-{cfg.loss.name}",
        f"s{cfg.seed}",
    ]
    slug = "__".join(parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", slug)  # safe as a folder name


def run_id(cfg: Config) -> str:
    """Directory name for this run: readable slug + unique hash."""
    return f"{run_slug(cfg)}__{config_hash(cfg)}"