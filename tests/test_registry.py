"""Step 1 tests for the registry: register, look up, and fail loudly."""

import re

import pytest

from cle import registry
from cle.registry import available, get, register


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    """Give every test in this file its own empty registry."""
    monkeypatch.setattr(registry, "_REGISTRIES", {})


def test_register_then_get_returns_the_class():
    @register("loss", "psp")
    class PSP:
        pass

    assert get("loss", "psp") is PSP


def test_decorator_returns_class_unchanged():
    class Plain:
        pass

    decorated = register("loss", "plain")(Plain)
    assert decorated is Plain


def test_unknown_name_lists_known_names():
    @register("loss", "psp")
    class PSP:
        pass

    @register("loss", "triplet")
    class Triplet:
        pass

    expected = "unknown loss 'pssp'; known: ['psp', 'triplet']"
    with pytest.raises(KeyError, match=re.escape(expected)):
        get("loss", "pssp")


def test_unknown_category_raises_with_empty_known_list():
    with pytest.raises(KeyError, match=re.escape("known: []")):
        get("nonexistent", "anything")


def test_duplicate_name_raises_and_keeps_original():
    @register("loss", "psp")
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register("loss", "psp")
        class Second:
            pass

    assert get("loss", "psp") is First


def test_same_name_in_different_categories_is_fine():
    @register("loss", "mlp")
    class LossMLP:
        pass

    @register("encoder", "mlp")
    class EncoderMLP:
        pass

    assert get("loss", "mlp") is LossMLP
    assert get("encoder", "mlp") is EncoderMLP


def test_available_is_sorted():
    for name in ["c", "a", "b"]:
        register("loss", name)(type(name, (), {}))

    assert available("loss") == ["a", "b", "c"]
    assert available("unknown") == []