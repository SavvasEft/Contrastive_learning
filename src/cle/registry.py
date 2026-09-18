"""Name -> class lookup tables, one per category ("loss", "encoder", "dataset", ...).

Usage:
    @register("loss", "psp")
    class PSPLoss: ...

    cls = get("loss", "psp")

create a registry like the following, for easy retrieval:
{
  "loss": {"psp": PSPLoss, "mse": MSELoss},
  "encoder": {"cnn": CNNEncoder, "transformer": TransformerEncoder},
  "dataset": {"mnist": MNISTDataset, ...}
}
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# category -> (name -> class). Module-level, so it is shared by the whole process.
_REGISTRIES: dict[str, dict[str, type]] = {}


def register(category: str, name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: store `cls` under (category, name). Duplicate names raise."""

    def deco(cls: type[T]) -> type[T]:
        table = _REGISTRIES.setdefault(category, {})
        if name in table:
            raise ValueError(
                f"{category} {name!r} is already registered "
                f"(to {table[name].__module__}.{table[name].__qualname__})"
            )
        table[name] = cls
        return cls  # return the class unchanged, so the decorator is invisible

    return deco


def get(category: str, name: str) -> type:
    """Look up a registered class. Unknown names raise, listing what *is* known."""
    try:
        return _REGISTRIES[category][name]
    except KeyError:
        known = sorted(_REGISTRIES.get(category, {}))
        raise KeyError(f"unknown {category} {name!r}; known: {known}") from None


def available(category: str) -> list[str]:
    """Sorted names registered in a category (empty list if the category is unknown)."""
    return sorted(_REGISTRIES.get(category, {}))