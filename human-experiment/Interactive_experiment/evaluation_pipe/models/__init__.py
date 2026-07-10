"""Model registry and factory for VLM wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseVLM

MODEL_REGISTRY: dict[str, type[BaseVLM]] = {}


def register_model(name: str):
    """Class decorator that registers a VLM wrapper under *name*."""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def create_model(name: str, **kwargs) -> BaseVLM:
    """Instantiate a registered model by *name*.

    Parameters
    ----------
    name : str
        Key in MODEL_REGISTRY (e.g. "smolvlm", "internvl").
    **kwargs
        Forwarded to the wrapper's __init__.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown model {name!r}. Available: {available}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(MODEL_REGISTRY)


# Auto-register local model wrappers on import.
from . import local_models as _local_models  # noqa: E402, F401
