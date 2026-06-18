from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

from ruamel.yaml import YAML


_YAML = YAML(typ="safe")

_REGISTRY: dict[str, dict[str, Any]] | None = None


def _load() -> dict[str, dict[str, Any]]:
    global _REGISTRY
    if _REGISTRY is None:
        path = Path(__file__).resolve().parent.parent / "models" / "registry.yaml"
        with open(path) as f:
            _REGISTRY = _YAML.load(f).get("models", {})
    return _REGISTRY


def get_generate_func(model_family: str) -> Callable[[dict], None]:
    """Return the generate() function for a model family.

    The returned callable has signature: generate(cfg: dict) -> None
    """
    registry = _load()
    if model_family not in registry:
        raise KeyError(
            f"Unknown model family '{model_family}'. Available: {list(registry)}"
        )
    entry = registry[model_family]
    module_path, _, func_name = entry["generate_func"].rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)
