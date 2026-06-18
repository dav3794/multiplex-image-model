from __future__ import annotations

from typing import Callable

from loaders.immuvis import load_embeddings as _load_immuvis
from loaders.virtues import load_embeddings as _load_virtues


LOADER_REGISTRY: dict[str, Callable] = {
    "immuvis": _load_immuvis,
    "virtues": _load_virtues,
}


def get_loader(format: str) -> Callable:
    """Return the loader function for a given embedding format.

    The returned callable has signature:
        (cfg, model_cfg, panel) -> pd.DataFrame
    """
    fn = LOADER_REGISTRY.get(format)
    if fn is None:
        raise ValueError(
            f"Unknown embedding format '{format}'. "
            f"Available: {list(LOADER_REGISTRY)}"
        )
    return fn
