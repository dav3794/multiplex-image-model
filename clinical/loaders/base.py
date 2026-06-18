from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from omegaconf import DictConfig


class EmbeddingLoader(ABC):
    """Abstract base for embedding loaders.

    Each subclass implements load() to read precomputed embeddings from disk
    and return a DataFrame indexed by image filename with feature columns.
    """

    @abstractmethod
    def load(
        self,
        cfg: DictConfig,
        model_cfg: DictConfig,
        panel: str,
    ) -> pd.DataFrame:
        ...
