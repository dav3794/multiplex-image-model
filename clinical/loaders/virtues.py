"""Virtues/Eva pre-split embedding loader.

Reads the pre-split output format produced by clinical.models.virtues or eva:
  {output_root}/{panel}/train_metadata.csv + train_embeddings.npy
  {output_root}/{panel}/test_metadata.csv  + test_embeddings.npy
"""

import os

import numpy as np
import pandas as pd
from omegaconf import DictConfig


IMG_COL = "img"
SUBSET_COL = "subset"
TRAIN_LABEL = "train"
TEST_LABEL = "test"


def load_embeddings(
    cfg: DictConfig,
    model_cfg: DictConfig,
    panel: str,
) -> pd.DataFrame:
    prefix = cfg.preprocessing.feature_prefix
    panel_path = os.path.join(model_cfg.path, panel)
    train_meta = pd.read_csv(os.path.join(panel_path, "train_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(panel_path, "test_metadata.csv"))
    train_emb = np.load(os.path.join(panel_path, "train_embeddings.npy"))
    test_emb = np.load(os.path.join(panel_path, "test_embeddings.npy"))

    feat_cols = [f"{prefix}{i}" for i in range(train_emb.shape[1])]

    train_df = pd.concat([
        train_meta, pd.DataFrame(train_emb, columns=feat_cols),
    ], axis=1).assign(**{SUBSET_COL: TRAIN_LABEL})

    test_df = pd.concat([
        test_meta, pd.DataFrame(test_emb, columns=feat_cols),
    ], axis=1).assign(**{SUBSET_COL: TEST_LABEL})

    full = pd.concat([train_df, test_df], ignore_index=True)
    full[IMG_COL] = full[model_cfg.img_path_col].apply(lambda x: x.split("/")[-1])
    img_df = full.groupby(IMG_COL)[feat_cols].mean()
    return img_df
