"""ImmuVis batch-based embedding loader.

Reads the batch-based output format produced by clinical.models.immuvis:
  {model}_{split}_image_patches_embeddings_batch_{N}.npy
  {model}_{split}_image_patches_metadata_batch_{N}.csv
"""

import os

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


IMG_COL = "img"
SUBSET_COL = "subset"
TRAIN_LABEL = "train"
TEST_LABEL = "test"


def _arcsin_pre_mean(x):
    np.arcsinh(x, out=x)
    return x.mean(axis=(2, 3))


_BATCH_AGG = {
    "mean": lambda x: x.mean(axis=(2, 3)),
    "arcsin-post-mean": lambda x: np.arcsinh(x.mean(axis=(2, 3)) / 5.0),
    "arcsin-pre-mean": _arcsin_pre_mean,
}

_PATCH_AGG = {
    "mean": "mean",
    "arcsin-post-mean": lambda x: np.arcsinh(x / 5.0).mean(),
    "arcsin-pre-mean": lambda x: np.arcsinh(x.mean() / 5.0),
}


def load_embeddings(
    cfg: DictConfig,
    model_cfg: DictConfig,
    panel: str,
) -> pd.DataFrame:
    prefix = cfg.preprocessing.feature_prefix
    batch_agg_f = _BATCH_AGG[model_cfg.batch_agg]

    all_dfs = []
    for subset_label in (TRAIN_LABEL, TEST_LABEL):
        subset_name = f"{model_cfg.name}_{subset_label}"
        meta_files = sorted([
            os.path.join(model_cfg.path, f)
            for f in os.listdir(model_cfg.path)
            if subset_name in f and "metadata" in f
        ])
        emb_map = {}
        for f in os.listdir(model_cfg.path):
            if subset_name in f and "embeddings_batch" in f:
                emb_map[int(f.split("_")[-1].split(".")[0])] = os.path.join(model_cfg.path, f)

        metadatas = []
        for mf in meta_files:
            cur = pd.read_csv(mf)
            batch_idx = int(mf.split("_")[-1].split(".")[0])
            cur["batch"] = batch_idx
            cur["index"] = list(range(len(cur)))
            metadatas.append(cur)
        full_meta = pd.concat(metadatas, ignore_index=True)
        panel_meta = full_meta[full_meta["panel"] == panel]

        batch_dfs = []
        for batch_id in tqdm(panel_meta["batch"].unique(), desc=f"{subset_label} batches"):
            batch_meta = panel_meta[panel_meta["batch"] == batch_id].copy().reset_index(drop=True)
            batch_emb = batch_agg_f(np.load(emb_map[batch_id]))
            batch_sel = batch_emb[batch_meta["index"].values]
            cols = [f"{prefix}{i}" for i in range(batch_sel.shape[1])]
            batch_df = pd.concat([batch_meta, pd.DataFrame(batch_sel, columns=cols)], axis=1)
            batch_dfs.append(batch_df)

        subset_df = pd.concat(batch_dfs, ignore_index=True)
        subset_df[IMG_COL] = subset_df[model_cfg.img_path_col].apply(lambda x: x.split("/")[-1])
        subset_df[SUBSET_COL] = subset_label
        all_dfs.append(subset_df)

    full = pd.concat(all_dfs, ignore_index=True)
    feat_cols = [c for c in full.columns if c.startswith(prefix)]
    img_df = full.groupby(IMG_COL)[feat_cols].agg(_PATCH_AGG[model_cfg.patch_agg])

    img_df["img_name"] = [el.split(".")[0] + ".tiff" for el in img_df.index]
    img_df = img_df.groupby("img_name")[feat_cols].mean()

    return img_df
