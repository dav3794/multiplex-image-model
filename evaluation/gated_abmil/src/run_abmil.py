from __future__ import annotations

import sys
from pathlib import Path
import argparse
import os

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from models.ABMIL.gated_abmil import GatedABMILClassifierWithValidation
import gc
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score 

_EMB_MEMMAP_CACHE: dict[str, np.ndarray] = {}

def _load_memmap(path: str) -> np.ndarray:
    """
    Cache `np.load(..., mmap_mode="r")` across the whole run.
    This avoids repeatedly re-opening the same embedding files for each feature/dataset.
    """
    arr = _EMB_MEMMAP_CACHE.get(path)
    if arr is None:
        arr = np.load(path, mmap_mode="r")
        _EMB_MEMMAP_CACHE[path] = arr
    return arr

class MILDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]

    def collate_fn(self, batch):
        bags, labels = zip(*batch)
        max_len = max(b.shape[0] for b in bags)
        emb_dim = bags[0].shape[1]

        padded_bags = []
        masks = []

        for b in bags:
            s = b.shape[0]
            padded = torch.zeros(max_len, emb_dim)
            padded[:s] = torch.from_numpy(b).float()
    
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[s:] = 1 

            padded_bags.append(padded)
            masks.append(mask)

        return (
            torch.stack(padded_bags),   # B x S x D
            torch.stack(masks),         # B x S
            torch.tensor(labels)        # B
        )

def balance_meta_df(
    meta_df: pd.DataFrame,
    min_class_freq: float = 0.05,
) -> pd.DataFrame:
    """
    Image-level filtering for a *single* pool (e.g. train+test combined):
    - Drop NaN labels (image-level)
    - Keep classes whose frequency in the pool is >= min_class_freq
    """
    img_labels = meta_df.groupby("img_path", sort=False)["feature_value"].first()
    img_labels = img_labels[~pd.isna(img_labels)].astype(str)
    if len(img_labels) == 0:
        return meta_df.iloc[0:0]

    unique_classes, counts = np.unique(img_labels.values, return_counts=True)
    freq = counts / counts.sum()
    keep_classes = set(unique_classes[freq >= float(min_class_freq)])

    keep_imgs = img_labels[img_labels.isin(keep_classes)].index
    return meta_df[meta_df["img_path"].isin(keep_imgs)]


def build_image_bags(meta_df, class_to_idx, model=None, debug=False):
    bags = []
    labels = []

    for img_path, group in meta_df.groupby("img_path", sort=False):
        raw_label = group["feature_value"].iloc[0]
        # print(group[["img_path", "image_paths"]].head(1))
        if pd.isna(raw_label):
            continue
        raw_label = str(raw_label)
        label = class_to_idx.get(raw_label, None)

        if label is None or pd.isna(label):
            continue

        for emb_file in group["embeddings_file"].unique():
            emb_array = _load_memmap(emb_file)

            idx = group.loc[
                group["embeddings_file"] == emb_file, "embedding_idx"
            ].values

            crop_embs = emb_array[idx]
            # crop_embs = np.arcsinh(crop_embs / 5)

            if model.startswith("immuvis"):
                if crop_embs.ndim == 4:
                    crop_embs = crop_embs.mean(axis=(2, 3))
                elif crop_embs.ndim == 3:
                    crop_embs = crop_embs.mean(axis=2)
                elif crop_embs.ndim != 2:
                    crop_embs = crop_embs.reshape(crop_embs.shape[0], crop_embs.shape[1], -1).mean(axis=-1)
            else:
                if crop_embs.ndim > 2:
                    crop_embs = crop_embs.reshape(crop_embs.shape[0], crop_embs.shape[1], -1).mean(axis=-1)

            if debug:
                print(emb_file, crop_embs.shape)
       
        bags.append(crop_embs)
        labels.append(label)
    return bags, torch.tensor(labels)


def get_a_subset(meta_table: pd.DataFrame, column: str, value: str):
    return meta_table[meta_table[column]==value]

def get_unique_values(tables: list[pd.DataFrame], column: str) -> set:
    intersection = None
    for table in tables:
        values = set(pd.unique(table[column]))
        if intersection is None:
            intersection = values
        else:
            intersection &= values
    return intersection 

def drop_nan_labels(meta_df: pd.DataFrame) -> pd.DataFrame:
    valid_imgs = (
        meta_df
        .dropna(subset=["feature_value"])["img_path"]
        .unique()
    )
    return meta_df[meta_df["img_path"].isin(valid_imgs)]


FEATURE_FILTER = ['Grade', 'Relapse', 'DX.name', 'ERStatus', 'ERBB2_pos', 'PAM50'] 

def filter_features(features: set, filter: list):
    return set(features) & set(filter)

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def _acc_macro_f1(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1).astype(int, copy=False)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int, copy=False)
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro"))

def _auc(y_true: np.ndarray, logits: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_true_int = y_true.astype(int, copy=False)
    if len(np.unique(y_true_int)) < 2:
        return float("nan")

    logits = np.asarray(logits)

    if int(num_classes) <= 2:
        scores = _sigmoid_np(logits.reshape(-1))
        try:
            return float(roc_auc_score(y_true_int, scores))
        except ValueError:
            return float("nan")

    if logits.ndim != 2:
        return float("nan")

    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    probs = expz / expz.sum(axis=1, keepdims=True)
    try:
        return float(roc_auc_score(y_true_int, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")

def run(
    train_meta_table: pd.DataFrame,
    test_meta_table: pd.DataFrame,
    RESULTS_DIR: str,
    model_name: str,
    num_epochs: int = 100,
    batch_size: int = 16,
    device: str = "cuda",
    patience=100,
    num_folds: int = 10,
    reuse_checkpoints: bool = True,
):
    datasets = [
        'danenberg', 
        'cords'
    ]
    results = {}
    cv_fold_rows = []
    cv_summary_rows = []

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset} ===", flush=True)

        pooled_meta = pd.concat(
            [
                train_meta_table[train_meta_table["dataset"] == dataset],
                test_meta_table[test_meta_table["dataset"] == dataset],
            ],
            ignore_index=True,
        )

        features = filter_features(
            set(pooled_meta["feature"].unique()),
            FEATURE_FILTER
        )

        for feature in features:
            print(f"\n--- Feature: {feature} ---", flush=True)

            feat_meta = pooled_meta[pooled_meta["feature"] == feature]
            feat_meta = drop_nan_labels(feat_meta)
            feat_meta = balance_meta_df(feat_meta, min_class_freq=0.05)
            if len(feat_meta) == 0:
                print("Skipping (no data after filtering).", flush=True)
                continue

            classes = sorted(feat_meta["feature_value"].astype(str).unique())
            class_to_idx = {c: i for i, c in enumerate(classes)}
            num_classes = len(classes)

            print(f"Classes: {classes}")
            # ---------- build MIL bags ----------
            bags, labels = build_image_bags(feat_meta, class_to_idx, model_name)

            embedding_dim = bags[0].shape[1]

            num_classes = len(torch.unique(labels))

            print(f"Images (pooled): {len(bags)}")
            print(f"Classes: {num_classes}")

            # ---------- cross-validation (IMAGE LEVEL) ----------
            labels_np = labels.detach().cpu().numpy()
            _, class_counts = np.unique(labels_np, return_counts=True)
            min_class_count = int(class_counts.min()) if len(class_counts) else 0

            desired_folds = int(num_folds)
            if min_class_count >= 2:
                actual_folds = min(desired_folds, min_class_count, len(bags))
                if actual_folds < desired_folds:
                    print(
                        f"Warning: requested {desired_folds}-fold CV but smallest class has "
                        f"{min_class_count} samples; using {actual_folds} folds instead.",
                        flush=True,
                    )
                splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
                split_iter = splitter.split(np.zeros(len(labels_np)), labels_np)
            else:
                print(
                    "Warning: some class has <2 samples; cannot do stratified CV. "
                    "Falling back to a single non-stratified 70/30 train/val split.",
                    flush=True,
                )
                tr_idx, va_idx = train_test_split(
                    np.arange(len(bags)),
                    test_size=0.1,
                    random_state=42,
                    shuffle=True,
                )
                split_iter = [(tr_idx, va_idx)]

            fold_val_metrics = []
            fold_val_aucs = []
            oof_pred = np.full(len(bags), -1, dtype=int)
            logits_dim = 1 if num_classes <= 2 else int(num_classes)
            oof_logits = np.full((len(bags), logits_dim), np.nan, dtype=np.float32)
            oof_y_true = labels_np.astype(int, copy=False)

            for fold, (train_idx, val_idx) in enumerate(split_iter):
                fold_name = f"{dataset}_{feature}_fold{fold}"
                ckpt_path = Path(RESULTS_DIR) / f"{fold_name}.pt"

                train_ds = MILDataset(
                    [bags[i] for i in train_idx],
                    labels[train_idx]
                )
                val_ds = MILDataset(
                    [bags[i] for i in val_idx],
                    labels[val_idx]
                )

                train_dl = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=train_ds.collate_fn
                )
                val_dl = DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=val_ds.collate_fn
                )

                print(f"Building model (fold {fold})...", flush=True)
                model = GatedABMILClassifierWithValidation(
                    input_dim=embedding_dim,
                    hidden_dim=256,
                    num_heads=8,
                    num_classes=num_classes,
                    device=device,
                    patience=patience,
                    save_path=RESULTS_DIR,
                    name=fold_name,
                )

                loss_fn = torch.nn.CrossEntropyLoss() if num_classes > 2 else torch.nn.BCEWithLogitsLoss()
                optimizer = None

                if reuse_checkpoints and ckpt_path.exists():
                    print(f"Reusing checkpoint (fold {fold}): {ckpt_path}", flush=True)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    print(f"train model (fold {fold})...", flush=True)
                    model.train_model(
                        train_dl=train_dl,
                        valid_dl=val_dl,
                        num_epochs=num_epochs,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        monitor="valid_loss",
                    )

                model.load_best_model()

                val_outputs = model.get_outputs(val_dl, load_best=False)
                val_acc, val_f1 = _acc_macro_f1(val_outputs["ground_truth"], val_outputs["predictions"])
                val_auc = _auc(val_outputs["ground_truth"], val_outputs["logits"], num_classes=num_classes)
                fold_val_metrics.append((val_acc, val_f1))
                fold_val_aucs.append(val_auc)
                
                oof_pred[np.asarray(val_idx)] = np.asarray(val_outputs["predictions"]).astype(int, copy=False)
                
                v_logits = np.asarray(val_outputs["logits"])
                if v_logits.ndim == 1:
                    v_logits = v_logits.reshape(-1, 1)
                oof_logits[np.asarray(val_idx)] = v_logits.astype(np.float32, copy=False)

                cv_fold_rows.append({
                    "dataset": dataset,
                    "feature": feature,
                    "fold": fold,
                    "cv_accuracy": val_acc,
                    "cv_macro_f1": val_f1,
                    "cv_auc": float(val_auc),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                })

                del model
                if optimizer is not None:
                    del optimizer
                del train_ds, val_ds
                del train_dl, val_dl
                del val_outputs
                torch.cuda.empty_cache()
                gc.collect()

            eval_mask = oof_pred != -1
            oof_acc, oof_f1 = _acc_macro_f1(oof_y_true[eval_mask], oof_pred[eval_mask])
            oof_auc = _auc(oof_y_true[eval_mask], oof_logits[eval_mask], num_classes=num_classes)

            results[(dataset, feature)] = {
                "ground_truth": oof_y_true[eval_mask],
                "predictions": oof_pred[eval_mask],
                "logits": oof_logits[eval_mask],
            }

            cv_accs = np.array([m[0] for m in fold_val_metrics], dtype=float)
            cv_f1s = np.array([m[1] for m in fold_val_metrics], dtype=float)
            cv_aucs = np.asarray(fold_val_aucs, dtype=float)
            cv_auc_mean = float(np.nanmean(cv_aucs)) if cv_aucs.size else float("nan")
            _non_nan = int(np.sum(~np.isnan(cv_aucs)))
            cv_auc_std = float(np.nanstd(cv_aucs, ddof=1)) if _non_nan > 1 else 0.0
            cv_acc_std = float(cv_accs.std(ddof=1)) if len(cv_accs) > 1 else 0.0
            cv_f1_std = float(cv_f1s.std(ddof=1)) if len(cv_f1s) > 1 else 0.0

            cv_summary_rows.append({
                "dataset": dataset,
                "feature": feature,
                "folds_used": int(len(fold_val_metrics)),
                "cv_accuracy_mean": float(cv_accs.mean()),
                "cv_accuracy_std": cv_acc_std,
                "cv_macro_f1_mean": float(cv_f1s.mean()),
                "cv_macro_f1_std": cv_f1_std,
                "cv_auc_mean": cv_auc_mean,
                "cv_auc_std": cv_auc_std,
                "oof_accuracy": float(oof_acc),
                "oof_macro_f1": float(oof_f1),
                "oof_auc": float(oof_auc),
                "oof_accuracy_std": cv_acc_std,
                "oof_macro_f1_std": cv_f1_std,
                "oof_auc_std": cv_auc_std,
                "min_class_count": int(min_class_count),
                "n_total": int(len(bags)),
                "n_eval": int(eval_mask.sum()),
            })

            del bags, labels
            del oof_pred, oof_logits, oof_y_true
            torch.cuda.empty_cache()
            gc.collect()


    cv_folds_df = pd.DataFrame(cv_fold_rows)
    cv_summary_df = pd.DataFrame(cv_summary_rows)
    return results, cv_folds_df, cv_summary_df

def summarize_results(results):
    rows = []

    for (dataset, feature), res in results.items():
        y_true = res['ground_truth']  # already numpy
        y_pred = res['predictions']   # numpy array of predicted classes
        logits = res.get("logits", None)
        num_classes = 2
        if logits is not None:
            logits_arr = np.asarray(logits)
            if logits_arr.ndim == 2 and logits_arr.shape[1] > 1:
                num_classes = int(logits_arr.shape[1])

        # No need to check ndim anymore
        rows.append({
            "dataset": dataset,
            "feature": feature,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "auc": _auc(y_true, logits, num_classes=num_classes) if logits is not None else float("nan"),
        })

    return pd.DataFrame(rows)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", 
    required=True,
    choices=["immuvis_475", "immuvis_473", "virtues_new", "swin200", "vitl200", "vitm200", "vits200"],
    help="Choose the embedding model."
)

args = parser.parse_args()
model_name = args.model

train_meta_table_path = f"/home/tnocon/mil/{model_name}/train_meta_table.csv"
test_meta_table_path = f"/home/tnocon/mil/{model_name}/test_meta_table.csv"

RESULTS_DIR = f"/home/tnocon/mil/abmil/results/{model_name}"
os.makedirs(RESULTS_DIR, exist_ok=True)
train_meta_table = pd.read_csv(train_meta_table_path)
test_meta_table = pd.read_csv(test_meta_table_path)
results, cv_folds_df, cv_summary_df = run(
    train_meta_table,
    test_meta_table,
    RESULTS_DIR,
    model_name,
    reuse_checkpoints=True,
)

df = summarize_results(results)
df.to_csv(f"{RESULTS_DIR}/abmil_2losses_results.csv", index=False)
cv_folds_df.to_csv(f"{RESULTS_DIR}/abmil_2losses_cv_folds.csv", index=False)
cv_summary_df.to_csv(f"{RESULTS_DIR}/abmil_2losses_cv_summary_auc.csv", index=False)
print(df)
