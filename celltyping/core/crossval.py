import argparse
import glob
import json
import os
import time
import yaml

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    roc_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_mapping(mapping_path):
    """Loads the class name to integer ID mapping."""
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"Missing celltypes mapping at: {mapping_path}")

    with open(mapping_path) as f:
        data = json.load(f)

    mapping = data.get("type_to_id", {})
    print(f"Loaded mapping ({len(mapping)} classes)")
    return mapping


def map_labels(y_raw, mapping):
    """Converts raw string/byte labels into integer IDs based on the mapping."""
    y = np.full(len(y_raw), -1, dtype=np.int64)

    for i, lbl in enumerate(y_raw):
        if isinstance(lbl, (str, bytes, np.str_)):
            lbl = str(lbl).strip()
            if lbl in mapping:
                y[i] = mapping[lbl]
        else:
            try:
                y[i] = int(lbl)
            except ValueError:
                pass

    return y


def load_and_filter_data(emb_path, mapping):
    """Loads paired embeddings and labels, maps classes, and filters out invalid data."""
    emb_files = sorted(glob.glob(os.path.join(emb_path, "*_embeddings.npy")))

    if not emb_files:
        raise FileNotFoundError(f"No '*_embeddings.npy' files found in {emb_path}")

    X_all, y_all = [], []

    for emb_file in emb_files:
        prefix = os.path.basename(emb_file).replace("_embeddings.npy", "")
        lab_file = os.path.join(emb_path, prefix + "_labels.npy")

        if not os.path.exists(lab_file):
            continue

        X = np.load(emb_file, allow_pickle=True)
        y = np.load(lab_file, allow_pickle=True)

        if len(X) != len(y):
            print(
                f"[WARNING] Mismatch in {prefix}: embeddings {X.shape[0]}, labels {y.shape[0]}"
            )
            continue

        y = map_labels(y, mapping)

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise ValueError(f"Failed to load any valid paired data from {emb_path}")

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    total = len(y)
    print(f"Loaded {total} datapoints")

    undefined_id = mapping.get("undefined", -1)

    nan_mask = np.isnan(X).any(axis=1)
    if (nan_count := np.sum(nan_mask)) > 0:
        print(f"Removing {nan_count} datapoints with NaNs")

    neg1_mask = y == -1
    if (neg1_count := np.sum(neg1_mask)) > 0:
        print(f"Removing {neg1_count} datapoints with label -1 (unmapped)")

    if undefined_id != -1:
        undefined_mask = y == undefined_id
        if (undefined_count := np.sum(undefined_mask)) > 0:
            print(f"Removing {undefined_count} datapoints with 'undefined' label")
    else:
        undefined_mask = np.zeros_like(y, dtype=bool)

    mask = (~nan_mask) & (~neg1_mask) & (~undefined_mask)
    kept = np.sum(mask)
    print(f"Kept {kept} / {total} datapoints (removed {total - kept})")

    return X[mask], y[mask]


def crossval(X, y, class_names, class_labels, n_splits=10):
    """Executes stratified K-Fold cross validation and calculates metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {
        "class_names": class_names,
        "class_labels": class_labels,
        "auc_per_class": [],
        "f1_per_class_ovr": [],
        "f1_per_class_multiclass": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "ap_per_class": [],
    }

    print(f"\nStarting {n_splits}-fold cross-validation")
    print(f"Samples: {len(X)}")
    print(f"Class balance: {np.bincount(y)}")

    start_total = time.time()

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print("\n---------------------------------")
        print(f"Split {i}/{n_splits}")
        print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")

        start_split = time.time()

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=42
        )
        clf.fit(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        proba = clf.predict_proba(X_val)
        class_to_col = {c: j for j, c in enumerate(clf.classes_)}

        auc_this = []
        f1_ovr_this = []
        ap_this = []

        for name, label in zip(class_names, class_labels):
            y_true_binary = (y_val == label).astype(int)
            y_pred_binary = (y_val_pred == label).astype(int)

            if label not in class_to_col:
                auc_this.append(np.nan)
                ap_this.append(np.nan)
                f1_ovr_this.append(
                    f1_score(y_true_binary, y_pred_binary, zero_division=0)
                )
                continue

            y_score = proba[:, class_to_col[label]]
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc_this.append(auc(fpr, tpr))
            ap_this.append(average_precision_score(y_true_binary, y_score))
            f1_ovr_this.append(f1_score(y_true_binary, y_pred_binary, zero_division=0))

        results["auc_per_class"].append(auc_this)
        results["ap_per_class"].append(ap_this)
        results["f1_per_class_ovr"].append(f1_ovr_this)

        f1_multi_this = f1_score(
            y_val, y_val_pred, average=None, labels=class_labels, zero_division=0
        )
        results["f1_per_class_multiclass"].append(f1_multi_this.tolist())

        acc = accuracy_score(y_val, y_val_pred)
        bal = balanced_accuracy_score(y_val, y_val_pred)
        mac = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        wei = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

        results["accuracy"].append(acc)
        results["balanced_accuracy"].append(bal)
        results["macro_f1"].append(mac)
        results["weighted_f1"].append(wei)

        print(f"Split metrics: acc={acc:.4f} | bal_acc={bal:.4f} | macro_f1={mac:.4f}")
        print(f"Split time: {time.time() - start_split:.2f}s")

    print(f"\nTotal time: {time.time() - start_total:.2f}s")

    return results


def run_experiment(run_name, emb_path, mapping_path, crossval_save):
    print(f"\n=== {run_name} ===")

    mapping = load_mapping(mapping_path)
    class_names = list(mapping.keys())
    class_labels = [mapping[name] for name in class_names]

    X, y = load_and_filter_data(emb_path, mapping)
    print(f"Data ready for cross-validation: X={X.shape}, y={y.shape}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    results = crossval(X, y, class_names, class_labels)

    os.makedirs(os.path.dirname(crossval_save), exist_ok=True)
    np.save(crossval_save, results, allow_pickle=True)
    print(f"Saved results to: {crossval_save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the dataset YAML config"
    )
    parser.add_argument(
        "--run",
        nargs="+",
        help="Specific runs to evaluate. Evaluates all in embeddings/ if left blank.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    base_path = conf["base_path"]

    embeddings_dir = os.path.join(base_path, "embeddings")
    crossval_dir = os.path.join(base_path, "crossval")

    mapping_path = os.path.join(base_path, "celltypes_map.json")
    if not os.path.exists(mapping_path) and "processed_dir" in conf:
        mapping_path = os.path.join(conf["processed_dir"], "celltypes_map.json")

    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory missing: {embeddings_dir}")

    available_runs = [
        d
        for d in os.listdir(embeddings_dir)
        if os.path.isdir(os.path.join(embeddings_dir, d))
    ]

    selected_runs = available_runs
    if args.run:
        for r in args.run:
            if r not in available_runs:
                raise ValueError(f"Requested run '{r}' not found in {embeddings_dir}")
        selected_runs = args.run

    if not selected_runs:
        print("No runs found in embeddings directory. Exiting.")
        exit(0)

    for run_name in selected_runs:
        run_experiment(
            run_name=run_name,
            emb_path=os.path.join(embeddings_dir, run_name),
            mapping_path=mapping_path,
            crossval_save=os.path.join(crossval_dir, f"crossval_{run_name}.npy"),
        )
