"""Stratified k-fold cross-validation with logistic regression."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_pipeline(variant: str, seed: int) -> Pipeline:
    registry = {
        "standard": Pipeline([
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "scaled": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "pca50": Pipeline([
            ("pca50", PCA(0.5)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "pca75": Pipeline([
            ("pca75", PCA(0.75)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "pca99": Pipeline([
            ("pca99", PCA(0.99)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "pca50w": Pipeline([
            ("pca50", PCA(0.5, whiten=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "pca75w": Pipeline([
            ("pca75", PCA(0.75, whiten=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "pca99w": Pipeline([
            ("pca99", PCA(0.99, whiten=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
    }
    if variant not in registry:
        raise ValueError(f"Unknown preprocessing variant: {variant}")
    return registry[variant]


def run_cv(
    img_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    feature_prefix: str,
    variant: str,
    n_folds: int,
    rare_threshold: float,
    seed: int,
) -> pd.DataFrame:
    common_idx = clinical_df.index.intersection(img_df.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping indices between clinical data and embeddings.")

    y = clinical_df.loc[common_idx, target_col]
    feat_cols = [c for c in img_df.columns if c.startswith(feature_prefix)]
    if not feat_cols:
        raise ValueError(f"No feature columns with prefix '{feature_prefix}'.")

    X = img_df.loc[common_idx, feat_cols].copy()
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    freq = y.value_counts(normalize=True)
    keep_classes = freq[freq >= rare_threshold].index
    keep_mask = y.isin(keep_classes)
    X = X.loc[keep_mask]
    y = y.loc[keep_mask].astype("category")

    if y.nunique() < 2:
        raise ValueError("Fewer than 2 classes remain after filtering rare categories.")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pipeline = _build_pipeline(variant, seed)
    rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)

        roc_auc = np.nan
        try:
            y_prob = pipeline.predict_proba(X_te)
            if y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_te, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_te, y_prob, multi_class="ovr")
        except Exception:
            pass

        rows.append({
            "fold": fold,
            "n_train": len(tr),
            "n_test": len(te),
            "accuracy": accuracy_score(y_te, y_pred),
            "f1_macro": f1_score(y_te, y_pred, average="macro"),
            "roc_auc": roc_auc,
        })

    return pd.DataFrame(rows)
