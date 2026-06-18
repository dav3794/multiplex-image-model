"""Per-model evaluation pipeline: load embeddings, run CV per panel/feature/variant."""

import pandas as pd
from omegaconf import DictConfig

from loaders import get_loader
from .crossval import run_cv


def evaluate_model(
    cfg: DictConfig,
    model_cfg: DictConfig,
) -> pd.DataFrame:
    all_rows = []
    load_embeddings = get_loader(model_cfg.format)

    for panel in model_cfg.panels:
        print(f"\n  Panel: {panel}")
        img_df = load_embeddings(cfg, model_cfg, panel)

        clinical_data = pd.read_csv(cfg.clinical_data.path)
        panel_clinical = clinical_data[clinical_data["dataset"] == panel].copy()
        features = cfg.datasets[panel].features

        variants = model_cfg.get("variants", cfg.preprocessing.variants)

        for feature in features:
            feat_clinical = panel_clinical[panel_clinical["feature"] == feature]
            feat_img_df = (
                feat_clinical.groupby("img_path")["feature_value"]
                .first()
                .to_frame()
            )

            for variant in variants:
                try:
                    fold_results = run_cv(
                        img_df=img_df,
                        clinical_df=feat_img_df,
                        feature_col="feature_value",
                        target_col="feature_value",
                        feature_prefix=cfg.preprocessing.feature_prefix,
                        variant=variant,
                        n_folds=cfg.cross_validation.n_folds,
                        rare_threshold=cfg.preprocessing.rare_class_threshold,
                        seed=cfg.cross_validation.seed,
                    )
                except (ValueError, KeyError) as e:
                    print(f"      Skipping {feature}/{variant}: {e}")
                    continue

                for _, row in fold_results.iterrows():
                    all_rows.append({
                        "model": model_cfg.name,
                        "model_type": model_cfg.format,
                        "panel": panel,
                        "feature": feature,
                        "variant": variant,
                        "fold": row["fold"],
                        "n_train": row["n_train"],
                        "n_test": row["n_test"],
                        "accuracy": row["accuracy"],
                        "f1_macro": row["f1_macro"],
                        "roc_auc": row["roc_auc"],
                    })

    return pd.DataFrame(all_rows)


def print_summary(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        print("(no results)")
        return

    mean_f1 = (
        results_df.groupby(["model", "variant", "panel", "feature"])["f1_macro"]
        .mean()
        .reset_index()
    )
    pivot = mean_f1.pivot_table(
        index=["model", "variant"],
        columns=["panel", "feature"],
        values="f1_macro",
        aggfunc="first",
    )
    pivot["Mean"] = pivot.mean(axis=1)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)
    print(pivot.to_string())
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.precision")
