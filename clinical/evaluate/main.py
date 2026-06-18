"""Stage 2 entry point: evaluate embeddings on clinical outcome prediction."""

import argparse
import os
import sys

import pandas as pd

from core.config import load_evaluation_config
from .pipeline import evaluate_model, print_summary


EXCLUDED_FEATURES = ["DeathBreast"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate embeddings on clinical outcome prediction."
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_evaluation_config(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.detailed_results_dir:
        os.makedirs(cfg.detailed_results_dir, exist_ok=True)

    all_results = []
    for model_cfg in cfg.models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_cfg.name}  (format: {model_cfg.format})")
        print(f"{'=' * 60}")
        results = evaluate_model(cfg, model_cfg)
        if not results.empty:
            all_results.append(results)
            print(f"  -> {len(results)} result rows")
            if cfg.detailed_results_dir:
                safe = model_cfg.name.replace("/", "_")
                results.to_csv(
                    os.path.join(cfg.detailed_results_dir, f"{safe}.csv"),
                    index=False,
                )

    if not all_results:
        print("No results generated.")
        sys.exit(0)

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[~combined["feature"].isin(EXCLUDED_FEATURES)]

    out = os.path.join(cfg.output_dir, "clinical_cv_results.csv")
    combined.to_csv(out, index=False)
    print(f"\nPer-fold results saved to: {out}")

    summary_path = os.path.join(cfg.output_dir, "clinical_summary.csv")
    _save_summary(combined, summary_path)

    print("\nSummary table (mean F1-macro per model/variant/panel/feature):")
    print_summary(combined)


def _save_summary(results_df: pd.DataFrame, path: str) -> None:
    if results_df.empty:
        return
    mean_df = (
        results_df.groupby(["model", "model_type", "variant", "panel", "feature"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_f1_macro=("f1_macro", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .reset_index()
    )
    mean_all = (
        mean_df.groupby(["model", "model_type", "variant"])
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            mean_f1_macro=("mean_f1_macro", "mean"),
            mean_roc_auc=("mean_roc_auc", "mean"),
        )
        .reset_index()
    )
    mean_all["panel"] = "Mean"
    mean_all["feature"] = "Mean"
    summary = pd.concat([mean_df, mean_all], ignore_index=True)
    summary.to_csv(path, index=False)
    print(f"Summary saved to: {path}")


if __name__ == "__main__":
    main()
