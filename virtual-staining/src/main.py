"""Entry point logic: parse args, run pipeline, print/save results."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.cli import parse_args
from src.metrics import finalize_pearson
from src.models import PipelineConfig, load_config
from src.pipeline import run_pipeline, save_results
from src.plotting import boxplot_virtual_staining


def main():
    args = parse_args()

    if args.command == "plot":
        boxplot_virtual_staining(
            results_csv=args.results,
            output_path=args.output,
            log_mse=args.log_mse,
            figsize=tuple(args.figsize),
        )
        return

    if args.command == "all":
        pass

    cfg = load_config(args.config)

    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.crop_size is not None:
        cfg.crop_size = tuple(args.crop_size)
    if args.plot is not None:
        cfg.plot = args.plot

    if not cfg.models:
        print("No models defined in config.", file=sys.stderr)
        sys.exit(1)

    df, accums, _ = run_pipeline(cfg)

    if not args.no_save:
        save_results(df, accums, list(df["img"].unique()), cfg.output_dir)
    else:
        print("--no-save set; skipping CSV output.")

    if args.table:
        _print_summary_table(df, accums, cfg)

    print("\nDone. To generate a box plot, run:")
    print(
        f"  python evaluate_all.py plot --results {cfg.output_dir}/per_point_results.csv"
    )


def _print_summary_table(
    df: pd.DataFrame,
    accums: dict,
    cfg: PipelineConfig | None = None,
) -> None:
    mse_summary = df.groupby("model")["mse"].mean()
    pearson_summary = {
        name: float(np.mean(list(finalize_pearson(stats).values())))
        for name, stats in accums.items()
    }
    display_map: dict[str, str] = {}
    if cfg:
        for m in cfg.models:
            display_map[m.name] = m.display_name or m.name
    print("\n" + "=" * 72)
    print(f"{'Model':<48s} {'MSE':>10s}  {'Pearson':>10s}")
    print("=" * 72)
    for model_name in sorted(mse_summary.index):
        label = display_map.get(model_name, model_name)
        mse_val = mse_summary[model_name]
        r_val = pearson_summary.get(model_name, float("nan"))
        print(f"{label:<48s} {mse_val:>10.6f}  {r_val:>10.6f}")
    print("=" * 72)
