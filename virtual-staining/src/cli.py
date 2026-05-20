"""Command-line interface."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate virtual-staining models against ground truth.",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    p.add_argument(
        "--crop-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override crop size (default: config value or 128 128)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        default=None,
        help="Enable per-pair diagnostic plots (off by default).",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving CSV outputs (useful for quick checks).",
    )
    p.add_argument(
        "--table",
        action="store_true",
        help="Print per-model summary table (mean MSE | Pearson).",
    )
    sub = p.add_subparsers(dest="command", help="Sub-commands")

    plot_p = sub.add_parser("plot", help="Generate box plots from results.")
    plot_p.add_argument(
        "--results",
        default="results/per_point_results.csv",
        help="Path to per_point_results.csv (default: results/per_point_results.csv)",
    )
    plot_p.add_argument(
        "--output",
        default="results/boxplot.pdf",
        help="Output path for the figure.",
    )
    plot_p.add_argument(
        "--log-mse",
        action="store_true",
        default=True,
        help="Plot log(MSE) on y-axis (default: true).",
    )
    plot_p.add_argument(
        "--no-log-mse",
        action="store_false",
        dest="log_mse",
        help="Plot raw MSE on y-axis.",
    )
    plot_p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(20, 6),
        metavar=("W", "H"),
        help="Figure size in inches (default: 20 6).",
    )

    return p


def parse_args(argv: list[str] | None = None):
    return build_parser().parse_args(argv)
