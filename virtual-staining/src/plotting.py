"""Box-plot generation for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def boxplot_virtual_staining(
    results_csv: str,
    output_path: str = "boxplot.pdf",
    log_mse: bool = True,
    figsize: tuple[float, float] = (20, 6),
    dpi: int = 150,
):
    df = pd.read_csv(results_csv)

    markers = sorted(df["marker"].unique())
    models = sorted(df["model"].unique())
    n_markers = len(markers)
    n_models = len(models)

    cmap = matplotlib.colormaps["tab10"]
    model_colors = {m: cmap(i / max(n_models, 1)) for i, m in enumerate(models)}

    value_col = "log_mse" if log_mse else "mse"
    if log_mse:
        df[value_col] = np.log10(df["mse"].clip(lower=1e-10))

    box_width = 0.8 / n_models
    gap_between_markers = 0.4
    marker_spacing = 1.0
    positions = []
    group_centers = []
    start = 0.0
    for mi in range(n_markers):
        for ni in range(n_models):
            offset = (ni - (n_models - 1) / 2) * box_width
            positions.append(start + marker_spacing / 2 + offset)
        group_centers.append(start + marker_spacing / 2)
        start += marker_spacing + gap_between_markers

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-marker_spacing / 2, start - gap_between_markers + marker_spacing / 2)

    boxes_data: list[list[float]] = [[] for _ in range(n_markers * n_models)]
    model_at_pos: list[str] = []
    for mi, marker in enumerate(markers):
        sub = df[df["marker"] == marker]
        for ni, model in enumerate(models):
            idx = mi * n_models + ni
            vals = sub[sub["model"] == model][value_col].dropna().tolist()
            boxes_data[idx] = vals
            model_at_pos.append(model)

    non_empty = [(i, d) for i, d in enumerate(boxes_data) if len(d) > 0]
    plot_positions = [positions[i] for i, _ in non_empty]
    plot_data = [d for _, d in non_empty]
    plot_model_at = [model_at_pos[i] for i, _ in non_empty]

    bp = ax.boxplot(
        plot_data,
        positions=plot_positions,
        widths=box_width * 0.9,
        patch_artist=True,
        manage_ticks=False,
    )

    for i, (patch, model_name) in enumerate(zip(bp["boxes"], plot_model_at)):
        patch.set_facecolor(model_colors[model_name])
        patch.set_alpha(0.8)

    ylabel = r"$\log_{10}(\mathrm{MSE})$" if log_mse else "MSE"
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(markers, rotation=45, ha="right", fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=model_colors[m], alpha=0.8)
        for m in models
    ]
    ax.legend(
        handles, models,
        title="Model",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Box plot saved to: {output_path}")
