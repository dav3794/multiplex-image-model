import matplotlib.pyplot as plt
import numpy as np
import os


def auc_f1_values_with_uncertainty(path, metric="f1_multiclass", display_map=None):
    results = np.load(path, allow_pickle=True).item()
    class_names = list(results["class_names"])

    if metric == "auc":
        values_list = results["auc_per_class"]
    elif metric == "f1_multiclass":
        values_list = results["f1_per_class_multiclass"]
    elif metric == "ap":
        values_list = results["ap_per_class"]
    else:
        raise ValueError("Metric must be 'auc', 'f1_multiclass', or 'ap'")

    values_bootstrap = np.asarray(values_list)

    if display_map:
        valid_idx = [
            i
            for i, name in enumerate(class_names)
            if name in display_map and name != "Mean"
        ]
        mean_over_classes = values_bootstrap[:, valid_idx].mean(axis=1, keepdims=True)
    else:
        mean_over_classes = values_bootstrap.mean(axis=1, keepdims=True)

    values_bootstrap = np.concatenate([values_bootstrap, mean_over_classes], axis=1)
    class_names.append("Mean")

    mean_values = values_bootstrap.mean(axis=0)
    ci_low, ci_high = np.percentile(values_bootstrap, [2.5, 97.5], axis=0)

    return mean_values, mean_values - ci_low, ci_high - mean_values, class_names


def generate_metric_bar_figure(
    paths: dict,
    model_cfg: list,
    target_order: list,
    display_map: dict,
    metric: str = "f1_multiclass",
    save_path: str | None = None,
    figsize=(6.2, 6.5),
    ylim=(0, 1.05),
    ylabel=None,
    rotation=60,
):
    if ylabel is None:
        ylabel = (
            "F1 Score"
            if metric == "f1_multiclass"
            else "AUC"
            if metric == "auc"
            else "Average Precision"
            if metric == "ap"
            else metric.upper()
        )

    active_models = sorted(
        [m for m in model_cfg if m["short_key"] in paths],
        key=lambda x: x.get("order", 0),
    )
    if not active_models:
        raise ValueError("No valid models found in provided paths")
    n_methods = len(active_models)

    means_dict, yerr_low, yerr_high = {}, {}, {}
    for cfg in active_models:
        mean, low, high, class_names = auc_f1_values_with_uncertainty(
            paths[cfg["short_key"]], metric=metric, display_map=display_map
        )

        valid_indices = [i for i, name in enumerate(class_names) if name in display_map]
        valid_class_names = [class_names[i] for i in valid_indices]
        mapped_names = [display_map[name] for name in valid_class_names]

        reorder_idx = [mapped_names.index(name) for name in target_order]

        means_dict[cfg["display_name"]] = mean[valid_indices][reorder_idx]
        yerr_low[cfg["display_name"]] = low[valid_indices][reorder_idx]
        yerr_high[cfg["display_name"]] = high[valid_indices][reorder_idx]

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    x_spacing = 0.72
    x = np.arange(len(target_order)) * x_spacing
    group_width = 0.55
    bar_width = group_width / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_width

    for i, cfg in enumerate(active_models):
        ax.bar(
            x + offsets[i],
            means_dict[cfg["display_name"]],
            width=bar_width,
            yerr=[yerr_low[cfg["display_name"]], yerr_high[cfg["display_name"]]],
            error_kw=dict(ecolor="black", capsize=1.4, lw=0.7, capthick=0.7),
            color=cfg.get("color", "#000000"),
            edgecolor=cfg.get("edgecolor", "none"),
            alpha=cfg.get("alpha", 1.0),
            label=cfg["display_name"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        target_order, rotation=rotation, ha="right", va="top", fontsize=8.5
    )
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(4, n_methods),
        fontsize=8.5,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved figure → {save_path}")

    return fig
