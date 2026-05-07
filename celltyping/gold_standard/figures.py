import os
import copy
import numpy as np
import pandas as pd
from core.bar_plot import generate_metric_bar_figure
from core.utils import load_config

CONFIG_PATH = "./gold_standard/gs_config.yaml"
config = load_config(CONFIG_PATH)

base_path = config["base_path"]
crossval_dir = os.path.join(base_path, "crossval")
csv_output_dir = "./gold_standard/csv"

MODEL_CONFIG = [
    {
        "short_key": "virtues_patch",
        "model_base": "virtues",
        "variant": "patch",
        "display_name": "VirTues Patch",
        "color": "#6085D3",
        "alpha": 1.0,
        "order": 5,
    },
    {
        "short_key": "virtues_context",
        "model_base": "virtues",
        "variant": "context",
        "display_name": "VirTues Context",
        "color": "#6085D3",
        "alpha": 1.0,
        "order": 6,
    },
    # VIRTUES OURS
    {
        "short_key": "virtuesours_patch",
        "model_base": "virtuesours",
        "variant": "patch",
        "display_name": "VirTuesOurs Patch",
        "color": "#3F5FA8",
        "alpha": 1.0,
        "order": 5,
    },
    {
        "short_key": "virtuesours_context",
        "model_base": "virtuesours",
        "variant": "context",
        "display_name": "VirTuesOurs Context",
        "color": "#3F5FA8",
        "alpha": 1.0,
        "order": 6,
    },
    {
        "short_key": "virtuesours_arcsinh_patch",
        "model_base": "virtuesours_arcsinh",
        "variant": "patch",
        "display_name": "VirTues Arcsinh Patch",
        "color": "#2F4F90",
        "alpha": 1.0,
        "order": 9,
    },
    {
        "short_key": "virtuesours_arcsinh_context",
        "model_base": "virtuesours_arcsinh",
        "variant": "context",
        "display_name": "VirTues Arcsinh Context",
        "color": "#2F4F90",
        "alpha": 1.0,
        "order": 10,
    },
    # BETA
    {
        "short_key": "beta603_patch",
        "model_base": "beta603",
        "variant": "patch",
        "display_name": "Beta-603 Patch",
        "color": "#d81b60",
        "alpha": 1.0,
        "order": 13,
    },
    {
        "short_key": "beta603_context",
        "model_base": "beta603",
        "variant": "context",
        "display_name": "Beta-603 Context",
        "color": "#f48fb1",
        "alpha": 1.0,
        "order": 14,
    },
    {
        "short_key": "beta610_patch",
        "model_base": "beta610",
        "variant": "patch",
        "display_name": "Beta-610 Patch",
        "color": "#1e88e5",
        "alpha": 1.0,
        "order": 15,
    },
    {
        "short_key": "beta610_context",
        "model_base": "beta610",
        "variant": "context",
        "display_name": "Beta-610 Context",
        "color": "#90caf9",
        "alpha": 1.0,
        "order": 16,
    },
    {
        "short_key": "beta614_patch",
        "model_base": "beta614",
        "variant": "patch",
        "display_name": "Beta-614 Patch",
        "color": "#43a047",
        "alpha": 1.0,
        "order": 17,
    },
    {
        "short_key": "beta614_context",
        "model_base": "beta614",
        "variant": "context",
        "display_name": "Beta-614 Context",
        "color": "#a5d6a7",
        "alpha": 1.0,
        "order": 18,
    },
    # VIRTUES PREPROCESSING
    {
        "short_key": "beta658_patch",
        "model_base": "beta658",
        "variant": "patch",
        "display_name": "Beta-658 Patch",
        "color": "#3949AB",  # Indigo
        "alpha": 1.0,
        "order": 27,
    },
    {
        "short_key": "beta658_context",
        "model_base": "beta658",
        "variant": "context",
        "display_name": "Beta-658 Context",
        "color": "#9FA8DA",  # Light Indigo
        "alpha": 1.0,
        "order": 28,
    },
    {
        "short_key": "beta659_patch",
        "model_base": "beta659",
        "variant": "patch",
        "display_name": "Beta-659 Patch",
        "color": "#8D6E63",  # Brown
        "alpha": 1.0,
        "order": 29,
    },
    {
        "short_key": "beta659_context",
        "model_base": "beta659",
        "variant": "context",
        "display_name": "Beta-659 Context",
        "color": "#D7CCC8",  # Light Brown
        "alpha": 1.0,
        "order": 30,
    },
    # EVA
    {
        "short_key": "eva_patch",
        "model_base": "eva",
        "variant": "patch",
        "display_name": "EVA Patch",
        "color": "#00695C",  # Teal
        "alpha": 1.0,
        "order": 41,
    },
    {
        "short_key": "eva_context",
        "model_base": "eva",
        "variant": "context",
        "display_name": "EVA Context",
        "color": "#80CBC4",  # Light Teal
        "alpha": 1.0,
        "order": 42,
    },
]


FIGURE_CONFIG = [
    {
        "name": "f1_all_patch.png",
        "models": [
            "virtues_patch",
            "virtuesours_patch",
            "virtuesours_arcsinh_patch",
            "beta603_patch",
            "beta610_patch",
            "beta614_patch",
            "beta658_patch",
            "beta659_patch",
            "eva_patch",
        ],
    },
    {
        "name": "f1_all_context.png",
        "models": [
            "virtues_context",
            "virtuesours_context",
            "virtuesours_arcsinh_context",
            "beta603_context",
            "beta610_context",
            "beta614_context",
            "beta658_context",
            "beta659_context",
            "eva_context",
        ],
    },
    {
        "name": "ap_all_patch.png",
        "models": [
            "virtues_patch",
            "virtuesours_patch",
            "virtuesours_arcsinh_patch",
            "beta603_patch",
            "beta610_patch",
            "beta614_patch",
            "beta658_patch",
            "beta659_patch",
            "eva_patch",
        ],
    },
    {
        "name": "ap_all_context.png",
        "models": [
            "virtues_context",
            "virtuesours_context",
            "virtuesours_arcsinh_context",
            "beta603_context",
            "beta610_context",
            "beta614_context",
            "beta658_context",
            "beta659_context",
            "eva_context",
        ],
    },
]


DISPLAY_MAP = {
    "B": "B",
    "BnT": "BnT",
    "CD4": "CD4",
    "CD8": "CD8",
    "DC": "DC",
    "HLADR": "HLADR",
    "MacCD163": "MacCD163",
    "Mural": "Mural",
    "NK": "NK",
    "Neutrophil": "Neutrophil",
    "Treg": "Treg",
    "Tumor": "Tumor",
    "pDC": "pDC",
    "plasma": "plasma",
    # "undefined": "undefined",
    "Mean": "Mean",
}

TARGET_ORDER = [
    "B",
    "BnT",
    "CD4",
    "CD8",
    "DC",
    "HLADR",
    "MacCD163",
    "Mural",
    "NK",
    "Neutrophil",
    "Treg",
    "Tumor",
    "pDC",
    "plasma",
    # "undefined",
    "Mean",
]



def get_crossval_path(crossval_dir: str, model_base: str, variant: str) -> str:
    filename = f"crossval_{model_base}_{variant}.npy"
    full_path = os.path.join(crossval_dir, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Cross-validation file not found for model '{model_base}' ({variant}):\n"
            f"   {full_path}\n"
            f"Make sure the cross-validation has been run and the .npy file exists."
        )
    return full_path


def get_raw_data(path, metric="f1_multiclass", display_map=None):
    """Loads the numpy dictionary and extracts raw metric folds and means."""
    results = np.load(path, allow_pickle=True).item()
    class_names = results["class_names"]

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

    return mean_values, class_names, values_bootstrap


def export_metric_csvs(
    active_models,
    paths_dict,
    target_order,
    display_map,
    metric="f1_multiclass",
    prefix="f1",
    output_dir=".",
):
    """Generates the 3 requested CSV variants using the structured model configs for a specific metric."""
    os.makedirs(output_dir, exist_ok=True)

    csv_data_var1 = []
    csv_data_var2 = []
    csv_data_var3 = []

    for cfg in active_models:
        short_key = cfg["short_key"]
        display_name = cfg["display_name"]

        if short_key not in paths_dict:
            continue

        mean_vals, class_names, raw_vals = get_raw_data(
            paths_dict[short_key], metric=metric, display_map=display_map
        )

        valid_indices = [i for i, name in enumerate(class_names) if name in display_map]
        mapped_names = [display_map[class_names[i]] for i in valid_indices]

        reorder_idx = [mapped_names.index(name) for name in target_order]
        ordered_mean = mean_vals[valid_indices][reorder_idx]
        ordered_raw = raw_vals[:, valid_indices][:, reorder_idx]

        # 1. First CSV: per-fold and per-class values
        for fold_idx in range(ordered_raw.shape[0]):
            row_var1 = {"Model": display_name, "short_key": short_key, "fold": fold_idx}
            for col_idx, cls_name in enumerate(target_order):
                row_var1[cls_name] = ordered_raw[fold_idx, col_idx]
            csv_data_var1.append(row_var1)

        # 2. Second CSV: per-class values (mean across folds)
        row_var2 = {"Model": display_name, "short_key": short_key}
        for col_idx, cls_name in enumerate(target_order):
            row_var2[cls_name] = ordered_mean[col_idx]
        csv_data_var2.append(row_var2)

        # 3. Third CSV: completely aggregated mean values per short_key
        # Safely grab the exact "Mean" value calculated by get_raw_data
        true_mean_idx = class_names.index("Mean")
        true_overall_mean = mean_vals[true_mean_idx]

        csv_data_var3.append(
            {
                "Model": display_name,
                "short_key": short_key,
                f"mean_{prefix}": true_overall_mean,
            }
        )

    if not csv_data_var1:
        print(f"No valid data processed for {prefix}. CSVs will not be generated.")
        return

    df1 = pd.DataFrame(csv_data_var1)
    df2 = pd.DataFrame(csv_data_var2)
    df3 = pd.DataFrame(csv_data_var3)

    path1 = os.path.join(output_dir, f"{prefix}_per_fold_per_class.csv")
    path2 = os.path.join(output_dir, f"{prefix}_per_class.csv")
    path3 = os.path.join(output_dir, f"{prefix}_aggregated.csv")

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)
    df3.to_csv(path3, index=False)

    print(
        f"\nSuccessfully saved 3 CSVs for {prefix.upper()} to {os.path.abspath(output_dir)}:"
    )
    print(f"  - {os.path.basename(path1)}")
    print(f"  - {os.path.basename(path2)}")
    print(f"  - {os.path.basename(path3)}")


if __name__ == "__main__":
    print("--- GENERATING FIGURES ---")

    for cfg in FIGURE_CONFIG:
        active_paths = {}
        active_model_cfg = []

        for model_def in MODEL_CONFIG:
            if model_def["short_key"] in cfg["models"]:
                try:
                    path = get_crossval_path(
                        crossval_dir, model_def["model_base"], model_def["variant"]
                    )
                    active_paths[model_def["short_key"]] = path
                    active_model_cfg.append(copy.deepcopy(model_def))
                except FileNotFoundError as e:
                    print(e)

        if not active_model_cfg:
            print(f"Warning: No models found for figure {cfg['name']}")
            continue

        if "color_override" in cfg:
            for m in active_model_cfg:
                if m["short_key"] in cfg["color_override"]:
                    m["color"] = cfg["color_override"][m["short_key"]]

        metric = "ap" if "ap" in cfg["name"].lower() else "f1_multiclass"
        ylabel = "Average Precision (AP)" if metric == "ap" else "F1 Score"

        print(f"Generating {cfg['name']} with {len(active_model_cfg)} models...")

        generate_metric_bar_figure(
            paths=active_paths,
            model_cfg=active_model_cfg,
            target_order=TARGET_ORDER,
            display_map=DISPLAY_MAP,
            metric=metric,
            save_path=f"./gold_standard/figures/{cfg['name']}",
            figsize=(14, 6.5),
            ylabel=ylabel,
            rotation=65,
        )

    print("\nAll figures generated successfully!")

    print("\n--- EXPORTING CSV DATA ---")

    MODELS_TO_EXPORT_CSV = [
        "virtues_patch",
        "virtuesours_patch",
        "virtuesours_arcsinh_patch",
        "beta603_patch",
        "beta610_patch",
        "beta614_patch",
        "beta658_patch",
        "beta659_patch",
        "eva_patch",
        #
        "virtues_context",
        "virtuesours_context",
        "virtuesours_arcsinh_context",
        "beta603_context",
        "beta610_context",
        "beta614_context",
        "beta658_context",
        "beta659_context",
        "eva_context",
    ]

    csv_paths_dict = {}
    csv_active_models = []

    for model_def in MODEL_CONFIG:
        if model_def["short_key"] in MODELS_TO_EXPORT_CSV:
            try:
                path = get_crossval_path(
                    crossval_dir, model_def["model_base"], model_def["variant"]
                )
                csv_paths_dict[model_def["short_key"]] = path
                csv_active_models.append(model_def)
            except FileNotFoundError:
                pass

    if csv_active_models:
        metrics_to_export = [("f1_multiclass", "f1"), ("auc", "auc"), ("ap", "ap")]
        # metrics_to_export = [("f1_multiclass", "f1"), ("auc", "auc")]

        for metric_key, file_prefix in metrics_to_export:
            export_metric_csvs(
                active_models=csv_active_models,
                paths_dict=csv_paths_dict,
                target_order=TARGET_ORDER,
                display_map=DISPLAY_MAP,
                metric=metric_key,
                prefix=file_prefix,
                output_dir=csv_output_dir,
            )
    else:
        print("No valid models found to export CSVs.")
