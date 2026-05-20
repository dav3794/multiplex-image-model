"""Evaluation pipeline: load data, transform, compute metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.loaders import (
    PredictionSet,
    TargetSet,
    get_loader,
    load_ground_truth,
)
from src.metrics import (
    accumulate,
    center_crop,
    finalize_pearson,
    mse,
    pearson_r,
)
from src.models import PipelineConfig
from src.transforms import build_transform_pair, load_marker_stats


def resolve_target_sources(cfg: PipelineConfig) -> dict[str, TargetSet]:
    target_sets: dict[str, TargetSet] = {}
    for m in cfg.models:
        if m.format == "immuvis_npz" and m.name not in target_sets:
            target_sets[m.name] = load_ground_truth(m)

    result: dict[str, TargetSet] = {}
    for m in cfg.models:
        src_name = m.ground_truth.source_model or m.name
        if src_name not in target_sets:
            raise ValueError(
                f"Model '{m.name}' needs GT from '{src_name}' but that model is "
                f"not an immuvis_npz type or is missing from config."
            )
        result[m.name] = target_sets[src_name]
    return result


def run_pipeline(cfg: PipelineConfig) -> tuple[pd.DataFrame, dict, dict[str, int]]:
    eval_models = [m for m in cfg.models if m.evaluate]
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_sources = resolve_target_sources(cfg)
    print(f"Ground truth sources resolved for {len(gt_sources)} models.")

    marker_stats = None
    if cfg.marker_stats_csv and Path(cfg.marker_stats_csv).exists():
        marker_stats = load_marker_stats(cfg.marker_stats_csv)

    pred_sets: dict[str, PredictionSet] = {}
    for m in eval_models:
        loader = get_loader(m.format)
        pred_sets[m.name] = loader.load(m)
        print(f"Loaded: {m.name} — {len(pred_sets[m.name].pairs)} pairs")

    rows = []
    accums = {}
    model_pair_counts: dict[str, int] = {}

    for m in eval_models:
        print(f"\nEvaluating: {m.name}")
        ps = pred_sets[m.name]

        pairs = list(ps.pairs)
        if m.filter_pairs_to:
            ref_name = m.filter_pairs_to
            if ref_name not in pred_sets:
                raise ValueError(
                    f"Model '{m.name}' has filter_pairs_to='{ref_name}' "
                    f"but that model is not in the evaluation set."
                )
            ref_pairs = set(pred_sets[ref_name].pairs)
            own_set = set(pairs)
            dropped = own_set - ref_pairs
            if dropped:
                print(f"  Dropping {len(dropped)} pairs not in {ref_name}")
                for img, mrk in sorted(dropped):
                    print(f"    {img}  {mrk}")
            pairs = sorted(own_set & ref_pairs)

        gt_tfm = build_transform_pair(
            m.ground_truth.transform if m.ground_truth.source_model else m.prediction_transform,
            cfg.marker_stats_csv,
        ).target

        pred_tfm = build_transform_pair(
            m.prediction_transform, cfg.marker_stats_csv
        ).pred

        ts = gt_sources[m.name]

        pairs = [(img, mrk) for (img, mrk) in pairs if ts.has_image(img)]
        model_pair_counts[m.name] = len(pairs)

        accums[m.name] = {}

        for img_id, marker in tqdm(pairs, desc=f"  {m.name}"):
            pred_raw = ps.get_prediction(img_id, marker)
            if pred_raw is None or pred_raw.size == 0:
                continue

            raw_gt = ts.get_target(img_id, marker)
            if raw_gt.size == 0:
                continue

            gt_t = gt_tfm(raw_gt, marker)
            crop = center_crop(gt_t, cfg.crop_size)

            pred_t = pred_tfm(pred_raw, marker)

            mse_val = mse(pred_t, crop)
            r_val = pearson_r(pred_t, crop)

            rows.append({
                "model": m.name,
                "img": img_id,
                "marker": marker,
                "mse": mse_val,
                "pearson": r_val,
            })

            accumulate(accums[m.name], marker, pred_t, crop)

    df = pd.DataFrame(rows)
    return df, accums, model_pair_counts


def save_results(
    df: pd.DataFrame,
    accums: dict,
    common_pairs: list,
    output_dir: str,
    add_all_marker: bool = True,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pearson_rows = []
    for model_name, stats in accums.items():
        per_marker = finalize_pearson(stats)
        for marker, r in per_marker.items():
            pearson_rows.append({"model": model_name, "marker": marker, "pearson": r})

    pdf = pd.DataFrame(pearson_rows)
    pdf.to_csv(out / "per_marker_pearson.csv", index=False)

    df.to_csv(out / "per_point_results.csv", index=False)

    summary = df.groupby(["model", "marker"])[["mse", "pearson"]].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")

    if add_all_marker:
        all_df = df.copy()
        all_df["marker"] = "All"
        all_summary = all_df.groupby(["model", "marker"])[["mse", "pearson"]].agg(
            ["mean", "std"]
        )
        all_summary.to_csv(out / "summary_all.csv")
        combined = pd.concat(
            [summary.reset_index(), all_summary.reset_index()], ignore_index=True
        )
        combined.to_csv(out / "summary_with_all.csv", index=False)

    print(f"\nResults saved to {out.resolve()}/")
    print(f"  per_point_results.csv        — per (image, marker, model) metrics")
    print(f"  per_marker_pearson.csv       — per-marker aggregated Pearson")
    print(f"  summary.csv                  — per (model, marker) mean ± std")
