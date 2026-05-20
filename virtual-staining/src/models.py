from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

TransformType = Literal[
    "immu_offset",
    "virtues_zs",
    "virtues_zs_op_pred",
    "none",
]


@dataclass
class TransformConfig:
    type: TransformType = "immu_offset"
    params: dict = field(default_factory=dict)


@dataclass
class GroundTruthConfig:
    source_model: str | None = None
    transform: TransformConfig = field(default_factory=TransformConfig)


@dataclass
class ModelConfig:
    name: str
    path: str
    format: Literal["immuvis_npz", "virtues_npy"] = "immuvis_npz"
    dataset_filter: str | None = "hn"
    display_name: str | None = None
    evaluate: bool = True
    filter_pairs_to: str | None = None
    prediction_transform: TransformConfig = field(default_factory=TransformConfig)
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)


@dataclass
class PipelineConfig:
    models: list[ModelConfig] = field(default_factory=list)
    marker_stats_csv: str | None = None
    crop_size: tuple[int, int] = (128, 128)
    output_dir: str = "results"
    plot: bool = False
    max_plots: int = 5


def load_config(path: str) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        print(f"Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = []
    for m in raw.get("models", []):
        gt = m.get("ground_truth", {}) or {}
        pred_t = m.get("prediction_transform", {}) or {}
        gt_t = gt.get("transform", {}) or {}

        models.append(
            ModelConfig(
                name=m["name"],
                path=m["path"],
                format=m.get("format", "immuvis_npz"),
                dataset_filter=m.get("dataset_filter") if "dataset_filter" in m else "hn",
                display_name=m.get("display_name"),
                evaluate=m.get("evaluate", True),
                filter_pairs_to=m.get("filter_pairs_to"),
                prediction_transform=TransformConfig(
                    type=pred_t.get("type", "immu_offset"),
                    params=pred_t.get("params", {}),
                ),
                ground_truth=GroundTruthConfig(
                    source_model=gt.get("source_model"),
                    transform=TransformConfig(
                        type=gt_t.get("type", "immu_offset"),
                        params=gt_t.get("params", {}),
                    ),
                ),
            )
        )

    crop = raw.get("crop_size", [128, 128])
    return PipelineConfig(
        models=models,
        marker_stats_csv=raw.get("marker_stats_csv"),
        crop_size=tuple(crop),
        output_dir=raw.get("output_dir", "results"),
        plot=raw.get("plot", False),
        max_plots=raw.get("max_plots", 5),
    )
