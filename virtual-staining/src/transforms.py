"""Transforms mapping predictions and targets to a common evaluation space."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import TransformConfig


def load_marker_stats(csv_path: str | Path, dataset: str = "hn"):
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == dataset]
    return {
        row["marker"]: {"mean": row["mean"], "std": row["std"]}
        for _, row in df.iterrows()
    }


class PredictionTransform(ABC):
    @abstractmethod
    def __call__(self, pred: np.ndarray, marker: str) -> np.ndarray:
        ...


class TargetTransform(ABC):
    @abstractmethod
    def __call__(self, target: np.ndarray, marker: str) -> np.ndarray:
        ...


class NoOpTransform(PredictionTransform, TargetTransform):
    def __call__(self, arr: np.ndarray, marker: str = "") -> np.ndarray:
        return arr


class ScalingTransform(PredictionTransform, TargetTransform):
    def __init__(self, scale: float = 3.4):
        self.scale = scale

    def __call__(self, arr: np.ndarray, marker: str = "") -> np.ndarray:
        return arr * self.scale


class VirtuesZSPredictionTransform(PredictionTransform):
    def __init__(
        self,
        sinh_factor: float = 5.0,
        pre_scale: float | None = None,
        marker_stats: dict | None = None,
    ):
        self.sinh_factor = sinh_factor
        self.pre_scale = pre_scale
        self.marker_stats = marker_stats or {}

    def __call__(self, pred: np.ndarray, marker: str) -> np.ndarray:
        x = pred
        if self.pre_scale is not None:
            x = x * self.pre_scale
        x = np.log1p(np.sinh(x) * self.sinh_factor)
        if marker in self.marker_stats:
            mu = self.marker_stats[marker]["mean"]
            sd = self.marker_stats[marker]["std"]
            if sd > 0:
                x = (x - mu) / sd
        return x


class NoTransformTarget(TargetTransform):
    def __call__(self, target: np.ndarray, marker: str = "") -> np.ndarray:
        return target


@dataclass
class TransformPair:
    pred: PredictionTransform
    target: TargetTransform


def build_transform_pair(
    transform_cfg: TransformConfig,
    marker_stats_csv: str | None = None,
) -> TransformPair:
    ttype = transform_cfg.type
    params = transform_cfg.params

    if ttype == "none":
        return TransformPair(NoOpTransform(), NoOpTransform())

    if ttype == "immu_offset":
        scale = params.get("scale", 3.4)
        t = ScalingTransform(scale=scale)
        return TransformPair(t, t)

    if ttype == "virtues_zs":
        sinh_factor = params.get("sinh_factor", 5.0)
        pre_scale = params.get("pre_scale", None)
        marker_stats = None
        stats_path = params.get("marker_stats_csv") or marker_stats_csv
        if stats_path:
            marker_stats = load_marker_stats(stats_path)
        return TransformPair(
            pred=VirtuesZSPredictionTransform(
                sinh_factor=sinh_factor,
                pre_scale=pre_scale,
                marker_stats=marker_stats,
            ),
            target=NoTransformTarget(),
        )

    if ttype == "virtues_zs_op_pred":
        log1p_offset = params.get("log1p_offset", 5.4)
        sinh_factor = params.get("sinh_factor", 5.0)
        marker_stats = None
        stats_path = params.get("marker_stats_csv") or marker_stats_csv
        if stats_path:
            marker_stats = load_marker_stats(stats_path)
        return TransformPair(
            pred=VirtuesZSPredictionTransform(
                sinh_factor=sinh_factor,
                pre_scale=log1p_offset,
                marker_stats=marker_stats,
            ),
            target=NoTransformTarget(),
        )

    raise ValueError(f"Unknown transform type: {ttype}")
