"""Data loaders for immuvis_npz and virtues_npy reconstruction formats."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models import ModelConfig


@dataclass
class PredictionSet:
    model_name: str
    pairs: list[tuple[str, str]]
    _get_prediction_fn: callable

    def get_prediction(self, image_id: str, marker: str) -> np.ndarray:
        return self._get_prediction_fn(image_id, marker)


@dataclass
class TargetSet:
    model_name: str
    image_ids: set[str]
    marker_names: list[str]
    _get_target_fn: callable

    def get_target(self, image_id: str, marker: str) -> np.ndarray:
        return self._get_target_fn(image_id, marker)

    def resolve_marker_index(self, marker: str) -> int:
        return self.marker_names.index(marker)

    def has_image(self, image_id: str) -> bool:
        return image_id in self.image_ids


class Loader(ABC):
    @abstractmethod
    def load(self, cfg: ModelConfig) -> PredictionSet:
        ...


class ImmuvisNpzLoader(Loader):
    def load(self, cfg: ModelConfig) -> PredictionSet:
        path = Path(cfg.path)
        dataset_filter = cfg.dataset_filter
        pairs: list[tuple[str, str]] = []
        id_to_fname: dict[str, str] = {}

        for fname in os.listdir(path):
            if not fname.endswith(".npz"):
                continue
            fpath = path / fname
            with np.load(fpath) as data:
                meta = json.loads(data["metadata"].item())
                if dataset_filter and meta.get("dataset_name") != dataset_filter:
                    continue
                img_id = meta["image_path"].split("/")[-1].split(".")[0]
                id_to_fname[img_id] = fname
                for m in data["marker_names"]:
                    m_str = str(m)
                    pairs.append((img_id, m_str))

        pairs = sorted(set(pairs))

        def get_pred(image_id: str, marker: str) -> np.ndarray:
            fname = id_to_fname.get(image_id)
            if fname is None:
                return np.array([])
            fpath = path / fname
            with np.load(fpath) as data:
                marker_names = list(data["marker_names"])
                idx = marker_names.index(marker)
                return data["recon"][idx].astype(np.float64)

        return PredictionSet(
            model_name=cfg.name,
            pairs=pairs,
            _get_prediction_fn=get_pred,
        )


class VirtuesNpyLoader(Loader):
    def load(self, cfg: ModelConfig) -> PredictionSet:
        path = Path(cfg.path)
        pairs: list[tuple[str, str]] = []

        def get_pred(image_id: str, marker: str) -> np.ndarray:
            return self._read_prediction(path, image_id, marker)

        for fname in os.listdir(path):
            if not fname.endswith("_recon.npy"):
                continue
            stem = fname.replace("_recon.npy", "")
            parts = stem.split("_", 1)
            if len(parts) < 2:
                continue
            img_id, marker = parts
            marker = self._normalise_marker(marker)
            pairs.append((img_id, marker))

        pairs = sorted(set(pairs))
        return PredictionSet(
            model_name=cfg.name,
            pairs=pairs,
            _get_prediction_fn=get_pred,
        )

    @staticmethod
    def _normalise_marker(marker: str) -> str:
        if marker == "Carbonic":
            return "Carbonic Anhydrase"
        if marker == "PARP":
            return "cl.PARP"
        if marker in ("H3", "Histone"):
            return "Histone H3"
        return marker.replace("_", " ")

    @staticmethod
    def _read_prediction(path: Path, image_id: str, marker: str) -> np.ndarray:
        for candidate in (marker, _denormalise_marker(marker)):
            fname = f"{image_id}_{candidate}_recon.npy".replace(" ", "_")
            fpath = path / fname
            if fpath.exists():
                return np.load(fpath).astype(np.float64)
        raise FileNotFoundError(
            f"Neither canonical nor short form found for "
            f"{image_id}_{marker}_recon.npy"
        )


def _denormalise_marker(marker: str) -> str:
    mapping = {
        "Carbonic Anhydrase": "Carbonic",
        "cl.PARP": "PARP",
        "Histone H3": "H3",
    }
    return mapping.get(marker, marker)


def load_ground_truth(cfg: ModelConfig) -> TargetSet:
    path = Path(cfg.path)
    dataset_filter = cfg.dataset_filter
    image_ids: set[str] = set()
    marker_names: list[str] = []
    id_to_fname: dict[str, str] = {}

    for fname in os.listdir(path):
        if not fname.endswith(".npz"):
            continue
        fpath = path / fname
        with np.load(fpath) as data:
            meta = json.loads(data["metadata"].item())
            if dataset_filter and meta.get("dataset_name") != dataset_filter:
                continue
            img_id = meta["image_path"].split("/")[-1].split(".")[0]
            id_to_fname[img_id] = fname
            image_ids.add(img_id)
            if not marker_names:
                marker_names = [str(m) for m in data["marker_names"]]

    def get_target(image_id: str, marker: str) -> np.ndarray:
        fname = id_to_fname.get(image_id)
        if fname is None:
            return np.array([])
        fpath = path / fname
        with np.load(fpath) as data:
            idx = list(data["marker_names"]).index(marker)
            return data["target"][idx].astype(np.float64)

    return TargetSet(
        model_name=cfg.name,
        image_ids=image_ids,
        marker_names=marker_names,
        _get_target_fn=get_target,
    )


LOADER_REGISTRY: dict[str, type[Loader]] = {
    "immuvis_npz": ImmuvisNpzLoader,
    "virtues_npy": VirtuesNpyLoader,
}


def get_loader(format: str) -> Loader:
    cls = LOADER_REGISTRY.get(format)
    if cls is None:
        raise ValueError(f"Unknown format '{format}'. Available: {list(LOADER_REGISTRY)}")
    return cls()
