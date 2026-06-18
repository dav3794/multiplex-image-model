from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


_YAML = YAML(typ="safe")


def load_generation_config(path: str) -> dict[str, Any]:
    """Load a generation config YAML.

    Returns a dict with top-level shared defaults and a 'generation' list
    of per-model entries.  Shared fields are merged into each entry before
    being passed to a model's generate() function.
    """
    with open(path) as f:
        raw = _YAML.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Generation config must be a mapping: {path}")

    shared = {k: v for k, v in raw.items() if k != "generation"}
    entries = raw.get("generation", [])
    merged = []
    for entry in entries:
        merged.append({**shared, **entry})
    return {"entries": merged}


@dataclass
class ClinicalDataConfig:
    path: str = ""


@dataclass
class DatasetConfig:
    features: list[str] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    variants: list[str] = field(
        default_factory=lambda: [
            "standard",
            "scaled",
            "pca50",
            "pca75",
            "pca99",
            "pca50w",
            "pca75w",
            "pca99w",
        ]
    )
    rare_class_threshold: float = 0.05
    seed: int = 42
    feature_prefix: str = "f_"


@dataclass
class CrossValidationConfig:
    n_folds: int = 10
    seed: int = 42


@dataclass
class ModelConfig:
    name: str
    format: str
    path: str
    panels: list[str]
    img_path_col: str
    variants: list[str] | None = None
    batch_agg: str = "mean"
    patch_agg: str = "mean"


@dataclass
class EvalConfig:
    output_dir: str = "./clinical/results"
    detailed_results_dir: str = "./clinical/results/detailed"
    clinical_data: ClinicalDataConfig = field(default_factory=ClinicalDataConfig)
    datasets: dict[str, DatasetConfig] = field(default_factory=dict)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )
    models: list[ModelConfig] = field(default_factory=list)


def load_evaluation_config(path: str) -> EvalConfig:
    """Load a structured evaluation config via OmegaConf merge."""
    from omegaconf import OmegaConf

    schema = OmegaConf.structured(EvalConfig)
    user = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, user)
    OmegaConf.resolve(merged)
    return merged
