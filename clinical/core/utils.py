from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


_YAML = YAML(typ="safe")


def add_repo(repo_path: str) -> None:
    """Prepend a vendored repo to sys.path if not already there."""
    abspath = str(Path(repo_path).resolve())
    if abspath not in sys.path:
        sys.path.insert(0, abspath)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        payload = _YAML.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def normalize_optional(value: str | None) -> str | None:
    if value is None or value == "none":
        return None
    return value
