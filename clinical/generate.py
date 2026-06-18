#!/usr/bin/env python3
"""Stage 1: Generate patch embeddings for clinical prediction.

Usage:
  python generate.py --config configs/generate/603.yaml
"""

from __future__ import annotations

import argparse

from core.registry import get_generate_func
from core.config import load_generation_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate clinical patch embeddings (leave-one-out).",
    )
    p.add_argument("--config", required=True, help="Path to YAML generation config.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_generation_config(args.config)

    for entry in cfg["entries"]:
        family = entry.get("family") or entry.get("type")
        name = entry.get("name", "unnamed")

        if family is None:
            print(f"  Skipping '{name}': no 'family' or 'type' field.")
            continue

        try:
            generate_func = get_generate_func(family)
        except KeyError as e:
            print(f"  Skipping '{name}': {e}")
            continue

        print(f"\n{'=' * 72}")
        print(f"Generating embeddings for: {name} (family: {family})")
        print(f"{'=' * 72}")
        generate_func(entry)
        print(f"Finished: {name}\n")


if __name__ == "__main__":
    main()
