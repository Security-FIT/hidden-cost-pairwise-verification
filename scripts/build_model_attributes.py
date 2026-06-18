#!/usr/bin/env python3
"""
Utility to extract model attribute metadata from existing *_meta.csv files.

It consolidates all unique `meta_model_name` values and records the associated
architecture, model_type, and model_family information, producing a JSON file
that can be consumed by compile_mlaad_metadata.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_ARCHITECTURE = "unknown_architecture"
DEFAULT_MODEL_TYPE = "type_unknown"
DEFAULT_MODEL_FAMILY = "family_unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model_attributes.json from MLAAD *_meta.csv files.")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        required=True,
        help="Directory containing *_meta.csv files (e.g., /path/to/mlaad4sourcetracing).",
    )
    parser.add_argument(
        "--splits",
        default="dev_meta,eval_meta",
        help="Comma-separated list of *_meta CSV basenames to include (default: dev_meta,eval_meta).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/model_attributes.json"),
        help="Path to write the consolidated JSON (default: configs/model_attributes.json).",
    )
    return parser.parse_args()


def load_metadata_frames(root: Path, splits: List[str]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for split in splits:
        csv_path = root / f"{split}.csv"
        if not csv_path.exists():
            print(f"Warning: metadata file {csv_path} not found, skipping.")
            continue
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise FileNotFoundError("No metadata CSVs loaded; please check --metadata-root/--splits.")
    return frames

def normalize_attribute(value: str | float | None, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower().startswith("unknown"):
        return default
    return text


def build_attribute_map(frames: List[pd.DataFrame]) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for df in frames:
        for _, row in df.iterrows():
            model_name = row.get("meta_model_name")
            if pd.isna(model_name):
                continue
            model_key = str(model_name)
            candidate = {
                "architecture": normalize_attribute(row.get("meta_architecture"), DEFAULT_ARCHITECTURE),
                "model_type": normalize_attribute(row.get("model_type"), DEFAULT_MODEL_TYPE),
                "model_family": normalize_attribute(row.get("model_family"), DEFAULT_MODEL_FAMILY),
            }
            if model_key not in records:
                records[model_key] = candidate
            else:
                stored = records[model_key]
                for field, default in [
                    ("architecture", DEFAULT_ARCHITECTURE),
                    ("model_type", DEFAULT_MODEL_TYPE),
                    ("model_family", DEFAULT_MODEL_FAMILY),
                ]:
                    if stored.get(field) == default and candidate[field] != default:
                        stored[field] = candidate[field]
    return records


def main() -> None:
    args = parse_args()
    metadata_root = args.metadata_root.expanduser().resolve()
    if not metadata_root.exists():
        raise FileNotFoundError(f"Metadata root {metadata_root} not found.")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    frames = load_metadata_frames(metadata_root, splits)
    attr_map = build_attribute_map(frames)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(attr_map, handle, indent=2, sort_keys=True)
    print(f"Wrote {len(attr_map)} model entries to {output_path}")


if __name__ == "__main__":
    main()
