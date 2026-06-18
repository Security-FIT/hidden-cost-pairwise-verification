#!/usr/bin/env python3
"""
Compile MLAAD metadata tables by joining the base train/dev/eval protocols with
per-model meta.csv files (fake/<lang>/<model>/meta.csv).

Each output CSV (e.g., train_meta.csv) contains the original protocol columns
plus every metadata field discovered for the referenced utterances.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_ATTRIBUTES = {
    "architecture": "unknown_architecture",
    "model_type": "type_unknown",
    "model_family": "family_unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment MLAAD protocols with meta.csv attributes.")
    parser.add_argument(
        "--protocol-root",
        type=Path,
        required=True,
        help="Directory containing train/dev/eval CSV metadata (e.g., mlaad4sourcetracing).",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="Root directory containing the fake/<lang>/<model>/meta.csv hierarchy. "
        "Defaults to the parent of --protocol-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store the augmented CSVs. Defaults to --protocol-root.",
    )
    parser.add_argument(
        "--path-column",
        default="path",
        help="Column in the base CSV containing relative waveform paths (default: path).",
    )
    parser.add_argument(
        "--splits",
        default="train,dev,eval",
        help="Comma-separated list of split names to process (default: train,dev,eval).",
    )
    parser.add_argument(
        "--model-attributes",
        type=Path,
        default=Path("configs/model_attributes.json"),
        help="Path to JSON mapping of model_name -> architecture/type/family metadata.",
    )
    parser.add_argument(
        "--meta-glob",
        default="fake/*/*/meta.csv",
        help="Glob (relative to --metadata-root) to discover per-model metadata files.",
    )
    return parser.parse_args()


def read_meta_file(meta_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            meta_path,
            sep="|",
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )
    except (pd.errors.ParserError, UnicodeDecodeError):
        return pd.read_csv(meta_path, on_bad_lines="skip")


def load_all_metadata(metadata_root: Path, glob_pattern: str) -> pd.DataFrame:
    meta_paths = sorted(metadata_root.glob(glob_pattern))
    if not meta_paths:
        raise FileNotFoundError(
            f"No metadata files matched pattern '{glob_pattern}' under {metadata_root}."
        )

    frames: List[pd.DataFrame] = []
    for meta_path in meta_paths:
        df = read_meta_file(meta_path)
        if "path" not in df.columns:
            print(f"  Warning: {meta_path} skipped (missing 'path' column).")
            continue
        df["normalized_path"] = df["path"].astype(str).str.lstrip("./")
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            if col in {"path", "normalized_path"}:
                continue
            rename_map[col] = f"meta_{col}"
        df = df.rename(columns=rename_map)
        df = df.drop(columns=["path"])
        df["meta_source_file"] = str(meta_path.relative_to(metadata_root))
        frames.append(df)

    if not frames:
        raise RuntimeError("No usable metadata rows were loaded.")
    return pd.concat(frames, ignore_index=True)


def load_model_attribute_map(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Model attribute file {path} not found.")
    with path.open() as handle:
        data = json.load(handle)
    out: Dict[str, Dict[str, str]] = {}
    for key, attrs in data.items():
        if not isinstance(attrs, dict):
            continue
        normalized = {
            "architecture": attrs.get("architecture") or DEFAULT_ATTRIBUTES["architecture"],
            "model_type": attrs.get("model_type") or DEFAULT_ATTRIBUTES["model_type"],
            "model_family": attrs.get("model_family") or DEFAULT_ATTRIBUTES["model_family"],
        }
        out[key] = normalized
    return out


def load_seen_partitions(protocol_root: Path) -> Dict[str, pd.DataFrame]:
    fine_dir = protocol_root / "fine"
    if not fine_dir.is_dir():
        return {}
    partitions: Dict[str, List[pd.DataFrame]] = {}
    for csv_path in fine_dir.glob("*.csv"):
        stem = csv_path.stem
        parts = stem.split("___")
        if len(parts) != 3:
            continue
        split_name, lang_part, model_part = parts
        if not lang_part.startswith("lang_") or not model_part.startswith("model_"):
            continue
        lang_status_raw = lang_part.replace("lang_", "")
        model_status_raw = model_part.replace("model_", "")
        status_map = {"seen": 1, "not_seen": 0}
        lang_status = status_map.get(lang_status_raw, lang_status_raw)
        model_status = status_map.get(model_status_raw, model_status_raw)
        df = pd.read_csv(csv_path)
        if "path" not in df.columns:
            continue
        subset = pd.DataFrame(
            {
                "normalized_path": df["path"].astype(str).str.lstrip("./"),
                "lang_seen": lang_status,
                "model_seen": model_status,
            }
        )
        partitions.setdefault(split_name, []).append(subset)
    lookup: Dict[str, pd.DataFrame] = {}
    for split_name, frames in partitions.items():
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset="normalized_path", keep="last")
        lookup[split_name] = merged
    return lookup


def augment_split(
    split_name: str,
    protocol_root: Path,
    output_dir: Path,
    path_column: str,
    meta_df: pd.DataFrame,
    seen_lookup: Dict[str, pd.DataFrame],
    model_attributes: Dict[str, Dict[str, str]],
    training_architectures: set[str],
    training_model_types: set[str],
    training_model_families: set[str],
) -> None:
    csv_path = protocol_root / f"{split_name}.csv"
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found, skipping split '{split_name}'.")
        return

    base_df = pd.read_csv(csv_path)
    if path_column not in base_df.columns:
        raise ValueError(f"{csv_path} missing required column '{path_column}'.")

    base_df["normalized_path"] = base_df[path_column].astype(str).str.lstrip("./")
    merged = base_df.merge(meta_df, on="normalized_path", how="left")

    extra = seen_lookup.get(split_name)
    if extra is not None:
        merged = merged.merge(extra, on="normalized_path", how="left")

    # Enrich with architecture/type/family tags derived from the original model_name.
    source_model = merged["model_name"].fillna("unknown")

    def lookup_attributes(name: str) -> Dict[str, str]:
        return model_attributes.get(name, DEFAULT_ATTRIBUTES)

    attributes = source_model.map(lookup_attributes)
    merged["model_architecture"] = attributes.map(lambda attr: attr["architecture"])
    merged["model_type"] = attributes.map(lambda attr: attr["model_type"])
    merged["model_family"] = attributes.map(lambda attr: attr["model_family"])

    if split_name in {"dev", "eval"}:
        merged["model_architecture_seen"] = merged["model_architecture"].isin(training_architectures).astype(int)
        merged["model_type_seen"] = merged["model_type"].isin(training_model_types).astype(int)
        merged["model_family_seen"] = merged["model_family"].isin(training_model_families).astype(int)

    missing = merged["meta_source_file"].isna().sum()
    if missing:
        print(f"  Warning: {missing:,} entries in {split_name}.csv lacked metadata matches.")

    merged = merged.drop(columns=["normalized_path"])
    out_path = output_dir / f"{split_name}_meta.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  Wrote {len(merged):,} rows with metadata to {out_path}")


def main() -> None:
    args = parse_args()
    protocol_root = args.protocol_root.expanduser().resolve()
    if not protocol_root.exists():
        raise FileNotFoundError(f"Protocol root {protocol_root} does not exist.")

    metadata_root = (
        args.metadata_root.expanduser().resolve()
        if args.metadata_root
        else protocol_root.parent
    )
    if not metadata_root.exists():
        raise FileNotFoundError(f"Metadata root {metadata_root} does not exist.")

    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else protocol_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_attribute_path = args.model_attributes.expanduser().resolve()
    model_attributes = load_model_attribute_map(model_attribute_path)
    print(f"Loaded {len(model_attributes)} model attribute entries from {model_attribute_path}")

    print(f"Loading metadata from {metadata_root}...")
    meta_df = load_all_metadata(metadata_root, args.meta_glob)
    print(f"  Loaded {len(meta_df):,} rows of metadata.")
    seen_lookup = load_seen_partitions(protocol_root)
    train_csv = protocol_root / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Training protocol {train_csv} not found.")
    train_df = pd.read_csv(train_csv)
    if "model_name" not in train_df.columns:
        raise ValueError(f"{train_csv} missing 'model_name' column.")
    training_models = set(train_df["model_name"].dropna().unique())
    training_architectures = {
        model_attributes.get(name, DEFAULT_ATTRIBUTES)["architecture"]
        for name in training_models
    }
    training_model_types = {
        model_attributes.get(name, DEFAULT_ATTRIBUTES)["model_type"]
        for name in training_models
    }
    training_model_families = {
        model_attributes.get(name, DEFAULT_ATTRIBUTES)["model_family"]
        for name in training_models
    }

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No splits specified via --splits.")

    for split in splits:
        print(f"Processing split '{split}'...")
        augment_split(
            split,
            protocol_root,
            output_dir,
            args.path_column,
            meta_df,
            seen_lookup,
            model_attributes,
            training_architectures,
            training_model_types,
            training_model_families,
        )

    print("Done.")


if __name__ == "__main__":
    main()
