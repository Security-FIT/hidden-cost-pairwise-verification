#!/usr/bin/env python3
"""
Combine intermediate pairs (positives + negatives) with hard-mined negatives.

Keeps only positives from the intermediate CSV and appends all pairs from the
hard-mined CSV (expected to be negatives). Optionally shuffles the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _require_columns(df: pd.DataFrame, path: Path, cols: set[str]) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge positives from an intermediate CSV with hard-mined negatives."
    )
    parser.add_argument("--intermediate-csv", type=Path, required=True, help="Intermediate pairs CSV.")
    parser.add_argument("--hardneg-csv", type=Path, required=True, help="Hard-mined negatives CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the combined CSV (same protocol columns).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional seed to shuffle the combined rows; leave unset to preserve order.",
    )
    args = parser.parse_args()

    required_cols = {"path_A", "model_name_A", "path_B", "model_name_B", "same_model"}

    pbar = tqdm(total=3, desc="Combining", unit="step")

    interm_df = pd.read_csv(args.intermediate_csv)
    _require_columns(interm_df, args.intermediate_csv, required_cols)
    pos_df = interm_df[interm_df["same_model"] == 1]
    if pos_df.empty:
        raise ValueError(f"No positives found in {args.intermediate_csv}; check input.")
    pbar.update(1)

    hardneg_df = pd.read_csv(args.hardneg_csv)
    _require_columns(hardneg_df, args.hardneg_csv, required_cols)
    pbar.update(1)

    combined = pd.concat([pos_df, hardneg_df], ignore_index=True)
    if args.shuffle_seed is not None:
        combined = combined.sample(frac=1.0, random_state=args.shuffle_seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    pbar.update(1)
    pbar.close()
    pos_count = (combined["same_model"] == 1).sum()
    neg_count = (combined["same_model"] == 0).sum()
    print(f"Wrote {args.output} (positives={pos_count:,}, negatives={neg_count:,}, total={len(combined):,})")


if __name__ == "__main__":
    main()
