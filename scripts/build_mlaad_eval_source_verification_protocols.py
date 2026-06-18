#!/usr/bin/env python3
"""
Build source-verification protocols for MLAAD eval (single-utterance).

Inputs:
  - eval_meta.csv with columns: utt_id, path, model_id
  - seed integer
  - output root dir

Outputs per scenario folder (R in {1, 5} by default):
  - references.csv
  - trials.csv or trials.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


def _stable_model_seed(seed: int, model_id: str) -> int:
    payload = f"{seed}:{model_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _load_eval_meta(
    meta_path: Path, utt_id_col: str, path_col: str, model_id_col: str
) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(f"eval_meta.csv not found: {meta_path}")
    df = pd.read_csv(meta_path)
    required = {utt_id_col, path_col, model_id_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{meta_path} missing required columns: {', '.join(sorted(missing))}"
        )
    df = df.dropna(subset=[utt_id_col, path_col, model_id_col]).copy()
    df["utt_id"] = df[utt_id_col].astype(str)
    df["path"] = df[path_col].astype(str)
    df["model_id"] = df[model_id_col].astype(str)
    df = df[["utt_id", "path", "model_id"]]
    before = len(df)
    df = df.drop_duplicates(subset=["utt_id", "path", "model_id"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Warning: dropped {dropped} duplicate rows from eval_meta.csv.")
    df = df.sort_values(["utt_id", "path"]).reset_index(drop=True)
    return df


def _select_references(
    df: pd.DataFrame, seed: int, r_value: int
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    rows = []
    ref_map: Dict[str, List[str]] = {}
    for model_id, group in df.groupby("model_id", sort=True):
        group = group.sort_values(["utt_id", "path"]).reset_index(drop=True)
        utt_ids = group["utt_id"].tolist()
        paths = group["path"].tolist()
        if len(utt_ids) < r_value:
            raise ValueError(
                f"Model {model_id} has {len(utt_ids)} utterances, but R={r_value} requested."
            )
        rng = random.Random(_stable_model_seed(seed, model_id))
        sampled_indices = rng.sample(range(len(utt_ids)), r_value)
        selected_utts = []
        for rank, idx in enumerate(sampled_indices, start=1):
            utt_id = utt_ids[idx]
            path = paths[idx]
            rows.append(
                {
                    "claim_id": model_id,
                    "ref_rank": rank,
                    "utt_id": utt_id,
                    "path": path,
                }
            )
            selected_utts.append(utt_id)
        ref_map[model_id] = selected_utts

    ref_df = pd.DataFrame(rows)
    return ref_df, ref_map


def _write_references(
    ref_df: pd.DataFrame, protocol_id: str, output_dir: Path
) -> Path:
    ref_df = ref_df.copy()
    ref_df.insert(0, "protocol_id", protocol_id)
    out_path = output_dir / "references.csv"
    ref_df.to_csv(out_path, index=False)
    return out_path


def _write_trials(
    df: pd.DataFrame,
    claims: Sequence[str],
    ref_map: Dict[str, List[str]],
    protocol_id: str,
    output_dir: Path,
    gzip_threshold: int,
    force_gzip: bool,
    r_value: int,
) -> Tuple[Path, Dict[str, int], int]:
    total_utt = len(df)
    total_claims = len(claims)
    expected_trials = total_claims * total_utt - total_claims * r_value
    use_gzip = force_gzip or (expected_trials >= gzip_threshold)
    out_path = output_dir / ("trials.csv.gz" if use_gzip else "trials.csv")

    positives_per_claim: Dict[str, int] = {claim: 0 for claim in claims}
    total_written = 0

    open_fn = gzip.open if use_gzip else open
    open_kwargs = (
        {"mode": "wt", "newline": "", "encoding": "utf-8"}
        if use_gzip
        else {"mode": "w", "newline": "", "encoding": "utf-8"}
    )
    with open_fn(out_path, **open_kwargs) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protocol_id",
                "claim_id",
                "query_utt_id",
                "query_path",
                "query_model_id",
                "label",
            ]
        )
        for claim_id in claims:
            ref_set = set(ref_map[claim_id])
            for _, row in df.iterrows():
                query_utt_id = row["utt_id"]
                if query_utt_id in ref_set:
                    continue
                query_model_id = row["model_id"]
                label = 1 if query_model_id == claim_id else 0
                if label == 1:
                    positives_per_claim[claim_id] += 1
                writer.writerow(
                    [
                        protocol_id,
                        claim_id,
                        query_utt_id,
                        row["path"],
                        query_model_id,
                        label,
                    ]
                )
                total_written += 1
    return out_path, positives_per_claim, total_written


def _run_sanity_checks(
    df: pd.DataFrame,
    claims: Sequence[str],
    ref_map: Dict[str, List[str]],
    positives_per_claim: Dict[str, int],
    total_written: int,
    r_value: int,
) -> None:
    for claim in claims:
        if len(ref_map.get(claim, [])) != r_value:
            raise ValueError(f"Claim {claim} has {len(ref_map.get(claim, []))} refs; expected {r_value}.")
    expected_trials = len(claims) * len(df) - len(claims) * r_value
    if total_written != expected_trials:
        raise ValueError(f"Total trials {total_written} != expected {expected_trials}.")
    counts = df["model_id"].value_counts()
    for claim in claims:
        expected_pos = int(counts.get(claim, 0)) - r_value
        if positives_per_claim.get(claim, 0) != expected_pos:
            raise ValueError(
                f"Claim {claim} positives {positives_per_claim.get(claim, 0)} != expected {expected_pos}."
            )


def build_protocols(
    eval_meta: Path,
    seed: int,
    output_root: Path,
    r_values: Sequence[int],
    gzip_threshold: int,
    force_gzip: bool,
    utt_id_col: str,
    path_col: str,
    model_id_col: str,
) -> None:
    df = _load_eval_meta(eval_meta, utt_id_col, path_col, model_id_col)
    claims = sorted(df["model_id"].unique())
    if not claims:
        raise ValueError("No model_id values found in eval_meta.csv.")

    for r_value in r_values:
        scenario_dir = output_root / "protocols" / f"mlaad_eval_R{r_value}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        protocol_id = f"mlaad_eval_R{r_value}_seed{seed}"

        ref_df, ref_map = _select_references(df, seed, r_value)
        _write_references(ref_df, protocol_id, scenario_dir)

        _, positives_per_claim, total_written = _write_trials(
            df,
            claims,
            ref_map,
            protocol_id,
            scenario_dir,
            gzip_threshold,
            force_gzip,
            r_value,
        )
        _run_sanity_checks(df, claims, ref_map, positives_per_claim, total_written, r_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MLAAD eval source-verification protocols (R=1 and R=5 by default)."
    )
    parser.add_argument("--eval-meta", required=True, type=Path, help="Path to eval_meta.csv.")
    parser.add_argument("--seed", required=True, type=int, help="Seed for deterministic sampling.")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output root directory to create protocols/ subdirectories.",
    )
    parser.add_argument(
        "--r-values",
        default="1,5",
        help="Comma-separated list of R values to build (default: 1,5).",
    )
    parser.add_argument(
        "--utt-id-col",
        default="utt_id",
        help="Column name to use as utt_id (default: utt_id).",
    )
    parser.add_argument(
        "--path-col",
        default="path",
        help="Column name to use as path (default: path).",
    )
    parser.add_argument(
        "--model-id-col",
        default="model_id",
        help="Column name to use as model_id (default: model_id).",
    )
    parser.add_argument(
        "--gzip-threshold",
        type=int,
        default=5_000_000,
        help="Use gzip when expected trials >= threshold (default: 5000000).",
    )
    parser.add_argument(
        "--force-gzip",
        action="store_true",
        help="Always gzip trials.csv output.",
    )
    args = parser.parse_args()

    r_values = [int(part) for part in args.r_values.split(",") if part.strip()]
    if not r_values:
        raise ValueError("At least one R value must be provided via --r-values.")

    build_protocols(
        eval_meta=args.eval_meta,
        seed=args.seed,
        output_root=args.output_root,
        r_values=r_values,
        gzip_threshold=args.gzip_threshold,
        force_gzip=args.force_gzip,
        utt_id_col=args.utt_id_col,
        path_col=args.path_col,
        model_id_col=args.model_id_col,
    )


if __name__ == "__main__":
    main()
