#!/usr/bin/env python3
"""
Analyze EER errors to understand which models/pairs dominate the error budget.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

PATH_A_CANDIDATES = ("path_A", "pathA", "path_a", "path1", "path_ref", "path_reference")
PATH_B_CANDIDATES = ("path_B", "pathB", "path_b", "path2", "path_query", "path_test")
SCORE_CANDIDATES = ("score", "score_same", "score_same_model", "prob_same", "score_pair")
LABEL_CANDIDATES = ("label", "same_model", "is_same", "target", "y_true")
CLAIM_CANDIDATES = ("model_name_A", "model_nameA", "model_A", "claim_model", "claim_id")
QUERY_CANDIDATES = ("model_name_B", "model_nameB", "model_B", "query_model", "query_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EER error attribution.")
    parser.add_argument("--eval-scores", type=Path, required=True, help="Eval scored pairs CSV.")
    parser.add_argument("--score-column", help="Score column name (defaults to auto-detect).")
    parser.add_argument("--label-column", help="Label column name (defaults to auto-detect).")
    parser.add_argument("--pos-label", default="1", help="Positive label value (default: 1).")
    parser.add_argument("--pathA-column", help="Path A column name (defaults to auto-detect).")
    parser.add_argument("--pathB-column", help="Path B column name (defaults to auto-detect).")
    parser.add_argument("--claim-column", help="Claim/model column (defaults to auto-detect).")
    parser.add_argument("--query-column", help="Query/model column (defaults to auto-detect).")
    parser.add_argument(
        "--path-model-csv",
        type=Path,
        help="Optional CSV mapping paths to model names (columns: path, model_name).",
    )
    parser.add_argument(
        "--path-model-path-column",
        default="path",
        help="Path column in --path-model-csv (default: path).",
    )
    parser.add_argument(
        "--path-model-source-column",
        default="model_name",
        help="Model column in --path-model-csv (default: model_name).",
    )
    parser.add_argument(
        "--score-direction",
        choices=("high", "low"),
        default="high",
        help="Whether higher scores indicate positives (default: high).",
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.9,
        help="Quantile for tail errors based on margin (default: 0.9).",
    )
    parser.add_argument(
        "--centroid-similarity-csv",
        type=Path,
        help="Optional centroid similarity CSV (model_name_A, model_name_B, cosine_similarity).",
    )
    parser.add_argument(
        "--similarity-bins",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated similarity bin edges (default: 0.0,0.2,0.4,0.6,0.8,1.0).",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: project_root/tmp/eer_analysis).")
    return parser.parse_args()


def _default_output_dir(script_path: Path) -> Path:
    return script_path.resolve().parents[1] / "tmp" / "eer_analysis"


def _infer_column(df: pd.DataFrame, candidates: Sequence[str], provided: str | None, label: str) -> str:
    if provided:
        if provided in df.columns:
            return provided
        raise ValueError(f"{label} column '{provided}' not found in CSV.")
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {label} column (tried {', '.join(candidates)}).")


def _load_path_model_map(path_csv: Path, path_col: str, model_col: str) -> Dict[str, str]:
    df = pd.read_csv(path_csv)
    missing = {path_col, model_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path_csv} missing columns: {', '.join(sorted(missing))}")
    df = df.dropna(subset=[path_col, model_col])
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        mapping[str(row[path_col])] = str(row[model_col])
    return mapping


def _compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float]:
    if scores.size == 0:
        raise ValueError("No scores provided.")
    if labels.size != scores.size:
        raise ValueError("Score/label size mismatch.")
    pos_total = int(labels.sum())
    neg_total = int(labels.size - pos_total)
    if pos_total == 0 or neg_total == 0:
        raise ValueError("Need both positive and negative samples to compute EER.")

    order = np.argsort(scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    pos_cum = np.cumsum(labels_sorted)
    neg_cum = np.cumsum(1 - labels_sorted)

    unique_scores, first_idx = np.unique(scores_sorted, return_index=True)
    idxs = first_idx - 1
    pos_below = np.where(idxs >= 0, pos_cum[idxs], 0)
    neg_below = np.where(idxs >= 0, neg_cum[idxs], 0)

    fnr = pos_below / pos_total
    fpr = (neg_total - neg_below) / neg_total
    diff = np.abs(fpr - fnr)
    best = int(np.argmin(diff))
    thr = float(unique_scores[best])
    fpr_val = float(fpr[best])
    fnr_val = float(fnr[best])
    eer = 0.5 * (fpr_val + fnr_val)
    return thr, fpr_val, fnr_val, eer


def _parse_similarity_bins(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("Need at least two similarity bin edges.")
    bins = [float(p) for p in parts]
    if any(b2 <= b1 for b1, b2 in zip(bins, bins[1:])):
        raise ValueError("Similarity bins must be strictly increasing.")
    return bins


def _load_centroid_similarity(sim_csv: Path) -> Dict[Tuple[str, str], float]:
    df = pd.read_csv(sim_csv)
    required = {"model_name_A", "model_name_B", "cosine_similarity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{sim_csv} missing columns: {', '.join(sorted(missing))}")
    sim_map: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        a = str(row["model_name_A"])
        b = str(row["model_name_B"])
        key = tuple(sorted((a, b)))
        sim_map[key] = float(row["cosine_similarity"])
    return sim_map


def _build_pair_table(
    df: pd.DataFrame,
    claim_col: str,
    query_col: str,
    mask_total: np.ndarray,
    mask_error: np.ndarray,
    label: str,
) -> pd.DataFrame:
    total_counts = (
        df.loc[mask_total].groupby([claim_col, query_col]).size().rename("total_count")
    )
    err_counts = df.loc[mask_error].groupby([claim_col, query_col]).size().rename("error_count")
    table = pd.concat([total_counts, err_counts], axis=1).fillna(0)
    table["error_rate"] = table["error_count"] / table["total_count"].replace(0, np.nan)
    table = table.reset_index().rename(columns={claim_col: "model_name_A", query_col: "model_name_B"})
    table = table.sort_values(by=["error_count", "error_rate"], ascending=[False, False])
    table.insert(0, "error_type", label)
    return table


def _build_model_table(
    df: pd.DataFrame,
    model_col: str,
    mask_total: np.ndarray,
    mask_error: np.ndarray,
    label: str,
) -> pd.DataFrame:
    total_counts = df.loc[mask_total].groupby(model_col).size().rename("total_count")
    err_counts = df.loc[mask_error].groupby(model_col).size().rename("error_count")
    table = pd.concat([total_counts, err_counts], axis=1).fillna(0)
    table["error_rate"] = table["error_count"] / table["total_count"].replace(0, np.nan)
    table = table.reset_index().rename(columns={model_col: "model_name"})
    table = table.sort_values(by=["error_count", "error_rate"], ascending=[False, False])
    table.insert(0, "error_type", label)
    return table


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.eval_scores)

    score_col = _infer_column(df, SCORE_CANDIDATES, args.score_column, "score")
    label_col = _infer_column(df, LABEL_CANDIDATES, args.label_column, "label")
    path_a_col = _infer_column(df, PATH_A_CANDIDATES, args.pathA_column, "path_A")
    path_b_col = _infer_column(df, PATH_B_CANDIDATES, args.pathB_column, "path_B")
    claim_col = None
    query_col = None
    try:
        claim_col = _infer_column(df, CLAIM_CANDIDATES, args.claim_column, "claim/model")
        query_col = _infer_column(df, QUERY_CANDIDATES, args.query_column, "query/model")
    except ValueError:
        if args.path_model_csv:
            mapping = _load_path_model_map(
                args.path_model_csv, args.path_model_path_column, args.path_model_source_column
            )
            claim_values = []
            query_values = []
            missing = 0
            for path_a, path_b in zip(df[path_a_col].astype(str), df[path_b_col].astype(str)):
                model_a = mapping.get(path_a)
                model_b = mapping.get(path_b)
                if model_a is None:
                    missing += 1
                    model_a = path_a
                if model_b is None:
                    missing += 1
                    model_b = path_b
                claim_values.append(model_a)
                query_values.append(model_b)
            df["__claim_model"] = claim_values
            df["__query_model"] = query_values
            claim_col = "__claim_model"
            query_col = "__query_model"
            if missing:
                print(f"Warning: {missing} paths missing in --path-model-csv; using raw paths for those rows.")
        else:
            print("Warning: claim/query columns not found; falling back to path columns.")
            claim_col = path_a_col
            query_col = path_b_col

    scores = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    labels_raw = df[label_col].astype(str)
    pos_mask = labels_raw == str(args.pos_label)
    labels = pos_mask.astype(int).to_numpy()

    if np.any(~np.isfinite(scores)):
        raise ValueError("Non-finite scores found; clean or filter the input CSV.")

    scores_adj = scores if args.score_direction == "high" else -scores
    thr_adj, fpr_val, fnr_val, eer = _compute_eer(scores_adj, labels)
    thr = thr_adj if args.score_direction == "high" else -thr_adj

    pred_pos = scores_adj >= thr_adj
    fa_mask = (~pos_mask.to_numpy()) & pred_pos
    fr_mask = pos_mask.to_numpy() & (~pred_pos)
    margin = np.abs(scores_adj - thr_adj)

    output_dir = args.output_dir or _default_output_dir(Path(__file__))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "threshold": thr,
        "threshold_adjusted": thr_adj,
        "score_direction": args.score_direction,
        "eer": eer,
        "fpr_at_eer": fpr_val,
        "fnr_at_eer": fnr_val,
        "total_pairs": int(len(df)),
        "total_pos": int(pos_mask.sum()),
        "total_neg": int((~pos_mask).sum()),
        "false_accepts": int(fa_mask.sum()),
        "false_rejects": int(fr_mask.sum()),
    }
    summary_path = output_dir / "eer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    neg_mask = ~pos_mask.to_numpy()
    pos_mask_arr = pos_mask.to_numpy()

    fa_by_pair = _build_pair_table(df, claim_col, query_col, neg_mask, fa_mask, "FA")
    fr_by_model = _build_model_table(df, claim_col, pos_mask_arr, fr_mask, "FR")

    fa_by_pair.to_csv(output_dir / "eer_fa_by_pair.csv", index=False)
    fr_by_model.to_csv(output_dir / "eer_fr_by_model.csv", index=False)

    fa_by_claim = _build_model_table(df, claim_col, neg_mask, fa_mask, "FA_as_claim")
    fa_by_query = _build_model_table(df, query_col, neg_mask, fa_mask, "FA_as_query")
    fa_by_model = pd.concat([fa_by_claim, fa_by_query], ignore_index=True)
    fa_by_model.to_csv(output_dir / "eer_fa_by_model.csv", index=False)

    if fa_mask.any() or fr_mask.any():
        tail_q = float(args.tail_quantile)
        if not (0.0 < tail_q < 1.0):
            raise ValueError("--tail-quantile must be between 0 and 1.")
        fa_margin = margin[fa_mask]
        fr_margin = margin[fr_mask]
        fa_thr = float(np.quantile(fa_margin, tail_q)) if fa_margin.size else np.nan
        fr_thr = float(np.quantile(fr_margin, tail_q)) if fr_margin.size else np.nan

        fa_tail_mask = fa_mask & (margin >= fa_thr) if np.isfinite(fa_thr) else np.zeros_like(fa_mask)
        fr_tail_mask = fr_mask & (margin >= fr_thr) if np.isfinite(fr_thr) else np.zeros_like(fr_mask)

        fa_tail_by_pair = _build_pair_table(df, claim_col, query_col, fa_mask, fa_tail_mask, "FA_tail")
        fr_tail_by_model = _build_model_table(df, claim_col, fr_mask, fr_tail_mask, "FR_tail")

        fa_tail_by_pair.to_csv(output_dir / "eer_fa_tail_by_pair.csv", index=False)
        fr_tail_by_model.to_csv(output_dir / "eer_fr_tail_by_model.csv", index=False)

    if args.centroid_similarity_csv:
        sim_map = _load_centroid_similarity(args.centroid_similarity_csv)
        bins = _parse_similarity_bins(args.similarity_bins)
        neg_df = df.loc[neg_mask].copy()
        sims: List[float] = []
        for _, row in neg_df.iterrows():
            a = str(row[claim_col])
            b = str(row[query_col])
            key = tuple(sorted((a, b)))
            sims.append(sim_map.get(key, np.nan))
        neg_df["centroid_similarity"] = sims
        neg_df["is_fa"] = fa_mask[neg_mask]
        neg_df = neg_df[np.isfinite(neg_df["centroid_similarity"])]
        if not neg_df.empty:
            neg_df["sim_bin"] = pd.cut(neg_df["centroid_similarity"], bins=bins, include_lowest=True)
            bin_total = neg_df.groupby("sim_bin").size().rename("total_neg")
            bin_fa = neg_df.groupby("sim_bin")["is_fa"].sum().rename("fa_count")
            bin_table = pd.concat([bin_total, bin_fa], axis=1).fillna(0)
            bin_table["fa_rate"] = bin_table["fa_count"] / bin_table["total_neg"].replace(0, np.nan)
            bin_table = bin_table.reset_index()
            bin_table.to_csv(output_dir / "eer_fa_by_similarity_bin.csv", index=False)

    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
