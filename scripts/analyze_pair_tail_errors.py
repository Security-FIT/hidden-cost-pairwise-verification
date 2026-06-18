#!/usr/bin/env python3
"""
Tail error attribution for pairwise source verification.

Inputs:
  - scored pairs CSV (eval) with paths + score + label
  - optional dev scored pairs CSV to set thresholds using negatives only
  - optional metadata CSV to map paths -> model_id + nuisance attributes

Outputs (per FPR):
  - FA/FR pair lists
  - per-claim FA/FR/TPR stats
  - top claim/impostor FA pairs
  - cohort same-rate stats (speaker/prompt/language/etc if available)
  - optional embedding neighborhood summary
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import det_curve

PATH_A_CANDIDATES = ("path_A", "pathA", "path_a", "path1", "path_ref", "path_reference")
PATH_B_CANDIDATES = ("path_B", "pathB", "path_b", "path2", "path_query", "path_test")
SCORE_CANDIDATES = ("score", "score_same", "score_same_model", "prob_same", "score_pair")
LABEL_CANDIDATES = ("label", "same_model", "is_same", "target", "y_true")
CLAIM_CANDIDATES = ("model_name_A", "model_nameA", "model_A", "claim_model", "claim_id")
QUERY_CANDIDATES = ("model_name_B", "model_nameB", "model_B", "query_model", "query_id")

META_PATH_CANDIDATES = ("path", "normalized_path", "relpath", "relative_path", "file", "audio_file_name")
META_MODEL_CANDIDATES = ("model_name", "model_source", "meta_model_name", "model_id", "source")

ATTRIBUTE_ALIASES: Mapping[str, Sequence[str]] = {
    "speaker": ("speaker", "speaker_id", "spk_id", "meta_speaker", "meta_reference_speaker"),
    "prompt": ("prompt", "prompt_id", "text_id", "meta_prompt_id", "meta_text_id", "meta_transcript_id", "meta_transcript"),
    "language": ("language", "meta_language"),
    "codec": ("codec", "meta_codec"),
    "post_processing": ("post_processing", "postprocess", "meta_post_processing", "meta_postprocess", "post_proc"),
}

DURATION_CANDIDATES = ("meta_duration", "duration", "dur", "length", "meta_length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tail errors (FA/FR) for pairwise source verification scores."
    )
    parser.add_argument("--eval-scores", type=Path, required=True, help="Eval scored pairs CSV.")
    parser.add_argument("--dev-scores", type=Path, help="Dev scored pairs CSV (negatives used for thresholds).")
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable score calibration (Platt). Uses --dev-scores unless --calibrate-from is provided.",
    )
    parser.add_argument(
        "--calibrate-from",
        type=Path,
        help="Optional scored pairs CSV to fit calibration (overrides --dev-scores).",
    )
    parser.add_argument("--score-column", help="Score column name (defaults to auto-detect).")
    parser.add_argument("--label-column", help="Label column name (defaults to auto-detect).")
    parser.add_argument("--pos-label", default="1", help="Positive label value (default: 1).")
    parser.add_argument("--pathA-column", help="Path A column name (defaults to auto-detect).")
    parser.add_argument("--pathB-column", help="Path B column name (defaults to auto-detect).")
    parser.add_argument("--claim-column", help="Claim/model column in scores CSV (defaults to auto-detect).")
    parser.add_argument("--query-column", help="Query/model column in scores CSV (defaults to auto-detect).")
    parser.add_argument("--metadata", type=Path, help="Metadata CSV with per-utterance attributes.")
    parser.add_argument("--metadata-path-column", help="Path column in metadata CSV (defaults to auto-detect).")
    parser.add_argument("--metadata-model-column", help="Model column in metadata CSV (defaults to auto-detect).")
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Prefix to strip from paths (repeatable).",
    )
    parser.add_argument(
        "--fpr",
        default="0.001,0.0001",
        help="Comma-separated FPRs (e.g., 0.001,0.0001).",
    )
    parser.add_argument(
        "--score-direction",
        choices=("high", "low"),
        default="high",
        help="Whether higher scores indicate positives (default: high).",
    )
    parser.add_argument(
        "--threshold-method",
        choices=("det", "neg-quantile"),
        default="det",
        help="How to pick thresholds for target FPRs (default: det).",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory to write CSV/JSON outputs.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K impostor pairs to list (default: 20).")
    parser.add_argument(
        "--attributes",
        default="auto",
        help="Comma-separated attribute names/columns to compare for FA same-rates; 'auto' picks known fields.",
    )
    parser.add_argument(
        "--duration-column",
        default="auto",
        help="Duration column name for FA distributions; 'auto' picks common names.",
    )
    parser.add_argument("--embedding-npz", type=Path, help="Optional .npz with embeddings + utt_ids.")
    parser.add_argument("--embedding-key", default="embeddings", help="Embeddings key in the NPZ (default: embeddings).")
    parser.add_argument("--embedding-ids-key", default="utt_ids", help="Utterance ids key in the NPZ (default: utt_ids).")
    parser.add_argument(
        "--embedding-metric",
        choices=("cosine", "l2"),
        default="cosine",
        help="Distance metric for embedding analysis (default: cosine).",
    )
    parser.add_argument("--embedding-k", type=int, default=5, help="k for local density (default: 5).")
    parser.add_argument(
        "--embedding-max-claim",
        type=int,
        default=500,
        help="Max claim samples for density stats (default: 500).",
    )
    parser.add_argument(
        "--embedding-max-refs",
        type=int,
        default=5000,
        help="Max reference embeddings per claim for min-distance search (default: 5000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42).")
    parser.add_argument("--embedding-details-out", type=Path, help="Optional CSV for per-FA embedding distances.")
    return parser.parse_args()


def _infer_column(df: pd.DataFrame, candidates: Sequence[str], provided: str | None, label: str) -> str:
    if provided:
        if provided in df.columns:
            return provided
        raise ValueError(f"{label} column '{provided}' not found in CSV.")
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {label} column (tried {', '.join(candidates)}).")


def _infer_optional_column(df: pd.DataFrame, candidates: Sequence[str], provided: str | None) -> str | None:
    if provided:
        if provided in df.columns:
            return provided
        raise ValueError(f"Column '{provided}' not found in CSV.")
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_fprs(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("No FPR values provided.")
    fprs: List[float] = []
    for part in parts:
        fpr = float(part)
        if not (0.0 < fpr < 1.0):
            raise ValueError(f"Invalid FPR {fpr}; must be between 0 and 1.")
        fprs.append(fpr)
    return fprs


def _normalize_path(path: str, prefixes: Sequence[str]) -> str:
    text = str(path).strip().replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    for prefix in prefixes:
        if not prefix:
            continue
        pref = prefix.replace("\\", "/")
        if text.startswith(pref):
            text = text[len(pref) :]
            if text.startswith("/"):
                text = text[1:]
    return text


def _quantile(scores: np.ndarray, q: float, mode: str) -> float:
    try:
        return float(np.quantile(scores, q, method=mode))
    except TypeError:
        return float(np.quantile(scores, q, interpolation=mode))


def _compute_threshold(neg_scores: np.ndarray, fpr: float, direction: str) -> Tuple[float, float]:
    if neg_scores.size == 0:
        raise ValueError("No negative scores available for threshold calibration.")
    if direction == "high":
        thr = _quantile(neg_scores, 1.0 - fpr, "higher")
        fpr_est = float(np.mean(neg_scores >= thr))
    else:
        thr = _quantile(neg_scores, fpr, "lower")
        fpr_est = float(np.mean(neg_scores <= thr))
    return thr, fpr_est


def _compute_det_threshold(
    scores: np.ndarray, labels: np.ndarray, fpr_target: float, direction: str
) -> Tuple[float, float, bool]:
    if scores.size == 0:
        raise ValueError("No scores available for DET thresholding.")
    labels_int = labels.astype(int)
    scores_use = -scores if direction == "low" else scores
    fpr, fnr, thresholds = det_curve(labels_int, scores_use, pos_label=1)
    finite = np.isfinite(fpr) & np.isfinite(thresholds)
    if not np.any(finite):
        raise ValueError("No finite DET points available for thresholding.")
    fpr_f = fpr[finite]
    thr_f = thresholds[finite]
    valid = np.where(fpr_f <= fpr_target)[0]
    if valid.size > 0:
        idx = int(valid[np.nanargmax(fpr_f[valid])])
        met = True
    else:
        idx = int(np.nanargmin(fpr_f))
        met = False
    thr_use = float(thr_f[idx])
    thr = -thr_use if direction == "low" else thr_use
    if direction == "high":
        pred_pos = scores >= thr
    else:
        pred_pos = scores <= thr
    neg_mask = labels_int != 1
    fpr_est = float(np.mean(pred_pos[neg_mask])) if np.any(neg_mask) else float("nan")
    return thr, fpr_est, met


def _summarize_numeric(series: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "count": float(values.count()),
        "min": float(values.min()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _format_fpr(fpr: float) -> str:
    label = f"{fpr:.6f}".rstrip("0").rstrip(".")
    return label.replace(".", "p")


def _load_scores(path: Path, args: argparse.Namespace, tag: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{tag} scores CSV not found: {path}")
    df = pd.read_csv(path)
    score_col = _infer_column(df, SCORE_CANDIDATES, args.score_column, "score")
    label_col = _infer_column(df, LABEL_CANDIDATES, args.label_column, "label")
    path_a_col = _infer_column(df, PATH_A_CANDIDATES, args.pathA_column, "pathA")
    path_b_col = _infer_column(df, PATH_B_CANDIDATES, args.pathB_column, "pathB")
    claim_col = _infer_optional_column(df, CLAIM_CANDIDATES, args.claim_column)
    query_col = _infer_optional_column(df, QUERY_CANDIDATES, args.query_column)

    keep_cols = [path_a_col, path_b_col, score_col, label_col]
    for col in (claim_col, query_col):
        if col and col not in keep_cols:
            keep_cols.append(col)

    out = df[keep_cols].copy()
    out = out.rename(
        columns={
            path_a_col: "pathA",
            path_b_col: "pathB",
            score_col: "score",
            label_col: "label_raw",
        }
    )
    if claim_col:
        out = out.rename(columns={claim_col: "claim_id"})
    if query_col:
        out = out.rename(columns={query_col: "query_id"})

    out["pathA"] = out["pathA"].apply(lambda x: _normalize_path(x, args.strip_prefix))
    out["pathB"] = out["pathB"].apply(lambda x: _normalize_path(x, args.strip_prefix))
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["score", "pathA", "pathB"]).reset_index(drop=True)
    label_num = pd.to_numeric(out["label_raw"], errors="coerce")
    pos_num = pd.to_numeric(pd.Series([args.pos_label]), errors="coerce").iloc[0]
    if not math.isnan(pos_num) and label_num.notna().any():
        out["is_pos"] = label_num == pos_num
    else:
        out["label_str"] = out["label_raw"].astype(str)
        out["is_pos"] = out["label_str"] == str(args.pos_label)
    return out


def _fit_logistic_calibration(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    x = scores.reshape(-1, 1)
    y = labels.astype(int)
    lr = LogisticRegression(class_weight="balanced", solver="lbfgs")
    lr.fit(x, y)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return a, b


def _apply_logistic_calibration(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * scores + b


def _load_metadata(
    path: Path, args: argparse.Namespace
) -> Tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    path_col = _infer_column(df, META_PATH_CANDIDATES, args.metadata_path_column, "metadata path")
    model_col = _infer_column(df, META_MODEL_CANDIDATES, args.metadata_model_column, "metadata model")
    df = df.copy()
    df[path_col] = df[path_col].apply(lambda x: _normalize_path(x, args.strip_prefix))
    df = df.drop_duplicates(subset=[path_col])
    return df, path_col, model_col


def _resolve_attributes(attr_arg: str, meta_columns: Sequence[str]) -> Dict[str, str]:
    available = set(meta_columns)
    resolved: Dict[str, str] = {}
    if attr_arg.strip().lower() == "auto":
        for name, candidates in ATTRIBUTE_ALIASES.items():
            for col in candidates:
                if col in available:
                    resolved[name] = col
                    break
        return resolved

    requested = [a.strip() for a in attr_arg.split(",") if a.strip()]
    for item in requested:
        if item in available:
            resolved[item] = item
            continue
        if item in ATTRIBUTE_ALIASES:
            for col in ATTRIBUTE_ALIASES[item]:
                if col in available:
                    resolved[item] = col
                    break
            continue
        matched = False
        for name, candidates in ATTRIBUTE_ALIASES.items():
            if item in candidates and item in available:
                resolved[name] = item
                matched = True
                break
        if not matched:
            raise ValueError(f"Attribute '{item}' not found in metadata columns.")
    return resolved


def _resolve_duration(duration_arg: str, meta_columns: Sequence[str]) -> str | None:
    if duration_arg.strip().lower() == "none":
        return None
    if duration_arg.strip().lower() != "auto":
        if duration_arg in meta_columns:
            return duration_arg
        raise ValueError(f"Duration column '{duration_arg}' not found in metadata.")
    for col in DURATION_CANDIDATES:
        if col in meta_columns:
            return col
    return None


def _attach_metadata(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    meta_path_col: str,
    columns: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    if not columns:
        return df
    meta_sub = meta_df[[meta_path_col] + list(columns)].copy()
    meta_sub = meta_sub.drop_duplicates(subset=[meta_path_col]).set_index(meta_path_col)
    out = df.copy()
    for col in columns:
        out[f"{prefix}{col}"] = meta_sub[col].reindex(out["pathA" if prefix == "A_" else "pathB"]).to_numpy()
    return out


def _same_rate(df: pd.DataFrame, col_a: str, col_b: str) -> Tuple[float | None, int]:
    valid = df[col_a].notna() & df[col_b].notna()
    if valid.sum() == 0:
        return None, 0
    rate = float((df.loc[valid, col_a] == df.loc[valid, col_b]).mean())
    return rate, int(valid.sum())


def _normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embedding encountered during normalization.")
    return emb / norms


def _min_distance(query: np.ndarray, refs: np.ndarray, metric: str) -> float:
    if refs.size == 0:
        return float("inf")
    if metric == "cosine":
        sims = refs @ query
        return float(1.0 - np.max(sims))
    diffs = refs - query
    return float(np.min(np.sum(diffs * diffs, axis=1)))


def _knn_stats(refs: np.ndarray, k: int, metric: str) -> Tuple[float, float] | None:
    n = refs.shape[0]
    if n <= k:
        return None
    if metric == "cosine":
        sims = refs @ refs.T
        dist = 1.0 - sims
    else:
        norms = np.sum(refs * refs, axis=1, keepdims=True)
        dist = norms + norms.T - 2.0 * (refs @ refs.T)
        dist = np.maximum(dist, 0.0)
    np.fill_diagonal(dist, np.inf)
    kth = np.partition(dist, k, axis=1)[:, k]
    return float(np.median(kth)), float(np.quantile(kth, 0.9))


def _embedding_analysis(
    fa_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    meta_path_col: str,
    meta_model_col: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], pd.DataFrame | None]:
    if args.embedding_npz is None:
        return {}, None
    if meta_df is None:
        print("  [warn] Skipping embedding analysis (metadata required to map paths to models).")
        return {}, None

    data = np.load(args.embedding_npz)
    if args.embedding_key not in data or args.embedding_ids_key not in data:
        raise KeyError(
            f"Embeddings NPZ missing keys {args.embedding_key} and/or {args.embedding_ids_key}."
        )
    emb = data[args.embedding_key].astype(np.float32)
    utt_ids = [str(x) for x in data[args.embedding_ids_key]]
    utt_ids = [_normalize_path(x, args.strip_prefix) for x in utt_ids]
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {emb.shape}.")

    emb_map = {utt_id: idx for idx, utt_id in enumerate(utt_ids)}
    if args.embedding_metric == "cosine":
        emb = _normalize_embeddings(emb)

    path_to_model = (
        meta_df[[meta_path_col, meta_model_col]]
        .dropna(subset=[meta_path_col, meta_model_col])
        .drop_duplicates(subset=[meta_path_col])
        .set_index(meta_path_col)[meta_model_col]
        .to_dict()
    )
    model_to_indices: Dict[str, List[int]] = defaultdict(list)
    for path, model in path_to_model.items():
        idx = emb_map.get(path)
        if idx is not None:
            model_to_indices[str(model)].append(idx)

    rng = np.random.default_rng(args.seed)
    claim_knn_stats: Dict[str, Tuple[float, float]] = {}
    for claim_id in fa_df["claim_id"].dropna().astype(str).unique():
        indices = model_to_indices.get(claim_id, [])
        if len(indices) <= args.embedding_k:
            continue
        sample = np.array(indices, dtype=int)
        if len(sample) > args.embedding_max_claim:
            sample = rng.choice(sample, size=args.embedding_max_claim, replace=False)
        stats = _knn_stats(emb[sample], args.embedding_k, args.embedding_metric)
        if stats:
            claim_knn_stats[claim_id] = stats

    all_indices = np.arange(len(emb))
    non_claim_cache: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    for _, row in fa_df.iterrows():
        claim_id = str(row.get("claim_id", ""))
        query_path = row["pathB"]
        q_idx = emb_map.get(query_path)
        if q_idx is None:
            continue
        claim_indices = model_to_indices.get(claim_id, [])
        if not claim_indices:
            continue
        claim_idx_arr = np.array(claim_indices, dtype=int)
        if len(claim_idx_arr) > args.embedding_max_refs:
            claim_idx_arr = rng.choice(claim_idx_arr, size=args.embedding_max_refs, replace=False)
        if claim_id in non_claim_cache:
            non_claim_idx = non_claim_cache[claim_id]
        else:
            mask = np.ones(len(emb), dtype=bool)
            mask[np.array(model_to_indices.get(claim_id, []), dtype=int)] = False
            non_claim_idx = all_indices[mask]
            if len(non_claim_idx) > args.embedding_max_refs:
                non_claim_idx = rng.choice(non_claim_idx, size=args.embedding_max_refs, replace=False)
            non_claim_cache[claim_id] = non_claim_idx

        q_vec = emb[q_idx]
        d_claim = _min_distance(q_vec, emb[claim_idx_arr], args.embedding_metric)
        d_other = _min_distance(q_vec, emb[non_claim_idx], args.embedding_metric)
        inlier = None
        if claim_id in claim_knn_stats:
            _, p90 = claim_knn_stats[claim_id]
            inlier = d_claim <= p90
        rows.append(
            {
                "pathA": row["pathA"],
                "pathB": row["pathB"],
                "claim_id": row.get("claim_id"),
                "query_id": row.get("query_id"),
                "d_claim_min": d_claim,
                "d_other_min": d_other,
                "closer_to_claim": d_claim < d_other,
                "inlier_vs_claim_p90": inlier,
            }
        )

    details = pd.DataFrame(rows)
    summary: Dict[str, object] = {
        "total_fa": int(len(fa_df)),
        "with_embeddings": int(len(details)),
        "closer_to_claim": int(details["closer_to_claim"].sum()) if not details.empty else 0,
        "closer_rate": float(details["closer_to_claim"].mean()) if not details.empty else 0.0,
    }
    if not details.empty:
        summary["d_claim_min"] = _summarize_numeric(details["d_claim_min"])
        summary["d_other_min"] = _summarize_numeric(details["d_other_min"])
    if "inlier_vs_claim_p90" in details:
        valid = details["inlier_vs_claim_p90"].notna()
        if valid.any():
            summary["inlier_vs_claim_p90_rate"] = float(details.loc[valid, "inlier_vs_claim_p90"].mean())
            summary["inlier_vs_claim_p90_count"] = int(valid.sum())
    return summary, details if not details.empty else None


def main() -> None:
    args = parse_args()
    fprs = _parse_fprs(args.fpr)

    eval_df = _load_scores(args.eval_scores, args, "Eval")
    dev_df = _load_scores(args.dev_scores, args, "Dev") if args.dev_scores else None
    calibration_requested = args.calibrate or args.calibrate_from is not None
    calibration_info = None
    if calibration_requested:
        calib_path = args.calibrate_from if args.calibrate_from else args.dev_scores
        if calib_path is None:
            raise ValueError("Calibration requested but no --dev-scores or --calibrate-from provided.")
        calib_df = _load_scores(Path(calib_path), args, "Calibration")
        a_cal, b_cal = _fit_logistic_calibration(
            calib_df["score"].to_numpy(dtype=float),
            calib_df["is_pos"].to_numpy(dtype=int),
        )
        eval_df["score_raw"] = eval_df["score"]
        eval_df["score"] = _apply_logistic_calibration(eval_df["score"].to_numpy(dtype=float), a_cal, b_cal)
        if dev_df is not None:
            dev_df["score_raw"] = dev_df["score"]
            dev_df["score"] = _apply_logistic_calibration(dev_df["score"].to_numpy(dtype=float), a_cal, b_cal)
        calibration_info = {
            "a": a_cal,
            "b": b_cal,
            "source": str(calib_path),
        }
        print(f"[info] Applied calibration from {calib_path} (a={a_cal:.6f}, b={b_cal:.6f}).")

    meta_df = None
    meta_path_col = ""
    meta_model_col = ""
    if args.metadata:
        meta_df, meta_path_col, meta_model_col = _load_metadata(args.metadata, args)

    if meta_df is not None:
        path_to_model = meta_df.set_index(meta_path_col)[meta_model_col].to_dict()
        if "claim_id" not in eval_df.columns:
            eval_df["claim_id"] = eval_df["pathA"].map(path_to_model)
        else:
            eval_df["claim_id"] = eval_df["claim_id"].fillna(eval_df["pathA"].map(path_to_model))
        if "query_id" not in eval_df.columns:
            eval_df["query_id"] = eval_df["pathB"].map(path_to_model)
        else:
            eval_df["query_id"] = eval_df["query_id"].fillna(eval_df["pathB"].map(path_to_model))
    else:
        if "claim_id" not in eval_df.columns or eval_df["claim_id"].isna().all():
            print("  [warn] claim_id missing and no metadata provided; per-claim stats will be skipped.")

    if meta_df is not None:
        attr_map = _resolve_attributes(args.attributes, meta_df.columns)
        duration_col = _resolve_duration(args.duration_column, meta_df.columns)
    else:
        attr_map = {}
        duration_col = None

    if meta_df is not None and attr_map:
        eval_df = _attach_metadata(eval_df, meta_df, meta_path_col, attr_map.values(), "A_")
        eval_df = _attach_metadata(eval_df, meta_df, meta_path_col, attr_map.values(), "B_")
    if meta_df is not None and duration_col:
        eval_df = _attach_metadata(eval_df, meta_df, meta_path_col, [duration_col], "A_")
        eval_df = _attach_metadata(eval_df, meta_df, meta_path_col, [duration_col], "B_")

    thresholds: List[Dict[str, object]] = []
    calib_df = dev_df if dev_df is not None else eval_df
    calib_scores = calib_df["score"].to_numpy(dtype=float)
    calib_labels = calib_df["is_pos"].to_numpy(dtype=int)
    neg_scores = calib_scores[calib_labels != 1]
    for fpr in fprs:
        if args.threshold_method == "det":
            thr, fpr_est, met_target = _compute_det_threshold(
                calib_scores, calib_labels, fpr, args.score_direction
            )
        else:
            thr, fpr_est = _compute_threshold(neg_scores, fpr, args.score_direction)
            met_target = True
        thresholds.append(
            {
                "fpr": fpr,
                "threshold": thr,
                "calibration_fpr": fpr_est,
                "calibration_negatives": int(len(neg_scores)),
                "calibration_split": "dev" if dev_df is not None else "eval",
                "threshold_method": args.threshold_method,
                "met_target": met_target,
            }
        )

    output_dir = None
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
        if calibration_info is not None:
            (output_dir / "calibration.json").write_text(json.dumps(calibration_info, indent=2))

    for entry in thresholds:
        fpr = entry["fpr"]
        thr = entry["threshold"]
        met_target = entry.get("met_target")
        threshold_method = entry.get("threshold_method")
        calibration_split = entry.get("calibration_split")
        calibration_fpr = entry.get("calibration_fpr")
        if args.score_direction == "high":
            pred_pos = eval_df["score"] >= thr
        else:
            pred_pos = eval_df["score"] <= thr
        fa_mask = (~eval_df["is_pos"]) & pred_pos
        fr_mask = eval_df["is_pos"] & (~pred_pos)
        fa_df = eval_df.loc[fa_mask].copy()
        fr_df = eval_df.loc[fr_mask].copy()

        print(f"\nFPR {fpr:.6f} -> threshold {thr:.6f}")
        if threshold_method:
            method_str = f"  Threshold method: {threshold_method}"
            if met_target is not None:
                method_str += f" (met target: {met_target})"
            print(method_str)
        if calibration_split is not None and calibration_fpr is not None:
            print(
                f"  Calibration FPR ({calibration_split}): {calibration_fpr*100:.4f}%"
            )
        print(f"  FA: {len(fa_df):,} / {int((~eval_df['is_pos']).sum()):,}")
        print(f"  FR: {len(fr_df):,} / {int(eval_df['is_pos'].sum()):,}")
        pos_total = int(eval_df["is_pos"].sum())
        if pos_total > 0:
            tpr = 100.0 * (pos_total - len(fr_df)) / pos_total
            print(f"  TPR: {tpr:.2f}%")
        else:
            print("  TPR: n/a (no positives)")

        if output_dir:
            slug = _format_fpr(fpr)
            fa_df.to_csv(output_dir / f"fa_pairs_fpr_{slug}.csv", index=False)
            fr_df.to_csv(output_dir / f"fr_pairs_fpr_{slug}.csv", index=False)

        if "claim_id" in eval_df.columns and eval_df["claim_id"].notna().any():
            claim_stats = []
            pos_df = eval_df[eval_df["is_pos"]]
            pos_by_claim = pos_df.groupby("claim_id", dropna=True)
            neg_by_claim = eval_df[~eval_df["is_pos"]].groupby("claim_id", dropna=True)
            fa_by_claim = fa_df.groupby("claim_id", dropna=True)
            fr_by_claim = fr_df.groupby("claim_id", dropna=True)
            for claim_id, pos_group in pos_by_claim:
                total_pos = len(pos_group)
                fr_count = len(fr_by_claim.get_group(claim_id)) if claim_id in fr_by_claim.groups else 0
                tpr = (total_pos - fr_count) / total_pos if total_pos else math.nan
                fa_count = len(fa_by_claim.get_group(claim_id)) if claim_id in fa_by_claim.groups else 0
                neg_total = len(neg_by_claim.get_group(claim_id)) if claim_id in neg_by_claim.groups else 0
                claim_stats.append(
                    {
                        "claim_id": claim_id,
                        "fa_count": fa_count,
                        "fr_count": fr_count,
                        "tpr": tpr,
                        "pos_total": total_pos,
                        "neg_total": neg_total,
                    }
                )
            claim_stats_df = pd.DataFrame(claim_stats).sort_values(
                by=["fa_count", "fr_count"], ascending=False
            )
            if output_dir:
                claim_stats_df.to_csv(output_dir / f"claim_stats_fpr_{_format_fpr(fpr)}.csv", index=False)

            if "query_id" in eval_df.columns and fa_df["query_id"].notna().any():
                fa_pairs = fa_df[fa_df["claim_id"].notna() & fa_df["query_id"].notna()]
                fa_pairs = fa_pairs[fa_pairs["claim_id"] != fa_pairs["query_id"]]
                pair_counts = (
                    fa_pairs.groupby(["claim_id", "query_id"])
                    .size()
                    .reset_index(name="fa_count")
                    .sort_values("fa_count", ascending=False)
                )
                top_pairs = pair_counts.head(args.top_k)
                if output_dir:
                    top_pairs.to_csv(
                        output_dir / f"impostor_pairs_fpr_{_format_fpr(fpr)}.csv", index=False
                    )
                print("  Top impostor pairs:")
                for _, row in top_pairs.iterrows():
                    print(f"    {row['claim_id']} x {row['query_id']}: {row['fa_count']}")

        if attr_map:
            attr_rows = []
            neg_df = eval_df[~eval_df["is_pos"]]
            for name, col in attr_map.items():
                fa_rate, fa_n = _same_rate(fa_df, f"A_{col}", f"B_{col}")
                neg_rate, neg_n = _same_rate(neg_df, f"A_{col}", f"B_{col}")
                lift = (fa_rate / neg_rate) if (fa_rate is not None and neg_rate) else None
                attr_rows.append(
                    {
                        "attribute": name,
                        "column": col,
                        "fa_same_rate": fa_rate,
                        "fa_pairs": fa_n,
                        "neg_same_rate": neg_rate,
                        "neg_pairs": neg_n,
                        "lift": lift,
                    }
                )
            attr_df = pd.DataFrame(attr_rows)
            if output_dir:
                attr_df.to_csv(
                    output_dir / f"attribute_same_rates_fpr_{_format_fpr(fpr)}.csv", index=False
                )

        if duration_col:
            dur_a = f"A_{duration_col}"
            dur_b = f"B_{duration_col}"
            if dur_a in fa_df.columns and dur_b in fa_df.columns:
                dur_stats = {
                    "duration_A": _summarize_numeric(fa_df[dur_a]),
                    "duration_B": _summarize_numeric(fa_df[dur_b]),
                    "duration_abs_diff": _summarize_numeric(
                        (pd.to_numeric(fa_df[dur_a], errors="coerce") - pd.to_numeric(fa_df[dur_b], errors="coerce")).abs()
                    ),
                }
                if output_dir:
                    (output_dir / f"duration_stats_fpr_{_format_fpr(fpr)}.json").write_text(
                        json.dumps(dur_stats, indent=2)
                    )

        if args.embedding_npz:
            if "claim_id" not in fa_df.columns:
                print("  [warn] Skipping embedding analysis (claim_id missing).")
                continue
            summary, details = _embedding_analysis(fa_df, meta_df, meta_path_col, meta_model_col, args)
            if summary:
                if output_dir:
                    (output_dir / f"embedding_summary_fpr_{_format_fpr(fpr)}.json").write_text(
                        json.dumps(summary, indent=2)
                    )
                print(f"  Embedding FA summary: {summary}")
            if details is not None and args.embedding_details_out:
                details.to_csv(args.embedding_details_out, index=False)


if __name__ == "__main__":
    main()
