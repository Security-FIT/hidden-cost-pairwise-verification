#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def _infer_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _normalize_path(value: str) -> str:
    return str(value).strip().replace("\\", "/")


def _extract_source_model_from_path(path: str, include_lang: bool) -> str:
    """
    Default MLAAD path layout:
      ./fake/<lang>/<model>/<file>.wav
      ./real/<lang>/<model>/<file>.wav

    If that layout is not found, fall back to using the parent directory name.
    """
    p = _normalize_path(path).lstrip("./")
    parts = [x for x in p.split("/") if x]

    for token in ("fake", "real", "bonafide", "bona", "bonaf"):
        if token in parts:
            idx = parts.index(token)
            if idx + 2 < len(parts):
                lang = parts[idx + 1]
                model = parts[idx + 2]
                return f"{lang}/{model}" if include_lang else model

    if len(parts) >= 2:
        return parts[-2]
    return "UNKNOWN"


@dataclass(frozen=True)
class ScoreSchema:
    path_a: str
    path_b: str
    score: str
    label: str
    scenario_group: str | None


def _load_scores(path: Path) -> tuple[pd.DataFrame, ScoreSchema]:
    df = pd.read_csv(path)

    path_a = _infer_column(df, ["pathA", "path_a", "path_A", "hyp_wav", "path1", "path_1"])
    path_b = _infer_column(df, ["pathB", "path_b", "path_B", "trial_wav", "path2", "path_2"])
    score = _infer_column(df, ["score", "posterior", "prob", "p_same"])
    label = _infer_column(df, ["label", "same_model", "y", "target"])
    scenario_group = _infer_column(df, ["scenario_group", "group", "split"])

    missing: list[str] = []
    if path_a is None:
        missing.append("pathA")
    if path_b is None:
        missing.append("pathB")
    if score is None:
        missing.append("score")
    if label is None:
        missing.append("label")
    if missing:
        raise ValueError(
            f"Unsupported scores file schema: missing columns {', '.join(missing)}. "
            f"Present columns: {', '.join(df.columns)}"
        )

    schema = ScoreSchema(
        path_a=path_a,
        path_b=path_b,
        score=score,
        label=label,
        scenario_group=scenario_group,
    )

    keep = [schema.path_a, schema.path_b, schema.score, schema.label]
    if schema.scenario_group:
        keep.append(schema.scenario_group)
    df = df[keep].copy()

    df[schema.score] = pd.to_numeric(df[schema.score], errors="coerce")
    df[schema.label] = pd.to_numeric(df[schema.label], errors="coerce").astype("Int64")
    df = df.dropna(subset=[schema.score, schema.label])
    df[schema.label] = df[schema.label].astype(int)

    return df, schema


def _quantile_or_nan(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(series.quantile(q))


def analyze_pairs(
    df: pd.DataFrame,
    schema: ScoreSchema,
    include_lang: bool,
    threshold: float,
) -> pd.DataFrame:
    extractor: Callable[[str], str] = lambda p: _extract_source_model_from_path(p, include_lang=include_lang)

    model_a = df[schema.path_a].map(extractor)
    model_b = df[schema.path_b].map(extractor)
    m12 = np.sort(np.vstack([model_a.to_numpy(), model_b.to_numpy()]).T, axis=1)
    df = df.assign(model_1=m12[:, 0], model_2=m12[:, 1])

    score_col = schema.score
    label_col = schema.label

    neg = df[df[label_col] == 0]
    pos = df[df[label_col] == 1]

    group_cols = ["model_1", "model_2"]
    summary = df.groupby(group_cols, sort=False).size().rename("n_total").to_frame()
    summary["n_neg"] = neg.groupby(group_cols, sort=False).size()
    summary["n_pos"] = pos.groupby(group_cols, sort=False).size()
    summary[["n_neg", "n_pos"]] = summary[["n_neg", "n_pos"]].fillna(0).astype(int)

    if not neg.empty:
        neg_g = neg.groupby(group_cols, sort=False)[score_col]
        summary["neg_mean"] = neg_g.mean()
        summary["neg_p50"] = neg_g.median()
        summary["neg_p90"] = neg_g.apply(lambda s: _quantile_or_nan(s, 0.90))
        summary["neg_p95"] = neg_g.apply(lambda s: _quantile_or_nan(s, 0.95))
        summary["neg_p99"] = neg_g.apply(lambda s: _quantile_or_nan(s, 0.99))
        summary[f"neg_fa@{threshold:g}"] = (neg[score_col] >= threshold).groupby(
            [neg["model_1"], neg["model_2"]], sort=False
        ).mean()

    if not pos.empty:
        pos_g = pos.groupby(group_cols, sort=False)[score_col]
        summary["pos_mean"] = pos_g.mean()
        summary["pos_p50"] = pos_g.median()
        summary["pos_p10"] = pos_g.apply(lambda s: _quantile_or_nan(s, 0.10))
        summary["pos_p05"] = pos_g.apply(lambda s: _quantile_or_nan(s, 0.05))
        summary[f"pos_fr@{threshold:g}"] = (pos[score_col] < threshold).groupby(
            [pos["model_1"], pos["model_2"]], sort=False
        ).mean()

    if schema.scenario_group and schema.scenario_group in df.columns:
        scenario_counts = (
            df.groupby(group_cols + [schema.scenario_group], sort=False)
            .size()
            .rename("n")
            .reset_index()
        )
        top_scenario = (
            scenario_counts.sort_values(["model_1", "model_2", "n"], ascending=[True, True, False])
            .drop_duplicates(subset=group_cols, keep="first")
            .set_index(group_cols)[schema.scenario_group]
        )
        summary["top_scenario_group"] = top_scenario

    summary = summary.reset_index()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an eval scores.csv and rank unordered model pairs by how confusing they are.\n"
            "Unordered means (A,B) is treated the same as (B,A)."
        )
    )
    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="Path to scores CSV (e.g. eval_runs/.../scores.csv).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output CSV path for the aggregated per-(model_1,model_2) table.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="How many top confusing pairs to print (default: 50).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold in score domain (default: 0.5).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="neg_fa",
        choices=("neg_fa", "neg_p95", "neg_mean", "pos_fr", "pos_p05"),
        help=(
            "How to rank pairs. "
            "neg_fa = false-accept rate on negative (different-model) pairs at --threshold. "
            "pos_fr = false-reject rate on positive (same-model) pairs at --threshold."
        ),
    )
    parser.add_argument(
        "--include-lang",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use <lang>/<model> as the source model identifier (default: model only).",
    )
    parser.add_argument(
        "--exclude-same",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude model_1==model_2 (default: True; only compare different models).",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores).expanduser()
    df, schema = _load_scores(scores_path)
    summary = analyze_pairs(
        df,
        schema,
        include_lang=bool(args.include_lang),
        threshold=float(args.threshold),
    )

    if args.exclude_same:
        summary = summary[summary["model_1"] != summary["model_2"]].copy()

    metric = args.metric
    thr_key_neg = f"neg_fa@{float(args.threshold):g}"
    thr_key_pos = f"pos_fr@{float(args.threshold):g}"
    metric_col = {
        "neg_fa": thr_key_neg,
        "neg_p95": "neg_p95",
        "neg_mean": "neg_mean",
        "pos_fr": thr_key_pos,
        "pos_p05": "pos_p05",
    }[metric]

    if metric_col not in summary.columns:
        raise ValueError(
            f"Metric '{metric}' requires column '{metric_col}', but it is missing. "
            "This usually means the scores file contains no matching label rows for that metric."
        )

    summary = summary.sort_values(
        by=[metric_col, "n_total"],
        ascending=[False, False],
        na_position="last",
    )

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"Wrote: {out_path}")

    cols = ["model_1", "model_2", "n_total", "n_neg", "n_pos", metric_col]
    optional = [
        "neg_mean",
        "neg_p95",
        "neg_p99",
        "pos_mean",
        "pos_p05",
        "top_scenario_group",
    ]
    for col in optional:
        if col in summary.columns and col not in cols:
            cols.append(col)

    top = summary.head(int(args.top))[cols]
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 120):
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()

