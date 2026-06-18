#!/usr/bin/env python3
"""
Plot score distributions as CDFs (target vs non-target) for one or more scored-pairs CSV files.
Multiple CSVs are rendered as stacked subplots with shared axes for easy comparison.

Supports:
  - Generic score columns: score / score_raw
  - Cosine baseline outputs: cos_sim_raw / cos_dist_raw
  - Optional FFCosine "reverse" mapping from logit -> cosine similarity using checkpoint scale/bias

This is intended for diagnosing:
  - Saturation: probability/logit saturates but implied cosine similarity is moderate
  - Collapse: implied cosine similarity itself is ~1.0 for impostors
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FFCosineParams:
    scale: float
    bias: float


@dataclass
class ScoreTable:
    path: Path
    label: str
    df: pd.DataFrame
    label_col: str
    target_mask: np.ndarray
    non_mask: np.ndarray


def _load_ffcosine_params(checkpoint: Path) -> FFCosineParams:
    import torch

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")

    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(state)}")

    if "scale" not in state or "bias" not in state:
        raise ValueError(
            "Checkpoint does not contain FFCosine scale/bias. "
            f"Found keys: {', '.join(list(state)[:20])}"
        )

    scale = float(state["scale"].detach().cpu().item())
    bias = float(state["bias"].detach().cpu().item())
    return FFCosineParams(scale=scale, bias=bias)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _save_stacked_cdf(
    out_path: Path,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    bins: int = 120,
    width: float = 12.0,
    aspect: float = 4.0,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import ticker as mticker

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not series:
        return

    base_fontsize = 14
    with plt.rc_context(
        {
            "font.size": base_fontsize,
            "axes.labelsize": base_fontsize + 2,
            "xtick.labelsize": base_fontsize,
            "ytick.labelsize": base_fontsize,
            "legend.fontsize": base_fontsize,
        }
    ):
        def _format_symlog_tick(value: float, _pos: int) -> str:
            if abs(value) < 1e-12:
                return "0"
            text = f"{value:.6f}".rstrip("0").rstrip(".")
            return text if text else "0"

        all_values: list[np.ndarray] = []
        for _, target, non in series:
            if target.size:
                all_values.append(target)
            if non.size:
                all_values.append(non)

        if not all_values:
            return

        all_concat = np.concatenate(all_values, axis=0)
        all_concat = all_concat[np.isfinite(all_concat)]
        if all_concat.size == 0:
            return

        vmin = float(np.min(all_concat))
        vmax = float(np.max(all_concat))
        if vmin == vmax:
            pad = 1e-6 if vmin == 0 else abs(vmin) * 1e-3
            vmin -= pad
            vmax += pad
        n_points = max(bins * 4, 200)
        span = vmax - vmin
        pad = span * 0.01 if span > 0 else 1e-6
        x_min = vmin - pad
        x_max = vmax + pad
        abs_max = max(abs(x_min), abs(x_max))
        linthresh = max(abs_max * 0.1, 1e-6)
        linscale = 2.0
        x_grid = np.linspace(x_min, x_max, n_points)

        height_per = width / aspect
        fig_height = height_per * len(series)
        fig, axes = plt.subplots(
            nrows=len(series),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(width, fig_height),
        )
        if len(series) == 1:
            axes = [axes]

        def _cdf_at(sorted_vals: np.ndarray, ranks: np.ndarray, x_val: float) -> float:
            if sorted_vals.size == 0:
                return float("nan")
            return float(np.interp(x_val, sorted_vals, ranks, left=0.0, right=1.0))

        def _label_y(opp_y: float, own_y: float, base_offset: float = 0.06) -> float:
            y_base = opp_y
            if np.isfinite(own_y):
                y_base = max(y_base, own_y)
            if np.isfinite(opp_y):
                y_base = max(y_base, opp_y)
            y_val = y_base + base_offset
            return min(0.995, max(0.005, y_val))

        for ax, (_, target, non) in zip(axes, series):
            non_sorted = (
                np.sort(non[np.isfinite(non)]) if non.size else np.asarray([], dtype=float)
            )
            target_sorted = (
                np.sort(target[np.isfinite(target)]) if target.size else np.asarray([], dtype=float)
            )
            non_ranks = (
                np.arange(1, non_sorted.size + 1) / non_sorted.size
                if non_sorted.size
                else np.asarray([], dtype=float)
            )
            target_ranks = (
                np.arange(1, target_sorted.size + 1) / target_sorted.size
                if target_sorted.size
                else np.asarray([], dtype=float)
            )

            if non_sorted.size:
                non_cdf = np.interp(
                    x_grid,
                    non_sorted,
                    non_ranks,
                    left=0.0,
                    right=1.0,
                )
                non_line = ax.plot(
                    x_grid,
                    1.0 - non_cdf,
                    linewidth=2.0,
                    linestyle="-.",
                    label="non-target (1-CDF, label=0)",
                )[0]
                if non_sorted.size >= 2:
                    label_font = max(base_fontsize - 4, 8)
                    label_dx = 2
                    for q, style in ((0.95, "--"), (0.99, ":")):
                        qx = float(np.quantile(non_sorted, q))
                        own_y = 1.0 - _cdf_at(non_sorted, non_ranks, qx)
                        opp_y = (
                            _cdf_at(target_sorted, target_ranks, qx)
                            if target_sorted.size
                            else own_y
                        )
                        y_text = _label_y(opp_y, own_y)
                        ax.axvline(
                            qx,
                            color=non_line.get_color(),
                            linestyle=style,
                            linewidth=1.2,
                            alpha=0.6,
                            zorder=1,
                            label="_nolegend_",
                        )
                        ax.annotate(
                            f"{int(round(q * 100))}%",
                            xy=(qx, y_text),
                            xycoords="data",
                            xytext=(label_dx, 0),
                            textcoords="offset points",
                            rotation=45,
                            rotation_mode="anchor",
                            ha="right",
                            va="bottom",
                            fontsize=label_font,
                            color=non_line.get_color(),
                            clip_on=True,
                        )
                non_mean = float(np.mean(non_sorted))
                non_cdf_at_mean = float(_cdf_at(non_sorted, non_ranks, non_mean))
                ax.scatter(
                    [non_mean],
                    [1.0 - non_cdf_at_mean],
                    marker="D",
                    s=46,
                    facecolors="white",
                    edgecolors=non_line.get_color(),
                    linewidths=1.5,
                    zorder=3,
                    label="_nolegend_",
                )

            if target_sorted.size:
                target_cdf = np.interp(
                    x_grid,
                    target_sorted,
                    target_ranks,
                    left=0.0,
                    right=1.0,
                )
                target_line = ax.plot(
                    x_grid,
                    target_cdf,
                    linewidth=2.0,
                    linestyle="-",
                    label="target (CDF, label=1)",
                )[0]
                if target_sorted.size >= 2:
                    label_font = max(base_fontsize - 4, 8)
                    label_dx = 8
                    for q, style in ((0.05, "--"), (0.01, ":")):
                        qx = float(np.quantile(target_sorted, q))
                        own_y = _cdf_at(target_sorted, target_ranks, qx)
                        opp_y = (
                            1.0 - _cdf_at(non_sorted, non_ranks, qx)
                            if non_sorted.size
                            else own_y
                        )
                        y_text = _label_y(opp_y, own_y)
                        ax.axvline(
                            qx,
                            color=target_line.get_color(),
                            linestyle=style,
                            linewidth=1.2,
                            alpha=0.6,
                            zorder=1,
                            label="_nolegend_",
                        )
                        ax.annotate(
                            f"{int(round((1.0 - q) * 100))}%",
                            xy=(qx, y_text),
                            xycoords="data",
                            xytext=(label_dx, 0),
                            textcoords="offset points",
                            rotation=45,
                            rotation_mode="anchor",
                            ha="left",
                            va="bottom",
                            fontsize=label_font,
                            color=target_line.get_color(),
                            clip_on=True,
                        )
                target_mean = float(np.mean(target_sorted))
                target_cdf_at_mean = float(_cdf_at(target_sorted, target_ranks, target_mean))
                ax.scatter(
                    [target_mean],
                    [target_cdf_at_mean],
                    marker="D",
                    s=46,
                    facecolors="white",
                    edgecolors=target_line.get_color(),
                    linewidths=1.5,
                    zorder=3,
                    label="_nolegend_",
                )
            ax.set_ylabel("CDF / 1-CDF")
            ax.set_ylim(0.0, 1.0)
            ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale, base=10)
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_symlog_tick))
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect(1 / aspect)

        axes[-1].set_xlabel(xlabel)
        axes[0].legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def _summ(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "n": float(values.size),
        "min": float(np.min(values)),
        "p50": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _infer_label_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"label column '{preferred}' not found")
        return preferred
    for c in ("label", "same_model", "target", "y"):
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer label column from: {', '.join(df.columns)}")


def _infer_score_cols(df: pd.DataFrame, requested: Iterable[str] | None) -> list[str]:
    if requested:
        cols = [c for c in requested if c in df.columns]
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Requested score columns missing: {', '.join(missing)}")
        return cols

    candidates = ["score", "score_raw", "cos_sim_raw", "cos_dist_raw"]
    return [c for c in candidates if c in df.columns]


def _compute_implied_cos_sim(
    df: pd.DataFrame,
    logit_col: str,
    params: FFCosineParams,
    logit_mode: str,
) -> pd.Series:
    """
    FFCosine definition:
      similarity = cos_sim
      logit = similarity * scale + bias
      logits = [-logit, logit]
      prob_same = softmax(logits)[1] == sigmoid(2*logit)

    If input is "margin" (= logits[1]-logits[0] == 2*logit), we divide by 2.
    """
    logit = _as_float(df[logit_col])
    if logit_mode == "margin":
        logit = logit / 2.0
    cos = (logit - params.bias) / (params.scale if params.scale != 0 else 1e-12)
    return cos.clip(lower=-1.0, upper=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot score distributions for scored-pairs CSVs.")
    parser.add_argument("--scores", type=Path, nargs="+", required=True, help="One or more scores CSV paths.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write PNG plots + summary CSV.")
    parser.add_argument("--label-col", type=str, default=None, help="Label column name (default: infer).")
    parser.add_argument(
        "--score-cols",
        type=str,
        default=None,
        help="Comma-separated list of columns to plot (default: auto from score/score_raw/cos_*).",
    )
    parser.add_argument(
        "--score-cols-per-file",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Per-file score columns aligned with --scores order (optional). "
            "If set, only a single stacked plot is produced from these columns."
        ),
    )
    parser.add_argument(
        "--score-col-label",
        type=str,
        default="score",
        help=(
            "Label (and output stem) to use when --score-cols-per-file is set "
            "(default: score)."
        ),
    )
    parser.add_argument(
        "--ffcosine-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional FFCosine checkpoint to compute implied cosine similarity from a logit column "
            "(adds an 'implied_cos_sim' plot)."
        ),
    )
    parser.add_argument(
        "--ffcosine-logit-col",
        type=str,
        default="score_raw",
        help="Column to treat as FFCosine logit (default: score_raw).",
    )
    parser.add_argument(
        "--ffcosine-logit-mode",
        choices=("logit", "margin"),
        default="logit",
        help="Whether ffcosine-logit-col contains logit or logit_margin (=2*logit).",
    )
    parser.add_argument(
        "--saturation-prob-thr",
        type=float,
        default=0.999,
        help="Probability threshold for saturation check (default: 0.999).",
    )
    parser.add_argument(
        "--collapse-cos-thr",
        type=float,
        default=0.99,
        help="Cosine threshold for collapse check (default: 0.99).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser()
    _ensure_dir(out_dir)

    score_cols = None
    if args.score_cols:
        score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip()]
    score_cols_per_file = None
    if args.score_cols_per_file:
        score_cols_per_file = [c.strip() for c in args.score_cols_per_file if c.strip()]

    if score_cols and score_cols_per_file:
        raise ValueError("Use either --score-cols or --score-cols-per-file, not both.")
    if score_cols_per_file and len(score_cols_per_file) != len(args.scores):
        raise ValueError(
            "--score-cols-per-file must have the same number of entries as --scores."
        )

    ff_params = None
    if args.ffcosine_checkpoint is not None:
        ff_params = _load_ffcosine_params(args.ffcosine_checkpoint.expanduser())
        print(f"[ffcosine] scale={ff_params.scale:.6f} bias={ff_params.bias:.6f}")

    summaries: list[dict[str, object]] = []
    tables: list[ScoreTable] = []
    available_cols_per_table: list[list[str]] = []

    for path in args.scores:
        path = path.expanduser()
        df = pd.read_csv(path)
        label_col = _infer_label_col(df, args.label_col)
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
        df = df.dropna(subset=[label_col]).copy()
        df[label_col] = df[label_col].astype(int)

        target_mask = (df[label_col] == 1).to_numpy()
        non_mask = (df[label_col] == 0).to_numpy()

        if score_cols is None and score_cols_per_file is None:
            available_cols_per_table.append(_infer_score_cols(df, None))

        tables.append(
            ScoreTable(
                path=path,
                label=path.name,
                df=df,
                label_col=label_col,
                target_mask=target_mask,
                non_mask=non_mask,
            )
        )

    used_per_file = score_cols_per_file is not None

    if used_per_file:
        series: list[tuple[str, np.ndarray, np.ndarray]] = []
        for table, col in zip(tables, score_cols_per_file):
            if col not in table.df.columns:
                raise ValueError(f"Requested score column '{col}' missing in {table.path}.")
            values = _as_float(table.df[col]).to_numpy()
            target = values[table.target_mask]
            non = values[table.non_mask]
            series.append((table.label, target, non))

            summaries.append(
                {
                    "file": str(table.path),
                    "column": col,
                    "target": "label=1",
                    **_summ(target),
                }
            )
            summaries.append(
                {
                    "file": str(table.path),
                    "column": col,
                    "target": "label=0",
                    **_summ(non),
                }
            )

        label = (args.score_col_label or "score").strip()
        if not label:
            label = "score"
        out_stem = label.replace(" ", "_")
        _save_stacked_cdf(
            out_dir / f"stacked.{out_stem}.png",
            series=series,
            xlabel=label,
        )
    else:
        if score_cols:
            for table in tables:
                missing = [c for c in score_cols if c not in table.df.columns]
                if missing:
                    raise ValueError(
                        f"Requested score columns missing in {table.path}: {', '.join(missing)}"
                    )
            plot_cols = score_cols
        else:
            candidates = ["score", "score_raw", "cos_sim_raw", "cos_dist_raw"]
            plot_cols = [
                c for c in candidates if all(c in cols for cols in available_cols_per_table)
            ]

        if not plot_cols and ff_params is None:
            raise ValueError("No known score columns found across the provided CSVs.")

        for col in plot_cols:
            series = []
            for table in tables:
                values = _as_float(table.df[col]).to_numpy()
                target = values[table.target_mask]
                non = values[table.non_mask]
                series.append((table.label, target, non))

                summaries.append(
                    {
                        "file": str(table.path),
                        "column": col,
                        "target": "label=1",
                        **_summ(target),
                    }
                )
                summaries.append(
                    {
                        "file": str(table.path),
                        "column": col,
                        "target": "label=0",
                        **_summ(non),
                    }
                )

            _save_stacked_cdf(
                out_dir / f"stacked.{col}.png",
                series=series,
                xlabel=col,
            )

    if ff_params is not None:
        missing_logit = [
            table.path for table in tables if args.ffcosine_logit_col not in table.df.columns
        ]
        if missing_logit:
            missing_paths = ", ".join(str(p) for p in missing_logit)
            print(
                f"[ffcosine] Skipping implied_cos_sim plot; missing "
                f"'{args.ffcosine_logit_col}' in: {missing_paths}"
            )
        else:
            implied_series: list[tuple[str, np.ndarray, np.ndarray]] = []
            sat_prob_thr = float(args.saturation_prob_thr)
            collapse_cos_thr = float(args.collapse_cos_thr)

            for table in tables:
                implied = _compute_implied_cos_sim(
                    table.df,
                    logit_col=args.ffcosine_logit_col,
                    params=ff_params,
                    logit_mode=args.ffcosine_logit_mode,
                )
                implied_np = implied.to_numpy(dtype=float)
                implied_t = implied_np[table.target_mask]
                implied_n = implied_np[table.non_mask]
                implied_series.append((table.label, implied_t, implied_n))

                # Diagnostics for your two scenarios.
                prob_col = "score" if "score" in table.df.columns else None
                sat_rate = None
                if prob_col is not None:
                    prob = _as_float(table.df[prob_col]).to_numpy()
                    sat_rate = float(
                        np.mean(
                            table.non_mask
                            & (prob >= sat_prob_thr)
                            & (implied_np < collapse_cos_thr)
                        )
                    )

                collapse_rate = float(
                    np.mean(table.non_mask & (implied_np >= collapse_cos_thr))
                )

                summaries.append(
                    {
                        "file": str(table.path),
                        "column": "diagnostic",
                        "target": "negatives",
                        "collapse_rate@cos>=" + str(collapse_cos_thr): collapse_rate,
                        "saturation_rate@prob>=" + str(sat_prob_thr): sat_rate,
                    }
                )

            _save_stacked_cdf(
                out_dir / "stacked.implied_cos_sim.png",
                series=implied_series,
                xlabel="implied_cos_sim",
                bins=160,
            )

    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote plots + summary to: {out_dir}")


if __name__ == "__main__":
    main()
