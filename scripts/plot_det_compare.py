#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.metrics import det_curve

DET_TICKS = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8])
SCORE_COL_CANDIDATES = ["score", "score_raw", "posterior", "prob", "p_same", "cos_sim_raw", "logit", "logits"]
LABEL_COL_CANDIDATES = ["label", "y", "target", "same_generator", "same_model", "same_source"]
MANIFEST_DATASET_COLS = ["dataset", "corpus", "domain"]
MANIFEST_LABEL_COLS = ["label", "name", "system"]
MANIFEST_SCORES_COLS = ["scores", "scores_path", "score_path", "path"]
MANIFEST_GROUP_COLS = ["group", "category", "family"]
MANIFEST_OBJECTIVE_COLS = ["objective", "loss"]
MANIFEST_SEED_COLS = ["seed"]


def probit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    inv = NormalDist().inv_cdf
    return np.vectorize(inv, otypes=[float])(p)


def fmt_pct_tick(p: float) -> str:
    x = p * 100.0
    if x < 0.1:
        s = f"{x:.3f}".rstrip("0").rstrip(".")
    elif x < 1:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
    elif x < 10:
        s = f"{x:.1f}".rstrip("0").rstrip(".")
    else:
        s = f"{x:.0f}"
    return s


def infer_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer column from {candidates}. Present: {list(df.columns)}")


def parse_rate(value: str) -> float:
    v = value.strip()
    if v.endswith("%"):
        return float(v[:-1].strip()) / 100.0
    x = float(v)
    return x / 100.0 if x > 1.0 else x


def parse_rate_list(value: str) -> list[float]:
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(parse_rate(part))
    return out


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    dataset_col = infer_col(df, MANIFEST_DATASET_COLS)
    label_col = infer_col(df, MANIFEST_LABEL_COLS)
    scores_col = infer_col(df, MANIFEST_SCORES_COLS)

    group_col = next((c for c in MANIFEST_GROUP_COLS if c in df.columns), None)
    objective_col = next((c for c in MANIFEST_OBJECTIVE_COLS if c in df.columns), None)
    seed_col = next((c for c in MANIFEST_SEED_COLS if c in df.columns), None)

    df = df.rename(
        columns={
            dataset_col: "dataset",
            label_col: "label",
            scores_col: "scores",
            **({group_col: "group"} if group_col else {}),
            **({objective_col: "objective"} if objective_col else {}),
            **({seed_col: "seed"} if seed_col else {}),
        }
    )
    return df


def load_scores(path: Path, score_col: str | None, label_col: str | None) -> tuple[np.ndarray, np.ndarray, str, str]:
    df = pd.read_csv(path)
    score_col = score_col or infer_col(df, SCORE_COL_CANDIDATES)
    label_col = label_col or infer_col(df, LABEL_COL_CANDIDATES)

    s = pd.to_numeric(df[score_col], errors="coerce")
    y = pd.to_numeric(df[label_col], errors="coerce")
    ok = s.notna().to_numpy() & y.notna().to_numpy()

    scores = s.to_numpy(dtype=float)[ok]
    labels = y.to_numpy(dtype=int)[ok]
    return scores, labels, score_col, label_col


def eer_linear(fpr: np.ndarray, fnr: np.ndarray) -> float:
    d = fpr - fnr
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if idx.size == 0:
        i = int(np.argmin(np.abs(d)))
        return float((fpr[i] + fnr[i]) / 2.0)

    i = int(idx[0])
    x0, x1 = fpr[i], fpr[i + 1]
    y0, y1 = fnr[i], fnr[i + 1]
    d0, d1 = d[i], d[i + 1]
    t = 0.0 if (d1 - d0) == 0 else float(-d0 / (d1 - d0))
    fpr_i = x0 + t * (x1 - x0)
    fnr_i = y0 + t * (y1 - y0)
    return float((fpr_i + fnr_i) / 2.0)


def setup_interspeech_rc(fontsize: int = 9, linew: float = 1.1) -> None:
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "axes.linewidth": 0.8,
            "lines.linewidth": linew,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_rate(value: str) -> float:
    v = value.strip()
    if v.endswith("%"):
        return float(v[:-1].strip()) / 100.0
    x = float(v)
    return x / 100.0 if x > 1.0 else x


def parse_rate_range(value: str) -> tuple[float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Range must be 'min,max' (e.g. 2%,40% or 0.02,0.4).")
    lo = parse_rate(parts[0])
    hi = parse_rate(parts[1])
    if not (0.0 < lo < hi <= 1.0):
        raise ValueError(f"Invalid range: {lo:g},{hi:g} (expected 0<min<max<=1).")
    return lo, hi


def parse_sys(args: list[str]) -> list[dict]:
    out: list[dict] = []
    for raw in args:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=PATH, got: {raw}")
        name, p = raw.split("=", 1)
        name = name.strip()
        dataset = "all"
        label = name
        if ":" in name:
            dataset, label = name.split(":", 1)
        elif "/" in name:
            dataset, label = name.split("/", 1)
        out.append({"label": label.strip(), "scores": Path(p.strip()), "dataset": dataset.strip()})
    return out


def interp_fnr_at_fpr(fpr: np.ndarray, fnr: np.ndarray, x: float) -> float:
    # det_curve returns fpr sorted ascending, safe for interpolation
    x = float(np.clip(x, fpr.min(), fpr.max()))
    return float(np.interp(x, fpr, fnr))


def pick_det_ticks(lo: float, hi: float) -> np.ndarray:
    # Fewer, more readable ticks
    base = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1, 6e-1, 8e-1])
    return base[(base >= lo) & (base <= hi)]


def plot_det(
    series: list[dict],
    out_path: Path,
    *,
    figsize: tuple[float, float],
    pos_label: int,
    score_direction: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    score_col: str | None,
    label_col: str | None,
    show_inset: bool,
    inset_size: str,
    inset_xlim: tuple[float, float] | None,
    inset_ylim: tuple[float, float] | None,
    inset_loc: str,
    inset_connectors: bool,
    legend_ncol: int,
    use_verification_labels: bool,
    linestyle_by: str,
    marker_by: str,
    color_by: str,
    show_op_markers: bool,
    op_fprs: list[float],
    linewidth: float,
) -> None:
    setup_interspeech_rc(linew=float(linewidth))

    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.80)

    # Use fewer ticks in the visible range
    xt = pick_det_ticks(xlim[0], xlim[1])
    yt = pick_det_ticks(ylim[0], ylim[1])
    ax.set_xticks(probit(xt))
    ax.set_yticks(probit(yt))
    ax.set_xticklabels([fmt_pct_tick(t) for t in xt])
    ax.set_yticklabels([fmt_pct_tick(t) for t in yt])

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.22)
    ax.set_axisbelow(True)

    diag_p = np.array([max(xlim[0], ylim[0]), min(xlim[1], ylim[1])])
    ax.plot(
        probit(diag_p),
        probit(diag_p),
        linestyle="--",
        linewidth=max(0.6, float(linewidth) * 0.5),
        alpha=0.18,
        label="_nolegend_",
    )

    # First pass: compute curves and EERs, then sort for legend readability
    curves: list[dict] = []
    for s in series:
        name = s["label"]
        path = s["scores"]
        scores, labels, used_score, used_label = load_scores(path, score_col, label_col)
        if score_direction == "lower":
            scores = -scores

        tgt = scores[labels == pos_label]
        non = scores[labels != pos_label]
        if tgt.size > 0 and non.size > 0:
            mean_gap = float(np.mean(tgt) - np.mean(non))
            if mean_gap < 0:
                print(
                    f"[WARN] {name}: mean(target) < mean(nontarget) after score_direction handling "
                    f"(gap={mean_gap:.4g}). Check pos_label or score_direction. "
                    f"score_col={used_score} label_col={used_label}"
                )

        fpr, fnr, _ = det_curve(labels, scores, pos_label=pos_label)
        eer = eer_linear(fpr, fnr)

        curves.append(
            {
                "name": name,
                "path": path,
                "fpr": fpr,
                "fnr": fnr,
                "eer": eer,
                "meta": s,
            }
        )

    curves.sort(key=lambda d: d["eer"])  # best first

    def _coverage_in_range(fpr: np.ndarray, fnr: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> float:
        if fpr.size == 0:
            return 0.0
        m = (fpr >= xlim[0]) & (fpr <= xlim[1]) & (fnr >= ylim[0]) & (fnr <= ylim[1])
        return float(np.mean(m))

    coverages = [_coverage_in_range(d["fpr"], d["fnr"], xlim, ylim) for d in curves]
    if coverages and max(coverages) < 0.02:
        print(
            f"[WARN] Current xlim/ylim hides most of the curves (max coverage {max(coverages)*100:.1f}%). "
            f"Try wider ranges, e.g. --xlim 1%,40% and --ylim 5%,60% for STOPA."
        )

    def _value_key(d: dict, key: str, fallback: str) -> str:
        return str(d.get("meta", {}).get(key, fallback))

    def _build_map(key: str, palette: list[str]) -> dict[str, str]:
        values = []
        for d in curves:
            values.append(_value_key(d, key, d["name"]))
        out: dict[str, str] = {}
        j = 0
        for v in values:
            if v not in out:
                out[v] = palette[j % len(palette)]
                j += 1
        return out

    linestyle_map = _build_map(linestyle_by, linestyles) if linestyle_by != "none" else {}
    marker_map = _build_map(marker_by, markers) if marker_by != "none" else {}
    color_map = _build_map(color_by, colors) if color_by != "none" else {}

    # Operating points to visually compare at fixed FARs
    op_fprs = [float(x) for x in op_fprs]

    lines = []
    for i, d in enumerate(curves):
        name = d["name"]
        fpr = d["fpr"]
        fnr = d["fnr"]
        eer = float(d["eer"])

        ls_key = _value_key(d, linestyle_by, name)
        mk_key = _value_key(d, marker_by, name)
        col_key = _value_key(d, color_by, name)
        line_style = linestyle_map.get(ls_key, "-")
        marker_style = marker_map.get(mk_key, markers[i % len(markers)])
        color_style = color_map.get(col_key, colors[i % len(colors)])

        label = f"{name} ({eer*100:.2f}%)"
        line = ax.plot(
            probit(fpr),
            probit(fnr),
            linestyle=line_style,
            color=color_style,
            label=label,
        )[0]

        # EER point marker
        eer_pt = probit(np.array([eer]))[0]
        ax.plot(
            [eer_pt],
            [eer_pt],
            marker=marker_style,
            markersize=4.2,
            linestyle="None",
            color=line.get_color(),
            markerfacecolor="white",
            markeredgewidth=0.9,
            zorder=6,
        )

        # Annotate EER values lightly near the point
        # ax.text(
        #     eer_pt,
        #     eer_pt,
        #     f" {eer*100:.1f}",
        #     fontsize=7,
        #     va="bottom",
        #     ha="left",
        #     alpha=0.85,
        # )

        if show_op_markers:
            for op in op_fprs:
                if op < fpr.min() or op > fpr.max():
                    continue
                y = interp_fnr_at_fpr(fpr, fnr, op)
                ax.plot(
                    [probit(np.array([op]))[0]],
                    [probit(np.array([y]))[0]],
                    marker=marker_style,
                    markersize=3.6,
                    linestyle="None",
                    color=line.get_color(),
                    markerfacecolor=line.get_color(),
                    markeredgewidth=0.0,
                    alpha=0.9,
                    zorder=5,
                )

        lines.append(line)

    if use_verification_labels:
        ax.set_xlabel("False accept rate (%)")
        ax.set_ylabel("False reject rate (%)")
    else:
        ax.set_xlabel("False alarm rate (%)")
        ax.set_ylabel("Miss rate (%)")

    ax.set_xlim(probit(np.array([xlim[0], xlim[1]])))
    ax.set_ylim(probit(np.array([ylim[0], ylim[1]])))

    # Legend outside, sorted by EER, does not occlude curves
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=max(1, legend_ncol),
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
        labelspacing=0.3,
    )

    # Low-FAR inset that is always the forensic region
    if show_inset and len(curves) > 0:
        if inset_xlim is None:
            inset_xlo, inset_xhi = 1e-4, min(2e-2, xlim[1])     # 0.01% to 2%
        else:
            inset_xlo, inset_xhi = inset_xlim
        if inset_ylim is None:
            inset_ylo, inset_yhi = 1e-3, min(2e-1, ylim[1])     # 0.1% to 20%
        else:
            inset_ylo, inset_yhi = inset_ylim

        axins = inset_axes(
            ax,
            width="36%",
            height="36%",
            loc=inset_loc,
            borderpad=0.8,
        )
        axins.tick_params(labelsize=7)
        axins.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.22)

        for i, d in enumerate(curves):
            fpr = d["fpr"]
            fnr = d["fnr"]
            eer = float(d["eer"])
            color = lines[i].get_color()
            mk_key = _value_key(d, marker_by, d["name"])
            marker_style = marker_map.get(mk_key, markers[i % len(markers)])

            axins.plot(
                probit(fpr),
                probit(fnr),
                linestyle="-",
                color=color,
                linewidth=max(1.0, float(linewidth) * 0.85),
            )

            eer_pt = probit(np.array([eer]))[0]
            axins.plot(
                [eer_pt],
                [eer_pt],
                marker=marker_style,
                markersize=3.6,
                linestyle="None",
                color=color,
                markerfacecolor="white",
                markeredgewidth=0.8,
                zorder=6,
            )

        axins.plot(
            probit(diag_p),
            probit(diag_p),
            linestyle="--",
            linewidth=max(0.6, float(linewidth) * 0.5),
            alpha=0.18,
        )

        axins.set_xlim(probit(np.array([inset_xlo, inset_xhi])))
        axins.set_ylim(probit(np.array([inset_ylo, inset_yhi])))

        xt2 = pick_det_ticks(inset_xlo, inset_xhi)
        yt2 = pick_det_ticks(inset_ylo, inset_yhi)
        axins.set_xticks(probit(xt2))
        axins.set_yticks(probit(yt2))
        axins.set_xticklabels([fmt_pct_tick(t) for t in xt2])
        axins.set_yticklabels([fmt_pct_tick(t) for t in yt2])

        if inset_connectors:
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35", lw=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="DET plot with bottom-left legend and thinner lines.")
    ap.add_argument("--sys", action="append", help="System as NAME=PATH_TO_SCORES_CSV, repeat.")
    ap.add_argument("--manifest", type=str, default=None, help="CSV with dataset,label,scores[,group,objective,seed].")
    ap.add_argument("--split-by", choices=["none", "dataset"], default="none")
    ap.add_argument("-o", "--out", type=str, default="tmp/det/det_4sys.pdf")
    ap.add_argument("--figsize", type=str, default="3.35,3.0", help="Inches W,H.")
    ap.add_argument("--pos-label", type=int, default=1)
    ap.add_argument("--score-direction", choices=["higher", "lower"], default="higher")
    ap.add_argument("--score-col", type=str, default=None)
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--xlim", type=str, default="0.0003,0.9")
    ap.add_argument("--ylim", type=str, default="0.0003,0.95")
    ap.add_argument("--legend-ncol", type=int, default=1)
    ap.add_argument("--inset", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--inset-size", type=str, default="45%")
    ap.add_argument("--inset-xlim", type=str, default=None)
    ap.add_argument("--inset-ylim", type=str, default=None)
    ap.add_argument("--inset-loc", type=str, default="lower left")
    ap.add_argument("--inset-connectors", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--verification-labels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--linestyle-by", choices=["none", "group", "objective", "seed", "label"], default="group")
    ap.add_argument("--marker-by", choices=["none", "seed", "group", "objective", "label"], default="seed")
    ap.add_argument("--color-by", choices=["label", "objective", "group", "seed", "none"], default="label")
    ap.add_argument("--op-markers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--op-fprs", type=str, default="0.1%,1%,10%")
    ap.add_argument("--linewidth", type=float, default=1.6)
    args = ap.parse_args()

    w, h = [float(x.strip()) for x in args.figsize.split(",")]
    xlo, xhi = parse_rate_range(args.xlim)
    ylo, yhi = parse_rate_range(args.ylim)
    inset_xlim = parse_rate_range(args.inset_xlim) if args.inset_xlim else None
    inset_ylim = parse_rate_range(args.inset_ylim) if args.inset_ylim else None

    if not args.manifest and not args.sys:
        raise ValueError("Provide --sys entries or a --manifest CSV.")

    if args.manifest:
        df = load_manifest(Path(args.manifest))
        if args.split_by == "dataset":
            groups = dict(tuple(df.groupby("dataset", dropna=False)))
        else:
            groups = {"all": df}

        for key, g in groups.items():
            series = []
            for _, row in g.iterrows():
                series.append(
                    {
                        "label": str(row["label"]),
                        "scores": Path(str(row["scores"])),
                        "dataset": str(row.get("dataset", "")),
                        "group": str(row.get("group", "")),
                        "objective": str(row.get("objective", "")),
                        "seed": str(row.get("seed", "")),
                    }
                )

            out_base = Path(args.out)
            if args.split_by != "none":
                tag = str(key).strip().replace(" ", "_")
                out_path = out_base.with_name(f"{out_base.stem}_{tag}{out_base.suffix}")
            else:
                out_path = out_base

            plot_det(
                series,
                out_path,
                figsize=(w, h),
                pos_label=args.pos_label,
                score_direction=args.score_direction,
                xlim=(xlo, xhi),
                ylim=(ylo, yhi),
                score_col=args.score_col,
                label_col=args.label_col,
                show_inset=bool(args.inset),
                inset_size=str(args.inset_size),
                inset_xlim=inset_xlim,
                inset_ylim=inset_ylim,
                inset_loc=str(args.inset_loc),
                inset_connectors=bool(args.inset_connectors),
                legend_ncol=int(args.legend_ncol),
                use_verification_labels=bool(args.verification_labels),
                linestyle_by=args.linestyle_by,
                marker_by=args.marker_by,
                color_by=args.color_by,
                show_op_markers=bool(args.op_markers),
                op_fprs=parse_rate_list(args.op_fprs),
                linewidth=float(args.linewidth),
            )
    else:
        linestyle_by = args.linestyle_by
        marker_by = args.marker_by
        if linestyle_by in {"group", "objective", "seed"}:
            linestyle_by = "label"
        if marker_by in {"group", "objective", "seed"}:
            marker_by = "label"
        series_all = parse_sys(args.sys)
        if args.split_by == "dataset":
            by_ds: dict[str, list[dict]] = {}
            for s in series_all:
                by_ds.setdefault(s.get("dataset", "all"), []).append(s)
        else:
            by_ds = {"all": series_all}

        for key, series in by_ds.items():
            out_base = Path(args.out)
            if args.split_by != "none":
                tag = str(key).strip().replace(" ", "_")
                out_path = out_base.with_name(f"{out_base.stem}_{tag}{out_base.suffix}")
            else:
                out_path = out_base

            plot_det(
                series,
                out_path,
                figsize=(w, h),
                pos_label=args.pos_label,
                score_direction=args.score_direction,
                xlim=(xlo, xhi),
                ylim=(ylo, yhi),
            score_col=args.score_col,
            label_col=args.label_col,
            show_inset=bool(args.inset),
            inset_size=str(args.inset_size),
            inset_xlim=inset_xlim,
            inset_ylim=inset_ylim,
            inset_loc=str(args.inset_loc),
            inset_connectors=bool(args.inset_connectors),
            legend_ncol=int(args.legend_ncol),
            use_verification_labels=bool(args.verification_labels),
            linestyle_by=linestyle_by,
            marker_by=marker_by,
            color_by=args.color_by,
            show_op_markers=bool(args.op_markers),
            op_fprs=parse_rate_list(args.op_fprs),
            linewidth=float(args.linewidth),
        )


if __name__ == "__main__":
    main()
