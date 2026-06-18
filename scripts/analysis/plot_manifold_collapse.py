#!/usr/bin/env python3
"""
Plot cumulative explained variance (PCA spectrum) for embedding tables.

This is intended for the "Manifold Collapse" verification:
  - Global (MLAAD)
  - Global (STOPA)
  - Pairwise (MLAAD)
  - Pairwise (STOPA)

Edit the EMBEDDING_SPECS list below or pass --sys to point at your embeddings.
Embeddings are expected to be NPZ files with `embeddings` + `utt_ids`
or a dict-like NPZ/NPY mapping of id -> vector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class EmbeddingSpec:
    label: str
    path: Path
    color: str
    linestyle: str


# TODO: Update these paths to your four embedding tables.
EMBEDDING_SPECS: list[EmbeddingSpec] = [
    EmbeddingSpec(
        label="Global (MLAAD)",
        path=Path("runs/cosine_refset_R1_s123_embs.npz"),
        color="tab:blue",
        linestyle="-",
    ),
    EmbeddingSpec(
        label="Global (STOPA)",
        path=Path("runs/stopa_cosine_refset_R1_s123_embs_1324982.npz"),
        color="tab:blue",
        linestyle="--",
    ),
    EmbeddingSpec(
        label="Pairwise (MLAAD)",
        path=Path("runs/baseline_pairwise_refset_FFCosine_rival_R1_MHFA_FFCosine_FFCosine_26_embs_XLSR_300M_MHFA_FFCosine_FFCosine_26_seg4.npz"),
        color="tab:orange",
        linestyle="-",
    ),
    EmbeddingSpec(
        label="Pairwise (STOPA)",
        path=Path("runs/stopa_pairwise_refset_FFCosine_rival_s42_R1_MHFA_FFCosine_FFCosine_26_embs_XLSR_300M_MHFA_FFCosine_FFCosine_26_seg4.npz"),
        color="tab:orange",
        linestyle="--",
    ),
]

def _load_embedding_matrix(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile) and "embeddings" in data:
        emb = np.asarray(data["embeddings"])
    elif isinstance(data, np.lib.npyio.NpzFile):
        mapping = {k: np.asarray(data[k]) for k in data.files}
        utt_ids = sorted(mapping)
        emb = np.stack([mapping[u] for u in utt_ids], axis=0)
    elif isinstance(data, np.ndarray) and data.dtype == object:
        if data.ndim == 0 or data.size == 1:
            maybe_dict = data.item()
            if isinstance(maybe_dict, dict):
                utt_ids = sorted(maybe_dict)
                emb = np.stack([np.asarray(maybe_dict[u]) for u in utt_ids], axis=0)
            else:
                raise ValueError(f"Unsupported embedding array shape in {path}: {data.shape}.")
        elif data.ndim == 2:
            emb = data
        else:
            raise ValueError(f"Unsupported embedding array shape in {path}: {data.shape}.")
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        emb = data
    elif isinstance(data, dict):
        utt_ids = sorted(data)
        emb = np.stack([np.asarray(data[u]) for u in utt_ids], axis=0)
    else:
        raise ValueError(
            f"Unsupported embedding table format in {path}. "
            "Expect npz with embeddings/utt_ids or a dict of id -> vector."
        )

    X = np.asarray(emb, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2D [N, D], got shape {X.shape} in {path}.")
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] < 2:
        raise ValueError(f"Not enough rows after filtering NaNs in {path}.")
    return X


def _components_for_threshold(cum: np.ndarray, threshold: float) -> float:
    idx = int(np.searchsorted(cum, threshold, side="left"))
    if idx >= cum.size:
        return float("nan")
    return float(idx + 1)


def _pca_metrics(X: np.ndarray, max_plot_components: int) -> tuple[np.ndarray, dict[str, float]]:
    n_samples, n_features = X.shape
    n_components = min(n_features, n_samples - 1)
    if n_components < 1:
        raise ValueError(f"Need at least 1 PCA component; got n={n_samples}, d={n_features}.")

    # Center features for PCA.
    X = X - X.mean(axis=0, keepdims=True)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(X)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    eigs = np.asarray(pca.explained_variance_, dtype=float)
    cum = np.cumsum(evr)

    if cum.size < max_plot_components:
        pad = np.full(max_plot_components - cum.size, cum[-1], dtype=float)
        cum_plot = np.concatenate([cum, pad], axis=0)
    else:
        cum_plot = cum[:max_plot_components]

    # Effective rank from eigenvalue entropy.
    total = float(np.sum(eigs))
    p = eigs / total if total > 0 else eigs
    entropy = float(-np.sum(np.where(p > 0, p * np.log(p), 0.0))) if p.size else float("nan")
    eff_rank = float(np.exp(entropy)) if p.size else float("nan")
    participation_ratio = float((total * total) / float(np.sum(eigs * eigs))) if total > 0 else float("nan")

    stats = {
        "n_samples": float(n_samples),
        "n_features": float(n_features),
        "n_components": float(n_components),
        "effective_rank": eff_rank,
        "participation_ratio": participation_ratio,
        "k90": _components_for_threshold(cum, 0.90),
        "k95": _components_for_threshold(cum, 0.95),
        "k99": _components_for_threshold(cum, 0.99),
        "top1_var": float(evr[0]) if evr.size else float("nan"),
        "top10_var": float(np.sum(evr[: min(10, evr.size)])) if evr.size else float("nan"),
    }
    return cum_plot, stats


def _ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_sys_specs(raw_specs: list[str]) -> list[EmbeddingSpec]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    linestyles = ["-", "--", ":", "-."]
    specs: list[EmbeddingSpec] = []
    for i, raw in enumerate(raw_specs):
        if "=" not in raw:
            raise ValueError(f"Expected NAME=PATH, got: {raw}")
        label, p = raw.split("=", 1)
        specs.append(
            EmbeddingSpec(
                label=label.strip(),
                path=Path(p.strip()),
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
            )
        )
    return specs


def _rank_report(results: list[tuple[EmbeddingSpec, dict[str, float]]]) -> None:
    ranked = sorted(
        results,
        key=lambda r: float(np.nan_to_num(r[1]["effective_rank"], nan=-np.inf)),
        reverse=True,
    )
    print("Rank by effective rank (higher = less collapse):")
    for idx, (spec, stats) in enumerate(ranked, start=1):
        print(f"  {idx}. {spec.label} (eff_rank={stats['effective_rank']:.2f})")

    label_to_rank = {spec.label: i + 1 for i, (spec, _) in enumerate(ranked)}
    keywords = ("global", "pairwise", "intermediate")
    for keyword in keywords:
        matches = [spec for spec, _ in results if keyword in spec.label.lower()]
        if not matches:
            continue
        for spec in matches:
            rank = label_to_rank.get(spec.label)
            if rank is None:
                continue
            print(f"  -> {spec.label} rank: {rank}/{len(ranked)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative explained variance (PCA spectrum) for embedding tables."
    )
    parser.add_argument("--sys", action="append", help="System as NAME=PATH_TO_EMB_NPZ, repeat.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for the plot.")
    parser.add_argument(
        "--max-components",
        type=int,
        default=50,
        help="Max PCA components to plot (default: 50).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser()
    _ensure_out_dir(out_dir)

    if args.sys:
        specs = _parse_sys_specs(args.sys)
    else:
        specs = EMBEDDING_SPECS

    base_fontsize = 14
    plt.rcParams.update(
        {
            "font.size": base_fontsize,
            "axes.labelsize": base_fontsize + 2,
            "axes.titlesize": base_fontsize + 2,
            "xtick.labelsize": base_fontsize,
            "ytick.labelsize": base_fontsize,
            "legend.fontsize": base_fontsize,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    results: list[tuple[EmbeddingSpec, dict[str, float]]] = []

    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {spec.path}")
        X = _load_embedding_matrix(spec.path)
        cum, stats = _pca_metrics(X, args.max_components)
        results.append((spec, stats))
        print(
            f"{spec.label}: n={int(stats['n_samples'])} d={int(stats['n_features'])} "
            f"components={int(stats['n_components'])} eff_rank={stats['effective_rank']:.2f} "
            f"pr={stats['participation_ratio']:.2f} k95={stats['k95']:.0f} k99={stats['k99']:.0f}"
        )
        x = np.arange(1, args.max_components + 1, dtype=int)
        ax.plot(x, cum, label=spec.label, color=spec.color, linestyle=spec.linestyle, linewidth=2.0)

    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_xlim(1, args.max_components)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    out_path = out_dir / "manifold_collapse_cumvar.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")
    if results:
        _rank_report(results)


if __name__ == "__main__":
    main()
