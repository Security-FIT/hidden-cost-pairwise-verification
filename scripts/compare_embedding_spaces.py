#!/usr/bin/env python3
"""
Compare embedding spaces between two checkpoints on a single-utterance protocol.

This script is intended for MLAAD-style protocols (CSV with at least a `path` column
and a class column like `model_name`). It will:
  - load two checkpoints via `common.build_model()`
  - extract per-utterance embeddings at a requested layer for each model
  - compute embedding-space metrics and save plots (incl. t-SNE)
  - (by default) also run L2-normalized cosine diagnostics + tail margin analysis
  - reuse `embeddings_{A,B}.npz` from --out-dir when compatible (disable with --no-cache)

Example:
  PYTHONPATH=. python scripts/compare_embedding_spaces.py \
    --data-root /mnt/expansion/Datasets/MLAAD \
    --protocol-csv /mnt/expansion/Datasets/MLAAD/mlaad4sourcetracing/eval_single.csv \
    --path-column path --class-column model_name \
    --exclude-classes bonafide,real \
    --max-per-class 200 \
    --a-checkpoint runs/ckpts/FFMulticlass_10.pt --a-extractor XLSR_300M --a-processor MHFA --a-classifier FFMulticlass \
    --a-bottleneck-dim 256 --a-num-classes 120 --a-embed-layer bottleneck \
    --b-checkpoint runs/ckpts/FFCosineRaw_10.pt --b-extractor XLSR_300M --b-processor MHFA --b-classifier FFCosineRaw \
    --b-embed-layer processor \
    --out-dir analyses/embedding_compare/run1
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from classifiers.FFBase import FFBase
from common import build_model
from datasets.utils import custom_single_batch_create


try:
    import soundfile as sf  # type: ignore

    def _read_audio_mono(path: str) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(sr)

except ModuleNotFoundError:

    def _read_audio_mono(path: str) -> tuple[np.ndarray, int]:
        sr, audio = wavfile.read(path)
        # wavfile returns int16/int32/float; normalize ints to [-1, 1]
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            max_val = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)
        return audio, int(sr)


def _maybe_resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    if sr <= 0 or target_sr <= 0:
        raise ValueError(f"Invalid sample rates: sr={sr}, target_sr={target_sr}")
    # Use rational resampling.
    g = math.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts: list[str] = []
    for chunk in str(value).split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts or None


def _parse_float_pair(value: str, name: str) -> tuple[float, float]:
    raw = str(value).strip().lower().replace("x", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be 'W,H' in inches (e.g., 3.2,3.2).")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc


PLOT_ALPHA = 0.9
PLOT_EDGE_COLOR = "black"
PLOT_EDGE_WIDTH = 0.5
PLOT_CLASS_SETS = {
    "forensic-hard": ["tts_models/en/ljspeech/vits", "suno/bark-small", "tts_models/en/ljspeech/vits--neon", "suno/bark", "parler_tts_large_v1", "parler_tts_mini_v1"],
    "global-hard": ["facebook/mms-tts-fra", "facebook/mms-tts-hun",  "facebook/mms-tts-fin", "facebook/mms-tts-deu"],
    "easy": ["tts_models/en/multi-dataset/tortoise-v2", "tts_models/zh-CN/baker/tacotron2-DDC-GST", "OpenVoiceV2", 'f5-tts', 'e2-tts', 'WhisperSpeech'],
}
PLOT_CLASS_COLORS = {
    "tts_models/en/ljspeech/vits": "#1f77b4",
    "suno/bark-small": "#17becf",
    "tts_models/en/ljspeech/vits--neon": "#2ca02c",
    "suno/bark": "#bcbd22",
    "facebook/mms-tts-fra": "#ff7f0e",
    "tts_models/zh-CN/baker/tacotron2-DDC-GST": "#e377c2",
    "facebook/mms-tts-deu": "#000000",
    "tts_models/en/multi-dataset/tortoise-v2": "#d62728",
    "facebook/mms-tts-hun": "#8c564b",
    "facebook/mms-tts-hun": "#9467bd",
    "OpenVoiceV2": "#9467bd",
    "parler_tts_large_v1": "#8c564b",
    "parler_tts_mini_v1": "#e377c2",
    "griffin_lim": "#7f7f7f",
    'f5-tts': '#ff9896',
    'e2-tts': '#c5b0d5',
    'WhisperSpeech': '#c49c94',
}
PLOT_CLASS_SET_MARKERS = {
    "forensic-hard": "o",
    "global-hard": "o",
    "easy": "o",
}
PLOT_CLASS_SET_LABELS = {
    "easy": "easy",
    "global-hard": "medium",
    "forensic-hard": "hard",
}
PLOT_CLASS_SET_ORDER = ["easy", "global-hard", "forensic-hard"]
PLOT_FAMILY_GROUPS = [
    {
        "name": "VITS",
        "members": ["tts_models/en/ljspeech/vits", "tts_models/en/ljspeech/vits--neon"],
        "color": "#1f77b4",
    },
    {
        "name": "Bark",
        "members": ["suno/bark", "suno/bark-small"],
        "color": "#2ca02c",
    },
    {
        "name": "MMS-TTS",
        "prefixes": ["facebook/mms-tts-"],
        "color": "#ff7f0e",
    },
    {
        "name": "Parler",
        "members": ["parler_tts_large_v1", "parler_tts_mini_v1"],
        "color": "#8c564b",
    },
]
FAMILY_LABEL_FONTSIZE = 15.0
FAMILY_EDGE_ALPHA = 0.5
FAMILY_FILL_ALPHA = 0.04
FAMILY_LINEWIDTH = 1.0
FAMILY_RADIUS_QUANTILE = 0.9
FAMILY_RADIUS_PAD = 0.08
FAMILY_TRIM_QUANTILE = 0.95
FAMILY_SMOOTHING_ITER = 2
FAMILY_LABEL_BG_ALPHA = 0.8
FAMILY_LABEL_PAD_FRACTION = 0.02

def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _amp_dtype(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError("amp_dtype must be one of: bf16, fp16")


def _safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz_meta_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(npz_path.suffix + ".meta.json")


def _current_data_meta(args: argparse.Namespace) -> dict[str, object]:
    return {
        "data_root": str(Path(args.data_root).resolve()),
        "protocol_csv": str(Path(args.protocol_csv).resolve()),
        "path_column": str(args.path_column),
        "class_column": str(args.class_column),
        "include_classes": _csv_list(args.include_classes),
        "exclude_classes": _csv_list(args.exclude_classes),
        "segment_seconds": args.segment_seconds,
        "sample_rate": int(args.sample_rate),
        "max_per_class": args.max_per_class,
        "max_total": args.max_total,
        "seed": int(args.seed),
    }


def _current_model_meta(prefix: str, args: argparse.Namespace) -> dict[str, object]:
    ckpt = getattr(args, f"{prefix}_checkpoint", None)
    src = None
    if ckpt is not None:
        ckpt_p = Path(ckpt).resolve()
        src = {"path": str(ckpt_p), "size": ckpt_p.stat().st_size if ckpt_p.exists() else None}
    return {
        "source": src,
        "extractor": getattr(args, f"{prefix}_extractor", None),
        "processor": getattr(args, f"{prefix}_processor", None),
        "classifier": getattr(args, f"{prefix}_classifier", None),
        "embed_layer": getattr(args, f"{prefix}_embed_layer", None),
        "bottleneck_dim": getattr(args, f"{prefix}_bottleneck_dim", None),
        "num_classes": getattr(args, f"{prefix}_num_classes", None),
        "l2_normalize": getattr(args, f"{prefix}_l2_normalize", None),
        "post_l2_normalize": bool(getattr(args, f"{prefix}_post_l2_normalize", False)),
    }


def _cache_compatible_by_meta(meta_path: Path, expected: dict[str, object]) -> bool:
    if not meta_path.exists():
        return False
    try:
        existing = _load_json(meta_path)
    except Exception:
        return False
    return _stable_json(existing) == _stable_json(expected)


class PathListDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        paths: list[str],
        sample_rate: int,
        segment_seconds: float | None,
    ):
        self.data_root = data_root
        self.paths = list(paths)
        self.sample_rate = int(sample_rate)
        self.segment_samples: int | None = None
        if segment_seconds and segment_seconds > 0:
            self.segment_samples = int(round(float(segment_seconds) * float(sample_rate)))

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, rel_path: str) -> str:
        normalized = str(rel_path).lstrip("./")
        return str(self.data_root / normalized)

    def _load_waveform(self, abs_path: str) -> torch.Tensor:
        waveform, sr = _read_audio_mono(abs_path)
        waveform = _maybe_resample(waveform, sr=sr, target_sr=self.sample_rate)
        if self.segment_samples is not None:
            if waveform.shape[0] >= self.segment_samples:
                waveform = waveform[: self.segment_samples]
            else:
                pad = self.segment_samples - waveform.shape[0]
                waveform = np.pad(waveform, (0, pad), mode="constant")
        waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        rel_path = str(self.paths[int(idx)])
        wav = self._load_waveform(self._resolve_path(rel_path))
        return rel_path, wav, 0


def _merge_metadata_on_path(
    base: pd.DataFrame,
    meta: pd.DataFrame,
    base_path_col: str,
    meta_path_col: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    base_path_col = str(base_path_col)
    meta_path_col = str(meta_path_col)
    df = base.copy()
    meta2 = meta.copy()
    meta2[meta_path_col] = meta2[meta_path_col].astype(str)
    meta2 = meta2.drop_duplicates(subset=[meta_path_col], keep="first")
    meta2 = meta2.rename(columns={meta_path_col: base_path_col})
    if prefix:
        protected = {base_path_col}
        rename_cols = {
            c: f"{prefix}{c}"
            for c in meta2.columns
            if c not in protected and c in df.columns
        }
        meta2 = meta2.rename(columns=rename_cols)
    return df.merge(meta2, on=base_path_col, how="left", sort=False)


def load_metadata_table(
    *,
    protocol_csv: Path,
    path_column: str,
    utt_ids_ordered: list[str],
    extra_metadata_csvs: list[Path],
    metadata_join_column: str | None,
) -> pd.DataFrame:
    path_column = str(path_column)
    proto = pd.read_csv(protocol_csv)
    if path_column not in proto.columns:
        raise ValueError(f"Protocol CSV missing path column: {path_column}")
    proto[path_column] = proto[path_column].astype(str)
    proto = proto.dropna(subset=[path_column]).copy()
    proto = proto.drop_duplicates(subset=[path_column], keep="first")

    base = pd.DataFrame({path_column: [str(u) for u in utt_ids_ordered]})
    base = _merge_metadata_on_path(base, proto, base_path_col=path_column, meta_path_col=path_column, prefix=None)

    join_col = str(metadata_join_column) if metadata_join_column else path_column
    for i, meta_path in enumerate(extra_metadata_csvs):
        meta = pd.read_csv(meta_path)
        if join_col not in meta.columns:
            raise ValueError(f"Metadata CSV {meta_path} missing join column: {join_col}")
        base = _merge_metadata_on_path(
            base,
            meta,
            base_path_col=path_column,
            meta_path_col=join_col,
            prefix=f"meta{i}_",
        )

    return base


def _choose_factor_columns(
    meta_df: pd.DataFrame,
    *,
    path_column: str,
    class_column: str,
    factors: str,
    max_factors: int,
    max_categories: int,
    min_count: int,
) -> list[str]:
    factors = str(factors).strip()
    if factors.lower() in {"", "none", "off"}:
        return []
    if factors.lower() != "auto":
        cols = [c.strip() for c in factors.split(",") if c.strip()]
        missing = [c for c in cols if c not in meta_df.columns]
        if missing:
            raise ValueError(f"Requested factor columns missing from metadata table: {', '.join(missing)}")
        return cols

    ignore = {str(path_column), str(class_column), "__label"}
    candidates: list[str] = []
    for c in meta_df.columns:
        if c in ignore:
            continue
        s = meta_df[c]
        # Treat common missing tokens as missing for factor selection.
        if not pd.api.types.is_numeric_dtype(s):
            cleaned = s.astype("object").fillna("").astype(str).str.strip()
            lowered = cleaned.str.lower()
            cleaned = cleaned.mask(lowered.isin({"", "nan", "none", "null"}), other=np.nan)
            s = cleaned
        # Skip very-high-cardinality text columns by default.
        nunique = int(s.dropna().astype(str).nunique())
        if nunique < 2 or nunique > int(max_categories):
            continue
        # Require at least `min_count` samples for the top category to avoid noise.
        counts = s.dropna().astype(str).value_counts()
        if counts.empty:
            continue
        if int(counts.iloc[0]) < int(min_count):
            continue
        candidates.append(c)
    # Heuristic ordering: prefer common MLAAD-style columns.
    preferred = [
        "speaker",
        "speaker_id",
        "meta_reference_speaker",
        "orig_speaker",
        "orig_gender",
        "orig_subset",
        "orig_locale",
        "meta_language",
        "language",
        "meta_training_data",
        "training_data",
        "model_architecture",
        "architecture",
        "model_type",
        "model_family",
        "lang_seen",
        "model_seen",
        "seen",
        "meta_is_original_language",
        "is_original_language",
    ]
    ordered: list[str] = []
    for pcol in preferred:
        if pcol in candidates:
            ordered.append(pcol)
    for c in candidates:
        if c not in ordered:
            ordered.append(c)
    return ordered[: int(max_factors)]


def _encode_categories(series: pd.Series, top_k: int) -> tuple[np.ndarray, list[str]]:
    s = series.astype("object")
    # Avoid leading underscores: matplotlib ignores legend labels that start with '_'.
    vals = s.where(~s.isna(), other="MISSING").astype(str)
    counts = vals.value_counts()
    keep = set(counts.head(int(top_k)).index.tolist())
    mapped = vals.where(vals.isin(list(keep)), other="OTHER")
    cats = sorted(mapped.unique().tolist())
    cat_to_id = {c: i for i, c in enumerate(cats)}
    ids = mapped.map(cat_to_id).to_numpy(dtype=np.int32)
    return ids, cats


def knn_purity(coords: np.ndarray, cat_ids: np.ndarray, k: int = 20, metric: str = "euclidean") -> float:
    coords = np.asarray(coords, dtype=np.float32)
    cat_ids = np.asarray(cat_ids, dtype=int)
    if coords.shape[0] < k + 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(coords)
    idxs = nn.kneighbors(return_distance=False)[:, 1:]
    purities: list[float] = []
    for i, row in enumerate(idxs):
        same = np.mean(cat_ids[row] == cat_ids[i])
        purities.append(float(same))
    return float(np.mean(purities)) if purities else float("nan")


def factor_separation_2d(coords: np.ndarray, cat_ids: np.ndarray) -> dict[str, float]:
    coords = np.asarray(coords, dtype=np.float32)
    cat_ids = np.asarray(cat_ids, dtype=int)
    if np.unique(cat_ids).size < 2:
        return {"silhouette_2d": float("nan"), "davies_bouldin_2d": float("nan"), "calinski_harabasz_2d": float("nan")}
    sil = float(silhouette_score(coords, cat_ids, metric="euclidean"))
    db = float(davies_bouldin_score(coords, cat_ids))
    ch = float(calinski_harabasz_score(coords, cat_ids))
    return {"silhouette_2d": sil, "davies_bouldin_2d": db, "calinski_harabasz_2d": ch}


def _style_scatter_axes(
    ax,
    show_labels: bool = False,
    show_frame: bool = False,
    frame_color: str = "black",
    frame_linewidth: float = 0.6,
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    if not show_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(show_frame)
        if show_frame:
            spine.set_linewidth(frame_linewidth)
            spine.set_color(frame_color)


def _axis_limits_from_coords(coords_list: list[np.ndarray], pad_fraction: float = 0.03) -> tuple[float, float, float, float]:
    stacked = np.vstack([np.asarray(c, dtype=np.float32) for c in coords_list if c is not None])
    xmin = float(np.min(stacked[:, 0]))
    xmax = float(np.max(stacked[:, 0]))
    ymin = float(np.min(stacked[:, 1]))
    ymax = float(np.max(stacked[:, 1]))
    span_x = xmax - xmin
    span_y = ymax - ymin
    span = max(span_x, span_y, 1e-6)
    pad = span * float(pad_fraction)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _inline_key_handles() -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    for set_name in PLOT_CLASS_SET_ORDER:
        marker = PLOT_CLASS_SET_MARKERS.get(set_name, "o")
        label = PLOT_CLASS_SET_LABELS.get(set_name, set_name)
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="None",
                markersize=6.0,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=0.6,
            )
        )
        labels.append(label)
    return handles, labels


def _add_inline_key(ax, fontsize: float):
    handles, labels = _inline_key_handles()
    if not handles:
        return None
    key = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        fontsize=fontsize,
        handletextpad=0.4,
        borderpad=0.25,
        labelspacing=0.2,
    )
    frame = key.get_frame()
    frame.set_alpha(0.7)
    frame.set_linewidth(0.4)
    frame.set_edgecolor("black")
    frame.set_facecolor("white")
    ax.add_artist(key)
    return key


def _build_family_groups(class_names: list[str]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for spec in PLOT_FAMILY_GROUPS:
        members = set(spec.get("members", []))
        prefixes = spec.get("prefixes", [])
        ids = []
        for idx, name in enumerate(class_names):
            if name in members or any(name.startswith(p) for p in prefixes):
                ids.append(idx)
        if ids:
            groups.append(
                {
                    "name": str(spec.get("name", "family")),
                    "class_ids": ids,
                    "color": str(spec.get("color", "#666666")),
                }
            )
    return groups


def _chaikin_smooth(points: np.ndarray, iterations: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3 or iterations <= 0:
        return pts
    for _ in range(iterations):
        new_pts: list[np.ndarray] = []
        n = pts.shape[0]
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        pts = np.vstack(new_pts)
    return pts


def _estimate_text_height_data(
    ax,
    axis_limits: tuple[float, float, float, float],
    fontsize: float,
) -> float:
    try:
        ax.figure.canvas.draw()
    except Exception:
        pass
    bbox = ax.get_window_extent()
    if bbox.height <= 0:
        return 0.0
    y_min, y_max = axis_limits[2], axis_limits[3]
    pixels = float(fontsize) * ax.figure.dpi / 72.0
    data_per_pixel = (y_max - y_min) / float(bbox.height)
    return pixels * data_per_pixel


def _clamp_label_position(
    label_x: float,
    label_y: float,
    axis_limits: tuple[float, float, float, float],
    pad_x: float,
    pad_y: float,
    text_height: float,
) -> tuple[float, float, str]:
    x_min, x_max, y_min, y_max = axis_limits
    label_x = float(np.clip(label_x, x_min + pad_x, x_max - pad_x))
    va = "bottom"
    if label_y + text_height > y_max - pad_y:
        label_y = y_max - pad_y
        va = "top"
    elif label_y - text_height < y_min + pad_y:
        label_y = y_min + pad_y
        va = "bottom"
    return label_x, float(label_y), va


def _draw_family_annotations(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    family_groups: list[dict[str, object]],
    label_fontsize: float,
    trim_quantile: float,
    axis_limits: tuple[float, float, float, float],
) -> None:
    x_min, x_max, y_min, y_max = axis_limits
    pad_x = (x_max - x_min) * FAMILY_LABEL_PAD_FRACTION
    pad_y = (y_max - y_min) * FAMILY_LABEL_PAD_FRACTION
    text_height = _estimate_text_height_data(ax, axis_limits, label_fontsize)
    ys = np.asarray(labels, dtype=int)
    for group in family_groups:
        class_ids = set(int(x) for x in group.get("class_ids", []))
        if not class_ids:
            continue
        mask = np.isin(ys, list(class_ids))
        if np.sum(mask) < 5:
            continue
        pts = np.asarray(coords, dtype=np.float32)[mask]
        if 0.0 < trim_quantile < 1.0 and pts.shape[0] >= 5:
            center = pts.mean(axis=0)
            dists = np.linalg.norm(pts - center, axis=1)
            cutoff = float(np.quantile(dists, trim_quantile))
            trimmed = pts[dists <= cutoff]
            if trimmed.shape[0] >= 3:
                pts = trimmed
        color = str(group.get("color", "#666666"))
        if pts.shape[0] >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                centroid = hull_pts.mean(axis=0)
                expanded = centroid + (hull_pts - centroid) * (1.0 + FAMILY_RADIUS_PAD)
                smooth = _chaikin_smooth(expanded, iterations=FAMILY_SMOOTHING_ITER)
                poly = plt.Polygon(
                    smooth,
                    closed=True,
                    edgecolor=mcolors.to_rgba(color, alpha=FAMILY_EDGE_ALPHA),
                    facecolor=mcolors.to_rgba(color, alpha=FAMILY_FILL_ALPHA),
                    linewidth=FAMILY_LINEWIDTH,
                    joinstyle="round",
                    zorder=1,
                )
                ax.add_patch(poly)
                y_span = max(float(smooth[:, 1].max() - smooth[:, 1].min()), 1e-6)
                label_x = float(smooth[:, 0].mean())
                label_y = float(smooth[:, 1].max() + y_span * 0.03)
                label_x, label_y, va = _clamp_label_position(
                    label_x,
                    label_y,
                    axis_limits,
                    pad_x,
                    pad_y,
                    text_height,
                )
            except Exception:
                hull = None
            else:
                ax.text(
                    label_x,
                    label_y,
                    str(group.get("name", "family")),
                    ha="center",
                    va=va,
                    fontsize=label_fontsize,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=FAMILY_LABEL_BG_ALPHA),
                    clip_on=True,
                    zorder=3,
                )
                continue
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        if dists.size == 0:
            continue
        radius = float(np.quantile(dists, FAMILY_RADIUS_QUANTILE)) * (1.0 + FAMILY_RADIUS_PAD)
        circ = plt.Circle(
            (float(center[0]), float(center[1])),
            radius,
            edgecolor=mcolors.to_rgba(color, alpha=FAMILY_EDGE_ALPHA),
            facecolor=mcolors.to_rgba(color, alpha=FAMILY_FILL_ALPHA),
            linewidth=FAMILY_LINEWIDTH,
            zorder=1,
        )
        ax.add_patch(circ)
        label_x, label_y, va = _clamp_label_position(
            float(center[0]),
            float(center[1] + radius * 1.05),
            axis_limits,
            pad_x,
            pad_y,
            text_height,
        )
        ax.text(
            label_x,
            label_y,
            str(group.get("name", "family")),
            ha="center",
            va=va,
            fontsize=label_fontsize,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=FAMILY_LABEL_BG_ALPHA),
            clip_on=True,
            zorder=3,
        )


def _generate_distinct_palette(n: int) -> list[str]:
    if n <= 0:
        return []
    golden_ratio = 0.61803398875
    colors: list[str] = []
    for i in range(n):
        h = (i * golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.9)
        colors.append(mcolors.to_hex((r, g, b)))
    return colors


def _class_centroids(coords: np.ndarray, labels: np.ndarray, keep_idx: list[int]) -> dict[int, np.ndarray]:
    ys = np.asarray(labels, dtype=int)[np.asarray(keep_idx, dtype=int)]
    centroids: dict[int, np.ndarray] = {}
    for cid in np.unique(ys):
        mask = ys == int(cid)
        if np.any(mask):
            centroids[int(cid)] = np.asarray(coords, dtype=np.float32)[mask].mean(axis=0)
    return centroids


def _build_adjacency(centroids: dict[int, np.ndarray], k: int) -> dict[int, set[int]]:
    class_ids = list(centroids.keys())
    adjacency: dict[int, set[int]] = {cid: set() for cid in class_ids}
    if len(class_ids) <= 1 or k <= 0:
        return adjacency
    for cid in class_ids:
        dists: list[tuple[float, int]] = []
        for other in class_ids:
            if other == cid:
                continue
            d = float(np.linalg.norm(centroids[cid] - centroids[other]))
            dists.append((d, other))
        dists.sort(key=lambda t: t[0])
        for _, other in dists[: min(k, len(dists))]:
            adjacency[cid].add(other)
            adjacency[other].add(cid)
    return adjacency


def _merge_adjacency(adjs: list[dict[int, set[int]]]) -> dict[int, set[int]]:
    merged: dict[int, set[int]] = {}
    for adj in adjs:
        for cid, neighbors in adj.items():
            merged.setdefault(cid, set()).update(neighbors)
    return merged


def _assign_colors_by_proximity(
    class_ids: list[int],
    adjacency: dict[int, set[int]],
    palette: list[str],
) -> dict[int, str]:
    if not class_ids:
        return {}
    palette_rgb = [np.asarray(mcolors.to_rgb(c), dtype=np.float32) for c in palette]
    if not palette_rgb:
        return {cid: "#4C78A8" for cid in class_ids}
    order = sorted(class_ids, key=lambda cid: len(adjacency.get(cid, set())), reverse=True)
    assigned: dict[int, np.ndarray] = {}
    for cid in order:
        neighbor_colors = [assigned[n] for n in adjacency.get(cid, set()) if n in assigned]
        best_idx = 0
        best_score = -1.0
        for idx, color in enumerate(palette_rgb):
            if neighbor_colors:
                min_dist = min(float(np.linalg.norm(color - other)) for other in neighbor_colors)
            elif assigned:
                min_dist = min(float(np.linalg.norm(color - other)) for other in assigned.values())
            else:
                min_dist = 0.0
            if min_dist > best_score:
                best_score = min_dist
                best_idx = idx
        assigned[cid] = palette_rgb[best_idx]
    return {cid: mcolors.to_hex(rgb) for cid, rgb in assigned.items()}


def _build_class_color_map(
    coords_list: list[np.ndarray],
    keep_list: list[list[int]],
    labels: np.ndarray,
    k: int = 4,
) -> dict[int, str]:
    centroids_list = []
    for coords, keep_idx in zip(coords_list, keep_list, strict=False):
        if coords is None or keep_idx is None:
            continue
        centroids_list.append(_class_centroids(coords, labels, keep_idx))
    if not centroids_list:
        return {}
    adjs = [_build_adjacency(centroids, k=k) for centroids in centroids_list]
    merged = _merge_adjacency(adjs)
    class_ids = sorted(merged.keys())
    palette = _generate_distinct_palette(max(len(class_ids), 1))
    return _assign_colors_by_proximity(class_ids, merged, palette)


def _plot_scatter_legend(
    class_ids: list[int],
    class_names: list[str],
    class_markers: dict[int, str] | None,
    out_path: Path,
    legend_figsize: tuple[float, float],
    legend_fontsize: float,
    legend_ncol: int,
    marker_size: float,
) -> None:
    if not class_ids:
        return
    plt.figure(figsize=legend_figsize)
    ax = plt.gca()
    ax.axis("off")
    cmap = plt.get_cmap("tab20")
    handles: list[Line2D] = []
    labels: list[str] = []
    marker_points = max(6.0, math.sqrt(marker_size) * 1.4)
    for i, cid in enumerate(class_ids):
        color = cmap(i % 20)
        marker = class_markers.get(int(cid), "o") if class_markers else "o"
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="None",
                markersize=marker_points,
                markerfacecolor=color,
                markeredgecolor=PLOT_EDGE_COLOR,
                markeredgewidth=PLOT_EDGE_WIDTH,
            )
        )
        labels.append(class_names[cid])
    ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=int(legend_ncol),
        fontsize=legend_fontsize,
        handletextpad=0.5,
        columnspacing=0.8,
    )
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def _plot_by_factor(
    coords: np.ndarray,
    series: pd.Series,
    out_path: Path,
    title: str,
    seed: int,
    top_k: int,
    legend_limit: int,
):
    rng = np.random.default_rng(int(seed))
    coords = np.asarray(coords, dtype=np.float32)
    cat_ids, cats = _encode_categories(series, top_k=top_k)

    # Shuffle draw order for nicer plots (avoid one category always on top).
    order = np.arange(coords.shape[0])
    rng.shuffle(order)
    coords = coords[order]
    cat_ids = cat_ids[order]

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")
    unique = np.unique(cat_ids)
    for i, cid in enumerate(unique.tolist()):
        mask = cat_ids == int(cid)
        if not np.any(mask):
            continue
        color = cmap(i % 20)
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=PLOT_ALPHA,
            color=color,
            edgecolors=PLOT_EDGE_COLOR,
            linewidths=PLOT_EDGE_WIDTH,
            rasterized=True,
        )
    _style_scatter_axes(plt.gca(), show_labels=False)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()


@dataclass(frozen=True)
class ProtocolSpec:
    data_root: Path
    protocol_csv: Path
    path_column: str
    class_column: str


class ProtocolSingleDataset(Dataset):
    def __init__(
        self,
        spec: ProtocolSpec,
        segment_seconds: float | None,
        sample_rate: int,
        include_classes: Iterable[str] | None,
        exclude_classes: Iterable[str] | None,
        max_per_class: int | None,
        max_total: int | None,
        seed: int,
    ):
        self.data_root = spec.data_root
        self.sample_rate = int(sample_rate)
        self.segment_samples: int | None = None
        if segment_seconds and segment_seconds > 0:
            self.segment_samples = int(round(float(segment_seconds) * float(sample_rate)))

        df = pd.read_csv(spec.protocol_csv)
        missing = {spec.path_column, spec.class_column} - set(df.columns)
        if missing:
            raise ValueError(
                f"{spec.protocol_csv} missing required columns: {', '.join(sorted(missing))}"
            )
        df = df.dropna(subset=[spec.path_column, spec.class_column]).copy()
        df[spec.path_column] = df[spec.path_column].astype(str)
        df[spec.class_column] = df[spec.class_column].astype(str)

        if include_classes:
            allowed = set(str(x) for x in include_classes)
            df = df[df[spec.class_column].isin(allowed)].reset_index(drop=True)
        if exclude_classes:
            banned = set(str(x) for x in exclude_classes)
            df = df[~df[spec.class_column].isin(banned)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Protocol filtering removed all rows.")

        rng = random.Random(seed)
        if max_per_class is not None and max_per_class > 0:
            kept = []
            for cls, sub in df.groupby(spec.class_column, sort=True):
                paths = sub[spec.path_column].tolist()
                if len(paths) > max_per_class:
                    chosen = rng.sample(paths, max_per_class)
                    sub = sub[sub[spec.path_column].isin(chosen)]
                kept.append(sub)
            df = pd.concat(kept, ignore_index=True)

        if max_total is not None and max_total > 0 and len(df) > max_total:
            indices = list(range(len(df)))
            rng.shuffle(indices)
            keep = sorted(indices[:max_total])
            df = df.iloc[keep].reset_index(drop=True)

        class_names = sorted(df[spec.class_column].unique().tolist())
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        df["__label"] = df[spec.class_column].map(self.class_to_id).astype(int)

        self.df = df[[spec.path_column, spec.class_column, "__label"]].copy()
        self.df = self.df.reset_index(drop=True)
        self.path_column = spec.path_column
        self.class_column = spec.class_column
        self.class_names = class_names

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> str:
        normalized = str(rel_path).lstrip("./")
        return str(self.data_root / normalized)

    def _load_waveform(self, abs_path: str) -> torch.Tensor:
        waveform, sr = _read_audio_mono(abs_path)
        waveform = _maybe_resample(waveform, sr=sr, target_sr=self.sample_rate)
        if self.segment_samples is not None:
            if waveform.shape[0] >= self.segment_samples:
                waveform = waveform[: self.segment_samples]
            else:
                pad = self.segment_samples - waveform.shape[0]
                waveform = np.pad(waveform, (0, pad), mode="constant")
        waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        row = self.df.iloc[int(idx)]
        rel_path = str(row[self.path_column])
        label = int(row["__label"])
        wav = self._load_waveform(self._resolve_path(rel_path))
        return rel_path, wav, label


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: Path
    extractor: str
    processor: str
    classifier: str
    embed_layer: str
    bottleneck_dim: int | None
    num_classes: int | None
    l2_normalize: bool | None
    post_l2_normalize: bool


def _build_torch_model(spec: ModelSpec, device: str, amp_eval: bool, amp_dtype: torch.dtype):
    args_ns = SimpleNamespace(
        extractor=spec.extractor,
        processor=spec.processor,
        classifier=spec.classifier,
        bottleneck_dim=spec.bottleneck_dim,
        num_classes=spec.num_classes,
        l2_normalize=spec.l2_normalize,
        kernel=None,
        n_components=None,
        covariance_type=None,
        finetune_ssl=False,
        extractor_lr=None,
    )
    model, trainer = build_model(args_ns)
    if hasattr(trainer, "set_amp_eval"):
        trainer.set_amp_eval(amp_eval, dtype=amp_dtype)
    trainer.load_model(str(spec.checkpoint))
    model.eval()
    model.to(device)
    return model


def _extract_batch_embeddings(model: torch.nn.Module, waveforms: torch.Tensor, layer: str) -> torch.Tensor:
    if not hasattr(model, "extractor"):
        raise RuntimeError("Model does not expose `extractor`; cannot extract embeddings.")

    feats = model.extractor.extract_features(waveforms)
    emb = feats
    if hasattr(model, "feature_processor"):
        emb = model.feature_processor(feats)

    layer = str(layer).lower().strip()
    if layer in {
        "processor",
        "processed",
        "pre_classifier",
        "pre-classifier",
        "feature_processor",
        "feature-processor",
        "featproc",
    }:
        return emb

    if layer in {"bottleneck", "bn"}:
        if hasattr(model, "bottleneck"):
            return model.bottleneck(emb)  # type: ignore[no-any-return]
        if hasattr(model, "extract_embedding") and not isinstance(model, FFBase):
            # e.g. FFMulticlass defines extract_embedding(waveforms)
            return model.extract_embedding(waveforms)  # type: ignore[no-any-return]
        if isinstance(model, FFBase):
            raise ValueError(
                "embed_layer=bottleneck requested, but this model is an FF/FFBase-style network (no `.bottleneck`). "
                "Use `--*-embed-layer penultimate` for FF/FFCosine1/FFCosineRaw2, or use classifier `FFMulticlass` "
                "if you want a true bottleneck embedding."
            )
        raise ValueError("embed_layer=bottleneck requested, but model has no `.bottleneck`.")

    if layer in {"penultimate", "ff_penultimate", "ffbase"}:
        if isinstance(model, FFBase):
            # Some subclasses (e.g. `classifiers/single_input/FF.FF`) override `extract_embedding`
            # with a different signature (taking waveforms). Call the base implementation
            # explicitly to ensure we extract from the FF classifier stack given processed features.
            return FFBase.extract_embedding(model, emb)  # type: ignore[no-any-return]
        raise ValueError("embed_layer=penultimate requested, but model is not an FFBase-derived network.")

    raise ValueError(f"Unknown embed_layer: {layer}")


def extract_embeddings(
    model_spec: ModelSpec,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    amp_eval: bool,
    amp_dtype: torch.dtype,
    oom_fallback: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model = _build_torch_model(model_spec, device=device, amp_eval=amp_eval, amp_dtype=amp_dtype)

    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "collate_fn": custom_single_batch_create,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.startswith("cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)

    autocast_ctx = (
        torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype)
        if amp_eval and device.startswith("cuda")
        else nullcontext()
    )

    all_ids: list[str] = []
    all_labels: list[int] = []
    all_emb: list[np.ndarray] = []
    with torch.no_grad():
        for file_names, waveforms, labels in tqdm(dataloader, desc=f"Embedding ({model_spec.classifier})"):
            waveforms = waveforms.to(device)
            try:
                with autocast_ctx:
                    emb = _extract_batch_embeddings(model, waveforms, model_spec.embed_layer)
                    if model_spec.post_l2_normalize:
                        emb = torch.nn.functional.normalize(emb, p=2, dim=1, eps=1e-12)
            except Exception as exc:
                is_cuda_oom = (
                    device.startswith("cuda")
                    and (
                        isinstance(exc, torch.cuda.OutOfMemoryError)
                        or ("out of memory" in str(exc).lower() and "cuda" in str(exc).lower())
                    )
                )
                if not (oom_fallback and is_cuda_oom):
                    raise

                # Retry with progressively smaller microbatches to handle rare long utterances.
                torch.cuda.empty_cache()
                micro = max(1, int(waveforms.shape[0]))
                succeeded = False
                last_exc: Exception | None = exc
                while micro >= 1 and not succeeded:
                    try:
                        parts: list[torch.Tensor] = []
                        for start in range(0, int(waveforms.shape[0]), micro):
                            chunk = waveforms[start : start + micro]
                            with autocast_ctx:
                                out = _extract_batch_embeddings(model, chunk, model_spec.embed_layer)
                                if model_spec.post_l2_normalize:
                                    out = torch.nn.functional.normalize(out, p=2, dim=1, eps=1e-12)
                            parts.append(out)
                        emb = torch.cat(parts, dim=0)
                        succeeded = True
                    except Exception as exc2:
                        last_exc = exc2
                        if not (
                            isinstance(exc2, torch.cuda.OutOfMemoryError)
                            or ("out of memory" in str(exc2).lower() and "cuda" in str(exc2).lower())
                        ):
                            raise
                        torch.cuda.empty_cache()
                        if micro == 1:
                            break
                        micro = max(1, micro // 2)
                if not succeeded:
                    assert last_exc is not None
                    raise last_exc
            emb_np = emb.detach().to(dtype=torch.float32).cpu().numpy()
            all_ids.extend([str(x) for x in file_names])
            all_labels.extend([int(x) for x in labels])
            all_emb.append(emb_np)

    emb_mat = np.concatenate(all_emb, axis=0).astype(np.float32, copy=False)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    return all_ids, emb_mat, labels_arr


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # average ranks for ties
    sorted_x = x[order]
    i = 0
    while i < sorted_x.size:
        j = i + 1
        while j < sorted_x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError("Spearman inputs must have same length.")
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _cosine_distance_matrix_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, eps, None)
    sim = Xn @ Xn.T
    return 1.0 - sim


def _pairwise_distances_sampled(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    intra_pairs_per_class: int,
    inter_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    metric = metric.lower().strip()

    by_class: dict[int, np.ndarray] = {}
    for cls in np.unique(labels):
        by_class[int(cls)] = np.where(labels == cls)[0]

    intra: list[float] = []
    for cls, idxs in by_class.items():
        if idxs.size < 2:
            continue
        n = min(intra_pairs_per_class, idxs.size * (idxs.size - 1) // 2)
        for _ in range(int(n)):
            i, j = rng.choice(idxs, size=2, replace=False)
            if metric == "euclidean":
                d = float(np.linalg.norm(X[i] - X[j]))
            elif metric == "cosine":
                a = X[i]
                b = X[j]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                d = 1.0 if denom == 0 else float(1.0 - np.dot(a, b) / denom)
            else:
                raise ValueError("distance_metric must be one of: euclidean, cosine")
            intra.append(d)

    classes = list(by_class.keys())
    if len(classes) < 2:
        return np.asarray(intra, dtype=float), np.asarray([], dtype=float)

    inter: list[float] = []
    for _ in range(int(inter_pairs)):
        c1, c2 = rng.choice(classes, size=2, replace=False)
        i = int(rng.choice(by_class[int(c1)]))
        j = int(rng.choice(by_class[int(c2)]))
        if metric == "euclidean":
            d = float(np.linalg.norm(X[i] - X[j]))
        else:
            a = X[i]
            b = X[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            d = 1.0 if denom == 0 else float(1.0 - np.dot(a, b) / denom)
        inter.append(d)
    return np.asarray(intra, dtype=float), np.asarray(inter, dtype=float)


def knn_overlap(
    Xa: np.ndarray,
    Xb: np.ndarray,
    k: int,
    metric: str,
) -> dict[str, float]:
    metric = metric.lower().strip()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("distance_metric must be one of: euclidean, cosine")
    if Xa.shape[0] != Xb.shape[0]:
        raise ValueError("Xa and Xb must have same number of points.")

    nn_a = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(Xa)
    nn_b = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(Xb)
    idx_a = nn_a.kneighbors(return_distance=False)[:, 1:]
    idx_b = nn_b.kneighbors(return_distance=False)[:, 1:]

    overlaps = []
    jaccards = []
    for a_row, b_row in zip(idx_a, idx_b, strict=False):
        a_set = set(int(x) for x in a_row.tolist())
        b_set = set(int(x) for x in b_row.tolist())
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        overlaps.append(inter / float(k))
        jaccards.append(inter / float(union) if union else 0.0)
    return {
        "knn_overlap_frac": float(np.mean(overlaps)) if overlaps else float("nan"),
        "knn_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
    }


def twonn_intrinsic_dim(X: np.ndarray, metric: str, max_points: int | None, seed: int) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 5:
        return {"twonn_id": float("nan"), "twonn_r2": float("nan")}
    metric = metric.lower().strip()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("distance_metric must be one of: euclidean, cosine")

    rng = np.random.default_rng(seed)
    if max_points is not None and max_points > 0 and X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]

    nn = NearestNeighbors(n_neighbors=3, metric=metric).fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    d1 = dists[:, 1]
    d2 = dists[:, 2]
    valid = (d1 > 0) & np.isfinite(d1) & np.isfinite(d2) & (d2 > 0)
    mu = (d2[valid] / d1[valid]).astype(np.float64)
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    if mu.size < 10:
        return {"twonn_id": float("nan"), "twonn_r2": float("nan")}

    mu.sort()
    n = mu.size
    x = np.log(mu)
    # Empirical CDF with a small offset to avoid inf.
    F = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / float(n)
    y = -np.log(1.0 - F)

    # Linear regression y = m*x + b (b included, but slope is the ID estimate)
    m, b = np.polyfit(x, y, deg=1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {"twonn_id": float(m), "twonn_r2": r2}


def pca_spectrum_metrics(X: np.ndarray, max_components: int = 512) -> dict[str, object]:
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    n, d = X.shape
    n_components = min(max_components, d, n - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(X)
    lam = np.asarray(pca.explained_variance_, dtype=np.float64)
    total = float(np.sum(lam))
    if total <= 0:
        return {"effective_rank": float("nan"), "participation_ratio": float("nan"), "pca": {"eigs": lam.tolist()}}
    p = lam / total
    entropy = -float(np.sum(np.where(p > 0, p * np.log(p), 0.0)))
    effective_rank = float(np.exp(entropy))
    participation_ratio = float((total * total) / float(np.sum(lam * lam)))
    top1 = float(lam[0] / total) if lam.size else float("nan")
    top10 = float(np.sum(lam[: min(10, lam.size)]) / total) if lam.size else float("nan")
    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "pca_top1_var": top1,
        "pca_top10_var": top10,
        "pca": {"eigs": lam.tolist(), "explained_variance_ratio": pca.explained_variance_ratio_.tolist()},
    }


def centroid_graph_metrics(X: np.ndarray, labels: np.ndarray, metric: str) -> dict[str, object]:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    classes = sorted(int(c) for c in np.unique(labels).tolist())
    if len(classes) < 2:
        return {"num_centroids": len(classes)}
    centroids = np.stack([X[labels == c].mean(axis=0) for c in classes], axis=0)
    metric = metric.lower().strip()
    if metric == "euclidean":
        # squared form isn't needed; keep Euclidean.
        diffs = centroids[:, None, :] - centroids[None, :, :]
        D = np.sqrt(np.sum(diffs * diffs, axis=-1))
    elif metric == "cosine":
        D = _cosine_distance_matrix_rows(centroids)
    else:
        raise ValueError("distance_metric must be one of: euclidean, cosine")
    upper = D[np.triu_indices(D.shape[0], k=1)]
    nearest = np.argsort(D, axis=1)[:, 1]
    return {
        "num_centroids": int(D.shape[0]),
        "centroid_distance_upper": upper.astype(np.float64),
        "centroid_nearest": nearest.astype(int),
    }


def _plot_tsne(
    X: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
    perplexity: float,
    seed: int,
    max_classes: int,
    samples_per_class: int,
    plot_class_ids: set[int] | None = None,
    tsne_metric: str = "euclidean",
    show_legend: bool = True,
    class_markers: dict[int, str] | None = None,
    class_order: list[int] | None = None,
    figsize: tuple[float, float] = (3.2, 3.2),
    marker_size: float = 16.0,
    legend_fontsize: float = 10.0,
    legend_ncol: int = 2,
    inline_key: bool = False,
    inline_key_fontsize: float = 8.0,
    density: bool = False,
    density_gridsize: int = 35,
    density_alpha: float = 0.12,
    show_frame: bool = False,
    frame_color: str = "black",
    frame_linewidth: float = 0.5,
    color_mode: str = "single",
    point_color: str | None = None,
    class_color_map: dict[int, str] | None = None,
    family_groups: list[dict[str, object]] | None = None,
    family_label_fontsize: float = FAMILY_LABEL_FONTSIZE,
    family_trim_quantile: float = FAMILY_TRIM_QUANTILE,
    family_annotations: bool = False,
    legend_mode: str = "inside",
    legend_out_path: Path | None = None,
    legend_figsize: tuple[float, float] | None = None,
    class_hardness_rank: dict[int, int] | None = None,
    axis_limits: tuple[float, float, float, float] | None = None,
    coords: np.ndarray | None = None,
    keep_idx: list[int] | None = None,
    compute_metrics: bool = True,
    save_plot: bool = True,
):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    tsne_metric = str(tsne_metric).lower().strip()
    metrics: dict[str, object] = {}
    if coords is None:
        X = np.asarray(X, dtype=np.float32)
        classes, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        classes = classes[order]
        if plot_class_ids:
            keep_classes = set(int(c) for c in classes.tolist() if int(c) in plot_class_ids)
        else:
            keep_classes = set(int(c) for c in classes[: max_classes].tolist())

        keep_idx = []
        for c in classes.tolist():
            idxs = np.where(labels == int(c))[0]
            if int(c) in keep_classes:
                if samples_per_class > 0 and idxs.size > samples_per_class:
                    idxs = rng.choice(idxs, size=samples_per_class, replace=False)
                keep_idx.extend(idxs.tolist())
        keep_idx = sorted(set(keep_idx))
        if len(keep_idx) < 10:
            raise ValueError("Too few samples left for t-SNE after plotting filters.")

        Xs = X[keep_idx]
        ys = labels[keep_idx]

        # PCA init to stabilize t-SNE; cap dims for speed.
        pca_dim = min(50, Xs.shape[1], Xs.shape[0] - 1)
        Xp = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed).fit_transform(Xs)
        if tsne_metric == "cosine":
            # For cosine, normalize after PCA to ensure the metric isn't dominated by norms.
            Xp = Xp / np.clip(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-12, None)

        tsne = TSNE(
            n_components=2,
            perplexity=float(perplexity),
            init="pca",
            learning_rate="auto",
            random_state=int(seed),
            metric=tsne_metric,
            method="barnes_hut" if tsne_metric == "euclidean" else "exact",
        )
        Z = tsne.fit_transform(Xp)
        if compute_metrics:
            tw = float(trustworthiness(Xp, Z, n_neighbors=10, metric=tsne_metric))
            metrics = {
                "tsne_metric": tsne_metric,
                "tsne_trustworthiness@10": tw,
                "tsne_points": int(Z.shape[0]),
                "tsne_classes": int(len(keep_classes)),
            }
    else:
        if keep_idx is None:
            raise ValueError("keep_idx must be provided when coords are precomputed.")
        keep_idx = list(keep_idx)
        ys = labels[keep_idx]
        keep_classes = set(int(c) for c in np.unique(ys).tolist())
        Z = np.asarray(coords, dtype=np.float32)

    if save_plot:
        # Plot only the kept classes; ignore others (already filtered).
        plt.figure(figsize=figsize)
        cmap = plt.get_cmap("tab20")
        if class_order:
            keep_set = set(keep_classes)
            ordered = [c for c in class_order if c in keep_set]
            remainder = sorted((c for c in keep_set if c not in set(ordered)), key=lambda cid: class_names[cid])
            plot_classes = ordered + remainder
        else:
            plot_classes = sorted(keep_classes)
        if class_hardness_rank:
            order_index = {c: i for i, c in enumerate(plot_classes)}
            plot_classes = sorted(
                plot_classes, key=lambda cid: (class_hardness_rank.get(int(cid), 0), order_index.get(int(cid), 0))
            )
        ax = plt.gca()
        if axis_limits is None:
            axis_limits = _axis_limits_from_coords([Z], pad_fraction=0.03)
        plt.xlim(axis_limits[0], axis_limits[1])
        plt.ylim(axis_limits[2], axis_limits[3])
        if density:
            ax.hexbin(
                Z[:, 0],
                Z[:, 1],
                gridsize=int(density_gridsize),
                mincnt=1,
                linewidths=0.0,
                cmap="Greys",
                alpha=float(density_alpha),
                zorder=0,
            )
        if family_annotations and family_groups:
            _draw_family_annotations(
                ax,
                Z,
                ys,
                family_groups,
                label_fontsize=family_label_fontsize,
                trim_quantile=family_trim_quantile,
                axis_limits=axis_limits,
            )
        for i, c in enumerate(plot_classes):
            idxs = np.where(ys == c)[0]
            if idxs.size == 0:
                continue
            class_name = class_names[int(c)]
            if color_mode != "single" and class_name in PLOT_CLASS_COLORS:
                color = PLOT_CLASS_COLORS[class_name]
            elif class_color_map and int(c) in class_color_map:
                color = class_color_map[int(c)]
            elif color_mode == "by-class":
                color = cmap(int(c) % 20)
            else:
                color = point_color or "#4C78A8"
            marker = class_markers.get(int(c), "o") if class_markers else "o"
            plt.scatter(
                Z[idxs, 0],
                Z[idxs, 1],
                s=marker_size,
                alpha=PLOT_ALPHA,
                color=color,
                linewidths=0.0,
                marker=marker,
                label=class_names[c] if show_legend else None,
                rasterized=True,
                zorder=2,
            )
        _style_scatter_axes(
            plt.gca(),
            show_labels=False,
            show_frame=show_frame,
            frame_color=frame_color,
            frame_linewidth=frame_linewidth,
        )
        class_legend = None
        if show_legend and legend_mode == "inside":
            class_legend = plt.legend(
                markerscale=2.0,
                fontsize=legend_fontsize,
                loc="upper left",
                bbox_to_anchor=(0.01, 0.99),
                frameon=False,
                ncol=int(legend_ncol),
                borderaxespad=0.0,
                handletextpad=0.5,
                columnspacing=0.8,
            )
        if inline_key:
            _add_inline_key(ax, fontsize=inline_key_fontsize)
            if class_legend is not None:
                ax.add_artist(class_legend)
        plt.tight_layout(pad=0.1)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        if show_legend and legend_mode == "separate" and legend_out_path is not None and legend_figsize is not None:
            _plot_scatter_legend(
                class_ids=plot_classes,
                class_names=class_names,
                class_markers=class_markers,
                out_path=legend_out_path,
                legend_figsize=legend_figsize,
                legend_fontsize=legend_fontsize,
                legend_ncol=legend_ncol,
                marker_size=marker_size,
            )

    return Z, metrics, keep_idx


def _plot_distance_distributions(
    intra_a: np.ndarray,
    inter_a: np.ndarray,
    intra_b: np.ndarray,
    inter_b: np.ndarray,
    out_path: Path,
    title: str,
    show_legend: bool,
):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    bins = 60
    if intra_a.size:
        plt.hist(intra_a, bins=bins, density=True, alpha=0.35, label="A intra")
    if inter_a.size:
        plt.hist(inter_a, bins=bins, density=True, alpha=0.35, label="A inter")
    if intra_b.size:
        plt.hist(intra_b, bins=bins, density=True, alpha=0.35, label="B intra")
    if inter_b.size:
        plt.hist(inter_b, bins=bins, density=True, alpha=0.35, label="B inter")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    if show_legend:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_pca_spectrum(eigs_a: list[float], eigs_b: list[float], out_path: Path, title: str, show_legend: bool):
    ea = np.asarray(eigs_a, dtype=float)
    eb = np.asarray(eigs_b, dtype=float)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ea.size:
        plt.semilogy(np.arange(1, ea.size + 1), ea, label="A")
    if eb.size:
        plt.semilogy(np.arange(1, eb.size + 1), eb, label="B")
    plt.xlabel("PCA component")
    plt.ylabel("Eigenvalue (log)")
    if show_legend:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_pca_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    out_pc12: Path,
    out_pc13: Path,
    title_prefix: str,
    seed: int,
    max_classes: int,
    samples_per_class: int,
    plot_class_ids: set[int] | None = None,
    show_legend: bool = True,
    class_markers: dict[int, str] | None = None,
    class_order: list[int] | None = None,
    figsize: tuple[float, float] = (3.2, 3.2),
    marker_size: float = 16.0,
    legend_fontsize: float = 10.0,
    legend_ncol: int = 2,
    inline_key: bool = False,
    inline_key_fontsize: float = 8.0,
    density: bool = False,
    density_gridsize: int = 35,
    density_alpha: float = 0.12,
    show_frame: bool = False,
    frame_color: str = "black",
    frame_linewidth: float = 0.5,
    color_mode: str = "single",
    point_color: str | None = None,
    class_color_map: dict[int, str] | None = None,
    class_hardness_rank: dict[int, int] | None = None,
    family_groups: list[dict[str, object]] | None = None,
    family_label_fontsize: float = FAMILY_LABEL_FONTSIZE,
    family_trim_quantile: float = FAMILY_TRIM_QUANTILE,
    family_annotations: bool = False,
):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    X = np.asarray(X, dtype=np.float32)

    classes, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    classes = classes[order]
    if plot_class_ids:
        keep_classes = set(int(c) for c in classes.tolist() if int(c) in plot_class_ids)
    else:
        keep_classes = set(int(c) for c in classes[: max_classes].tolist())

    keep_idx: list[int] = []
    for c in classes.tolist():
        idxs = np.where(labels == int(c))[0]
        if int(c) in keep_classes:
            if samples_per_class > 0 and idxs.size > samples_per_class:
                idxs = rng.choice(idxs, size=samples_per_class, replace=False)
            keep_idx.extend(idxs.tolist())
    keep_idx = sorted(set(keep_idx))
    if len(keep_idx) < 3:
        raise ValueError("Too few samples left for PCA scatter after plotting filters.")

    Xs = X[keep_idx]
    ys = labels[keep_idx]
    pca = PCA(n_components=3, svd_solver="randomized", random_state=int(seed))
    Z = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_

    def _scatter(out_path: Path, x_idx: int, y_idx: int, title: str):
        plt.figure(figsize=figsize)
        plt.title(title)
        cmap = plt.get_cmap("tab20")
        ax = plt.gca()
        axis_limits = _axis_limits_from_coords([Z[:, [x_idx, y_idx]]], pad_fraction=0.03)
        plt.xlim(axis_limits[0], axis_limits[1])
        plt.ylim(axis_limits[2], axis_limits[3])
        if density:
            ax.hexbin(
                Z[:, x_idx],
                Z[:, y_idx],
                gridsize=int(density_gridsize),
                mincnt=1,
                linewidths=0.0,
                cmap="Greys",
                alpha=float(density_alpha),
                zorder=0,
            )
        if family_annotations and family_groups:
            _draw_family_annotations(
                ax,
                Z[:, [x_idx, y_idx]],
                ys,
                family_groups,
                label_fontsize=family_label_fontsize,
                trim_quantile=family_trim_quantile,
                axis_limits=axis_limits,
            )
        if class_order:
            keep_set = set(keep_classes)
            ordered = [c for c in class_order if c in keep_set]
            remainder = sorted((c for c in keep_set if c not in set(ordered)), key=lambda cid: class_names[cid])
            plot_classes = ordered + remainder
        else:
            plot_classes = sorted(keep_classes)
        if class_hardness_rank:
            order_index = {c: i for i, c in enumerate(plot_classes)}
            plot_classes = sorted(
                plot_classes, key=lambda cid: (class_hardness_rank.get(int(cid), 0), order_index.get(int(cid), 0))
            )
        for i, c in enumerate(plot_classes):
            idxs = np.where(ys == c)[0]
            if idxs.size == 0:
                continue
            class_name = class_names[int(c)]
            if color_mode != "single" and class_name in PLOT_CLASS_COLORS:
                color = PLOT_CLASS_COLORS[class_name]
            elif class_color_map and int(c) in class_color_map:
                color = class_color_map[int(c)]
            elif color_mode == "by-class":
                color = cmap(int(c) % 20)
            else:
                color = point_color or "#4C78A8"
            marker = class_markers.get(int(c), "o") if class_markers else "o"
            plt.scatter(
                Z[idxs, x_idx],
                Z[idxs, y_idx],
                s=marker_size,
                alpha=PLOT_ALPHA,
                color=color,
                marker=marker,
                label=class_names[c],
                linewidths=0.0,
                zorder=2,
            )
        plt.xlabel(f"PC{x_idx+1} ({evr[x_idx]*100:.2f}%)")
        plt.ylabel(f"PC{y_idx+1} ({evr[y_idx]*100:.2f}%)")
        _style_scatter_axes(
            plt.gca(),
            show_labels=True,
            show_frame=show_frame,
            frame_color=frame_color,
            frame_linewidth=frame_linewidth,
        )
        class_legend = None
        if show_legend:
            class_legend = plt.legend(
                markerscale=2.0,
                fontsize=legend_fontsize,
                loc="upper left",
                bbox_to_anchor=(0.01, 0.99),
                frameon=False,
                ncol=int(legend_ncol),
                borderaxespad=0.0,
                handletextpad=0.5,
                columnspacing=0.8,
            )
        if inline_key:
            _add_inline_key(ax, fontsize=inline_key_fontsize)
            if class_legend is not None:
                ax.add_artist(class_legend)
        plt.tight_layout(pad=0.1)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close()

    _scatter(out_pc12, 0, 1, f"{title_prefix}: PCA PC1 vs PC2")
    _scatter(out_pc13, 0, 2, f"{title_prefix}: PCA PC1 vs PC3")
    return keep_idx, Z, evr


def _write_npz(out_path: Path, utt_ids: list[str], emb: np.ndarray, labels: np.ndarray, class_names: list[str]):
    np.savez_compressed(
        out_path,
        utt_ids=np.asarray(utt_ids, dtype=object),
        embeddings=np.asarray(emb, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(class_names, dtype=object),
    )


def _try_load_cached_embeddings(
    *,
    tag: str,
    out_dir: Path,
    expected_meta: dict[str, object],
    expected_utt_ids: list[str],
    expected_labels: np.ndarray,
    verify_samples: int,
    model_spec: ModelSpec | None,
    device: str,
    batch_size: int,
    num_workers: int,
    amp_eval: bool,
    amp_dtype: torch.dtype,
    segment_seconds: float | None,
    sample_rate: int,
    data_root: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]] | None:
    """
    Return cached embeddings if they look compatible; otherwise return None.
    """
    npz_path = out_dir / f"embeddings_{tag}.npz"
    if not npz_path.exists():
        return None

    meta_path = _npz_meta_path(npz_path)
    meta_ok = _cache_compatible_by_meta(meta_path, expected_meta)

    try:
        ids, X, y, class_names = _load_npz(npz_path)
    except Exception:
        return None

    # Always require utt_id/label match against *current* protocol selection to avoid silent mixups.
    if ids != expected_utt_ids:
        return None
    if y.shape != expected_labels.shape or not np.array_equal(y, expected_labels):
        return None
    if X.shape[0] != len(ids):
        return None

    if verify_samples and verify_samples > 0 and model_spec is not None:
        rng = np.random.default_rng(0)
        n = len(ids)
        k = min(int(verify_samples), n)
        sample_idx = rng.choice(n, size=k, replace=False)
        sample_ids = [ids[int(i)] for i in sample_idx.tolist()]
        sample_X = X[sample_idx]

        # Recompute those embeddings and compare by cosine similarity.
        subset = PathListDataset(
            data_root=data_root,
            paths=sample_ids,
            sample_rate=int(sample_rate),
            segment_seconds=segment_seconds,
        )
        _, X2, _ = extract_embeddings(
            model_spec,
            dataset=subset,
            device=device,
            batch_size=min(batch_size, k),
            num_workers=num_workers,
            amp_eval=amp_eval,
            amp_dtype=amp_dtype,
        )

        a = sample_X.astype(np.float64, copy=False)
        b = X2.astype(np.float64, copy=False)
        an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
        cos = np.sum(an * bn, axis=1)
        mean_cos = float(np.mean(cos)) if cos.size else float("nan")
        min_cos = float(np.min(cos)) if cos.size else float("nan")
        if not (mean_cos >= 0.9990 and min_cos >= 0.9950):
            # If the cache is stale (different checkpoint/settings), don't use it.
            return None

    # If meta mismatched but the strict protocol checks passed, allow cache use but warn.
    if not meta_ok:
        print(f"[cache] Using {npz_path} based on protocol match (metadata missing/mismatched).")
    else:
        print(f"[cache] Using {npz_path} (metadata match).")
    return ids, X, y, class_names


def _load_npz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    utt_ids = [str(x) for x in np.asarray(data["utt_ids"]).tolist()]
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    class_names = [str(x) for x in np.asarray(data["class_names"]).tolist()]
    return utt_ids, emb, labels, class_names


def _align_by_utt_id(
    ids_a: list[str],
    Xa: np.ndarray,
    ya: np.ndarray,
    ids_b: list[str],
    Xb: np.ndarray,
    yb: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    idx_a = {utt: i for i, utt in enumerate(ids_a)}
    idx_b = {utt: i for i, utt in enumerate(ids_b)}
    common = sorted(set(idx_a) & set(idx_b))
    if not common:
        raise ValueError("No overlapping utt_ids between model A and B embeddings.")
    ia = np.asarray([idx_a[u] for u in common], dtype=int)
    ib = np.asarray([idx_b[u] for u in common], dtype=int)
    Xa2 = Xa[ia]
    Xb2 = Xb[ib]
    # Prefer A labels; assert label consistency where available
    ya2 = ya[ia]
    yb2 = yb[ib]
    if not np.array_equal(ya2, yb2):
        mismatch = int(np.sum(ya2 != yb2))
        raise ValueError(f"Label mismatch after alignment for {mismatch} utterances (check protocol/filters).")
    return common, Xa2, Xb2, ya2


def _separability_metrics(X: np.ndarray, labels: np.ndarray, metric: str) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    metric = metric.lower().strip()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("distance_metric must be one of: euclidean, cosine")
    # Silhouette supports cosine; the others use Euclidean internally.
    sil = float(silhouette_score(X, labels, metric=metric))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


def _distance_rank_corr(
    Xa: np.ndarray,
    Xb: np.ndarray,
    metric: str,
    max_points: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = Xa.shape[0]
    if n < 5:
        return float("nan")
    if max_points > 0 and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        Xa = Xa[idx]
        Xb = Xb[idx]
        n = Xa.shape[0]

    metric = metric.lower().strip()
    if metric == "cosine":
        Da = _cosine_distance_matrix_rows(Xa)
        Db = _cosine_distance_matrix_rows(Xb)
    elif metric == "euclidean":
        diffa = Xa[:, None, :] - Xa[None, :, :]
        diffb = Xb[:, None, :] - Xb[None, :, :]
        Da = np.sqrt(np.sum(diffa * diffa, axis=-1))
        Db = np.sqrt(np.sum(diffb * diffb, axis=-1))
    else:
        raise ValueError("distance_metric must be one of: euclidean, cosine")
    iu = np.triu_indices(n, k=1)
    return spearman_corr(Da[iu], Db[iu])


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.clip(norms, eps, None)).astype(np.float32, copy=False)


def _percentiles(x: np.ndarray, ps: list[float]) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p*100):02d}": float("nan") for p in ps}
    values = np.percentile(x, [p * 100.0 for p in ps]).astype(float)
    return {f"p{int(p*100):02d}": float(v) for p, v in zip(ps, values, strict=False)}


def centroid_tail_margins(X: np.ndarray, labels: np.ndarray, metric: str) -> dict[str, object]:
    """
    For each sample:
      pos = dist(x, centroid(y))
      neg = min_{c!=y} dist(x, centroid(c))
      margin = neg - pos
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    classes = sorted(int(c) for c in np.unique(labels).tolist())
    if len(classes) < 2:
        return {"margin_p01": float("nan"), "num_classes": len(classes)}
    centroids = np.stack([X[labels == c].mean(axis=0) for c in classes], axis=0)
    metric = metric.lower().strip()

    if metric == "euclidean":
        # pos distances
        pos = np.zeros(X.shape[0], dtype=np.float64)
        for idx, c in enumerate(classes):
            mask = labels == c
            if not np.any(mask):
                continue
            diffs = X[mask] - centroids[idx]
            pos[mask] = np.sqrt(np.sum(diffs * diffs, axis=1))
        # neg centroid distance
        # D[i, j] = dist(x_i, centroid_j)
        diffs = X[:, None, :] - centroids[None, :, :]
        D = np.sqrt(np.sum(diffs * diffs, axis=-1))
    elif metric == "cosine":
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        Cn = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
        sim = Xn @ Cn.T
        D = 1.0 - sim
        pos = np.zeros(X.shape[0], dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            ci = class_to_idx[c]
            mask = labels == c
            pos[mask] = D[mask, ci]
    else:
        raise ValueError("distance_metric must be one of: euclidean, cosine")

    # Mask the true class to compute min negative
    class_to_idx = {c: i for i, c in enumerate(classes)}
    neg = np.full(X.shape[0], np.inf, dtype=np.float64)
    for i in range(X.shape[0]):
        ci = class_to_idx[int(labels[i])]
        row = D[i].copy()
        row[ci] = np.inf
        neg[i] = float(np.min(row))
    margin = neg - pos

    stats = {
        "pos_mean": float(np.mean(pos)),
        "neg_mean": float(np.mean(neg)),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_p01": float(np.percentile(margin, 1.0)),
        "margin_p05": float(np.percentile(margin, 5.0)),
        "margin_p95": float(np.percentile(margin, 95.0)),
        "num_classes": int(len(classes)),
        "num_samples": int(X.shape[0]),
    }
    return stats


def negative_neighbor_tail_margins(X: np.ndarray, labels: np.ndarray, metric: str, k: int = 64) -> dict[str, object]:
    """
    For each sample:
      pos = dist(x, centroid(y))
      neg_nn = distance to nearest neighbor with label != y (approx by querying k-NN list)
      margin_nn = neg_nn - pos
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size < 2 or X.shape[0] < 3:
        return {"margin_nn_p01": float("nan")}
    metric = metric.lower().strip()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("distance_metric must be one of: euclidean, cosine")

    # pos to centroid
    classes = sorted(int(c) for c in np.unique(labels).tolist())
    centroids = np.stack([X[labels == c].mean(axis=0) for c in classes], axis=0)
    if metric == "euclidean":
        pos = np.zeros(X.shape[0], dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            ci = class_to_idx[c]
            mask = labels == c
            diffs = X[mask] - centroids[ci]
            pos[mask] = np.sqrt(np.sum(diffs * diffs, axis=1))
    else:
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        Cn = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
        sim = Xn @ Cn.T
        D = 1.0 - sim
        pos = np.zeros(X.shape[0], dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            ci = class_to_idx[c]
            mask = labels == c
            pos[mask] = D[mask, ci]

    k_eff = min(int(k) + 1, X.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)
    neg_nn = np.full(X.shape[0], np.nan, dtype=np.float64)
    for i in range(X.shape[0]):
        for d, j in zip(dists[i][1:], idxs[i][1:], strict=False):
            if int(labels[int(j)]) != int(labels[i]):
                neg_nn[i] = float(d)
                break
    margin_nn = neg_nn - pos
    finite = np.isfinite(margin_nn)
    if not np.any(finite):
        return {"margin_nn_p01": float("nan"), "margin_nn_nan_frac": 1.0}
    stats = {
        "margin_nn_mean": float(np.nanmean(margin_nn)),
        "margin_nn_median": float(np.nanmedian(margin_nn)),
        "margin_nn_p01": float(np.nanpercentile(margin_nn, 1.0)),
        "margin_nn_p05": float(np.nanpercentile(margin_nn, 5.0)),
        "margin_nn_nan_frac": float(np.mean(~np.isfinite(margin_nn))),
        "knn_k": int(k),
    }
    return stats


def _write_points_csv(
    out_path: Path,
    *,
    utt_ids: list[str],
    labels: np.ndarray,
    class_names: list[str],
    keep_idx: list[int],
    coords: np.ndarray,
    coord_names: list[str],
    meta_df: pd.DataFrame | None,
) -> None:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != len(coord_names):
        raise ValueError("coords must be shape (N, len(coord_names))")
    keep = np.asarray(keep_idx, dtype=int)
    df = pd.DataFrame(
        {
            "utt_id": [utt_ids[int(i)] for i in keep.tolist()],
            "label_id": labels[keep].astype(int),
        }
    )
    df["label_name"] = df["label_id"].map(lambda i: class_names[int(i)] if int(i) < len(class_names) else str(i))
    for j, name in enumerate(coord_names):
        df[name] = coords[:, j]
    if meta_df is not None:
        # meta_df is aligned to utt_ids; subset by keep.
        meta_sub = meta_df.iloc[keep].reset_index(drop=True)
        # Avoid duplicating core columns.
        for col in meta_sub.columns:
            if col in {"utt_id", "label_id", "label_name"}:
                continue
            if col in df.columns:
                continue
            df[col] = meta_sub[col].values
    df.to_csv(out_path, index=False)


def _slug(value: str) -> str:
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        elif ch in {" ", "/", ":", ".", ","}:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "col"


def main() -> None:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data-root", type=Path, required=True, help="Root directory used to resolve `path` entries.")
    p.add_argument("--protocol-csv", type=Path, required=True, help="CSV with `path` and class columns.")
    p.add_argument("--path-column", type=str, default="path")
    p.add_argument("--class-column", type=str, default="model_name")
    p.add_argument("--include-classes", type=str, default=None, help="Comma-separated classes to keep.")
    p.add_argument("--exclude-classes", type=str, default="bonafide", help="Comma-separated classes to drop.")
    p.add_argument("--segment-seconds", type=float, default=None, help="Optional fixed segment length.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--max-total", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)

    # Extraction
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp-eval", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="bf16")
    p.add_argument(
        "--no-oom-fallback",
        action="store_true",
        help="Disable CUDA OOM microbatch retry (default: enabled).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reusing `embeddings_{A,B}.npz` from --out-dir when present.",
    )
    p.add_argument(
        "--cache-verify-samples",
        type=int,
        default=16,
        help="If using cached embeddings, recompute N random utterances to verify (0 disables).",
    )

    # Model A
    p.add_argument("--a-embeddings-npz", type=Path, default=None, help="Use precomputed embeddings instead of a checkpoint.")
    p.add_argument("--a-checkpoint", type=Path, default=None)
    p.add_argument("--a-extractor", type=str, default=None)
    p.add_argument("--a-processor", type=str, default=None)
    p.add_argument("--a-classifier", type=str, default=None)
    p.add_argument("--a-embed-layer", type=str, default="processor", help="processor|bottleneck|penultimate")
    p.add_argument("--a-bottleneck-dim", type=int, default=None)
    p.add_argument("--a-num-classes", type=int, default=None)
    p.add_argument("--a-l2-normalize", type=int, default=None, help="For FFCosineRaw/FFCosineRaw2 (0/1).")
    p.add_argument("--a-post-l2-normalize", action="store_true", help="L2-normalize extracted embeddings.")

    # Model B
    p.add_argument("--b-embeddings-npz", type=Path, default=None, help="Use precomputed embeddings instead of a checkpoint.")
    p.add_argument("--b-checkpoint", type=Path, default=None)
    p.add_argument("--b-extractor", type=str, default=None)
    p.add_argument("--b-processor", type=str, default=None)
    p.add_argument("--b-classifier", type=str, default=None)
    p.add_argument("--b-embed-layer", type=str, default="processor", help="processor|feature_processor|bottleneck|penultimate")
    p.add_argument("--b-bottleneck-dim", type=int, default=None)
    p.add_argument("--b-num-classes", type=int, default=None)
    p.add_argument("--b-l2-normalize", type=int, default=None, help="For FFCosineRaw/FFCosineRaw2 (0/1).")
    p.add_argument("--b-post-l2-normalize", action="store_true", help="L2-normalize extracted embeddings.")

    # Analysis knobs
    p.add_argument("--distance-metric", type=str, default="euclidean", help="euclidean|cosine")
    p.add_argument(
        "--also-cosine-l2",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Also run the same diagnostics on L2-normalized embeddings using cosine distance.",
    )
    p.add_argument("--knn-ks", type=str, default="5,10,20", help="Comma-separated k values.")
    p.add_argument("--rank-corr-max-points", type=int, default=1000, help="Max points for distance-rank Spearman.")
    p.add_argument("--intra-pairs-per-class", type=int, default=500)
    p.add_argument("--inter-pairs", type=int, default=20000)
    p.add_argument("--twonn-max-points", type=int, default=5000)
    p.add_argument("--margin-knn-k", type=int, default=64, help="k for nearest-negative-neighbor tail margin analysis.")

    # Plots
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--plot-max-classes", type=int, default=20)
    p.add_argument("--plot-samples-per-class", type=int, default=200)
    p.add_argument(
        "--scatter-figsize",
        type=str,
        default="6.0,6.0",
        help="Scatter plot size in inches (W,H).",
    )
    p.add_argument("--scatter-marker-size", type=float, default=10.0, help="Scatter marker size (points^2).")
    p.add_argument("--scatter-legend-fontsize", type=float, default=9.0)
    p.add_argument("--scatter-legend-ncol", type=int, default=2)
    p.add_argument("--scatter-inline-key-fontsize", type=float, default=8.0)
    p.add_argument(
        "--scatter-color-a",
        type=str,
        default="#4C78A8",
        help="Scatter color for model A when using single-color mode.",
    )
    p.add_argument(
        "--scatter-color-b",
        type=str,
        default="#F58518",
        help="Scatter color for model B when using single-color mode.",
    )
    p.add_argument(
        "--scatter-color-mode",
        type=str,
        default="by-class",
        choices=["single", "by-class", "by-proximity"],
        help="Coloring for scatter plots: single, by-class, or by-proximity.",
    )
    p.add_argument(
        "--scatter-density",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add a subtle density layer (hexbin) behind scatter points.",
    )
    p.add_argument("--scatter-density-alpha", type=float, default=0.06)
    p.add_argument("--scatter-density-gridsize", type=int, default=35)
    p.add_argument(
        "--scatter-frame",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Draw a thin frame around scatter plots (default: false).",
    )
    p.add_argument("--scatter-frame-width", type=float, default=0.5)
    p.add_argument("--scatter-frame-color", type=str, default="black")
    p.add_argument(
        "--family-annotations",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Annotate t-SNE/PCA with family highlight rings (default: true).",
    )
    p.add_argument("--family-label-fontsize", type=float, default=FAMILY_LABEL_FONTSIZE)
    p.add_argument(
        "--family-trim-quantile",
        type=float,
        default=FAMILY_TRIM_QUANTILE,
        help="Trim family highlights to this central quantile by distance (1.0 disables).",
    )
    p.add_argument(
        "--plot-legend-mode",
        type=str,
        default="inside",
        choices=["inside", "separate"],
        help="Legend placement for scatter plots: inside or separate.",
    )
    p.add_argument(
        "--legend-figsize",
        type=str,
        default="8.0,4.0",
        help="Legend-only size in inches (W,H) when using --plot-legend-mode=separate.",
    )
    p.add_argument(
        "--plot-legend",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Show legends on plots that support them (default: false).",
    )
    p.add_argument(
        "--plot-class-set",
        type=str,
        default=None,
        help="Named class list(s) to plot (comma-separated; hardcoded in script).",
    )
    p.add_argument("--plot-classes", type=str, default=None, help="Comma-separated classes to plot (plot-only).")
    p.add_argument(
        "--pca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PCA metrics and plots (default: true).",
    )
    p.add_argument(
        "--metadata-csv",
        action="append",
        default=[],
        help="Optional extra metadata CSV(s) to join on path for recoloring points (repeatable).",
    )
    p.add_argument(
        "--metadata-join-column",
        type=str,
        default=None,
        help="Join column name in --metadata-csv (default: same as --path-column).",
    )
    p.add_argument(
        "--factors",
        type=str,
        default="auto",
        help="Comma-separated metadata columns to recolor/score (or 'auto'/'none').",
    )
    p.add_argument("--factor-which", type=str, default="B", choices=["A", "B", "both"], help="Which model to analyze.")
    p.add_argument("--factor-max-factors", type=int, default=8)
    p.add_argument("--factor-max-categories", type=int, default=30)
    p.add_argument("--factor-min-count", type=int, default=50)
    p.add_argument("--factor-top-k", type=int, default=20, help="Top-k categories to plot (others -> __OTHER__).")
    p.add_argument("--factor-legend-limit", type=int, default=12)
    p.add_argument("--factor-knn-k", type=int, default=20, help="k for 2D kNN purity in factor scoring.")
    p.add_argument(
        "--write-points",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Write points CSVs (t-SNE/PCA coordinates) under --out-dir.",
    )

    # Output
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    rng = np.random.default_rng(int(args.seed))
    device = _resolve_device(args.device)
    amp_dtype = _amp_dtype(args.amp_dtype)
    scatter_figsize = _parse_float_pair(args.scatter_figsize, "--scatter-figsize")
    scatter_marker_size = float(args.scatter_marker_size)
    scatter_legend_fontsize = float(args.scatter_legend_fontsize)
    scatter_legend_ncol = int(args.scatter_legend_ncol)
    scatter_inline_key_fontsize = float(args.scatter_inline_key_fontsize)
    scatter_color_a = str(args.scatter_color_a)
    scatter_color_b = str(args.scatter_color_b)
    scatter_color_mode = str(args.scatter_color_mode).strip().lower()
    if scatter_color_mode not in {"single", "by-class", "by-proximity"}:
        raise ValueError("--scatter-color-mode must be one of: single, by-class, by-proximity")
    scatter_density = bool(args.scatter_density)
    scatter_density_alpha = float(args.scatter_density_alpha)
    scatter_density_gridsize = int(args.scatter_density_gridsize)
    scatter_frame = bool(args.scatter_frame)
    scatter_frame_width = float(args.scatter_frame_width)
    scatter_frame_color = str(args.scatter_frame_color)
    family_annotations = bool(args.family_annotations)
    family_label_fontsize = float(args.family_label_fontsize)
    family_trim_quantile = float(args.family_trim_quantile)
    plot_legend_mode = str(args.plot_legend_mode).strip().lower()
    legend_figsize = _parse_float_pair(args.legend_figsize, "--legend-figsize")

    spec = ProtocolSpec(
        data_root=args.data_root,
        protocol_csv=args.protocol_csv,
        path_column=str(args.path_column),
        class_column=str(args.class_column),
    )
    dataset = ProtocolSingleDataset(
        spec=spec,
        segment_seconds=args.segment_seconds,
        sample_rate=int(args.sample_rate),
        include_classes=_csv_list(args.include_classes),
        exclude_classes=_csv_list(args.exclude_classes),
        max_per_class=args.max_per_class,
        max_total=args.max_total,
        seed=int(args.seed),
    )

    out_dir = args.out_dir
    _safe_makedirs(out_dir)
    expected_utt_ids = dataset.df[dataset.path_column].astype(str).tolist()
    expected_labels = dataset.df["__label"].to_numpy(dtype=np.int64)
    meta_df = None
    factors: list[str] = []
    factor_scores: dict[str, dict[str, object]] = {}
    try:
        extra_meta_paths = [Path(p).expanduser() for p in (args.metadata_csv or []) if str(p)]
        if not extra_meta_paths:
            # Convenience: if a sibling <split>_meta.csv exists (e.g., eval_meta.csv),
            # use it automatically so recoloring works out-of-the-box.
            proto = Path(args.protocol_csv)
            auto_meta = proto.with_name(f"{proto.stem}_meta{proto.suffix}")
            if auto_meta.exists():
                extra_meta_paths = [auto_meta]
                print(f"[factors] Auto-using metadata CSV: {auto_meta}")
        meta_df = load_metadata_table(
            protocol_csv=Path(args.protocol_csv),
            path_column=str(args.path_column),
            utt_ids_ordered=expected_utt_ids,
            extra_metadata_csvs=extra_meta_paths,
            metadata_join_column=args.metadata_join_column,
        )
        # Derived quick buckets (optional, only if present).
        if "meta_transcript" in meta_df.columns:
            lengths = meta_df["meta_transcript"].astype("object").fillna("").astype(str).map(len)
            meta_df["meta_transcript_len"] = lengths
            # Fixed buckets for interpretability.
            bins = [0, 20, 40, 80, 160, 10_000]
            labels = [f"len<= {b}" for b in bins[1:]]
            meta_df["meta_transcript_len_bucket"] = pd.cut(lengths, bins=bins, labels=labels, include_lowest=True)
        if "meta_original_file" in meta_df.columns:
            # Common MLAAD original-file structure looks like:
            #   <locale>/<subset>/<gender>/<speaker>/<book>/...
            # Derive coarse proxies for speaker/subset which often explain manifold structure.
            orig = meta_df["meta_original_file"].astype("object").fillna("").astype(str).str.strip("/")
            parts = orig.str.split("/", n=5, expand=True)
            if parts.shape[1] >= 4:
                meta_df["orig_locale"] = parts[0].replace("", np.nan)
                meta_df["orig_subset"] = parts[1].replace("", np.nan)
                meta_df["orig_gender"] = parts[2].replace("", np.nan)
                meta_df["orig_speaker"] = parts[3].replace("", np.nan)
        factors = _choose_factor_columns(
            meta_df,
            path_column=str(args.path_column),
            class_column=str(args.class_column),
            factors=str(args.factors),
            max_factors=int(args.factor_max_factors),
            max_categories=int(args.factor_max_categories),
            min_count=int(args.factor_min_count),
        )
        if factors:
            print(f"[factors] Using: {', '.join(factors)}")
    except Exception as e:
        print(f"[factors] Skipping factor analysis (metadata load/select failed): {e}")
        meta_df = None
        factors = []

    # Embeddings
    if args.a_embeddings_npz is not None:
        ids_a, Xa, ya, class_names_a = _load_npz(args.a_embeddings_npz)
    else:
        if args.a_checkpoint is None:
            raise ValueError("Provide either --a-embeddings-npz or --a-checkpoint ...")
        if args.a_extractor is None or args.a_processor is None or args.a_classifier is None:
            raise ValueError("Model A requires --a-extractor, --a-processor, --a-classifier when using a checkpoint.")
        a_spec = ModelSpec(
            checkpoint=args.a_checkpoint,
            extractor=str(args.a_extractor),
            processor=str(args.a_processor),
            classifier=str(args.a_classifier),
            embed_layer=str(args.a_embed_layer),
            bottleneck_dim=args.a_bottleneck_dim,
            num_classes=args.a_num_classes,
            l2_normalize=None if args.a_l2_normalize is None else bool(int(args.a_l2_normalize)),
            post_l2_normalize=bool(args.a_post_l2_normalize),
        )
        a_npz = out_dir / "embeddings_A.npz"
        a_meta = {"data": _current_data_meta(args), "model": _current_model_meta("a", args), "amp": {"enabled": bool(args.amp_eval), "dtype": str(args.amp_dtype)}}
        cached = None
        if not bool(args.no_cache):
            cached = _try_load_cached_embeddings(
                tag="A",
                out_dir=out_dir,
                expected_meta=a_meta,
                expected_utt_ids=expected_utt_ids,
                expected_labels=expected_labels,
                verify_samples=int(args.cache_verify_samples),
                model_spec=a_spec,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                amp_eval=bool(args.amp_eval),
                amp_dtype=amp_dtype,
                segment_seconds=args.segment_seconds,
                sample_rate=int(args.sample_rate),
                data_root=Path(args.data_root),
            )
        if cached is not None:
            ids_a, Xa, ya, class_names_a = cached
        else:
            ids_a, Xa, ya = extract_embeddings(
                a_spec,
                dataset=dataset,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                amp_eval=bool(args.amp_eval),
                amp_dtype=amp_dtype,
                oom_fallback=not bool(args.no_oom_fallback),
            )
            class_names_a = dataset.class_names
            _write_npz(a_npz, ids_a, Xa, ya, class_names_a)
            _write_json(_npz_meta_path(a_npz), a_meta)

    if args.b_embeddings_npz is not None:
        ids_b, Xb, yb, class_names_b = _load_npz(args.b_embeddings_npz)
    else:
        if args.b_checkpoint is None:
            raise ValueError("Provide either --b-embeddings-npz or --b-checkpoint ...")
        if args.b_extractor is None or args.b_processor is None or args.b_classifier is None:
            raise ValueError("Model B requires --b-extractor, --b-processor, --b-classifier when using a checkpoint.")
        b_spec = ModelSpec(
            checkpoint=args.b_checkpoint,
            extractor=str(args.b_extractor),
            processor=str(args.b_processor),
            classifier=str(args.b_classifier),
            embed_layer=str(args.b_embed_layer),
            bottleneck_dim=args.b_bottleneck_dim,
            num_classes=args.b_num_classes,
            l2_normalize=None if args.b_l2_normalize is None else bool(int(args.b_l2_normalize)),
            post_l2_normalize=bool(args.b_post_l2_normalize),
        )
        b_npz = out_dir / "embeddings_B.npz"
        b_meta = {"data": _current_data_meta(args), "model": _current_model_meta("b", args), "amp": {"enabled": bool(args.amp_eval), "dtype": str(args.amp_dtype)}}
        cached = None
        if not bool(args.no_cache):
            cached = _try_load_cached_embeddings(
                tag="B",
                out_dir=out_dir,
                expected_meta=b_meta,
                expected_utt_ids=expected_utt_ids,
                expected_labels=expected_labels,
                verify_samples=int(args.cache_verify_samples),
                model_spec=b_spec,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                amp_eval=bool(args.amp_eval),
                amp_dtype=amp_dtype,
                segment_seconds=args.segment_seconds,
                sample_rate=int(args.sample_rate),
                data_root=Path(args.data_root),
            )
        if cached is not None:
            ids_b, Xb, yb, class_names_b = cached
        else:
            ids_b, Xb, yb = extract_embeddings(
                b_spec,
                dataset=dataset,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                amp_eval=bool(args.amp_eval),
                amp_dtype=amp_dtype,
                oom_fallback=not bool(args.no_oom_fallback),
            )
            class_names_b = dataset.class_names
            _write_npz(b_npz, ids_b, Xb, yb, class_names_b)
            _write_json(_npz_meta_path(b_npz), b_meta)

    # Align
    utt_ids, Xa2, Xb2, y = _align_by_utt_id(ids_a, Xa, ya, ids_b, Xb, yb)
    if class_names_a != class_names_b:
        raise ValueError("class_names differ between A and B. Use the same protocol/filters for both.")
    class_names = class_names_a

    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    family_groups = _build_family_groups(class_names) if family_annotations else []
    plot_class_ids: set[int] | None = None
    plot_class_markers: dict[int, str] | None = None
    plot_class_order: list[int] | None = None
    plot_class_hardness_rank: dict[int, int] | None = None
    inline_key = False
    plot_classes = _csv_list(args.plot_classes)
    plot_class_set_names = _csv_list(args.plot_class_set)
    if plot_classes and plot_class_set_names:
        print("[warn] Both --plot-classes and --plot-class-set provided; using --plot-classes.")
    if not plot_classes and plot_class_set_names:
        unknown = sorted(set(plot_class_set_names) - set(PLOT_CLASS_SETS))
        if unknown:
            raise ValueError(f"Unknown --plot-class-set entries: {', '.join(unknown)}")
        merged: list[str] = []
        for name in plot_class_set_names:
            merged.extend(PLOT_CLASS_SETS.get(name, []))
        plot_classes = merged
    if plot_classes:
        missing = sorted(set(plot_classes) - set(class_to_id))
        if missing:
            preview = ", ".join(missing[:5])
            print(f"[warn] Plot classes not found in protocol (e.g., {preview}).")
        plot_class_ids = {class_to_id[name] for name in plot_classes if name in class_to_id}
        if not plot_class_ids:
            raise ValueError("No plot classes matched protocol labels.")
        plot_class_order = []
        seen: set[int] = set()
        for name in plot_classes:
            cid = class_to_id.get(name)
            if cid is None or cid in seen:
                continue
            plot_class_order.append(cid)
            seen.add(cid)
        if not plot_class_order:
            plot_class_order = None
    if plot_class_set_names:
        plot_class_markers = {}
        plot_class_hardness_rank = {}
        conflicts: list[str] = []
        for rank, set_name in enumerate(PLOT_CLASS_SET_ORDER):
            if set_name not in plot_class_set_names:
                continue
            marker = PLOT_CLASS_SET_MARKERS.get(set_name, "o")
            for class_name in PLOT_CLASS_SETS.get(set_name, []):
                cid = class_to_id.get(class_name)
                if cid is None:
                    continue
                existing = plot_class_markers.get(cid)
                if existing is not None and existing != marker:
                    conflicts.append(class_name)
                    continue
                plot_class_markers[cid] = marker
                plot_class_hardness_rank[cid] = rank
        if plot_class_ids is not None:
            plot_class_markers = {cid: m for cid, m in plot_class_markers.items() if cid in plot_class_ids}
            plot_class_hardness_rank = {cid: r for cid, r in plot_class_hardness_rank.items() if cid in plot_class_ids}
        if not plot_class_markers:
            plot_class_markers = None
        if plot_class_hardness_rank and not plot_class_markers:
            plot_class_hardness_rank = None
        if conflicts:
            preview = ", ".join(sorted(set(conflicts))[:5])
            print(f"[warn] Some classes appear in multiple plot sets; keeping first marker (e.g., {preview}).")

    # Metrics
    metric_t0 = time.perf_counter()
    metric = str(args.distance_metric).lower().strip()
    sep_a = _separability_metrics(Xa2, y, metric=metric)
    sep_b = _separability_metrics(Xb2, y, metric=metric)

    intra_a, inter_a = _pairwise_distances_sampled(
        Xa2, y, metric=metric, rng=rng, intra_pairs_per_class=int(args.intra_pairs_per_class), inter_pairs=int(args.inter_pairs)
    )
    intra_b, inter_b = _pairwise_distances_sampled(
        Xb2, y, metric=metric, rng=rng, intra_pairs_per_class=int(args.intra_pairs_per_class), inter_pairs=int(args.inter_pairs)
    )

    rank_corr = _distance_rank_corr(
        Xa2, Xb2, metric=metric, max_points=int(args.rank_corr_max_points), seed=int(args.seed)
    )

    knn_ks = [int(k.strip()) for k in str(args.knn_ks).split(",") if k.strip()]
    knn = {f"k={k}": knn_overlap(Xa2, Xb2, k=k, metric=metric) for k in knn_ks}

    twonn_a = twonn_intrinsic_dim(Xa2, metric=metric, max_points=int(args.twonn_max_points), seed=int(args.seed))
    twonn_b = twonn_intrinsic_dim(Xb2, metric=metric, max_points=int(args.twonn_max_points), seed=int(args.seed))

    pca_enabled = bool(args.pca)
    pca_a = pca_spectrum_metrics(Xa2) if pca_enabled else None
    pca_b = pca_spectrum_metrics(Xb2) if pca_enabled else None

    cent_a = centroid_graph_metrics(Xa2, y, metric=metric)
    cent_b = centroid_graph_metrics(Xb2, y, metric=metric)
    if int(cent_a.get("num_centroids", 0)) >= 2 and int(cent_b.get("num_centroids", 0)) >= 2:
        da = np.asarray(cent_a["centroid_distance_upper"], dtype=float)
        db = np.asarray(cent_b["centroid_distance_upper"], dtype=float)
        centroid_dist_spearman = spearman_corr(da, db)
        nn_match = float(np.mean(np.asarray(cent_a["centroid_nearest"]) == np.asarray(cent_b["centroid_nearest"])))
    else:
        centroid_dist_spearman = float("nan")
        nn_match = float("nan")

    tail_centroid_a = centroid_tail_margins(Xa2, y, metric=metric)
    tail_centroid_b = centroid_tail_margins(Xb2, y, metric=metric)
    tail_neighbor_a = negative_neighbor_tail_margins(Xa2, y, metric=metric, k=int(args.margin_knn_k))
    tail_neighbor_b = negative_neighbor_tail_margins(Xb2, y, metric=metric, k=int(args.margin_knn_k))

    # L2-normalized cosine diagnostics (reviewer-proof for verifier embeddings)
    l2cos = None
    if bool(args.also_cosine_l2):
        Xa_l2 = _l2_normalize(Xa2)
        Xb_l2 = _l2_normalize(Xb2)
        metric_l2 = "cosine"
        sep_a_l2 = _separability_metrics(Xa_l2, y, metric=metric_l2)
        sep_b_l2 = _separability_metrics(Xb_l2, y, metric=metric_l2)
        intra_a_l2, inter_a_l2 = _pairwise_distances_sampled(
            Xa_l2,
            y,
            metric=metric_l2,
            rng=rng,
            intra_pairs_per_class=int(args.intra_pairs_per_class),
            inter_pairs=int(args.inter_pairs),
        )
        intra_b_l2, inter_b_l2 = _pairwise_distances_sampled(
            Xb_l2,
            y,
            metric=metric_l2,
            rng=rng,
            intra_pairs_per_class=int(args.intra_pairs_per_class),
            inter_pairs=int(args.inter_pairs),
        )
        rank_corr_l2 = _distance_rank_corr(
            Xa_l2, Xb_l2, metric=metric_l2, max_points=int(args.rank_corr_max_points), seed=int(args.seed)
        )
        knn_l2 = {f"k={k}": knn_overlap(Xa_l2, Xb_l2, k=k, metric=metric_l2) for k in knn_ks}
        twonn_a_l2 = twonn_intrinsic_dim(Xa_l2, metric=metric_l2, max_points=int(args.twonn_max_points), seed=int(args.seed))
        twonn_b_l2 = twonn_intrinsic_dim(Xb_l2, metric=metric_l2, max_points=int(args.twonn_max_points), seed=int(args.seed))
        pca_a_l2 = pca_spectrum_metrics(Xa_l2) if pca_enabled else None
        pca_b_l2 = pca_spectrum_metrics(Xb_l2) if pca_enabled else None
        cent_a_l2 = centroid_graph_metrics(Xa_l2, y, metric=metric_l2)
        cent_b_l2 = centroid_graph_metrics(Xb_l2, y, metric=metric_l2)
        if int(cent_a_l2.get("num_centroids", 0)) >= 2 and int(cent_b_l2.get("num_centroids", 0)) >= 2:
            da2 = np.asarray(cent_a_l2["centroid_distance_upper"], dtype=float)
            db2 = np.asarray(cent_b_l2["centroid_distance_upper"], dtype=float)
            centroid_dist_spearman_l2 = spearman_corr(da2, db2)
            nn_match_l2 = float(
                np.mean(np.asarray(cent_a_l2["centroid_nearest"]) == np.asarray(cent_b_l2["centroid_nearest"]))
            )
        else:
            centroid_dist_spearman_l2 = float("nan")
            nn_match_l2 = float("nan")
        tail_centroid_a_l2 = centroid_tail_margins(Xa_l2, y, metric=metric_l2)
        tail_centroid_b_l2 = centroid_tail_margins(Xb_l2, y, metric=metric_l2)
        tail_neighbor_a_l2 = negative_neighbor_tail_margins(Xa_l2, y, metric=metric_l2, k=int(args.margin_knn_k))
        tail_neighbor_b_l2 = negative_neighbor_tail_margins(Xb_l2, y, metric=metric_l2, k=int(args.margin_knn_k))

        l2cos = {
            "metric": metric_l2,
            "rank_distance_spearman": float(rank_corr_l2),
            "knn_overlap": knn_l2,
            "centroid_distance_spearman": float(centroid_dist_spearman_l2),
            "centroid_nearest_match": float(nn_match_l2),
            "A": {
                "separability": sep_a_l2,
                "intrinsic_dim": twonn_a_l2,
                "spectrum": None
                if not pca_enabled
                else {k: v for k, v in pca_a_l2.items() if k != "pca"} | {"pca_components": len(pca_a_l2["pca"]["eigs"])},
                "distance_samples": {
                    "intra_mean": float(np.mean(intra_a_l2)) if intra_a_l2.size else float("nan"),
                    "inter_mean": float(np.mean(inter_a_l2)) if inter_a_l2.size else float("nan"),
                    "intra_median": float(np.median(intra_a_l2)) if intra_a_l2.size else float("nan"),
                    "inter_median": float(np.median(inter_a_l2)) if inter_a_l2.size else float("nan"),
                },
                "tail_margins_centroid": tail_centroid_a_l2,
                "tail_margins_neg_neighbor": tail_neighbor_a_l2,
            },
            "B": {
                "separability": sep_b_l2,
                "intrinsic_dim": twonn_b_l2,
                "spectrum": None
                if not pca_enabled
                else {k: v for k, v in pca_b_l2.items() if k != "pca"} | {"pca_components": len(pca_b_l2["pca"]["eigs"])},
                "distance_samples": {
                    "intra_mean": float(np.mean(intra_b_l2)) if intra_b_l2.size else float("nan"),
                    "inter_mean": float(np.mean(inter_b_l2)) if inter_b_l2.size else float("nan"),
                    "intra_median": float(np.median(intra_b_l2)) if intra_b_l2.size else float("nan"),
                    "inter_median": float(np.median(inter_b_l2)) if inter_b_l2.size else float("nan"),
                },
                "tail_margins_centroid": tail_centroid_b_l2,
                "tail_margins_neg_neighbor": tail_neighbor_b_l2,
            },
            "centroids_A": {k: v for k, v in cent_a_l2.items() if k not in {"centroid_distance_upper", "centroid_nearest"}},
            "centroids_B": {k: v for k, v in cent_b_l2.items() if k not in {"centroid_distance_upper", "centroid_nearest"}},
        }

    print(f"[metrics] done in {time.perf_counter() - metric_t0:.1f}s", flush=True)

    # Plots
    plot_t0 = time.perf_counter()
    tsne_a_path = out_dir / "tsne_A.png"
    tsne_b_path = out_dir / "tsne_B.png"
    tsne_a_legend_path = out_dir / "tsne_A_legend.png"
    tsne_b_legend_path = out_dir / "tsne_B_legend.png"

    plot_steps: list[tuple[str, callable]] = []
    tsne_a_metrics: dict[str, object] = {}
    tsne_b_metrics: dict[str, object] = {}
    tsne_a_keep: list[int] | None = None
    tsne_b_keep: list[int] | None = None
    tsne_a_coords: np.ndarray | None = None
    tsne_b_coords: np.ndarray | None = None
    tsne_a_l2_keep: list[int] | None = None
    tsne_b_l2_keep: list[int] | None = None
    tsne_a_l2_coords: np.ndarray | None = None
    tsne_b_l2_coords: np.ndarray | None = None
    class_color_map: dict[int, str] | None = None
    pca_a_keep: list[int] | None = None
    pca_b_keep: list[int] | None = None
    pca_a_Z: np.ndarray | None = None
    pca_b_Z: np.ndarray | None = None
    pca_a_l2_keep: list[int] | None = None
    pca_b_l2_keep: list[int] | None = None
    pca_a_l2_Z: np.ndarray | None = None
    pca_b_l2_Z: np.ndarray | None = None

    def _step(name: str, fn: callable) -> None:
        plot_steps.append((name, fn))

    def _tsne_a():
        nonlocal tsne_a_metrics, tsne_a_keep, tsne_a_coords
        Z, tsne_a_metrics, keep = _plot_tsne(
            Xa2,
            y,
            class_names,
            tsne_a_path,
            title="t-SNE (A)",
            perplexity=float(args.tsne_perplexity),
            seed=int(args.seed),
            max_classes=int(args.plot_max_classes),
            samples_per_class=int(args.plot_samples_per_class),
            plot_class_ids=plot_class_ids,
            tsne_metric="euclidean",
            show_legend=bool(args.plot_legend),
            class_markers=plot_class_markers,
            class_order=plot_class_order,
            class_hardness_rank=plot_class_hardness_rank,
            figsize=scatter_figsize,
            marker_size=scatter_marker_size,
            legend_fontsize=scatter_legend_fontsize,
            legend_ncol=scatter_legend_ncol,
            inline_key=inline_key,
            inline_key_fontsize=scatter_inline_key_fontsize,
            density=scatter_density,
            density_gridsize=scatter_density_gridsize,
            density_alpha=scatter_density_alpha,
            show_frame=scatter_frame,
            frame_color=scatter_frame_color,
            frame_linewidth=scatter_frame_width,
            color_mode=scatter_color_mode,
            point_color=scatter_color_a,
            class_color_map=class_color_map if scatter_color_mode == "by-proximity" else None,
            legend_mode=plot_legend_mode,
            legend_out_path=tsne_a_legend_path if plot_legend_mode == "separate" else None,
            legend_figsize=legend_figsize,
            compute_metrics=True,
            save_plot=False,
        )
        tsne_a_keep = keep
        tsne_a_coords = Z
        if bool(args.write_points):
            _write_points_csv(
                out_dir / "points_tsne_A.csv",
                utt_ids=utt_ids,
                labels=y,
                class_names=class_names,
                keep_idx=keep,
                coords=Z,
                coord_names=["tsne_x", "tsne_y"],
                meta_df=meta_df,
            )
        _maybe_factor("tsne_A", "A", keep, Z)

    def _tsne_b():
        nonlocal tsne_b_metrics, tsne_b_keep, tsne_b_coords
        Z, tsne_b_metrics, keep = _plot_tsne(
            Xb2,
            y,
            class_names,
            tsne_b_path,
            title="t-SNE (B)",
            perplexity=float(args.tsne_perplexity),
            seed=int(args.seed),
            max_classes=int(args.plot_max_classes),
            samples_per_class=int(args.plot_samples_per_class),
            plot_class_ids=plot_class_ids,
            tsne_metric="euclidean",
            show_legend=bool(args.plot_legend),
            class_markers=plot_class_markers,
            class_order=plot_class_order,
            class_hardness_rank=plot_class_hardness_rank,
            figsize=scatter_figsize,
            marker_size=scatter_marker_size,
            legend_fontsize=scatter_legend_fontsize,
            legend_ncol=scatter_legend_ncol,
            inline_key=inline_key,
            inline_key_fontsize=scatter_inline_key_fontsize,
            density=scatter_density,
            density_gridsize=scatter_density_gridsize,
            density_alpha=scatter_density_alpha,
            show_frame=scatter_frame,
            frame_color=scatter_frame_color,
            frame_linewidth=scatter_frame_width,
            color_mode=scatter_color_mode,
            point_color=scatter_color_b,
            class_color_map=class_color_map if scatter_color_mode == "by-proximity" else None,
            legend_mode=plot_legend_mode,
            legend_out_path=tsne_b_legend_path if plot_legend_mode == "separate" else None,
            legend_figsize=legend_figsize,
            compute_metrics=True,
            save_plot=False,
        )
        tsne_b_keep = keep
        tsne_b_coords = Z
        if bool(args.write_points):
            _write_points_csv(
                out_dir / "points_tsne_B.csv",
                utt_ids=utt_ids,
                labels=y,
                class_names=class_names,
                keep_idx=keep,
                coords=Z,
                coord_names=["tsne_x", "tsne_y"],
                meta_df=meta_df,
            )
        _maybe_factor("tsne_B", "B", keep, Z)

    def _tsne_render():
        nonlocal class_color_map
        if tsne_a_coords is None or tsne_b_coords is None or tsne_a_keep is None or tsne_b_keep is None:
            return
        if scatter_color_mode == "by-proximity" and class_color_map is None:
            class_color_map = _build_class_color_map(
                coords_list=[tsne_a_coords, tsne_b_coords],
                keep_list=[tsne_a_keep, tsne_b_keep],
                labels=y,
                k=4,
            )
        limits = _axis_limits_from_coords([tsne_a_coords, tsne_b_coords], pad_fraction=0.03)
        _plot_tsne(
            Xa2,
            y,
            class_names,
            tsne_a_path,
            title="t-SNE (A)",
            perplexity=float(args.tsne_perplexity),
            seed=int(args.seed),
            max_classes=int(args.plot_max_classes),
            samples_per_class=int(args.plot_samples_per_class),
            plot_class_ids=plot_class_ids,
            tsne_metric="euclidean",
            show_legend=bool(args.plot_legend),
            class_markers=plot_class_markers,
            class_order=plot_class_order,
            class_hardness_rank=plot_class_hardness_rank,
            figsize=scatter_figsize,
            marker_size=scatter_marker_size,
            legend_fontsize=scatter_legend_fontsize,
            legend_ncol=scatter_legend_ncol,
            inline_key=inline_key,
            inline_key_fontsize=scatter_inline_key_fontsize,
            density=scatter_density,
            density_gridsize=scatter_density_gridsize,
            density_alpha=scatter_density_alpha,
            show_frame=scatter_frame,
            frame_color=scatter_frame_color,
            frame_linewidth=scatter_frame_width,
            color_mode=scatter_color_mode,
            point_color=scatter_color_a,
            family_groups=family_groups,
            family_label_fontsize=family_label_fontsize,
            family_trim_quantile=family_trim_quantile,
            family_annotations=family_annotations,
            class_color_map=class_color_map,
            legend_mode=plot_legend_mode,
            legend_out_path=tsne_a_legend_path if plot_legend_mode == "separate" else None,
            legend_figsize=legend_figsize,
            axis_limits=limits,
            coords=tsne_a_coords,
            keep_idx=tsne_a_keep,
            compute_metrics=False,
            save_plot=True,
        )
        _plot_tsne(
            Xb2,
            y,
            class_names,
            tsne_b_path,
            title="t-SNE (B)",
            perplexity=float(args.tsne_perplexity),
            seed=int(args.seed),
            max_classes=int(args.plot_max_classes),
            samples_per_class=int(args.plot_samples_per_class),
            plot_class_ids=plot_class_ids,
            tsne_metric="euclidean",
            show_legend=bool(args.plot_legend),
            class_markers=plot_class_markers,
            class_order=plot_class_order,
            class_hardness_rank=plot_class_hardness_rank,
            figsize=scatter_figsize,
            marker_size=scatter_marker_size,
            legend_fontsize=scatter_legend_fontsize,
            legend_ncol=scatter_legend_ncol,
            inline_key=inline_key,
            inline_key_fontsize=scatter_inline_key_fontsize,
            density=scatter_density,
            density_gridsize=scatter_density_gridsize,
            density_alpha=scatter_density_alpha,
            show_frame=scatter_frame,
            frame_color=scatter_frame_color,
            frame_linewidth=scatter_frame_width,
            color_mode=scatter_color_mode,
            point_color=scatter_color_b,
            family_groups=family_groups,
            family_label_fontsize=family_label_fontsize,
            family_trim_quantile=family_trim_quantile,
            family_annotations=family_annotations,
            class_color_map=class_color_map,
            legend_mode=plot_legend_mode,
            legend_out_path=tsne_b_legend_path if plot_legend_mode == "separate" else None,
            legend_figsize=legend_figsize,
            axis_limits=limits,
            coords=tsne_b_coords,
            keep_idx=tsne_b_keep,
            compute_metrics=False,
            save_plot=True,
        )

    _step("t-SNE A (euclidean)", _tsne_a)
    _step("t-SNE B (euclidean)", _tsne_b)
    _step("t-SNE render (euclidean)", _tsne_render)

    def _maybe_factor(space: str, model_tag: str, keep: list[int], coords2d: np.ndarray) -> None:
        if meta_df is None or not factors:
            return
        if args.factor_which != "both" and args.factor_which != model_tag:
            return
        sub = meta_df.iloc[np.asarray(keep, dtype=int)].reset_index(drop=True)
        for col in factors:
            if col not in sub.columns:
                continue
            ids, cats = _encode_categories(sub[col], top_k=int(args.factor_top_k))
            scores = factor_separation_2d(coords2d, ids)
            purity = knn_purity(coords2d, ids, k=int(args.factor_knn_k), metric="euclidean")
            factor_scores.setdefault(space, {})[col] = {
                **scores,
                "knn_purity_2d": float(purity),
                "num_categories_plot": int(len(cats)),
            }
            out_path = out_dir / f"recolor_{space}_{_slug(col)}.png"
            _plot_by_factor(
                coords2d,
                sub[col],
                out_path=out_path,
                title=f"{space} colored by {col}",
                seed=int(args.seed),
                top_k=int(args.factor_top_k),
                legend_limit=int(args.factor_legend_limit),
            )

    def _pca_a():
        nonlocal pca_a_keep, pca_a_Z
        keep, Z3, _ = _plot_pca_scatter(
            Xa2,
            y,
            class_names,
            out_pc12=out_dir / "pca_A_pc1_pc2.png",
            out_pc13=out_dir / "pca_A_pc1_pc3.png",
            title_prefix="A",
            seed=int(args.seed),
            max_classes=int(args.plot_max_classes),
            samples_per_class=int(args.plot_samples_per_class),
            plot_class_ids=plot_class_ids,
            show_legend=bool(args.plot_legend),
            class_markers=plot_class_markers,
            class_order=plot_class_order,
            class_hardness_rank=plot_class_hardness_rank,
            figsize=scatter_figsize,
            marker_size=scatter_marker_size,
            legend_fontsize=scatter_legend_fontsize,
            legend_ncol=scatter_legend_ncol,
            inline_key=inline_key,
            inline_key_fontsize=scatter_inline_key_fontsize,
            density=scatter_density,
            density_gridsize=scatter_density_gridsize,
            density_alpha=scatter_density_alpha,
            show_frame=scatter_frame,
            frame_color=scatter_frame_color,
            frame_linewidth=scatter_frame_width,
            color_mode=scatter_color_mode,
            point_color=scatter_color_a,
            class_color_map=class_color_map if scatter_color_mode == "by-proximity" else None,
            family_groups=family_groups,
            family_label_fontsize=family_label_fontsize,
            family_trim_quantile=family_trim_quantile,
            family_annotations=family_annotations,
        )
        pca_a_keep = keep
        pca_a_Z = Z3
        if bool(args.write_points):
            _write_points_csv(
                out_dir / "points_pca_A.csv",
                utt_ids=utt_ids,
                labels=y,
                class_names=class_names,
                keep_idx=keep,
                coords=Z3,
                coord_names=["pc1", "pc2", "pc3"],
                meta_df=meta_df,
            )
        _maybe_factor("pca_A_pc1_pc2", "A", keep, Z3[:, [0, 1]])
        _maybe_factor("pca_A_pc1_pc3", "A", keep, Z3[:, [0, 2]])

    def _pca_b():
        nonlocal pca_b_keep, pca_b_Z
        keep, Z3, _ = _plot_pca_scatter(
            Xb2,
            y,
            class_names,
            out_pc12=out_dir / "pca_B_pc1_pc2.png",
            out_pc13=out_dir / "pca_B_pc1_pc3.png",
            title_prefix="B",
            seed=int(args.seed),
            max_classes=int(args.plot_max_classes),
            samples_per_class=int(args.plot_samples_per_class),
            plot_class_ids=plot_class_ids,
            show_legend=bool(args.plot_legend),
            class_markers=plot_class_markers,
            class_order=plot_class_order,
            class_hardness_rank=plot_class_hardness_rank,
            figsize=scatter_figsize,
            marker_size=scatter_marker_size,
            legend_fontsize=scatter_legend_fontsize,
            legend_ncol=scatter_legend_ncol,
            inline_key=inline_key,
            inline_key_fontsize=scatter_inline_key_fontsize,
            density=scatter_density,
            density_gridsize=scatter_density_gridsize,
            density_alpha=scatter_density_alpha,
            show_frame=scatter_frame,
            frame_color=scatter_frame_color,
            frame_linewidth=scatter_frame_width,
            color_mode=scatter_color_mode,
            point_color=scatter_color_b,
            class_color_map=class_color_map if scatter_color_mode == "by-proximity" else None,
            family_groups=family_groups,
            family_label_fontsize=family_label_fontsize,
            family_trim_quantile=family_trim_quantile,
            family_annotations=family_annotations,
        )
        pca_b_keep = keep
        pca_b_Z = Z3
        if bool(args.write_points):
            _write_points_csv(
                out_dir / "points_pca_B.csv",
                utt_ids=utt_ids,
                labels=y,
                class_names=class_names,
                keep_idx=keep,
                coords=Z3,
                coord_names=["pc1", "pc2", "pc3"],
                meta_df=meta_df,
            )
        _maybe_factor("pca_B_pc1_pc2", "B", keep, Z3[:, [0, 1]])
        _maybe_factor("pca_B_pc1_pc3", "B", keep, Z3[:, [0, 2]])

    if pca_enabled:
        _step("PCA scatter A", _pca_a)
        _step("PCA scatter B", _pca_b)
    _step(
        "Distance distributions",
        lambda: _plot_distance_distributions(
            intra_a,
            inter_a,
            intra_b,
            inter_b,
            out_dir / "distance_distributions.png",
            title="Intra vs inter distances",
            show_legend=bool(args.plot_legend),
        ),
    )
    if pca_enabled:
        _step(
            "PCA spectrum",
            lambda: _plot_pca_spectrum(
                list(pca_a["pca"]["eigs"]),
                list(pca_b["pca"]["eigs"]),
                out_dir / "pca_spectrum.png",
                title="PCA spectrum",
                show_legend=bool(args.plot_legend),
            ),
        )

    if l2cos is not None:
        tsne_a_l2_legend_path = out_dir / "tsne_A_l2cos_legend.png"
        tsne_b_l2_legend_path = out_dir / "tsne_B_l2cos_legend.png"
        def _tsne_a_l2():
            nonlocal tsne_a_l2_keep, tsne_a_l2_coords
            Z, _, keep = _plot_tsne(
                Xa_l2,
                y,
                class_names,
                out_dir / "tsne_A_l2cos.png",
                title="t-SNE (A, L2 + cosine)",
                perplexity=float(args.tsne_perplexity),
                seed=int(args.seed),
                max_classes=int(args.plot_max_classes),
                samples_per_class=int(args.plot_samples_per_class),
                plot_class_ids=plot_class_ids,
                tsne_metric="cosine",
                show_legend=bool(args.plot_legend),
                class_markers=plot_class_markers,
                class_order=plot_class_order,
                class_hardness_rank=plot_class_hardness_rank,
                figsize=scatter_figsize,
                marker_size=scatter_marker_size,
                legend_fontsize=scatter_legend_fontsize,
                legend_ncol=scatter_legend_ncol,
                inline_key=inline_key,
                inline_key_fontsize=scatter_inline_key_fontsize,
                density=scatter_density,
                density_gridsize=scatter_density_gridsize,
                density_alpha=scatter_density_alpha,
                show_frame=scatter_frame,
                frame_color=scatter_frame_color,
                frame_linewidth=scatter_frame_width,
                color_mode=scatter_color_mode,
                point_color=scatter_color_a,
                legend_mode=plot_legend_mode,
                legend_out_path=tsne_a_l2_legend_path if plot_legend_mode == "separate" else None,
                legend_figsize=legend_figsize,
                compute_metrics=True,
                save_plot=False,
            )
            tsne_a_l2_keep = keep
            tsne_a_l2_coords = Z
            if bool(args.write_points):
                _write_points_csv(
                    out_dir / "points_tsne_A_l2cos.csv",
                    utt_ids=utt_ids,
                    labels=y,
                    class_names=class_names,
                    keep_idx=keep,
                    coords=Z,
                    coord_names=["tsne_x", "tsne_y"],
                    meta_df=meta_df,
                )
            _maybe_factor("tsne_A_l2cos", "A", keep, Z)

        def _tsne_b_l2():
            nonlocal tsne_b_l2_keep, tsne_b_l2_coords
            Z, _, keep = _plot_tsne(
                Xb_l2,
                y,
                class_names,
                out_dir / "tsne_B_l2cos.png",
                title="t-SNE (B, L2 + cosine)",
                perplexity=float(args.tsne_perplexity),
                seed=int(args.seed),
                max_classes=int(args.plot_max_classes),
                samples_per_class=int(args.plot_samples_per_class),
                plot_class_ids=plot_class_ids,
                tsne_metric="cosine",
                show_legend=bool(args.plot_legend),
                class_markers=plot_class_markers,
                class_order=plot_class_order,
                class_hardness_rank=plot_class_hardness_rank,
                figsize=scatter_figsize,
                marker_size=scatter_marker_size,
                legend_fontsize=scatter_legend_fontsize,
                legend_ncol=scatter_legend_ncol,
                inline_key=inline_key,
                inline_key_fontsize=scatter_inline_key_fontsize,
                density=scatter_density,
                density_gridsize=scatter_density_gridsize,
                density_alpha=scatter_density_alpha,
                show_frame=scatter_frame,
                frame_color=scatter_frame_color,
                frame_linewidth=scatter_frame_width,
                color_mode=scatter_color_mode,
                point_color=scatter_color_b,
                legend_mode=plot_legend_mode,
                legend_out_path=tsne_b_l2_legend_path if plot_legend_mode == "separate" else None,
                legend_figsize=legend_figsize,
                compute_metrics=True,
                save_plot=False,
            )
            tsne_b_l2_keep = keep
            tsne_b_l2_coords = Z
            if bool(args.write_points):
                _write_points_csv(
                    out_dir / "points_tsne_B_l2cos.csv",
                    utt_ids=utt_ids,
                    labels=y,
                    class_names=class_names,
                    keep_idx=keep,
                    coords=Z,
                    coord_names=["tsne_x", "tsne_y"],
                    meta_df=meta_df,
                )
            _maybe_factor("tsne_B_l2cos", "B", keep, Z)

        def _tsne_render_l2():
            nonlocal class_color_map
            if tsne_a_l2_coords is None or tsne_b_l2_coords is None or tsne_a_l2_keep is None or tsne_b_l2_keep is None:
                return
            if scatter_color_mode == "by-proximity" and class_color_map is None:
                class_color_map = _build_class_color_map(
                    coords_list=[tsne_a_l2_coords, tsne_b_l2_coords],
                    keep_list=[tsne_a_l2_keep, tsne_b_l2_keep],
                    labels=y,
                    k=4,
                )
            limits = _axis_limits_from_coords([tsne_a_l2_coords, tsne_b_l2_coords], pad_fraction=0.03)
            _plot_tsne(
                Xa_l2,
                y,
                class_names,
                out_dir / "tsne_A_l2cos.png",
                title="t-SNE (A, L2 + cosine)",
                perplexity=float(args.tsne_perplexity),
                seed=int(args.seed),
                max_classes=int(args.plot_max_classes),
                samples_per_class=int(args.plot_samples_per_class),
                plot_class_ids=plot_class_ids,
                tsne_metric="cosine",
                show_legend=bool(args.plot_legend),
                class_markers=plot_class_markers,
                class_order=plot_class_order,
                class_hardness_rank=plot_class_hardness_rank,
                figsize=scatter_figsize,
                marker_size=scatter_marker_size,
                legend_fontsize=scatter_legend_fontsize,
                legend_ncol=scatter_legend_ncol,
                inline_key=inline_key,
                inline_key_fontsize=scatter_inline_key_fontsize,
                density=scatter_density,
                density_gridsize=scatter_density_gridsize,
                density_alpha=scatter_density_alpha,
                show_frame=scatter_frame,
                frame_color=scatter_frame_color,
                frame_linewidth=scatter_frame_width,
                color_mode=scatter_color_mode,
                point_color=scatter_color_a,
                family_groups=family_groups,
                family_label_fontsize=family_label_fontsize,
                family_annotations=family_annotations,
                class_color_map=class_color_map,
                legend_mode=plot_legend_mode,
                legend_out_path=tsne_a_l2_legend_path if plot_legend_mode == "separate" else None,
                legend_figsize=legend_figsize,
                axis_limits=limits,
                coords=tsne_a_l2_coords,
                keep_idx=tsne_a_l2_keep,
                compute_metrics=False,
                save_plot=True,
            )
            _plot_tsne(
                Xb_l2,
                y,
                class_names,
                out_dir / "tsne_B_l2cos.png",
                title="t-SNE (B, L2 + cosine)",
                perplexity=float(args.tsne_perplexity),
                seed=int(args.seed),
                max_classes=int(args.plot_max_classes),
                samples_per_class=int(args.plot_samples_per_class),
                plot_class_ids=plot_class_ids,
                tsne_metric="cosine",
                show_legend=bool(args.plot_legend),
                class_markers=plot_class_markers,
                class_order=plot_class_order,
                class_hardness_rank=plot_class_hardness_rank,
                figsize=scatter_figsize,
                marker_size=scatter_marker_size,
                legend_fontsize=scatter_legend_fontsize,
                legend_ncol=scatter_legend_ncol,
                inline_key=inline_key,
                inline_key_fontsize=scatter_inline_key_fontsize,
                density=scatter_density,
                density_gridsize=scatter_density_gridsize,
                density_alpha=scatter_density_alpha,
                show_frame=scatter_frame,
                frame_color=scatter_frame_color,
                frame_linewidth=scatter_frame_width,
                color_mode=scatter_color_mode,
                point_color=scatter_color_b,
                family_groups=family_groups,
                family_label_fontsize=family_label_fontsize,
                family_annotations=family_annotations,
                class_color_map=class_color_map,
                legend_mode=plot_legend_mode,
                legend_out_path=tsne_b_l2_legend_path if plot_legend_mode == "separate" else None,
                legend_figsize=legend_figsize,
                axis_limits=limits,
                coords=tsne_b_l2_coords,
                keep_idx=tsne_b_l2_keep,
                compute_metrics=False,
                save_plot=True,
            )

        def _pca_a_l2():
            nonlocal pca_a_l2_keep, pca_a_l2_Z
            keep, Z3, _ = _plot_pca_scatter(
                Xa_l2,
                y,
                class_names,
                out_pc12=out_dir / "pca_A_l2_pc1_pc2.png",
                out_pc13=out_dir / "pca_A_l2_pc1_pc3.png",
                title_prefix="A (L2)",
                seed=int(args.seed),
                max_classes=int(args.plot_max_classes),
                samples_per_class=int(args.plot_samples_per_class),
                plot_class_ids=plot_class_ids,
                show_legend=bool(args.plot_legend),
                class_markers=plot_class_markers,
                class_order=plot_class_order,
                class_hardness_rank=plot_class_hardness_rank,
                figsize=scatter_figsize,
                marker_size=scatter_marker_size,
                legend_fontsize=scatter_legend_fontsize,
                legend_ncol=scatter_legend_ncol,
                inline_key=inline_key,
                inline_key_fontsize=scatter_inline_key_fontsize,
                density=scatter_density,
                density_gridsize=scatter_density_gridsize,
                density_alpha=scatter_density_alpha,
                color_mode=scatter_color_mode,
                point_color=scatter_color_a,
                class_color_map=class_color_map if scatter_color_mode == "by-proximity" else None,
                family_groups=family_groups,
                family_label_fontsize=family_label_fontsize,
                family_trim_quantile=family_trim_quantile,
                family_annotations=family_annotations,
            )
            pca_a_l2_keep = keep
            pca_a_l2_Z = Z3
            if bool(args.write_points):
                _write_points_csv(
                    out_dir / "points_pca_A_l2.csv",
                    utt_ids=utt_ids,
                    labels=y,
                    class_names=class_names,
                    keep_idx=keep,
                    coords=Z3,
                    coord_names=["pc1", "pc2", "pc3"],
                    meta_df=meta_df,
                )
            _maybe_factor("pca_A_l2_pc1_pc2", "A", keep, Z3[:, [0, 1]])
            _maybe_factor("pca_A_l2_pc1_pc3", "A", keep, Z3[:, [0, 2]])

        def _pca_b_l2():
            nonlocal pca_b_l2_keep, pca_b_l2_Z
            keep, Z3, _ = _plot_pca_scatter(
                Xb_l2,
                y,
                class_names,
                out_pc12=out_dir / "pca_B_l2_pc1_pc2.png",
                out_pc13=out_dir / "pca_B_l2_pc1_pc3.png",
                title_prefix="B (L2)",
                seed=int(args.seed),
                max_classes=int(args.plot_max_classes),
                samples_per_class=int(args.plot_samples_per_class),
                plot_class_ids=plot_class_ids,
                show_legend=bool(args.plot_legend),
                class_markers=plot_class_markers,
                class_order=plot_class_order,
                class_hardness_rank=plot_class_hardness_rank,
                figsize=scatter_figsize,
                marker_size=scatter_marker_size,
                legend_fontsize=scatter_legend_fontsize,
                legend_ncol=scatter_legend_ncol,
                inline_key=inline_key,
                inline_key_fontsize=scatter_inline_key_fontsize,
                density=scatter_density,
                density_gridsize=scatter_density_gridsize,
                density_alpha=scatter_density_alpha,
                color_mode=scatter_color_mode,
                point_color=scatter_color_b,
                class_color_map=class_color_map if scatter_color_mode == "by-proximity" else None,
                family_groups=family_groups,
                family_label_fontsize=family_label_fontsize,
                family_trim_quantile=family_trim_quantile,
                family_annotations=family_annotations,
            )
            pca_b_l2_keep = keep
            pca_b_l2_Z = Z3
            if bool(args.write_points):
                _write_points_csv(
                    out_dir / "points_pca_B_l2.csv",
                    utt_ids=utt_ids,
                    labels=y,
                    class_names=class_names,
                    keep_idx=keep,
                    coords=Z3,
                    coord_names=["pc1", "pc2", "pc3"],
                    meta_df=meta_df,
                )
            _maybe_factor("pca_B_l2_pc1_pc2", "B", keep, Z3[:, [0, 1]])
            _maybe_factor("pca_B_l2_pc1_pc3", "B", keep, Z3[:, [0, 2]])

        _step("t-SNE A (L2 + cosine)", _tsne_a_l2)
        _step("t-SNE B (L2 + cosine)", _tsne_b_l2)
        _step("t-SNE render (L2 + cosine)", _tsne_render_l2)
        if pca_enabled:
            _step("PCA scatter A (L2)", _pca_a_l2)
            _step("PCA scatter B (L2)", _pca_b_l2)
        _step(
            "Distance distributions (L2 + cosine)",
            lambda: _plot_distance_distributions(
                intra_a_l2,
                inter_a_l2,
                intra_b_l2,
                inter_b_l2,
                out_dir / "distance_distributions_l2cos.png",
                title="Intra vs inter distances (L2 + cosine)",
                show_legend=bool(args.plot_legend),
            ),
        )
        if pca_enabled:
            _step(
                "PCA spectrum (L2)",
                lambda: _plot_pca_spectrum(
                    list(pca_a_l2["pca"]["eigs"]),
                    list(pca_b_l2["pca"]["eigs"]),
                    out_dir / "pca_spectrum_l2.png",
                    title="PCA spectrum (L2 normalized)",
                    show_legend=bool(args.plot_legend),
                ),
            )

    with tqdm(total=len(plot_steps), desc="Plots", unit="step") as bar:
        for name, fn in plot_steps:
            bar.set_description(f"Plots: {name}")
            step_t0 = time.perf_counter()
            fn()
            bar.update(1)
            print(f"[plots] {name} done in {time.perf_counter() - step_t0:.1f}s", flush=True)

    print(f"[plots] all done in {time.perf_counter() - plot_t0:.1f}s", flush=True)

    # Summary JSON (remove big arrays)
    cent_a_slim = {k: v for k, v in cent_a.items() if k not in {"centroid_distance_upper", "centroid_nearest"}}
    cent_b_slim = {k: v for k, v in cent_b.items() if k not in {"centroid_distance_upper", "centroid_nearest"}}
    summary = {
        "data": {
            "data_root": str(args.data_root),
            "protocol_csv": str(args.protocol_csv),
            "path_column": str(args.path_column),
            "class_column": str(args.class_column),
            "include_classes": _csv_list(args.include_classes),
            "exclude_classes": _csv_list(args.exclude_classes),
            "segment_seconds": args.segment_seconds,
            "max_per_class": args.max_per_class,
            "max_total": args.max_total,
            "num_utts_aligned": int(len(utt_ids)),
            "num_classes": int(len(np.unique(y))),
        },
        "analysis": {
            "distance_metric": metric,
            "rank_distance_spearman": float(rank_corr),
            "knn_overlap": knn,
            "centroid_distance_spearman": float(centroid_dist_spearman),
            "centroid_nearest_match": float(nn_match),
            "tsne_A": tsne_a_metrics,
            "tsne_B": tsne_b_metrics,
            "l2cosine": l2cos,
            "factor_analysis_2d": factor_scores,
            "factors_used": factors,
            "metadata_csvs": [str(p) for p in (args.metadata_csv or [])],
            "metadata_join_column": args.metadata_join_column,
            "factor_which": args.factor_which,
        },
        "model_A": {
            "source": str(args.a_embeddings_npz) if args.a_embeddings_npz else str(args.a_checkpoint),
            "extractor": args.a_extractor,
            "processor": args.a_processor,
            "classifier": args.a_classifier,
            "embed_layer": args.a_embed_layer,
            "bottleneck_dim": args.a_bottleneck_dim,
            "num_classes": args.a_num_classes,
            "l2_normalize": args.a_l2_normalize,
            "post_l2_normalize": bool(args.a_post_l2_normalize),
            "separability": sep_a,
            "intrinsic_dim": twonn_a,
            "spectrum": None
            if not pca_enabled
            else {k: v for k, v in pca_a.items() if k != "pca"} | {"pca_components": len(pca_a["pca"]["eigs"])},
            "centroids": cent_a_slim,
            "distance_samples": {
                "intra_mean": float(np.mean(intra_a)) if intra_a.size else float("nan"),
                "inter_mean": float(np.mean(inter_a)) if inter_a.size else float("nan"),
                "intra_median": float(np.median(intra_a)) if intra_a.size else float("nan"),
                "inter_median": float(np.median(inter_a)) if inter_a.size else float("nan"),
            },
            "tail_margins_centroid": tail_centroid_a,
            "tail_margins_neg_neighbor": tail_neighbor_a,
        },
        "model_B": {
            "source": str(args.b_embeddings_npz) if args.b_embeddings_npz else str(args.b_checkpoint),
            "extractor": args.b_extractor,
            "processor": args.b_processor,
            "classifier": args.b_classifier,
            "embed_layer": args.b_embed_layer,
            "bottleneck_dim": args.b_bottleneck_dim,
            "num_classes": args.b_num_classes,
            "l2_normalize": args.b_l2_normalize,
            "post_l2_normalize": bool(args.b_post_l2_normalize),
            "separability": sep_b,
            "intrinsic_dim": twonn_b,
            "spectrum": None
            if not pca_enabled
            else {k: v for k, v in pca_b.items() if k != "pca"} | {"pca_components": len(pca_b["pca"]["eigs"])},
            "centroids": cent_b_slim,
            "distance_samples": {
                "intra_mean": float(np.mean(intra_b)) if intra_b.size else float("nan"),
                "inter_mean": float(np.mean(inter_b)) if inter_b.size else float("nan"),
                "intra_median": float(np.median(intra_b)) if intra_b.size else float("nan"),
                "inter_median": float(np.median(inter_b)) if inter_b.size else float("nan"),
            },
            "tail_margins_centroid": tail_centroid_b,
            "tail_margins_neg_neighbor": tail_neighbor_b,
        },
        "artifacts": {
            "tsne_A_png": str(tsne_a_path),
            "tsne_B_png": str(tsne_b_path),
            "tsne_A_legend_png": str(tsne_a_legend_path) if plot_legend_mode == "separate" else None,
            "tsne_B_legend_png": str(tsne_b_legend_path) if plot_legend_mode == "separate" else None,
            "tsne_A_l2cos_png": str(out_dir / "tsne_A_l2cos.png") if l2cos is not None else None,
            "tsne_B_l2cos_png": str(out_dir / "tsne_B_l2cos.png") if l2cos is not None else None,
            "tsne_A_l2cos_legend_png": str(out_dir / "tsne_A_l2cos_legend.png")
            if (l2cos is not None and plot_legend_mode == "separate")
            else None,
            "tsne_B_l2cos_legend_png": str(out_dir / "tsne_B_l2cos_legend.png")
            if (l2cos is not None and plot_legend_mode == "separate")
            else None,
            "pca_A_pc1_pc2_png": str(out_dir / "pca_A_pc1_pc2.png") if pca_enabled else None,
            "pca_A_pc1_pc3_png": str(out_dir / "pca_A_pc1_pc3.png") if pca_enabled else None,
            "pca_B_pc1_pc2_png": str(out_dir / "pca_B_pc1_pc2.png") if pca_enabled else None,
            "pca_B_pc1_pc3_png": str(out_dir / "pca_B_pc1_pc3.png") if pca_enabled else None,
            "pca_A_l2_pc1_pc2_png": str(out_dir / "pca_A_l2_pc1_pc2.png") if (pca_enabled and l2cos is not None) else None,
            "pca_A_l2_pc1_pc3_png": str(out_dir / "pca_A_l2_pc1_pc3.png") if (pca_enabled and l2cos is not None) else None,
            "pca_B_l2_pc1_pc2_png": str(out_dir / "pca_B_l2_pc1_pc2.png") if (pca_enabled and l2cos is not None) else None,
            "pca_B_l2_pc1_pc3_png": str(out_dir / "pca_B_l2_pc1_pc3.png") if (pca_enabled and l2cos is not None) else None,
            "distance_distributions_png": str(out_dir / "distance_distributions.png"),
            "distance_distributions_l2cos_png": str(out_dir / "distance_distributions_l2cos.png") if l2cos is not None else None,
            "pca_spectrum_png": str(out_dir / "pca_spectrum.png") if pca_enabled else None,
            "pca_spectrum_l2_png": str(out_dir / "pca_spectrum_l2.png") if (pca_enabled and l2cos is not None) else None,
            "embeddings_A_npz": str(out_dir / "embeddings_A.npz") if args.a_embeddings_npz is None else str(args.a_embeddings_npz),
            "embeddings_B_npz": str(out_dir / "embeddings_B.npz") if args.b_embeddings_npz is None else str(args.b_embeddings_npz),
            "embeddings_A_meta_json": str(_npz_meta_path(out_dir / "embeddings_A.npz")) if args.a_embeddings_npz is None else None,
            "embeddings_B_meta_json": str(_npz_meta_path(out_dir / "embeddings_B.npz")) if args.b_embeddings_npz is None else None,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Small, human-readable summary
    md = out_dir / "summary.md"
    lines = [
        "# Embedding space comparison",
        "",
        f"- Utterances (aligned): {summary['data']['num_utts_aligned']}",
        f"- Classes: {summary['data']['num_classes']}",
        f"- Distance metric: {metric}",
        "",
        "## Rank / neighborhood consistency (A vs B)",
        f"- Pairwise distance rank Spearman (sampled): {rank_corr:.4f}",
    ]
    for k in knn_ks:
        item = knn.get(f"k={k}", {})
        lines.append(
            f"- kNN overlap k={k}: overlap={item.get('knn_overlap_frac', float('nan')):.4f}, jaccard={item.get('knn_jaccard', float('nan')):.4f}"
        )
    lines += [
        "",
        "## Class separability",
        f"- A silhouette: {sep_a['silhouette']:.4f} | B silhouette: {sep_b['silhouette']:.4f}",
        f"- A Davies-Bouldin: {sep_a['davies_bouldin']:.4f} | B Davies-Bouldin: {sep_b['davies_bouldin']:.4f}",
        f"- A Calinski-Harabasz: {sep_a['calinski_harabasz']:.2f} | B Calinski-Harabasz: {sep_b['calinski_harabasz']:.2f}",
        "",
        "## Forensic tail margins (lower is worse)",
        f"- Centroid margin p01: A={tail_centroid_a.get('margin_p01', float('nan')):.6f} | B={tail_centroid_b.get('margin_p01', float('nan')):.6f}",
        f"- Nearest-negative-neighbor margin p01: A={tail_neighbor_a.get('margin_nn_p01', float('nan')):.6f} | B={tail_neighbor_b.get('margin_nn_p01', float('nan')):.6f}",
        "",
        "## Intrinsic dimension / spectrum",
        f"- A TwoNN ID: {twonn_a['twonn_id']:.3f} (R2={twonn_a['twonn_r2']:.3f}) | B TwoNN ID: {twonn_b['twonn_id']:.3f} (R2={twonn_b['twonn_r2']:.3f})",
    ]
    if pca_enabled:
        lines += [
            f"- A effective rank: {pca_a['effective_rank']:.2f} | B effective rank: {pca_b['effective_rank']:.2f}",
            f"- A participation ratio: {pca_a['participation_ratio']:.2f} | B participation ratio: {pca_b['participation_ratio']:.2f}",
            f"- A top-1 var: {pca_a['pca_top1_var']:.3f} | B top-1 var: {pca_b['pca_top1_var']:.3f}",
            "",
        ]
    else:
        lines += ["- PCA metrics: disabled (--no-pca)", ""]
    lines += [
        "## Centroid graph",
        f"- Centroid distance Spearman: {centroid_dist_spearman:.4f}",
        f"- Nearest-centroid match rate: {nn_match:.4f}",
        "",
    ]
    if l2cos is not None:
        lines += [
            "## L2-normalized + cosine diagnostics (reviewer-proof)",
            f"- Pairwise distance rank Spearman (sampled): {l2cos.get('rank_distance_spearman', float('nan')):.4f}",
        ]
        for k in knn_ks:
            item = l2cos.get("knn_overlap", {}).get(f"k={k}", {})
            lines.append(
                f"- kNN overlap k={k}: overlap={item.get('knn_overlap_frac', float('nan')):.4f}, jaccard={item.get('knn_jaccard', float('nan')):.4f}"
            )
        a_tail = l2cos.get("A", {}).get("tail_margins_centroid", {})
        b_tail = l2cos.get("B", {}).get("tail_margins_centroid", {})
        a_nn = l2cos.get("A", {}).get("tail_margins_neg_neighbor", {})
        b_nn = l2cos.get("B", {}).get("tail_margins_neg_neighbor", {})
        lines += [
            f"- Centroid margin p01: A={a_tail.get('margin_p01', float('nan')):.6f} | B={b_tail.get('margin_p01', float('nan')):.6f}",
            f"- Nearest-negative-neighbor margin p01: A={a_nn.get('margin_nn_p01', float('nan')):.6f} | B={b_nn.get('margin_nn_p01', float('nan')):.6f}",
            "",
        ]
    if factor_scores:
        lines += ["## Factor alignment (2D recolor scores)"]
        preferred_spaces = [
            "tsne_B_l2cos",
            "pca_B_l2_pc1_pc2",
            "pca_B_l2_pc1_pc3",
            "tsne_B",
            "pca_B_pc1_pc2",
            "pca_B_pc1_pc3",
        ]
        for space in preferred_spaces:
            if space not in factor_scores:
                continue
            items = []
            for col, stats in factor_scores.get(space, {}).items():
                sil = float(stats.get("silhouette_2d", float("nan")))
                purity = float(stats.get("knn_purity_2d", float("nan")))
                items.append((sil, purity, col))
            items.sort(key=lambda t: (t[0], t[1]), reverse=True)
            if not items:
                continue
            lines.append(f"- {space}:")
            for sil, purity, col in items[:5]:
                lines.append(f"  - {col}: silhouette={sil:.4f}, knn_purity@{int(args.factor_knn_k)}={purity:.4f}")
        lines.append("")
    lines += [
        "## Artifacts",
        "- `tsne_A.png`, `tsne_B.png` (top classes only)",
        "- `distance_distributions.png` (sampled intra/inter)",
        "- `points_*.csv` (2D coordinates + joined metadata, if --write-points)",
        "- `recolor_*.png` (same 2D coords recolored by metadata factors, if enabled)",
        "- `summary.json` (full metrics)",
    ]
    if plot_legend_mode == "separate" and bool(args.plot_legend):
        lines.append("- `tsne_A_legend.png`, `tsne_B_legend.png` (legend-only)")
    if pca_enabled:
        lines += [
            "- `pca_A_pc1_pc2.png`, `pca_A_pc1_pc3.png`, `pca_B_pc1_pc2.png`, `pca_B_pc1_pc3.png` (PCA scatter)",
            "- `pca_spectrum.png` (eigenvalue decay)",
        ]
    if l2cos is not None:
        lines += [
            "- `tsne_A_l2cos.png`, `tsne_B_l2cos.png` (t-SNE on L2-normalized embeddings with cosine metric)",
            "- `distance_distributions_l2cos.png` (sampled intra/inter on L2+cosine)",
        ]
        if plot_legend_mode == "separate" and bool(args.plot_legend):
            lines.append("- `tsne_A_l2cos_legend.png`, `tsne_B_l2cos_legend.png` (legend-only, L2+cosine)")
        if pca_enabled:
            lines += [
                "- `pca_A_l2_pc1_pc2.png`, `pca_A_l2_pc1_pc3.png`, `pca_B_l2_pc1_pc2.png`, `pca_B_l2_pc1_pc3.png` (PCA scatter, L2)",
                "- `pca_spectrum_l2.png` (eigenvalue decay on L2-normalized)",
            ]
    lines += [
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
