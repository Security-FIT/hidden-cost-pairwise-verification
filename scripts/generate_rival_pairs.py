#!/usr/bin/env python3
"""
Generate rival generator mappings using centroid cosine similarity.

Workflow:
  1) Sample up to N utterances per generator from the train protocol.
  2) Compute embeddings (from a checkpoint or a precomputed embeddings file).
  3) Average embeddings to get generator centroids.
  4) Pick the nearest neighbor centroid as the rival, then expand to a band of
     rivals with similarity >= (S_max - delta).
  5) Optionally force rivals that share dataset+language metadata.

Outputs a CSV with columns:
  model_name, rival_model_name, similarity, s_max, band_min, sample_count, is_forced_meta
"""

from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import build_model
from datasets.utils import custom_single_batch_create


class MLAADPathDataset(Dataset):
    def __init__(self, root_dir: str, paths: list[str]):
        self.root_dir = root_dir
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, rel_path: str) -> str:
        normalized = rel_path.lstrip("./")
        return str(Path(self.root_dir) / normalized)

    def _load_waveform(self, abs_path: str) -> torch.Tensor:
        waveform, _ = sf.read(abs_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        path = self.paths[idx]
        wav = self._load_waveform(self._resolve_path(path))
        return path, wav, 0


def load_protocol(csv_path: Path, path_column: str, source_column: str) -> Dict[str, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = {path_column, source_column} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} missing required columns: {', '.join(sorted(missing_cols))}")
    df = df.dropna(subset=[path_column, source_column])

    source_to_paths: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        path = str(row[path_column])
        source = str(row[source_column])
        source_to_paths.setdefault(source, []).append(path)
    for paths in source_to_paths.values():
        paths.sort()
    return source_to_paths


def _clean_value(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_dataset_constraints(
    df: pd.DataFrame,
    model_column: str,
    training_data_column: str,
    language_column: str,
) -> Dict[str, set[str]]:
    required = {model_column, training_data_column, language_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {', '.join(sorted(missing))}")

    exclusions = {"unknown", "multi-dataset", "none", "multilingual"}
    group_map: Dict[str, set[str]] = {}
    for _, row in df.iterrows():
        model = _clean_value(row.get(model_column))
        training_data = _clean_value(row.get(training_data_column)).lower()
        language = _clean_value(row.get(language_column)).lower()
        if not model or not training_data or not language:
            continue
        if training_data in exclusions:
            continue
        key = f"{training_data}::{language}"
        group_map.setdefault(key, set()).add(model)

    constraints: Dict[str, set[str]] = {}
    for models in group_map.values():
        if len(models) < 2:
            continue
        for model in models:
            rivals = models - {model}
            if rivals:
                constraints.setdefault(model, set()).update(rivals)

    bark_a = "suno/bark"
    bark_b = "suno/bark-small"
    constraints.setdefault(bark_a, set()).add(bark_b)
    constraints.setdefault(bark_b, set()).add(bark_a)

    total_constraints = sum(len(rivals) for rivals in constraints.values())
    print(f"Metadata forced constraints: {total_constraints}")
    return constraints


def sample_paths(
    source_to_paths: Dict[str, List[str]],
    samples_per_source: int,
    rng: random.Random,
) -> Dict[str, List[str]]:
    sampled: Dict[str, List[str]] = {}
    for source in sorted(source_to_paths):
        paths = source_to_paths[source]
        if not paths:
            continue
        if samples_per_source <= 0 or samples_per_source >= len(paths):
            chosen = paths[:]
        else:
            chosen = rng.sample(paths, samples_per_source)
        sampled[source] = chosen
    return sampled


def load_embedding_table(emb_path: Path) -> Tuple[List[str], np.ndarray]:
    data = np.load(emb_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile) and "embeddings" in data and "utt_ids" in data:
        emb = np.asarray(data["embeddings"])
        utt_ids_arr = np.asarray(data["utt_ids"])
        utt_ids = [str(u) for u in utt_ids_arr.tolist()]
        if emb.shape[0] != len(utt_ids):
            raise ValueError(
                f"Embedding/table shape mismatch in {emb_path}: {emb.shape[0]} rows vs {len(utt_ids)} ids."
            )
        return utt_ids, emb

    if isinstance(data, np.lib.npyio.NpzFile):
        mapping = {k: np.asarray(data[k]) for k in data.files}
        utt_ids = sorted(mapping)
        emb = np.stack([mapping[u] for u in utt_ids], axis=0)
        return utt_ids, emb

    if isinstance(data, np.ndarray) and data.dtype == object:
        maybe_dict = data.item()
        if isinstance(maybe_dict, dict):
            utt_ids = sorted(maybe_dict)
            emb = np.stack([np.asarray(maybe_dict[u]) for u in utt_ids], axis=0)
            return utt_ids, emb

    if isinstance(data, dict):
        utt_ids = sorted(data)
        emb = np.stack([np.asarray(data[u]) for u in utt_ids], axis=0)
        return utt_ids, emb

    raise ValueError(
        f"Unsupported embedding table format in {emb_path}. "
        "Expect npz with embeddings/utt_ids or a dict of id -> vector."
    )


def load_embeddings(emb_path: Path) -> Dict[str, np.ndarray]:
    utt_ids, emb = load_embedding_table(emb_path)
    return {utt_id: emb[idx] for idx, utt_id in enumerate(utt_ids)}


def extract_embeddings_from_checkpoint(
    paths: list[str],
    data_root: Path,
    checkpoint: Path,
    extractor: str,
    processor: str,
    classifier: str,
    bottleneck_dim: int | None,
    num_classes: int | None,
    batch_size: int,
    num_workers: int,
    amp_eval: bool,
    amp_dtype: str,
    device: str | None,
) -> Dict[str, np.ndarray]:
    if not paths:
        return {}
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    amp_torch_dtype = dtype_map.get(amp_dtype, torch.bfloat16)

    args_ns = SimpleNamespace(
        extractor=extractor,
        processor=processor,
        classifier=classifier,
        bottleneck_dim=bottleneck_dim,
        num_classes=num_classes,
        kernel=None,
        n_components=None,
        covariance_type=None,
    )
    model, trainer = build_model(args_ns)
    if hasattr(trainer, "set_amp_eval"):
        trainer.set_amp_eval(amp_eval, dtype=amp_torch_dtype)
    trainer.load_model(str(checkpoint))
    model.eval()
    model.to(resolved_device)

    if not hasattr(model, "feature_processor"):
        raise RuntimeError("Model does not expose feature_processor; cannot extract embeddings.")

    dataset = MLAADPathDataset(str(data_root), paths)
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": custom_single_batch_create,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": resolved_device.startswith("cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)

    emb_map: dict[str, np.ndarray] = {}
    autocast_ctx = (
        torch.autocast(device_type=resolved_device.split(":")[0], dtype=amp_torch_dtype)
        if amp_eval and resolved_device.startswith("cuda")
        else nullcontext()
    )
    with torch.no_grad():
        for file_names, waveforms, _ in tqdm(dataloader, desc="Extracting embeddings"):
            waveforms = waveforms.to(resolved_device)
            with autocast_ctx:
                feats = model.extractor.extract_features(waveforms)
                emb = model.feature_processor(feats)
            emb_cpu = emb.detach().to(dtype=torch.float32).cpu().numpy()
            for name, vec in zip(file_names, emb_cpu):
                emb_map[name] = vec
    return emb_map


def compute_centroids(
    sampled_paths: Dict[str, List[str]],
    embeddings: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    centroids: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    missing: List[str] = []
    embed_dim = None
    for source, paths in sampled_paths.items():
        vecs: List[np.ndarray] = []
        for path in paths:
            if path not in embeddings:
                missing.append(path)
                continue
            vec = np.asarray(embeddings[path], dtype=float).ravel()
            if embed_dim is None:
                embed_dim = vec.shape[0]
            elif vec.shape[0] != embed_dim:
                raise ValueError(
                    f"Embedding dimensionality mismatch for {path}: {vec.shape[0]} vs {embed_dim}."
                )
            vecs.append(vec)
        if vecs:
            mat = np.stack(vecs, axis=0)
            centroids[source] = mat.mean(axis=0)
            counts[source] = len(vecs)
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} sampled paths missing embeddings (e.g., {preview}).")
    return centroids, counts


def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < eps):
        raise ValueError("Zero-norm centroid encountered during normalization.")
    return matrix / norms


def compute_rivals(
    centroids: Dict[str, np.ndarray],
    delta: float,
    forced_map: Dict[str, set[str]] | None = None,
) -> Dict[str, List[Tuple[str, float, float, float, bool]]]:
    sources = sorted(centroids)
    if len(sources) < 2:
        return {}
    forced_map = forced_map or {}
    source_index = {source: idx for idx, source in enumerate(sources)}
    mat = np.stack([centroids[src] for src in sources], axis=0)
    mat = _l2_normalize_rows(mat)
    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)

    rivals: Dict[str, List[Tuple[str, float, float, float, bool]]] = {}
    for idx, source in enumerate(sources):
        row = sim[idx]
        best_idx = int(np.argmax(row))
        best_sim = float(row[best_idx])
        if not np.isfinite(best_sim):
            continue
        band_min = best_sim - delta
        band_indices = np.where(row >= band_min)[0]
        entry_map: Dict[str, Dict[str, float | bool]] = {}
        for j in band_indices:
            if j == idx:
                continue
            rival_name = sources[j]
            entry_map[rival_name] = {
                "similarity": float(row[j]),
                "s_max": best_sim,
                "band_min": band_min,
                "is_forced_meta": False,
            }
        for rival_name in sorted(forced_map.get(source, set())):
            if rival_name == source:
                continue
            j = source_index.get(rival_name)
            if j is None:
                continue
            sim_val = float(row[j]) if np.isfinite(row[j]) else np.nan
            if rival_name in entry_map:
                entry_map[rival_name]["is_forced_meta"] = True
            else:
                entry_map[rival_name] = {
                    "similarity": sim_val,
                    "s_max": best_sim,
                    "band_min": band_min,
                    "is_forced_meta": True,
                }
        if entry_map:
            entries = sorted(
                entry_map.items(),
                key=lambda item: item[1]["similarity"]
                if np.isfinite(item[1]["similarity"])
                else -np.inf,
                reverse=True,
            )
            rivals[source] = [
                (
                    name,
                    float(data["similarity"]),
                    float(data["s_max"]),
                    float(data["band_min"]),
                    bool(data["is_forced_meta"]),
                )
                for name, data in entries
            ]
    return rivals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rival generator mappings for MLAAD.")
    parser.add_argument("--train-csv", type=Path, required=True, help="Train protocol with path/model columns.")
    parser.add_argument("--path-column", default="path", help="CSV column with relative audio paths.")
    parser.add_argument("--source-column", default="model_name", help="CSV column with generator labels.")
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=50,
        help="Sample this many utterances per source (0 = all).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.10,
        help="Similarity band width: keep rivals with sim >= (S_max - delta).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Deprecated (ignored). Use --delta instead.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Optional embeddings file (npz with embeddings/utt_ids or dict-like npz).",
    )
    parser.add_argument("--data-root", type=Path, help="Root directory for MLAAD audio.")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint path.")
    parser.add_argument("--extractor", help="Extractor name (same as training).")
    parser.add_argument("--processor", help="Processor name (e.g., MHFA).")
    parser.add_argument("--classifier", help="Classifier name (same as training).")
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=None,
        help="Bottleneck dimension for FFMulticlass/FFCosine3 (match the checkpoint).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes for FF/FFMulticlass (match the checkpoint).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--amp-eval", action=argparse.BooleanOptionalAction, default=False, help="Enable AMP.")
    parser.add_argument("--amp-dtype", default="bf16", help="AMP dtype: bf16 or fp16.")
    parser.add_argument("--device", default=None, help="Force device, e.g., cuda or cpu.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: rivals.csv next to train CSV).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/train_meta.csv"),
        help="Metadata CSV used for dataset+language forced rivals.",
    )
    parser.add_argument(
        "--metadata-model-column",
        default="model_name",
        help="Column in metadata CSV with model names (default: model_name).",
    )
    parser.add_argument(
        "--metadata-training-data-column",
        default="meta_training_data",
        help="Metadata column with training dataset names (default: meta_training_data).",
    )
    parser.add_argument(
        "--metadata-language-column",
        default="meta_language",
        help="Metadata column with language tags (default: meta_language).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.similarity_threshold is not None:
        print("Warning: --similarity-threshold is deprecated and ignored; using --delta instead.")
    if args.delta < 0:
        raise ValueError("--delta must be non-negative.")
    source_to_paths = load_protocol(args.train_csv, args.path_column, args.source_column)
    if not source_to_paths:
        raise ValueError("No sources found in train CSV.")

    rng = random.Random(args.seed)
    sampled_paths = sample_paths(source_to_paths, args.samples_per_source, rng)
    flat_paths = sorted({p for paths in sampled_paths.values() for p in paths})
    if not flat_paths:
        raise ValueError("No paths sampled; check samples-per-source.")

    if args.embeddings:
        embeddings = load_embeddings(args.embeddings)
    else:
        required = {
            "data_root": args.data_root,
            "checkpoint": args.checkpoint,
            "extractor": args.extractor,
            "processor": args.processor,
            "classifier": args.classifier,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Checkpoint mode requires: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        embeddings = extract_embeddings_from_checkpoint(
            flat_paths,
            data_root=args.data_root,
            checkpoint=args.checkpoint,
            extractor=args.extractor,
            processor=args.processor,
            classifier=args.classifier,
            bottleneck_dim=args.bottleneck_dim,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp_eval=args.amp_eval,
            amp_dtype=args.amp_dtype,
            device=args.device,
        )

    centroids, counts = compute_centroids(sampled_paths, embeddings)
    forced_map: Dict[str, set[str]] = {}
    if args.metadata_csv:
        metadata_path = args.metadata_csv.expanduser().resolve()
        if metadata_path.exists():
            meta_df = pd.read_csv(metadata_path)
            forced_map = get_dataset_constraints(
                meta_df,
                args.metadata_model_column,
                args.metadata_training_data_column,
                args.metadata_language_column,
            )
            if forced_map:
                forced_map = {
                    model: rivals
                    for model, rivals in forced_map.items()
                    if model in sampled_paths
                }
                if not forced_map:
                    print("Metadata constraints did not match sampled sources; skipping.")
        else:
            print(f"Warning: metadata CSV not found at {metadata_path}; skipping metadata constraints.")

    rivals = compute_rivals(centroids, args.delta, forced_map)

    rows: List[Dict[str, object]] = []
    for source in sorted(sampled_paths):
        entries = rivals.get(source, [])
        if not entries:
            rows.append(
                {
                    "model_name": source,
                    "rival_model_name": "",
                    "similarity": np.nan,
                    "s_max": np.nan,
                    "band_min": np.nan,
                    "sample_count": counts.get(source, 0),
                    "is_forced_meta": 0,
                }
            )
            continue
        for rival_name, sim, s_max, band_min, forced in entries:
            rows.append(
                {
                    "model_name": source,
                    "rival_model_name": rival_name,
                    "similarity": float(sim),
                    "s_max": float(s_max),
                    "band_min": float(band_min),
                    "sample_count": counts.get(source, 0),
                    "is_forced_meta": int(forced),
                }
            )
    out_path = args.output or (args.train_csv.parent / "rivals.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    sources_with_rivals = len({r["model_name"] for r in rows if r["rival_model_name"]})
    total_rivals = sum(1 for r in rows if r["rival_model_name"])
    print(f"Wrote rivals for {len(sampled_paths):,} sources to {out_path}")
    print(f"  sources with rivals: {sources_with_rivals:,}, total rivals: {total_rivals:,}, delta={args.delta}")


if __name__ == "__main__":
    main()
