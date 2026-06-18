#!/usr/bin/env python3
"""
Extract per-utterance MHFA (processor) embeddings for a list of MLAAD paths.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

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


def _collect_paths(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {', '.join(missing_cols)}")
    paths: list[str] = []
    for col in columns:
        paths.extend(df[col].dropna().astype(str).tolist())
    return sorted(set(paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MHFA embeddings for MLAAD paths.")
    parser.add_argument("--pairs-csv", type=Path, required=True, help="CSV with path columns (e.g., candidates.csv).")
    parser.add_argument(
        "--path-columns",
        nargs="+",
        default=["path_A"],
        help="CSV columns to use as utterance paths (default: path_A).",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory for MLAAD audio.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--extractor", required=True, help="Extractor name (same as training).")
    parser.add_argument("--processor", required=True, help="Processor name (e.g., MHFA).")
    parser.add_argument("--classifier", required=True, help="Classifier name (same as training).")
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
    parser.add_argument("--output", type=Path, required=True, help="Output path for embeddings (npz).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.pairs_csv)
    paths = _collect_paths(df, args.path_columns)
    if not paths:
        raise ValueError("No paths found to extract embeddings for.")
    print(f"Preparing embeddings for {len(paths):,} unique paths.")

    resolved_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    amp_torch_dtype = dtype_map.get(args.amp_dtype, torch.bfloat16)

    args_ns = SimpleNamespace(
        extractor=args.extractor,
        processor=args.processor,
        classifier=args.classifier,
        bottleneck_dim=args.bottleneck_dim,
        num_classes=args.num_classes,
        kernel=None,
        n_components=None,
        covariance_type=None,
    )
    model, trainer = build_model(args_ns)
    if hasattr(trainer, "set_amp_eval"):
        trainer.set_amp_eval(args.amp_eval, dtype=amp_torch_dtype)
    trainer.load_model(str(args.checkpoint))
    model.eval()
    model.to(resolved_device)

    if not hasattr(model, "feature_processor"):
        raise RuntimeError("Model does not expose feature_processor; cannot extract MHFA embeddings.")

    dataset = MLAADPathDataset(str(args.data_root), paths)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": custom_single_batch_create,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": resolved_device.startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)

    emb_map: dict[str, np.ndarray] = {}
    autocast_ctx = (
        torch.autocast(device_type=resolved_device.split(":")[0], dtype=amp_torch_dtype)
        if args.amp_eval and resolved_device.startswith("cuda")
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

    if not emb_map:
        raise RuntimeError("No embeddings extracted; check paths and data root.")

    utt_ids = sorted(emb_map)
    emb_matrix = np.stack([emb_map[u] for u in utt_ids], axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=emb_matrix, utt_ids=np.array(utt_ids))
    print(f"Wrote embeddings for {len(utt_ids):,} items to {args.output}")


if __name__ == "__main__":
    main()
