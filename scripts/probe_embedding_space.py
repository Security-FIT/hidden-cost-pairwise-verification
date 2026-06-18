#!/usr/bin/env python3
"""
Linear probing utility for MLAAD-style single-utterance protocols.

Workflow:
1) Load a dev protocol CSV.
2) Filter rows/classes.
3) Extract embeddings from a trained checkpoint.
4) Train/test a small Logistic Regression probe on those embeddings.
5) Save metrics and metadata to JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import build_model
from config import karolina_config, local_config, sge_config
from datasets.utils import custom_single_batch_create

try:
    import soundfile as sf
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "soundfile is required for loading audio in this script."
    ) from exc


def _csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    parts: list[str] = []
    for chunk in str(value).split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _amp_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def _coerce_rel_path(path_value: object) -> str:
    return str(path_value).strip()


def _select_runtime_config(args: argparse.Namespace) -> tuple[dict, str]:
    if args.sge:
        return sge_config, "sge"
    if args.karolina:
        return karolina_config, "karolina"
    return local_config, "local"


def _resolve_dev_protocol(args: argparse.Namespace) -> tuple[Path, Path, str]:
    config, config_name = _select_runtime_config(args)
    dataset_cfg = config.get(args.dataset_config_key)
    if dataset_cfg is None:
        raise KeyError(
            f"Missing dataset config key '{args.dataset_config_key}' in selected config '{config_name}'."
        )
    if "dev_subdir" not in dataset_cfg or "dev_protocol" not in dataset_cfg:
        raise KeyError(
            f"Config key '{args.dataset_config_key}' must define dev_subdir and dev_protocol."
        )

    if args.data_root is not None:
        data_root = args.data_root.expanduser().resolve()
    else:
        data_root = (Path(config["data_dir"]).expanduser() / dataset_cfg["dev_subdir"]).resolve()

    protocol_name = args.dev_protocol or dataset_cfg["dev_protocol"]
    protocol_path = Path(protocol_name).expanduser()
    if not protocol_path.is_absolute():
        protocol_path = data_root / protocol_path
    protocol_path = protocol_path.resolve()

    return protocol_path, data_root, config_name


def _filter_protocol_rows(
    protocol_df: pd.DataFrame,
    path_column: str,
    label_column: str,
    exclude_classes: list[str],
    max_per_class: int,
    min_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if path_column not in protocol_df.columns:
        raise ValueError(f"Missing path column '{path_column}' in protocol CSV.")
    if label_column not in protocol_df.columns:
        raise ValueError(f"Missing label column '{label_column}' in protocol CSV.")

    initial_rows = int(len(protocol_df))

    rows = protocol_df[[path_column, label_column]].copy()
    rows = rows.dropna(subset=[path_column, label_column]).copy()
    rows[path_column] = rows[path_column].map(_coerce_rel_path)
    rows[label_column] = rows[label_column].map(lambda x: str(x).strip())
    rows = rows[(rows[path_column] != "") & (rows[label_column] != "")].copy()
    rows_after_cleanup = int(len(rows))

    dropped_excluded = 0
    if exclude_classes:
        exclude_set = set(exclude_classes)
        before = len(rows)
        rows = rows[~rows[label_column].isin(exclude_set)].copy()
        dropped_excluded = int(before - len(rows))

    class_counts = rows[label_column].value_counts()
    small_classes = class_counts[class_counts < min_per_class]
    dropped_small = int(small_classes.sum()) if not small_classes.empty else 0
    if dropped_small > 0:
        keep_classes = set(class_counts[class_counts >= min_per_class].index.tolist())
        rows = rows[rows[label_column].isin(keep_classes)].copy()

    sampled_parts: list[pd.DataFrame] = []
    dropped_capped = 0
    for _, class_df in rows.groupby(label_column, sort=True):
        if max_per_class > 0 and len(class_df) > max_per_class:
            sampled = class_df.sample(n=max_per_class, random_state=seed)
            sampled_parts.append(sampled)
            dropped_capped += int(len(class_df) - max_per_class)
        else:
            sampled_parts.append(class_df)
    if sampled_parts:
        rows = pd.concat(sampled_parts, axis=0, ignore_index=True)
    else:
        rows = rows.iloc[0:0].copy()

    rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    final_class_counts = rows[label_column].value_counts().sort_index()

    stats = {
        "rows_in_protocol": initial_rows,
        "rows_after_cleanup": rows_after_cleanup,
        "rows_after_filtering": int(len(rows)),
        "num_classes_after_filtering": int(final_class_counts.shape[0]),
        "dropped_excluded": dropped_excluded,
        "dropped_small_classes": dropped_small,
        "dropped_by_cap": dropped_capped,
    }
    return rows, stats


class FilteredSingleProtocolDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        path_column: str,
        label_column: str,
        protocol_dir: Path,
        data_root: Path | None,
        segment_seconds: float | None,
        sample_rate: int,
    ):
        if len(rows) == 0:
            raise ValueError("Filtered protocol is empty.")

        self.path_column = path_column
        self.label_column = label_column
        self.protocol_dir = protocol_dir
        self.data_root = data_root
        self.segment_samples = None
        if segment_seconds is not None and segment_seconds > 0:
            self.segment_samples = int(segment_seconds * sample_rate)

        self.rows = rows[[path_column, label_column]].copy().reset_index(drop=True)
        class_names = sorted(self.rows[label_column].astype(str).unique().tolist())
        self.label_map = {name: idx for idx, name in enumerate(class_names)}
        self.rows["label_id"] = self.rows[label_column].map(self.label_map).astype(int)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_audio_path(self, path_value: str) -> Path:
        raw = str(path_value).strip()
        raw_path = Path(raw)
        normalized = raw.lstrip("./")
        tried: list[Path] = []

        if raw_path.is_absolute():
            tried.append(raw_path)
            if raw_path.is_file():
                return raw_path
        else:
            tried.append(raw_path)
            if raw_path.is_file():
                return raw_path.resolve()

        if self.data_root is not None:
            candidate = (self.data_root / normalized).expanduser()
            tried.append(candidate)
            if candidate.is_file():
                return candidate

        protocol_candidate = (self.protocol_dir / normalized).expanduser()
        tried.append(protocol_candidate)
        if protocol_candidate.is_file():
            return protocol_candidate

        raise FileNotFoundError(
            f"Could not resolve audio path '{raw}'. Tried: "
            + ", ".join(str(path) for path in tried[:4])
        )

    def _load_waveform(self, audio_path: Path) -> torch.Tensor:
        waveform, _ = sf.read(str(audio_path), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if self.segment_samples is not None:
            if waveform.shape[0] >= self.segment_samples:
                waveform = waveform[: self.segment_samples]
            else:
                pad = self.segment_samples - waveform.shape[0]
                waveform = np.pad(waveform, (0, pad), mode="constant")
        waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, int]:
        row = self.rows.iloc[index]
        rel_path = str(row[self.path_column])
        label_id = int(row["label_id"])
        abs_path = self._resolve_audio_path(rel_path)
        waveform = self._load_waveform(abs_path)
        return rel_path, waveform, label_id


def _processor_embedding(model, waveforms: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "extractor") or not hasattr(model, "feature_processor"):
        raise RuntimeError(
            "Model does not expose extractor/feature_processor; cannot compute processor embeddings."
        )
    features = model.extractor.extract_features(waveforms)
    emb = model.feature_processor(features)
    return emb


def _pick_embedding_function(
    model,
    mode: str,
    probe_waveforms: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    if mode == "processor":
        return lambda wf: _processor_embedding(model, wf), "processor"

    if mode == "model":
        if not hasattr(model, "extract_embedding"):
            raise RuntimeError("Requested --embedding-mode model, but model has no extract_embedding().")
        fn = lambda wf: model.extract_embedding(wf)
        try:
            candidate = fn(probe_waveforms[:1])
        except Exception as exc:
            raise RuntimeError(
                "Requested --embedding-mode model, but model.extract_embedding() failed on waveform input. "
                "Use --embedding-mode processor or a model that supports waveform-based extract_embedding()."
            ) from exc
        if not isinstance(candidate, torch.Tensor) or candidate.shape[0] != 1:
            raise RuntimeError(
                "Requested --embedding-mode model, but model.extract_embedding() did not return "
                "a batch-aligned tensor."
            )
        return fn, "model.extract_embedding"

    if hasattr(model, "extract_embedding"):
        try:
            candidate = model.extract_embedding(probe_waveforms[:1])
            if isinstance(candidate, torch.Tensor) and candidate.shape[0] == 1:
                return lambda wf: model.extract_embedding(wf), "model.extract_embedding"
        except Exception as exc:
            print(
                "[warn] model.extract_embedding probe failed; "
                f"falling back to processor embeddings. Reason: {exc}"
            )

    return lambda wf: _processor_embedding(model, wf), "processor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe embedding space quality on a dev protocol.")
    runtime_group = parser.add_mutually_exclusive_group(required=False)
    runtime_group.add_argument("--local", action="store_true", help="Use local dataset config (default).")
    runtime_group.add_argument("--sge", action="store_true", help="Use SGE dataset config.")
    runtime_group.add_argument("--karolina", action="store_true", help="Use Karolina dataset config.")

    parser.add_argument(
        "--dataset-config-key",
        default="mlaad_single",
        help="Dataset config key used to resolve dev protocol/data root (default: mlaad_single).",
    )
    parser.add_argument(
        "--dev-protocol",
        type=str,
        default=None,
        help="Optional dev protocol override (absolute or relative to resolved data root).",
    )
    parser.add_argument("--out-path", type=Path, required=True, help="Output JSON path for probe results.")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional data root override used to resolve relative dev protocol and audio paths.",
    )
    parser.add_argument("--path-column", default="path", help="Audio path column name.")
    parser.add_argument("--label-column", default="model_name", help="Class label column name.")
    parser.add_argument(
        "--exclude-classes",
        default="bonafide,real",
        help="Comma-separated classes to drop before probing.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=300,
        help="Maximum samples to keep per class (<=0 disables cap).",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=2,
        help="Drop classes with fewer samples than this threshold.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Train/test split fraction for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument(
        "--extractor",
        default="XLSR_300M",
        help="Extractor name (default: XLSR_300M). Override to match your checkpoint.",
    )
    parser.add_argument(
        "--processor",
        default="MHFA",
        help="Processor name (default: MHFA). Override to match your checkpoint.",
    )
    parser.add_argument(
        "--classifier",
        default="FFCosine",
        help="Classifier name (default: FFCosine). Override to match your checkpoint.",
    )
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=None,
        help="Bottleneck dim for FFMulticlass/FFCosine3-style checkpoints.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes for FF/FFMulticlass-style checkpoints.",
    )
    parser.add_argument("--kernel", type=str, default=None, help="Kernel argument for SVMDiff checkpoints.")
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="n_components for GMM checkpoints.",
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default=None,
        help="covariance_type for GMM checkpoints.",
    )

    parser.add_argument(
        "--embedding-mode",
        choices=["auto", "processor", "model"],
        default="auto",
        help="Embedding source: processor output, model.extract_embedding, or automatic fallback.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding extraction batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Optional fixed segment length (seconds), cropped/padded from the start.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate used only with --segment-seconds.",
    )
    parser.add_argument(
        "--amp-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable AMP during embedding extraction.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="AMP dtype when --amp-eval is enabled.",
    )
    parser.add_argument("--device", default=None, help="Force device (e.g., cuda, cuda:0, cpu).")
    parser.add_argument(
        "--lr-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression.",
    )
    parser.add_argument(
        "--save-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save extracted embeddings next to out-path as NPZ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    protocol_path, resolved_data_root, config_name = _resolve_dev_protocol(args)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    print(f"Resolved dev protocol from config '{config_name}': {protocol_path}")

    protocol_df = pd.read_csv(protocol_path)
    filtered_df, filter_stats = _filter_protocol_rows(
        protocol_df=protocol_df,
        path_column=args.path_column,
        label_column=args.label_column,
        exclude_classes=_csv_list(args.exclude_classes),
        max_per_class=args.max_per_class,
        min_per_class=args.min_per_class,
        seed=args.seed,
    )
    if filtered_df.empty:
        raise ValueError("Filtering produced an empty protocol.")

    class_counts = filtered_df[args.label_column].value_counts().sort_index()
    if len(class_counts) < 2:
        raise ValueError(
            f"Need at least two classes after filtering; got {len(class_counts)}."
        )

    dataset = FilteredSingleProtocolDataset(
        rows=filtered_df,
        path_column=args.path_column,
        label_column=args.label_column,
        protocol_dir=protocol_path.parent,
        data_root=resolved_data_root,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
    )

    resolved_device = _resolve_device(args.device)
    amp_dtype = _amp_dtype(args.amp_dtype)
    use_amp = bool(args.amp_eval and resolved_device.startswith("cuda"))

    model_args = SimpleNamespace(
        extractor=args.extractor,
        processor=args.processor,
        classifier=args.classifier,
        bottleneck_dim=args.bottleneck_dim,
        num_classes=args.num_classes,
        kernel=args.kernel,
        n_components=args.n_components,
        covariance_type=args.covariance_type,
    )

    model, trainer = build_model(model_args)
    trainer.load_model(str(args.checkpoint))
    model.eval()
    model.to(resolved_device)

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

    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    embedding_mode_used = "unknown"

    print(f"Extracting embeddings from {len(dataset):,} filtered samples...")
    with torch.no_grad():
        for file_names, waveforms, label_batch in tqdm(dataloader, desc="Embedding extraction"):
            waveforms = waveforms.to(resolved_device)
            label_batch = label_batch.to(dtype=torch.long)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            )
            with autocast_context:
                if embedding_fn is None:
                    embedding_fn, embedding_mode_used = _pick_embedding_function(
                        model=model,
                        mode=args.embedding_mode,
                        probe_waveforms=waveforms,
                    )
                    print(f"Using embedding mode: {embedding_mode_used}")
                emb = embedding_fn(waveforms)

            emb = emb.detach().to(dtype=torch.float32).cpu()
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            elif emb.ndim > 2:
                emb = emb.flatten(start_dim=1)

            embeddings.append(emb.numpy())
            labels.append(label_batch.cpu().numpy().astype(np.int64))
            paths.extend([str(name) for name in file_names])

    if not embeddings:
        raise RuntimeError("No embeddings extracted.")

    X = np.vstack(embeddings)
    y = np.concatenate(labels)
    if X.shape[0] != len(paths):
        raise RuntimeError("Embedding/path count mismatch.")

    class_counts_post = pd.Series(y).value_counts().sort_index()
    if class_counts_post.min() < 2:
        raise ValueError(
            "At least one class has <2 samples after filtering; stratified split cannot proceed."
        )

    print(f"Training linear probe on embeddings: N={X.shape[0]}, D={X.shape[1]}")
    try:
        X_train, X_test, y_train, y_test, _, paths_test = train_test_split(
            X,
            y,
            np.asarray(paths),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
    except ValueError as exc:
        raise ValueError(
            f"train_test_split failed with stratify=y. "
            f"Check class counts and test_size (current test_size={args.test_size})."
        ) from exc

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=args.lr_max_iter,
        multi_class="multinomial",
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    id_to_label = {idx: label for label, idx in dataset.label_map.items()}
    predicted_labels = [id_to_label[int(v)] for v in y_pred.tolist()]
    target_labels = [id_to_label[int(v)] for v in y_test.tolist()]

    out_path = args.out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "protocol_path": str(protocol_path),
        "config_profile": config_name,
        "dataset_config_key": args.dataset_config_key,
        "dev_protocol_override": args.dev_protocol,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "data_root": str(resolved_data_root),
        "path_column": args.path_column,
        "label_column": args.label_column,
        "embedding_mode_requested": args.embedding_mode,
        "embedding_mode_used": embedding_mode_used,
        "embedding_dim": int(X.shape[1]),
        "filtering": {
            **filter_stats,
            "class_counts": {str(k): int(v) for k, v in class_counts.to_dict().items()},
        },
        "split": {
            "test_size": float(args.test_size),
            "train_size": int(X_train.shape[0]),
            "test_size_abs": int(X_test.shape[0]),
        },
        "probe": {
            "classifier": "LogisticRegression(lbfgs,multinomial)",
            "max_iter": int(args.lr_max_iter),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
        },
    }

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "path": paths_test.tolist(),
            "target_id": y_test.tolist(),
            "pred_id": y_pred.tolist(),
            "target_label": target_labels,
            "pred_label": predicted_labels,
        }
    )
    pred_path = out_path.with_suffix(".predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    if args.save_embeddings:
        emb_out = out_path.with_suffix(".embeddings.npz")
        np.savez_compressed(
            emb_out,
            embeddings=X.astype(np.float32),
            labels=y.astype(np.int64),
            utt_ids=np.asarray(paths),
        )
        print(f"Saved embeddings to {emb_out}")

    print("------------------------------------------------")
    print(f"FINAL PROBING ACCURACY: {acc * 100:.2f}%")
    print(f"Balanced accuracy: {bal_acc * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print("------------------------------------------------")
    print(f"Wrote report: {out_path}")
    print(f"Wrote predictions: {pred_path}")


if __name__ == "__main__":
    main()
