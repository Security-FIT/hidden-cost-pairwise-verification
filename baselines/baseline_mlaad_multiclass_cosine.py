#!/usr/bin/env python3
"""
MLAAD source verification baseline.

Train a multiclass classifier C for source attribution, then use its last
dense-layer embeddings for cosine verification. Supports:
  - Reference-set verification (R references per generator, max/mean cosine).
  - Pairwise verification (single reference, cosine between two tracks).
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import DetCurveDisplay, det_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import DATASETS, build_model
from config import karolina_config, local_config, sge_config
from datasets.utils import custom_single_batch_create
from trainers.utils import calculate_EER


@dataclass
class Trial:
    test_path: str
    ref_paths: list[str]
    label: int
    ref_model: str
    test_model: str


class PathDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        paths: list[str],
        segment_samples: int | None,
        skip_bad_paths: bool = False,
    ):
        self.root_dir = root_dir
        self.paths = paths
        self.segment_samples = segment_samples
        if skip_bad_paths:
            self.paths = self._filter_bad_paths(self.paths)

    def _filter_bad_paths(self, paths: list[str]) -> list[str]:
        good_paths: list[str] = []
        skipped = 0
        for rel_path in paths:
            abs_path = self._resolve_path(rel_path)
            try:
                sf.info(abs_path)
            except Exception:
                skipped += 1
                continue
            good_paths.append(rel_path)
        if skipped:
            print(f"Warning: skipped {skipped} unreadable paths during embedding.")
        return good_paths

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, rel_path: str) -> str:
        normalized = rel_path.lstrip("./")
        return str(Path(self.root_dir) / normalized)

    def _load_waveform(self, abs_path: str) -> torch.Tensor:
        try:
            waveform, _ = sf.read(abs_path, dtype="float32")
        except Exception as exc:
            raise RuntimeError(f"Failed to read audio: {abs_path}") from exc
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if self.segment_samples:
            if waveform.shape[0] >= self.segment_samples:
                waveform = waveform[: self.segment_samples]
            else:
                pad_width = self.segment_samples - waveform.shape[0]
                waveform = np.pad(waveform, (0, pad_width), mode="constant")
        waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        path = self.paths[idx]
        wav = self._load_waveform(self._resolve_path(path))
        return path, wav, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLAAD source verification with multiclass embeddings + cosine scoring."
    )

    env = parser.add_mutually_exclusive_group()
    env.add_argument("--karolina", action="store_true", help="Use Karolina paths/config.")
    env.add_argument("--metacentrum", action="store_true", help="Use MetaCentrum paths/config.")
    env.add_argument("--sge", action="store_true", help="Use SGE paths/config.")
    env.add_argument("--local", action="store_true", help="Use local paths/config (default).")

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Dataset name from common.DATASETS (MLAAD pair dataset for defaults).",
    )
    parser.add_argument("-e", "--extractor", required=True, help="Extractor name (e.g., XLSR_300M).")
    parser.add_argument(
        "-p",
        "--processor",
        required=True,
        help="Pooling/processor name (e.g., MHFA, AASIST, Mean, SLS).",
    )
    parser.add_argument(
        "-c",
        "--classifier",
        required=True,
        help="Classifier name used during training (expected: FF).",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the trained checkpoint.")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of source-model classes (defaults to protocol-derived).",
    )
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=None,
        help="Bottleneck dimension for FFMulticlass/FFCosine3 (match the checkpoint).",
    )

    protocol_group = parser.add_argument_group("Protocol selection")
    protocol_group.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help="Pairwise protocol (path_A,path_B,same_model) for cosine scoring.",
    )
    protocol_group.add_argument(
        "--reference-protocol",
        type=Path,
        default=None,
        help="Single-utterance reference protocol (path,model_name).",
    )
    protocol_group.add_argument(
        "--test-protocol",
        type=Path,
        default=None,
        help="Single-utterance test protocol (path,model_name).",
    )
    protocol_group.add_argument(
        "--reference-path-column",
        default="path",
        help="Column containing audio paths in reference protocol.",
    )
    protocol_group.add_argument(
        "--reference-group-column",
        default="model_name",
        help="Column containing reference group ID (e.g., model_name or claim_id).",
    )
    protocol_group.add_argument(
        "--test-path-column",
        default="path",
        help="Column containing audio paths in test protocol.",
    )
    protocol_group.add_argument(
        "--test-group-column",
        default="model_name",
        help="Column containing test group ID (e.g., model_name).",
    )
    protocol_group.add_argument(
        "--test-label-column",
        default="label",
        help="Column containing target/non-target labels in test protocol.",
    )
    protocol_group.add_argument(
        "--test-claim-column",
        default="claim_id",
        help="Column containing reference claim IDs in test protocol.",
    )
    protocol_group.add_argument(
        "--target-label",
        type=int,
        default=1,
        help="Label value for target trials (default: 1).",
    )
    protocol_group.add_argument(
        "--reference-size",
        type=int,
        default=5,
        help="Number of reference utterances per generator (default: 5).",
    )
    protocol_group.add_argument(
        "--negatives-per-test",
        type=int,
        default=1,
        help="Number of non-target trials per test sample (default: 1).",
    )
    protocol_group.add_argument(
        "--score-reduction",
        choices=("max", "mean"),
        default="max",
        help="Reduction over reference-set cosines (default: max).",
    )
    protocol_group.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    parser.add_argument("--data-root", type=Path, default=None, help="Override data root.")
    parser.add_argument("--batch-size", type=int, default=12, help="Eval batch size (default: 12).")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers (default: 8).")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor (default: 4).")
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable pin_memory (defaults to config).",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable persistent workers (defaults to config).",
    )
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., cuda:0, cpu).")
    parser.add_argument(
        "--amp-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.autocast during inference (default: on).",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="Autocast dtype when AMP is enabled (default: bf16).",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=4.0,
        help="Segment length in seconds (default: 4.0; set 0 to disable).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for segmenting (default: 16000).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to dump scores (trial_id, score, label).",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=None,
        help="Optional .npz cache for embeddings (embeddings, utt_ids).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to write metrics/plots (defaults to output-csv directory).",
    )
    parser.add_argument(
        "--plot-det",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot DET curve.",
    )
    parser.add_argument(
        "--plot-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot score distributions.",
    )
    parser.add_argument(
        "--p-target",
        type=float,
        default=0.5,
        help="Target prior for DCF/tDCF (default: 0.5).",
    )
    parser.add_argument(
        "--c-miss",
        type=float,
        default=1.0,
        help="Miss cost for DCF/tDCF (default: 1).",
    )
    parser.add_argument(
        "--c-fa",
        type=float,
        default=1.0,
        help="False-alarm cost for DCF/tDCF (default: 1).",
    )
    parser.add_argument(
        "--fixed-fprs",
        nargs="+",
        type=float,
        default=[0.0001, 0.001, 0.01, 0.05],
        help="Fixed FPR points for TPR@FPR reporting.",
    )
    parser.add_argument(
        "--skip-bad-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip paths that fail to load with soundfile.",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "eval", "both"),
        default="both",
        help="Which split(s) to score for pairwise defaults (default: both).",
    )

    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> dict:
    if args.karolina:
        return karolina_config
    if args.sge:
        return sge_config
    return local_config


def _resolve_dataset(dataset: str, config: dict) -> tuple[type, dict]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if "MLAAD" not in dataset:
        raise ValueError("This baseline expects an MLAAD dataset.")

    if "HardminedFFCosine" in dataset:
        dataset_config = config["mlaad_hardmined_ffcosine"]
    elif "HardminedFFConcat" in dataset:
        dataset_config = config["mlaad_hardmined_ffconcat"]
    elif "Minimal" in dataset:
        dataset_config = config["mlaad_minimal"]
    elif "Intermediate" in dataset:
        dataset_config = config["mlaad_intermediate"]
    elif "CuratedBalanced" in dataset:
        dataset_config = config["mlaad_curated_balanced"]
    elif "Curated" in dataset:
        dataset_config = config["mlaad_curated"]
    else:
        dataset_config = config["mlaad"]

    return DATASETS[dataset], dataset_config


def _infer_num_classes_from_protocol(protocol_path: Path, source_column: str) -> int:
    df = pd.read_csv(protocol_path)
    if source_column not in df.columns:
        raise ValueError(f"Protocol missing {source_column}: {protocol_path}")
    return int(df[source_column].dropna().astype(str).nunique())


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict | None:
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return None
    if isinstance(state, dict):
        nested_state = state.get("state_dict")
        if isinstance(nested_state, dict):
            return nested_state
        nested_state = state.get("model_state_dict")
        if isinstance(nested_state, dict):
            return nested_state
        return state
    return None


def _infer_num_classes_from_state_dict(state: dict | None) -> int | None:
    if not isinstance(state, dict):
        return None
    weight = state.get("classifier.6.weight")
    if weight is not None and hasattr(weight, "shape"):
        return int(weight.shape[0])
    bias = state.get("classifier.6.bias")
    if bias is not None and hasattr(bias, "shape"):
        return int(bias.shape[0])
    return None


def _infer_ff_embedding_dim_from_state_dict(state: dict | None) -> int | None:
    if not isinstance(state, dict):
        return None
    # FFBase second Linear: classifier.3 (in_dim//2 -> embedding_dim)
    layer2_weight = state.get("classifier.3.weight")
    if layer2_weight is not None and hasattr(layer2_weight, "shape"):
        return int(layer2_weight.shape[0])
    # Fallback: final Linear: classifier.6 (embedding_dim -> num_classes)
    head_weight = state.get("classifier.6.weight")
    if head_weight is not None and hasattr(head_weight, "shape"):
        return int(head_weight.shape[1])
    return None


def _amp_context(args, device: torch.device):
    if not args.amp_eval or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _embed_paths(
    model,
    device: torch.device,
    amp_ctx,
    data_root: str,
    paths: list[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    pin_memory: bool,
    persistent_workers: bool,
    segment_samples: int | None,
    skip_bad_paths: bool,
) -> dict[str, np.ndarray]:
    if not paths:
        return {}
    dataset = PathDataset(data_root, paths, segment_samples, skip_bad_paths=skip_bad_paths)
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": custom_single_batch_create,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    embeddings: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for file_names, waveforms, _ in tqdm(dataloader, desc="Embedding"):
            waveforms = waveforms.to(device)
            with amp_ctx:
                emb = model.extract_embedding(waveforms)
            emb = emb.detach().to(dtype=torch.float32).cpu().numpy()
            for name, vec in zip(file_names, emb):
                embeddings[name] = vec
    return embeddings


def _load_embedding_cache(cache_path: Path) -> dict[str, np.ndarray] | None:
    if cache_path is None or not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=False)
    utt_ids = data.get("utt_ids")
    embeddings = data.get("embeddings")
    if utt_ids is None or embeddings is None:
        raise ValueError(f"Embedding cache missing required arrays: {cache_path}")
    if len(utt_ids) != len(embeddings):
        raise ValueError(f"Embedding cache length mismatch: {cache_path}")
    return {str(k): embeddings[i] for i, k in enumerate(utt_ids)}


def _save_embedding_cache(cache_path: Path, embeddings: dict[str, np.ndarray]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    utt_ids = np.array(sorted(embeddings))
    emb_matrix = np.stack([embeddings[k] for k in utt_ids], axis=0)
    np.savez_compressed(cache_path, embeddings=emb_matrix, utt_ids=utt_ids)


def _reduce_scores(scores: np.ndarray, mode: str) -> float:
    if scores.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(scores))
    return float(np.max(scores))


def _score_trials(
    embeddings: dict[str, np.ndarray],
    trials: list[Trial],
    reduction: str,
    output_csv: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    rows = []
    for trial in tqdm(trials, desc="Scoring trials"):
        test_emb = embeddings[trial.test_path]
        ref_embs = np.stack([embeddings[p] for p in trial.ref_paths], axis=0)
        test_norm = test_emb / (np.linalg.norm(test_emb) + 1e-8)
        ref_norm = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)
        cosines = np.dot(ref_norm, test_norm)
        score = _reduce_scores(cosines, reduction)
        best_ref_idx = int(np.argmax(cosines)) if cosines.size else 0
        best_ref_path = trial.ref_paths[best_ref_idx] if trial.ref_paths else ""
        scores.append(score)
        labels.append(trial.label)
        rows.append(
            {
                "trial_id": f"{trial.test_path}|{trial.ref_model}",
                "pathA": best_ref_path,
                "pathB": trial.test_path,
                "path_A": best_ref_path,
                "path_B": trial.test_path,
                "score": score,
                "label": trial.label,
                "claim_id": trial.ref_model,
                "query_id": trial.test_model,
                "test_model": trial.test_model,
                "ref_model": trial.ref_model,
            }
        )

    scores_arr = np.asarray(scores, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)

    return scores_arr, labels_arr


def _build_reference_trials(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reference_path_col: str,
    reference_group_col: str,
    test_path_col: str,
    test_group_col: str,
    test_label_col: str,
    test_claim_col: str,
    reference_size: int,
    negatives_per_test: int,
    seed: int,
) -> tuple[list[Trial], dict[str, list[str]]]:
    rng = np.random.default_rng(seed)

    if reference_path_col not in reference_df.columns:
        raise ValueError(
            f"Reference protocol missing '{reference_path_col}'. Columns: {', '.join(reference_df.columns)}"
        )
    if reference_group_col not in reference_df.columns:
        raise ValueError(
            f"Reference protocol missing '{reference_group_col}'. Columns: {', '.join(reference_df.columns)}"
        )
    if test_path_col not in test_df.columns:
        raise ValueError(
            f"Test protocol missing '{test_path_col}'. Columns: {', '.join(test_df.columns)}"
        )

    reference_sets: dict[str, list[str]] = {}
    for model_name, group in reference_df.groupby(reference_group_col):
        paths = group[reference_path_col].dropna().astype(str).tolist()
        if len(paths) < reference_size:
            continue
        rng.shuffle(paths)
        reference_sets[str(model_name)] = paths[:reference_size]

    available_models = sorted(reference_sets)
    if not available_models:
        raise ValueError("No reference sets available; check protocol and reference_size.")

    trials: list[Trial] = []
    has_explicit_trials = test_label_col in test_df.columns and test_claim_col in test_df.columns
    if has_explicit_trials:
        for _, row in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Building trials", leave=False
        ):
            claim_id = str(row[test_claim_col])
            if claim_id not in reference_sets:
                continue
            test_path = str(row[test_path_col])
            test_model = str(row[test_group_col]) if test_group_col in test_df.columns else "unknown"
            label = int(row[test_label_col])
            trials.append(
                Trial(
                    test_path=test_path,
                    ref_paths=reference_sets[claim_id],
                    label=label,
                    ref_model=claim_id,
                    test_model=test_model,
                )
            )
    else:
        for _, row in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Building trials", leave=False
        ):
            test_path = str(row[test_path_col])
            test_model = str(row[test_group_col])
            if test_model not in reference_sets:
                continue
            trials.append(
                Trial(
                    test_path=test_path,
                    ref_paths=reference_sets[test_model],
                    label=1,
                    ref_model=test_model,
                    test_model=test_model,
                )
            )
            neg_models = [m for m in available_models if m != test_model]
            if not neg_models:
                continue
            for _ in range(negatives_per_test):
                ref_model = rng.choice(neg_models)
                trials.append(
                    Trial(
                        test_path=test_path,
                        ref_paths=reference_sets[ref_model],
                        label=0,
                        ref_model=ref_model,
                        test_model=test_model,
                    )
                )

    if not trials:
        raise ValueError("No trials generated; check protocol overlap and reference size.")
    return trials, reference_sets


def _resolve_column(name: str, columns: list[str], candidates: list[str]) -> str:
    if name in columns:
        return name
    for cand in candidates:
        if cand in columns:
            print(f"Warning: '{name}' not found, using '{cand}' instead.")
            return cand
    return name


def _score_pairwise(
    embeddings: dict[str, np.ndarray],
    pairs_df: pd.DataFrame,
    output_csv: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    required = ["path_A", "path_B", "same_model"]
    missing = [col for col in required if col not in pairs_df.columns]
    if missing:
        raise ValueError(f"Pairwise protocol missing columns: {', '.join(missing)}")

    scores = []
    labels = []
    rows = []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Scoring pairs"):
        path_a = str(row["path_A"])
        path_b = str(row["path_B"])
        label = int(row["same_model"])
        emb_a = embeddings[path_a]
        emb_b = embeddings[path_b]
        emb_a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
        emb_b = emb_b / (np.linalg.norm(emb_b) + 1e-8)
        score = float(np.dot(emb_a, emb_b))
        scores.append(score)
        labels.append(label)
        claim_id = None
        query_id = None
        if "claim_id" in pairs_df.columns:
            claim_id = row.get("claim_id")
        elif "model_name_A" in pairs_df.columns:
            claim_id = row.get("model_name_A")
        if "query_id" in pairs_df.columns:
            query_id = row.get("query_id")
        elif "model_name_B" in pairs_df.columns:
            query_id = row.get("model_name_B")
        out_row = {
            "trial_id": f"{path_a}|{path_b}",
            "pathA": path_a,
            "pathB": path_b,
            "path_A": path_a,
            "path_B": path_b,
            "score": score,
            "label": label,
            "same_model": label,
        }
        if claim_id is not None:
            out_row["claim_id"] = str(claim_id)
        if query_id is not None:
            out_row["query_id"] = str(query_id)
        for col in ("model_name_A", "model_name_B"):
            if col in pairs_df.columns:
                out_row[col] = str(row.get(col))
        rows.append(out_row)

    scores_arr = np.asarray(scores, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    return scores_arr, labels_arr


def _report_metrics(scores: np.ndarray, labels: np.ndarray, label: str, target_label: int):
    labels_arr = np.asarray(labels)
    if target_label != 0:
        labels_arr = (labels_arr != target_label).astype(int)
    eer = calculate_EER("MLAAD", labels_arr, scores, False, det_subtitle=label)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    print(f"[{label}] EER: {eer * 100:.2f}%  AUC: {auc * 100:.2f}%")
    return {
        "eer": float(eer),
        "auc": float(auc),
    }


def _dcf_best_cost(p_target: float, c_miss: float, c_fa: float) -> float:
    return float(min(c_miss * p_target, c_fa * (1.0 - p_target)))


def _dcf_norm(dcf: float, p_target: float, c_miss: float, c_fa: float) -> float:
    best = _dcf_best_cost(p_target, c_miss, c_fa)
    if best <= 0 or math.isnan(dcf):
        return float("nan")
    return float(dcf / best)


def _det_metrics(scores: np.ndarray, labels: np.ndarray, p_target: float, c_miss: float, c_fa: float):
    fpr, fnr, thresholds = det_curve(labels, scores, pos_label=1)
    dcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    best_idx = int(np.nanargmin(dcf))
    min_dcf = float(dcf[best_idx])
    min_thr = float(thresholds[best_idx])
    return {
        "fpr": fpr,
        "fnr": fnr,
        "thresholds": thresholds,
        "min_dcf": min_dcf,
        "min_dcf_threshold": min_thr,
        "min_dcf_norm": _dcf_norm(min_dcf, p_target, c_miss, c_fa),
    }


def _tpr_at_fpr(fpr: np.ndarray, fnr: np.ndarray, points: list[float]) -> dict[str, float]:
    tpr = 1.0 - fnr
    results = {}
    for target_fpr in points:
        mask = fpr <= target_fpr
        if not np.any(mask):
            results[str(target_fpr)] = float("nan")
        else:
            results[str(target_fpr)] = float(np.max(tpr[mask]))
    return results


def _plot_det(fpr: np.ndarray, fnr: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    disp = DetCurveDisplay(fpr=fpr, fnr=fnr, pos_label=1)
    disp.plot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)


def _plot_score_dist(scores: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    target = scores[labels == 1]
    non = scores[labels == 0]
    plt.figure(figsize=(8, 5))
    plt.hist(target, bins=60, alpha=0.5, label="target", density=True)
    plt.hist(non, bins=60, alpha=0.5, label="non-target", density=True)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("Score distributions")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)


def _resolve_pin_persist(args: argparse.Namespace, config: dict) -> tuple[bool, bool]:
    pin_memory = args.pin_memory if args.pin_memory is not None else config.get(
        "pin_memory", torch.cuda.is_available()
    )
    persistent_workers = args.persistent_workers if args.persistent_workers is not None else config.get(
        "persistent_workers", False
    )
    return pin_memory and torch.cuda.is_available(), persistent_workers


def main() -> None:
    args = parse_args()
    config = _resolve_config(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_state = _load_checkpoint_state_dict(args.checkpoint)

    dataset_class, dataset_config = _resolve_dataset(args.dataset, config)
    data_root = str(args.data_root or (Path(config["data_dir"]) / dataset_config["train_subdir"]))

    if args.num_classes is None:
        inferred = _infer_num_classes_from_state_dict(checkpoint_state)
        if inferred is not None:
            args.num_classes = inferred
            print(f"Inferred num_classes={args.num_classes} from checkpoint {args.checkpoint}.")
        else:
            default_protocol = args.reference_protocol or args.test_protocol
            if default_protocol is None:
                default_protocol = (
                    Path(config["data_dir"])
                    / dataset_config["train_subdir"]
                    / dataset_config["train_protocol"]
                )
            args.num_classes = _infer_num_classes_from_protocol(
                default_protocol, args.reference_group_column
            )
            print(f"Inferred num_classes={args.num_classes} from {default_protocol}.")

    embedding_dim = None
    if args.classifier == "FF":
        embedding_dim = _infer_ff_embedding_dim_from_state_dict(checkpoint_state)
        if embedding_dim is not None:
            print(f"Inferred embedding_dim={embedding_dim} from checkpoint {args.checkpoint}.")
        else:
            print(
                "[warning] Could not infer embedding_dim from checkpoint; "
                "falling back to classifier default."
            )

    args_ns = SimpleNamespace(
        extractor=args.extractor,
        processor=args.processor,
        classifier=args.classifier,
        kernel=None,
        n_components=None,
        covariance_type=None,
        num_classes=args.num_classes,
        embedding_dim=embedding_dim,
        bottleneck_dim=args.bottleneck_dim,
    )
    model, trainer = build_model(args_ns)
    trainer.load_model(str(args.checkpoint))
    model.to(device)
    if not hasattr(model, "extract_embedding"):
        raise RuntimeError("Model does not expose extract_embedding; expected FF multiclass model.")

    amp_ctx = _amp_context(args, device)
    pin_memory, persistent_workers = _resolve_pin_persist(args, config)
    segment_samples = None
    if args.segment_seconds and args.segment_seconds > 0:
        segment_samples = int(math.floor(args.segment_seconds * args.sample_rate))

    cached_embeddings = None
    if args.embeddings_cache is not None:
        cached_embeddings = _load_embedding_cache(args.embeddings_cache)

    if args.pairs_csv is not None:
        pairs_df = pd.read_csv(args.pairs_csv)
        paths = sorted(
            set(
                pairs_df["path_A"].dropna().astype(str).tolist()
                + pairs_df["path_B"].dropna().astype(str).tolist()
            )
        )
        embeddings = cached_embeddings or {}
        missing = [p for p in paths if p not in embeddings]
        if missing:
            new_embs = _embed_paths(
                model,
                device,
                amp_ctx,
                data_root,
                missing,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                segment_samples=segment_samples,
                skip_bad_paths=args.skip_bad_paths,
            )
            embeddings.update(new_embs)
            if args.embeddings_cache is not None:
                _save_embedding_cache(args.embeddings_cache, embeddings)
        if args.embeddings_cache is not None:
            cached_embeddings = embeddings
        scores, labels = _score_pairwise(embeddings, pairs_df, args.output_csv)
        metrics = _report_metrics(scores, labels, "pairwise", args.target_label)
        det = _det_metrics(
            scores,
            (np.asarray(labels) == args.target_label).astype(int),
            args.p_target,
            args.c_miss,
            args.c_fa,
        )
        metrics.update(
            {
                "min_dcf": det["min_dcf"],
                "min_dcf_threshold": det["min_dcf_threshold"],
                "min_dcf_norm": det["min_dcf_norm"],
                "tpr_at_fpr": _tpr_at_fpr(det["fpr"], det["fnr"], args.fixed_fprs),
            }
        )
        report_dir = args.report_dir or (args.output_csv.parent if args.output_csv else Path("."))
        report_dir.mkdir(parents=True, exist_ok=True)
        if args.plot_det:
            _plot_det(det["fpr"], det["fnr"], report_dir / "pairwise_det.png")
        if args.plot_scores:
            _plot_score_dist(
                scores,
                (np.asarray(labels) == args.target_label).astype(int),
                report_dir / "pairwise_scores.png",
            )
        with open(report_dir / "pairwise_metrics.json", "w") as handle:
            json.dump(metrics, handle, indent=2)
        return

    explicit_protocols = args.reference_protocol is not None or args.test_protocol is not None
    splits = ("protocol",) if explicit_protocols else (("dev", "eval") if args.split == "both" else (args.split,))
    for split in splits:
        reference_protocol = args.reference_protocol
        test_protocol = args.test_protocol
        if test_protocol is None:
            test_protocol = (
                Path(config["data_dir"])
                / dataset_config[f"{split}_subdir"]
                / dataset_config[f"{split}_protocol"]
            )
        if reference_protocol is None:
            reference_protocol = test_protocol

        reference_df = pd.read_csv(reference_protocol)
        test_df = pd.read_csv(test_protocol)
        reference_path_col = _resolve_column(
            args.reference_path_column,
            list(reference_df.columns),
            ["path", "ref_path", "utt_path"],
        )
        reference_group_col = _resolve_column(
            args.reference_group_column,
            list(reference_df.columns),
            ["claim_id", "model_name", "model_id"],
        )
        test_path_col = _resolve_column(
            args.test_path_column,
            list(test_df.columns),
            ["query_path", "path", "utt_path"],
        )
        test_group_col = _resolve_column(
            args.test_group_column,
            list(test_df.columns),
            ["query_model_id", "model_name", "model_id"],
        )
        test_label_col = _resolve_column(
            args.test_label_column,
            list(test_df.columns),
            ["label", "same_model", "target"],
        )
        test_claim_col = _resolve_column(
            args.test_claim_column,
            list(test_df.columns),
            ["claim_id", "ref_id", "reference_id"],
        )
        trials, reference_sets = _build_reference_trials(
            reference_df,
            test_df,
            reference_path_col,
            reference_group_col,
            test_path_col,
            test_group_col,
            test_label_col,
            test_claim_col,
            args.reference_size,
            args.negatives_per_test,
            args.seed,
        )
        unique_paths = sorted(
            set([t.test_path for t in trials] + [p for refs in reference_sets.values() for p in refs])
        )
        embeddings = cached_embeddings or {}
        missing = [p for p in unique_paths if p not in embeddings]
        if missing:
            new_embs = _embed_paths(
                model,
                device,
                amp_ctx,
                data_root,
                missing,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                segment_samples=segment_samples,
                skip_bad_paths=args.skip_bad_paths,
            )
            embeddings.update(new_embs)
            if args.embeddings_cache is not None:
                _save_embedding_cache(args.embeddings_cache, embeddings)
                cached_embeddings = embeddings
        output_csv = None
        if args.output_csv is not None:
            suffix = f"{split}_ref{args.reference_size}"
            output_csv = args.output_csv.with_name(f"{args.output_csv.stem}_{suffix}{args.output_csv.suffix}")
        scores, labels = _score_trials(embeddings, trials, args.score_reduction, output_csv)
        metrics = _report_metrics(scores, labels, split, args.target_label)
        det = _det_metrics(
            scores,
            (np.asarray(labels) == args.target_label).astype(int),
            args.p_target,
            args.c_miss,
            args.c_fa,
        )
        metrics.update(
            {
                "min_dcf": det["min_dcf"],
                "min_dcf_threshold": det["min_dcf_threshold"],
                "min_dcf_norm": det["min_dcf_norm"],
                "tpr_at_fpr": _tpr_at_fpr(det["fpr"], det["fnr"], args.fixed_fprs),
            }
        )
        report_dir = args.report_dir or (args.output_csv.parent if args.output_csv else Path("."))
        report_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{split}"
        if args.plot_det:
            _plot_det(det["fpr"], det["fnr"], report_dir / f"{suffix}_det.png")
        if args.plot_scores:
            _plot_score_dist(
                scores,
                (np.asarray(labels) == args.target_label).astype(int),
                report_dir / f"{suffix}_scores.png",
            )
        with open(report_dir / f"{suffix}_metrics.json", "w") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
