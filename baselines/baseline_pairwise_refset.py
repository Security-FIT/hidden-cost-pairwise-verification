#!/usr/bin/env python3
"""
Reference-set evaluation using pairwise classifiers.

For each trial (claim g, query x), score against R references for g using a pairwise
model (FFConcat/FFDiff/FFAttn/FFCosine/etc), then aggregate scores per trial.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import DetCurveDisplay, det_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import build_model
from config import karolina_config, local_config, sge_config
from datasets.utils import custom_single_batch_create


@dataclass
class TrialSpec:
    trial_id: str
    claim_id: str
    query_path: str
    query_model_id: str
    label: int


class PairwiseRefsetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        trials: list[TrialSpec],
        reference_sets: dict[str, list[str]],
        reference_size: int,
        segment_samples: int | None,
        skip_bad_paths: bool,
    ):
        self.root_dir = root_dir
        self.trials = trials
        self.reference_sets = reference_sets
        self.reference_size = reference_size
        self.segment_samples = segment_samples
        self.trial_count = len(self.trials)
        self.total_pairs = self.trial_count * self.reference_size
        if skip_bad_paths:
            self._filter_bad_paths()

    def __len__(self) -> int:
        return self.total_pairs

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

    def _filter_bad_paths(self) -> None:
        good_trials: list[TrialSpec] = []
        skipped = 0
        for trial in self.trials:
            refs = self.reference_sets.get(trial.claim_id, [])
            if len(refs) < self.reference_size:
                skipped += 1
                continue
            query_path = self._resolve_path(trial.query_path)
            try:
                sf.info(query_path)
            except Exception:
                skipped += 1
                continue
            keep = True
            for ref_path in refs[: self.reference_size]:
                try:
                    sf.info(self._resolve_path(ref_path))
                except Exception:
                    keep = False
                    break
            if keep:
                good_trials.append(trial)
            else:
                skipped += 1
        if skipped:
            print(f"Warning: skipped {skipped} trials due to unreadable paths.")
        self.trials = good_trials
        self.trial_count = len(self.trials)
        self.total_pairs = self.trial_count * self.reference_size

    def __getitem__(self, idx: int):
        trial_idx = idx // self.reference_size
        ref_idx = idx % self.reference_size
        trial = self.trials[trial_idx]
        ref_paths = self.reference_sets[trial.claim_id]
        ref_path = ref_paths[ref_idx]
        ref_waveform = self._load_waveform(self._resolve_path(ref_path))
        query_waveform = self._load_waveform(self._resolve_path(trial.query_path))
        return trial_idx, ref_path, trial.query_path, ref_waveform, query_waveform, trial.label


class PathDataset(Dataset):
    def __init__(self, root_dir: str, paths: list[str], segment_samples: int | None, skip_bad_paths: bool):
        self.root_dir = root_dir
        self.paths = paths
        self.segment_samples = segment_samples
        if skip_bad_paths:
            self.paths = self._filter_bad_paths(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, rel_path: str) -> str:
        normalized = rel_path.lstrip("./")
        return str(Path(self.root_dir) / normalized)

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


def collate_pairwise(batch):
    batch_size = len(batch)
    trial_indices = torch.tensor([item[0] for item in batch], dtype=torch.long)
    ref_paths = [item[1] for item in batch]
    query_paths = [item[2] for item in batch]
    lengths_ref = torch.tensor([item[3].size(1) for item in batch])
    lengths_query = torch.tensor([item[4].size(1) for item in batch])
    max_ref = int(torch.max(lengths_ref))
    max_query = int(torch.max(lengths_query))
    padded_refs = torch.zeros(batch_size, max_ref)
    padded_queries = torch.zeros(batch_size, max_query)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        ref = item[3]
        query = item[4]
        padded_refs[i] = torch.nn.functional.pad(ref, (0, max_ref - ref.size(1))).squeeze(0)
        padded_queries[i] = torch.nn.functional.pad(query, (0, max_query - query.size(1))).squeeze(0)
        labels[i] = torch.tensor(item[5])
    return trial_indices, ref_paths, query_paths, padded_refs, padded_queries, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise reference-set evaluation (R=1 or R=5) for MLAAD protocols."
    )

    env = parser.add_mutually_exclusive_group()
    env.add_argument("--karolina", action="store_true", help="Use Karolina paths/config.")
    env.add_argument("--metacentrum", action="store_true", help="Use MetaCentrum paths/config.")
    env.add_argument("--sge", action="store_true", help="Use SGE paths/config.")
    env.add_argument("--local", action="store_true", help="Use local paths/config (default).")

    parser.add_argument("-e", "--extractor", required=True, help="Extractor name (e.g., XLSR_300M).")
    parser.add_argument("-p", "--processor", required=True, help="Processor name (e.g., MHFA).")
    parser.add_argument("-c", "--classifier", required=True, help="Pairwise classifier name.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the trained checkpoint.")

    protocol_group = parser.add_argument_group("Protocol selection")
    protocol_group.add_argument("--reference-protocol", type=Path, required=True)
    protocol_group.add_argument("--test-protocol", type=Path, required=True)
    protocol_group.add_argument("--reference-path-column", default="path")
    protocol_group.add_argument("--reference-group-column", default="claim_id")
    protocol_group.add_argument("--reference-rank-column", default="ref_rank")
    protocol_group.add_argument("--test-path-column", default="query_path")
    protocol_group.add_argument("--test-claim-column", default="claim_id")
    protocol_group.add_argument("--test-label-column", default="label")
    protocol_group.add_argument("--test-id-column", default="query_utt_id")
    protocol_group.add_argument("--test-model-column", default="query_model_id")

    parser.add_argument("--data-root", type=Path, required=True, help="Root directory for MLAAD audio.")
    parser.add_argument("--reference-size", type=int, default=5)
    parser.add_argument("--aggregation", choices=("max", "mean", "min"), default="max")
    parser.add_argument(
        "--score-field",
        choices=("logit_margin", "logit_same", "logit_diff", "prob_same", "prob_diff", "cos_sim", "cos_dist"),
        default="logit_margin",
    )
    parser.add_argument("--swap-inputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--amp-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.autocast during inference (default: on).",
    )
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--skip-bad-paths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--embeddings-cache", type=Path, default=None)
    parser.add_argument(
        "--allow-stale-embeddings-cache",
        action="store_true",
        help=(
            "Reuse an existing embeddings cache even if it was built for a different "
            "checkpoint/extractor/processor/classifier/segment length. "
            "By default, mismatched caches are ignored and rebuilt."
        ),
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
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable logistic calibration using dev scores.",
    )
    parser.add_argument(
        "--calibrate-from",
        type=Path,
        default=None,
        help="CSV with dev scores (columns: trial_id, score, label).",
    )
    parser.add_argument(
        "--calibrate-reference-protocol",
        type=Path,
        default=None,
        help="Reference protocol for calibration scoring (dev).",
    )
    parser.add_argument(
        "--calibrate-test-protocol",
        type=Path,
        default=None,
        help="Test protocol for calibration scoring (dev).",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> dict:
    if args.karolina:
        return karolina_config
    if args.sge:
        return sge_config
    return local_config


def _amp_context(args, device: torch.device):
    if not args.amp_eval or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _build_reference_sets(
    reference_df: pd.DataFrame,
    group_col: str,
    path_col: str,
    rank_col: str,
    reference_size: int,
) -> dict[str, list[str]]:
    reference_sets: dict[str, list[str]] = {}
    for claim_id, group in reference_df.groupby(group_col):
        if rank_col in group.columns:
            group = group.sort_values(rank_col)
        paths = group[path_col].dropna().astype(str).tolist()
        if len(paths) < reference_size:
            continue
        reference_sets[str(claim_id)] = paths[:reference_size]
    return reference_sets


def _build_trials(test_df: pd.DataFrame, args: argparse.Namespace) -> list[TrialSpec]:
    trials: list[TrialSpec] = []
    for _, row in test_df.iterrows():
        trials.append(
            TrialSpec(
                trial_id=str(row[args.test_id_column]),
                claim_id=str(row[args.test_claim_column]),
                query_path=str(row[args.test_path_column]),
                query_model_id=str(row[args.test_model_column])
                if args.test_model_column in test_df.columns
                else "unknown",
                label=int(row[args.test_label_column]),
            )
        )
    return trials


def _score_from_logits(logits: torch.Tensor, score_field: str) -> torch.Tensor:
    if score_field == "logit_same":
        return logits[:, 1]
    if score_field == "logit_diff":
        return logits[:, 0]
    if score_field == "prob_same":
        return F.softmax(logits, dim=1)[:, 1]
    if score_field == "prob_diff":
        return F.softmax(logits, dim=1)[:, 0]
    return logits[:, 1] - logits[:, 0]


def _score_from_cosine_similarity(cos_sim: np.ndarray, score_field: str) -> np.ndarray:
    if score_field == "cos_sim":
        return cos_sim
    if score_field == "cos_dist":
        return 1.0 - cos_sim
    raise ValueError(f"Unsupported cosine score_field: {score_field}")


def _report_metrics(scores: np.ndarray, labels: np.ndarray, target_label: int):
    labels_arr = np.asarray(labels)
    labels_bin = (labels_arr == target_label).astype(int)
    if len(np.unique(labels_bin)) > 1:
        fpr, fnr, _ = det_curve(labels_bin, scores, pos_label=1)
        diff = np.abs(fnr - fpr)
        eer_idx = int(np.nanargmin(diff))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
        auc = float(roc_auc_score(labels_bin, scores))
    else:
        eer = float("nan")
        auc = float("nan")
    print(f"EER: {eer * 100:.2f}%  AUC: {auc * 100:.2f}%")
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


def _fit_logistic_calibration(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    x = scores.reshape(-1, 1)
    y = labels.astype(int)
    lr = LogisticRegression(class_weight="balanced", solver="lbfgs")
    lr.fit(x, y)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return a, b


def _apply_logistic_calibration(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    logits = a * scores + b
    # Stable sigmoid to avoid overflow on large logits.
    out = np.empty_like(logits, dtype=float)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_x = np.exp(logits[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _load_scores_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = {"score", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scores file {path} missing required columns: {', '.join(sorted(missing))}")
    scores = df["score"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)
    return scores, labels


def _score_trials_with_model(
    args: argparse.Namespace,
    model,
    device: torch.device,
    reference_sets: dict[str, list[str]],
    trials: list[TrialSpec],
    pin_memory: bool,
    persistent_workers: bool,
    segment_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    def _cache_meta() -> dict:
        try:
            stat = args.checkpoint.stat()
            mtime_ns = int(stat.st_mtime_ns)
            size = int(stat.st_size)
        except FileNotFoundError:
            mtime_ns = None
            size = None
        return {
            "cache_version": 1,
            "checkpoint": str(args.checkpoint),
            "checkpoint_mtime_ns": mtime_ns,
            "checkpoint_size": size,
            "model_class": type(model).__name__,
            "extractor": str(args.extractor),
            "processor": str(args.processor),
            "classifier": str(args.classifier),
            "segment_seconds": float(args.segment_seconds) if args.segment_seconds else 0.0,
            "sample_rate": int(args.sample_rate),
        }

    def _read_cache_meta(npz: np.lib.npyio.NpzFile) -> dict | None:
        meta = npz.get("cache_meta")
        if meta is None:
            return None
        if isinstance(meta, np.ndarray) and meta.shape == ():
            meta = meta.item()
        if isinstance(meta, (bytes, np.bytes_)):
            meta = meta.decode("utf-8")
        if not isinstance(meta, str):
            return None
        try:
            parsed = json.loads(meta)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _cache_compatible(existing: dict | None, expected: dict) -> bool:
        if not existing:
            return False
        keys = (
            "checkpoint",
            "checkpoint_mtime_ns",
            "checkpoint_size",
            "model_class",
            "extractor",
            "processor",
            "classifier",
            "segment_seconds",
            "sample_rate",
        )
        return all(existing.get(k) == expected.get(k) for k in keys)

    use_embedding_cache = False
    if (
        args.embeddings_cache is not None
        and hasattr(model, "forward_from_embeddings")
        and hasattr(model, "extractor")
        and hasattr(model, "feature_processor")
    ):
        use_embedding_cache = True

    agg_max = args.aggregation == "max"
    agg_min = args.aggregation == "min"
    agg_sum = args.aggregation == "mean"
    best_ref_paths: list[str] = [""] * len(trials)

    if use_embedding_cache:
        cache = {}
        if args.embeddings_cache.exists():
            with np.load(args.embeddings_cache, allow_pickle=False) as data:
                expected_meta = _cache_meta()
                existing_meta = _read_cache_meta(data)
                if not _cache_compatible(existing_meta, expected_meta) and not args.allow_stale_embeddings_cache:
                    print(
                        f"[warning] Embeddings cache {args.embeddings_cache} was built for a different "
                        "checkpoint/config; ignoring and rebuilding."
                    )
                else:
                    utt_ids = data.get("utt_ids")
                    embeddings = data.get("embeddings")
                    if utt_ids is None or embeddings is None:
                        raise ValueError(
                            f"Embedding cache missing required arrays: {args.embeddings_cache}"
                        )
                    cache = {str(k): embeddings[i] for i, k in enumerate(utt_ids)}

        paths = sorted(
            set([t.query_path for t in trials])
            | set(p for refs in reference_sets.values() for p in refs[: args.reference_size])
        )
        missing = [p for p in paths if p not in cache]
        if missing:
            dataset = PathDataset(str(args.data_root), missing, segment_samples, skip_bad_paths=args.skip_bad_paths)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=custom_single_batch_create,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers if args.num_workers > 0 else False,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )
            amp_ctx = _amp_context(args, device)
            model.eval()
            with torch.no_grad():
                for names, waveforms, _ in tqdm(dataloader, desc="Embedding"):
                    waveforms = waveforms.to(device)
                    with amp_ctx:
                        feats = model.extractor.extract_features(waveforms)
                        emb = model.feature_processor(feats)
                    emb = emb.detach().to(dtype=torch.float32).cpu().numpy()
                    for name, vec in zip(names, emb):
                        cache[name] = vec
            args.embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
            utt_ids = np.array(sorted(cache))
            emb_matrix = np.stack([cache[u] for u in utt_ids], axis=0)
            np.savez_compressed(
                args.embeddings_cache,
                embeddings=emb_matrix,
                utt_ids=utt_ids,
                cache_meta=np.array(json.dumps(_cache_meta(), sort_keys=True)),
            )

        agg_scores = np.full(len(trials), -np.inf if agg_max else np.inf if agg_min else 0.0, dtype=np.float32)
        def _scores_from_model_output(output, score_field: str) -> np.ndarray:
            if torch.is_tensor(output):
                if score_field in ("cos_sim", "cos_dist"):
                    return _score_from_cosine_similarity(output.detach().cpu().numpy(), score_field)
                raise RuntimeError("score_field requires logits, but model returned similarity.")
            if isinstance(output, tuple) and output:
                first = output[0]
                if torch.is_tensor(first) and first.dim() == 2 and first.size(1) == 2:
                    if score_field in ("cos_sim", "cos_dist"):
                        if hasattr(model, "scale") and hasattr(model, "bias"):
                            logit = first[:, 1]
                            scale = float(getattr(model, "scale").detach().cpu())
                            bias = float(getattr(model, "bias").detach().cpu())
                            cos = (logit - bias) / (scale if scale != 0.0 else 1.0)
                            return _score_from_cosine_similarity(cos.detach().cpu().numpy(), score_field)
                        raise RuntimeError("score_field requires similarity output, but model returned logits.")
                    return _score_from_logits(first, score_field).detach().cpu().numpy()
                if torch.is_tensor(first):
                    if score_field in ("cos_sim", "cos_dist"):
                        return _score_from_cosine_similarity(first.detach().cpu().numpy(), score_field)
                    raise RuntimeError("score_field requires logits, but model returned similarity.")
            raise RuntimeError("Unsupported model output for embedding-based scoring.")

        for i, trial in enumerate(tqdm(trials, desc="Scoring trials")):
            query_emb = cache[trial.query_path]
            ref_paths = reference_sets[trial.claim_id][: args.reference_size]
            ref_embs = np.stack([cache[p] for p in ref_paths], axis=0)
            q = torch.tensor(query_emb, device=device).unsqueeze(0)
            r = torch.tensor(ref_embs, device=device)
            output = model.forward_from_embeddings(r, q.repeat(r.shape[0], 1), label=None)
            scores = _scores_from_model_output(output, args.score_field)

            if agg_max:
                agg_scores[i] = float(np.max(scores))
                best_idx = int(np.argmax(scores))
            elif agg_min:
                agg_scores[i] = float(np.min(scores))
                best_idx = int(np.argmin(scores))
            else:
                agg_scores[i] = float(np.mean(scores))
                best_idx = int(np.argmax(scores))
            best_ref_paths[i] = ref_paths[best_idx]
    else:
        dataset = PairwiseRefsetDataset(
            root_dir=str(args.data_root),
            trials=trials,
            reference_sets=reference_sets,
            reference_size=args.reference_size,
            segment_samples=segment_samples,
            skip_bad_paths=args.skip_bad_paths,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_pairwise,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if args.num_workers > 0 else False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )

        agg_scores = np.full(dataset.trial_count, -np.inf if agg_max else np.inf if agg_min else 0.0, dtype=np.float32)
        best_scores = np.full(dataset.trial_count, np.inf if agg_min else -np.inf, dtype=np.float32)

        amp_ctx = _amp_context(args, device)

        model.eval()
        with torch.no_grad():
            for trial_idx, ref_paths, _, ref_batch, query_batch, labels in tqdm(dataloader, desc="Scoring pairs"):
                ref_batch = ref_batch.to(device)
                query_batch = query_batch.to(device)
                if args.swap_inputs:
                    ref_batch, query_batch = query_batch, ref_batch
                with amp_ctx:
                    if args.score_field in ("cos_sim", "cos_dist"):
                        feats_a = model.extractor.extract_features(ref_batch)
                        feats_b = model.extractor.extract_features(query_batch)
                        emb_a = model.feature_processor(feats_a)
                        emb_b = model.feature_processor(feats_b)
                        cos = F.cosine_similarity(emb_a, emb_b, dim=1, eps=1e-8).float()
                        if args.score_field == "cos_sim":
                            scores_t = cos
                        else:
                            scores_t = 1.0 - cos
                    else:
                        logits, _ = model(ref_batch, query_batch)
                        scores_t = _score_from_logits(logits.float(), args.score_field).float()
                scores = scores_t.detach().cpu().numpy()
                idx_np = trial_idx.numpy()
                if agg_max:
                    for i, t_idx in enumerate(idx_np):
                        if scores[i] > agg_scores[t_idx]:
                            agg_scores[t_idx] = scores[i]
                            best_ref_paths[t_idx] = ref_paths[i]
                elif agg_min:
                    for i, t_idx in enumerate(idx_np):
                        if scores[i] < agg_scores[t_idx]:
                            agg_scores[t_idx] = scores[i]
                            best_ref_paths[t_idx] = ref_paths[i]
                else:
                    for i, t_idx in enumerate(idx_np):
                        agg_scores[t_idx] += scores[i]
                        if scores[i] > best_scores[t_idx]:
                            best_scores[t_idx] = scores[i]
                            best_ref_paths[t_idx] = ref_paths[i]

        if agg_sum:
            agg_scores = agg_scores / args.reference_size

    labels = np.array([t.label for t in trials], dtype=np.int32)
    return agg_scores, labels, best_ref_paths


def main() -> None:
    args = parse_args()
    config = _resolve_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args_ns = argparse.Namespace(
        extractor=args.extractor,
        processor=args.processor,
        classifier=args.classifier,
        kernel=None,
        n_components=None,
        covariance_type=None,
    )
    args_ns.checkpoint = args.checkpoint
    model, trainer = build_model(args_ns)
    trainer.load_model(str(args.checkpoint))
    model.to(device)

    pin_memory = args.pin_memory if args.pin_memory is not None else config.get(
        "pin_memory", torch.cuda.is_available()
    )
    persistent_workers = args.persistent_workers if args.persistent_workers is not None else config.get(
        "persistent_workers", False
    )
    pin_memory = pin_memory and torch.cuda.is_available()

    segment_samples = None
    if args.segment_seconds and args.segment_seconds > 0:
        segment_samples = int(math.floor(args.segment_seconds * args.sample_rate))

    reference_df = pd.read_csv(args.reference_protocol)
    test_df = pd.read_csv(args.test_protocol)

    reference_sets = _build_reference_sets(
        reference_df,
        args.reference_group_column,
        args.reference_path_column,
        args.reference_rank_column,
        args.reference_size,
    )
    trials = _build_trials(test_df, args)
    trials = [t for t in trials if t.claim_id in reference_sets]
    if not trials:
        raise ValueError("No trials matched reference sets; check claim_id columns.")

    agg_scores_raw, labels, best_ref_paths = _score_trials_with_model(
        args,
        model,
        device,
        reference_sets,
        trials,
        pin_memory,
        persistent_workers,
        segment_samples,
    )

    calib_info = None
    agg_scores = agg_scores_raw
    calibration_requested = args.calibrate or args.calibrate_from is not None
    if calibration_requested:
        if args.calibrate_from is not None:
            dev_scores, dev_labels = _load_scores_csv(args.calibrate_from)
        else:
            if args.calibrate_reference_protocol is None or args.calibrate_test_protocol is None:
                raise ValueError("Calibration requested but no dev protocols provided.")
            dev_ref_df = pd.read_csv(args.calibrate_reference_protocol)
            dev_test_df = pd.read_csv(args.calibrate_test_protocol)
            dev_reference_sets = _build_reference_sets(
                dev_ref_df,
                args.reference_group_column,
                args.reference_path_column,
                args.reference_rank_column,
                args.reference_size,
            )
            dev_trials = _build_trials(dev_test_df, args)
            dev_trials = [t for t in dev_trials if t.claim_id in dev_reference_sets]
            dev_scores, dev_labels = _score_trials_with_model(
                args,
                model,
                device,
                dev_reference_sets,
                dev_trials,
                pin_memory,
                persistent_workers,
                segment_samples,
            )
        a_cal, b_cal = _fit_logistic_calibration(dev_scores, dev_labels)
        calib_info = {
            "a": a_cal,
            "b": b_cal,
            "calibration_source": str(args.calibrate_from)
            if args.calibrate_from is not None
            else "protocols",
        }
        print(
            f"[calibration] source={calib_info['calibration_source']} a={a_cal:.6f} b={b_cal:.6f}"
        )
        agg_scores = _apply_logistic_calibration(agg_scores_raw, a_cal, b_cal)

    metrics = _report_metrics(agg_scores, labels, args.target_label)

    det_raw = _det_metrics(
        agg_scores_raw,
        (labels == args.target_label).astype(int),
        args.p_target,
        args.c_miss,
        args.c_fa,
    )
    det_cal = _det_metrics(
        agg_scores,
        (labels == args.target_label).astype(int),
        args.p_target,
        args.c_miss,
        args.c_fa,
    )
    tpr_raw = _tpr_at_fpr(det_raw["fpr"], det_raw["fnr"], args.fixed_fprs)
    tpr_cal = _tpr_at_fpr(det_cal["fpr"], det_cal["fnr"], args.fixed_fprs)
    if calib_info is not None:
        metrics["calibration"] = calib_info

    metrics.update(
        {
            "raw": {
                "min_dcf": det_raw["min_dcf"],
                "min_dcf_threshold": det_raw["min_dcf_threshold"],
                "min_dcf_norm": det_raw["min_dcf_norm"],
                "tpr_at_fpr": tpr_raw,
            },
            "calibrated": {
                "min_dcf": det_cal["min_dcf"],
                "min_dcf_threshold": det_cal["min_dcf_threshold"],
                "min_dcf_norm": det_cal["min_dcf_norm"],
                "tpr_at_fpr": tpr_cal,
            },
        }
    )

    report_dir = args.report_dir
    if report_dir is None and args.output_csv is not None:
        report_dir = args.output_csv.parent
    if report_dir is None:
        report_dir = Path(".")
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_det:
        _plot_det(det_raw["fpr"], det_raw["fnr"], report_dir / "pairwise_refset_det_raw.png")
        _plot_det(det_cal["fpr"], det_cal["fnr"], report_dir / "pairwise_refset_det_cal.png")
    if args.plot_scores:
        _plot_score_dist(
            agg_scores_raw,
            (labels == args.target_label).astype(int),
            report_dir / "pairwise_refset_scores_raw.png",
        )
        _plot_score_dist(
            agg_scores,
            (labels == args.target_label).astype(int),
            report_dir / "pairwise_refset_scores_cal.png",
        )

    if args.output_csv is not None:
        rows = []
        for i, trial in enumerate(trials):
            cos_sim_raw = None
            cos_dist_raw = None
            if args.score_field == "cos_sim":
                cos_sim_raw = float(agg_scores_raw[i])
                cos_dist_raw = float(1.0 - agg_scores_raw[i])
            elif args.score_field == "cos_dist":
                cos_dist_raw = float(agg_scores_raw[i])
                cos_sim_raw = float(1.0 - agg_scores_raw[i])
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "claim_id": trial.claim_id,
                    "pathA": best_ref_paths[i],
                    "pathB": trial.query_path,
                    "query_path": trial.query_path,
                    "query_model_id": trial.query_model_id,
                    "query_id": trial.query_model_id,
                    "label": trial.label,
                    "score_field": args.score_field,
                    "score": float(agg_scores[i]),
                    "score_raw": float(agg_scores_raw[i]),
                    "cos_sim_raw": cos_sim_raw,
                    "cos_dist_raw": cos_dist_raw,
                }
            )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    with open(report_dir / "pairwise_refset_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
