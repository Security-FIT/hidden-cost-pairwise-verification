#!/usr/bin/env python3
"""
Evaluate a pairwise model (e.g., FFAttn1/2) on the eval split, emit per-pair scores,
and report EER, minDCF/actDCF (and normalized variants), C_llr, and fixed-FPR operating points.

The primary DCF settings come from --p-target/--c-miss/--c-fa; optionally report additional
application profiles (e.g., forensics vs intelligence) via --eval-profiles.
"""

from __future__ import annotations

import csv
import json
import math
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import det_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from common import build_model, get_dataloaders
from config import karolina_config, local_config, sge_config
from parse_arguments import parse_args
from trainers.embedding_cache import EmbeddingCache, build_pair_keys

try:
    # Optional Interspeech CI dependency (used during training)
    from confidence_intervals import evaluate_with_conf_int
except ImportError:
    evaluate_with_conf_int = None


DEFAULT_FIXED_FPRS = (0.0001, 0.001, 0.01, 0.05)

# NOTE: These are *reporting* presets (not "ground truth" priors). Adjust as needed.
DCF_PROFILE_PRESETS: dict[str, dict[str, float]] = {
    # Conservative evidence threshold (LR ≈ 99) appropriate for "linking" with low prior.
    "forensics": {"p_target": 0.01, "c_miss": 1.0, "c_fa": 1.0},
    # More permissive triage threshold (LR ≈ 9) useful for lead generation.
    "intel": {"p_target": 0.10, "c_miss": 1.0, "c_fa": 1.0},
    # Historical default used in this repo (ASVspoof5 Phase 2 style).
    "asvspoof5": {"p_target": 0.1125, "c_miss": 1.0, "c_fa": 10.0},
}

LABEL_COLUMNS = ("same_model", "model_type_same", "model_family_same")
DERIVED_ARCH_LABEL = "architecture_same"


def _select_config(args):
    if args.sge:
        return sge_config
    if args.karolina:
        return karolina_config
    return local_config


def _amp_context(args, device: torch.device):
    if not args.amp_eval or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _enable_memory_efficient_sdpa():
    """
    Enable PyTorch SDPA kernels (flash/mem-efficient) when available.
    Safe no-op on older torch builds.
    """
    if not torch.cuda.is_available():
        return
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        # Backend knobs are best-effort; keep eval running.
        return


def _maybe_disable_cross_attn_weights_for_eval(model: torch.nn.Module, args) -> bool:
    """
    FFAttn3/4/5 ignore attention weights but nn.MultiheadAttention still materializes them
    by default, which can OOM for long sequences. On SGE, force need_weights=False for the
    classifier cross-attention only (never touches extractor internals).
    """
    if not getattr(args, "sge", False):
        return False
    if getattr(args, "classifier", "") not in {"FFAttn3", "FFAttn4", "FFAttn5"}:
        return False
    attn = getattr(model, "attn", None)
    if not isinstance(attn, torch.nn.MultiheadAttention):
        return False

    _enable_memory_efficient_sdpa()

    orig_forward = attn.forward

    def _forward_no_weights(*inputs, **kwargs):
        kwargs["need_weights"] = False
        return orig_forward(*inputs, **kwargs)

    attn.forward = _forward_no_weights
    return True


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _with_tag(path: Path, tag: str | None) -> Path:
    if not tag:
        return path
    tag_clean = str(tag).strip()
    if not tag_clean:
        return path
    return path.with_name(f"{path.stem}_{tag_clean}{path.suffix}")


def _load_embedding_table(path: str | Path) -> tuple[list[str], np.ndarray]:
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"Embedding file not found: {src}")
    data = np.load(src, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile) and "embeddings" in data and "utt_ids" in data:
        emb = np.asarray(data["embeddings"])
        utt_ids = [str(x) for x in data["utt_ids"]]
        if emb.shape[0] != len(utt_ids):
            raise ValueError(
                f"Embedding/table shape mismatch in {src}: {emb.shape[0]} rows vs {len(utt_ids)} ids."
            )
        return utt_ids, emb
    if isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict):
        table = data.item()
        utt_ids = list(table.keys())
        emb = np.stack([table[k] for k in utt_ids], axis=0)
        return [str(x) for x in utt_ids], emb
    raise ValueError(
        f"Unsupported embedding table format in {src}. "
        "Expect npz with embeddings/utt_ids or a dict of id -> vector."
    )


def _normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embedding encountered during normalization.")
    return emb / norms


def _embedding_map(utt_ids: list[str], emb_matrix: np.ndarray) -> dict[str, np.ndarray]:
    if emb_matrix.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {emb_matrix.shape}.")
    emb_map: dict[str, np.ndarray] = {}
    for uid, vec in zip(utt_ids, emb_matrix):
        emb_map[_canonicalize_path(str(uid))] = np.asarray(vec, dtype=np.float32)
    return emb_map


def _extract_eval_embeddings(model, dataloader: DataLoader, device: torch.device, amp_ctx) -> dict[str, np.ndarray]:
    if not hasattr(model, "feature_processor"):
        raise RuntimeError("Model does not expose feature_processor; cannot extract embeddings for S-norm.")
    emb_map: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Embedding eval set", unit="batch")
        for pair_ids, gt, test, _ in iterator:
            gt = gt.to(device)
            test = test.to(device)
            with amp_ctx:
                feats_gt = model.extractor.extract_features(gt)
                feats_test = model.extractor.extract_features(test)
                emb_gt = model.feature_processor(feats_gt)
                emb_test = model.feature_processor(feats_test)
            emb_gt_np = emb_gt.detach().to(dtype=torch.float32).cpu().numpy()
            emb_test_np = emb_test.detach().to(dtype=torch.float32).cpu().numpy()
            for pid, vec_gt, vec_test in zip(pair_ids, emb_gt_np, emb_test_np):
                path_a, path_b = _split_pair_id(pid)
                key_a = _canonicalize_path(path_a)
                key_b = _canonicalize_path(path_b)
                if key_a not in emb_map:
                    emb_map[key_a] = vec_gt
                if key_b not in emb_map:
                    emb_map[key_b] = vec_test
    return emb_map


def _pair_embeddings_from_map(pair_ids: np.ndarray, emb_map: dict[str, np.ndarray]):
    emb_a = []
    emb_b = []
    mask = np.zeros(len(pair_ids), dtype=bool)
    for idx, pid in enumerate(pair_ids):
        path_a, path_b = _split_pair_id(pid)
        key_a = _canonicalize_path(path_a)
        key_b = _canonicalize_path(path_b)
        vec_a = emb_map.get(key_a)
        vec_b = emb_map.get(key_b)
        if vec_a is None or vec_b is None:
            continue
        emb_a.append(vec_a)
        emb_b.append(vec_b)
        mask[idx] = True
    if not emb_a:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), mask
    return np.stack(emb_a, axis=0), np.stack(emb_b, axis=0), mask


def _snorm_stats(emb: np.ndarray, cohort: np.ndarray, batch_size: int, scale: float, bias: float) -> tuple[np.ndarray, np.ndarray]:
    n = emb.shape[0]
    mu = np.empty(n, dtype=np.float32)
    sigma = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores = emb[start:end] @ cohort.T
        if scale != 1.0 or bias != 0.0:
            scores = scores * scale + bias
        mu[start:end] = scores.mean(axis=1)
        sigma[start:end] = scores.std(axis=1)
    return mu, sigma


def _snorm_scores(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    cohort: np.ndarray,
    scale: float,
    bias: float,
    eps: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_scores = np.sum(emb_a * emb_b, axis=1)
    if scale != 1.0 or bias != 0.0:
        raw_scores = raw_scores * scale + bias

    mu_a, sigma_a = _snorm_stats(emb_a, cohort, batch_size, scale, bias)
    mu_b, sigma_b = _snorm_stats(emb_b, cohort, batch_size, scale, bias)
    sigma_a = np.maximum(sigma_a, eps)
    sigma_b = np.maximum(sigma_b, eps)
    snorm = 0.5 * ((raw_scores - mu_a) / sigma_a + (raw_scores - mu_b) / sigma_b)
    return raw_scores.astype(np.float32), snorm.astype(np.float32)


def _canonicalize_path(path: str) -> str:
    """Normalize relative path strings for consistent lookups."""
    return Path(path).as_posix().lstrip("./")


def _canonicalize_pair_id(pid: str) -> str:
    if "|" in pid:
        a, b = pid.split("|", 1)
        return f"{_canonicalize_path(a)}|{_canonicalize_path(b)}"
    return _canonicalize_path(pid)


def _lookup_scenario(scenario_map: dict[str, str], pid: str) -> str | None:
    """
    Return a scenario for pid using a few canonical forms, including reversed pairs.
    """
    canonical = _canonicalize_pair_id(pid)
    if canonical in scenario_map:
        return scenario_map[canonical]
    if "|" in canonical:
        a, b = canonical.split("|", 1)
        reversed_pid = f"{b}|{a}"
        if reversed_pid in scenario_map:
            return scenario_map[reversed_pid]
    if pid in scenario_map:
        return scenario_map[pid]
    return None


def _bayes_threshold(p_target: float, c_miss: float, c_fa: float) -> float:
    """Posterior threshold for minimum Bayes risk."""
    return (c_fa * (1 - p_target)) / (c_fa * (1 - p_target) + c_miss * p_target)


def _bayes_threshold_llr(p_target: float, c_miss: float, c_fa: float) -> float:
    """LLR threshold for minimum Bayes risk."""
    denom = c_miss * p_target
    if denom <= 0:
        return float("inf")
    numer = c_fa * (1 - p_target)
    if numer <= 0:
        return float("-inf")
    return float(math.log(numer / denom))


def _sigmoid(x: float) -> float:
    # Stable sigmoid for scalar x.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-12) -> float:
    p_clip = min(max(p, eps), 1.0 - eps)
    return float(math.log(p_clip) - math.log1p(-p_clip))


def _rates(scores: np.ndarray, labels: np.ndarray, threshold: float, pos_label: int):
    preds = scores >= threshold
    pos = labels == pos_label
    neg = labels != pos_label
    tp = np.sum(preds & pos)
    fn = np.sum(~preds & pos)
    fp = np.sum(preds & neg)
    tn = np.sum(~preds & neg)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else math.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else math.nan
    return fnr, fpr


def _dcf_best_cost(p_target: float, c_miss: float, c_fa: float) -> float:
    return float(min(c_miss * p_target, c_fa * (1.0 - p_target)))


def _dcf_norm(dcf: float, p_target: float, c_miss: float, c_fa: float) -> float:
    best = _dcf_best_cost(p_target, c_miss, c_fa)
    if best <= 0 or math.isnan(dcf):
        return float("nan")
    return float(dcf / best)


def _det_metrics(scores: np.ndarray, labels: np.ndarray, p_target: float, c_miss: float, c_fa: float, pos_label: int):
    fpr, fnr, thresholds = det_curve(labels, scores, pos_label=pos_label)
    diff = np.abs(fnr - fpr)
    eer_idx = int(np.nanargmin(diff))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_thr = float(thresholds[eer_idx])

    # minDCF
    costs = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    min_idx = int(np.nanargmin(costs))
    min_dcf = float(costs[min_idx])
    min_dcf_thr = float(thresholds[min_idx])
    min_dcf_norm = _dcf_norm(min_dcf, p_target, c_miss, c_fa)

    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "min_dcf": min_dcf,
        "min_dcf_norm": min_dcf_norm,
        "min_dcf_threshold": min_dcf_thr,
        "fpr": fpr,
        "fnr": fnr,
        "thresholds": thresholds,
    }


def _tpr_at_fpr_targets(fpr: np.ndarray, fnr: np.ndarray, thresholds: np.ndarray, targets: Iterable[float]):
    tpr = 1.0 - fnr
    results = []
    for target in targets:
        if fpr.size == 0:
            results.append({"fpr_target": target, "fpr": None, "tpr": None, "threshold": None, "met_target": False})
            continue

        finite = np.isfinite(fpr) & np.isfinite(tpr) & np.isfinite(thresholds)
        if not np.any(finite):
            results.append({"fpr_target": target, "fpr": None, "tpr": None, "threshold": None, "met_target": False})
            continue

        fpr_f = fpr[finite]
        tpr_f = tpr[finite]
        thr_f = thresholds[finite]

        # Choose the best point under the constraint FPR <= target (never exceed if possible).
        valid = np.where(fpr_f <= target)[0]
        if valid.size > 0:
            idx = int(valid[np.nanargmax(fpr_f[valid])])
            met = True
        else:
            idx = int(np.nanargmin(fpr_f))
            met = False

        fpr_val = float(fpr_f[idx]) if not math.isnan(fpr_f[idx]) else None
        tpr_val = float(tpr_f[idx]) if not math.isnan(tpr_f[idx]) else None
        thr_val = float(thr_f[idx]) if not math.isnan(thr_f[idx]) else None
        results.append({"fpr_target": target, "fpr": fpr_val, "tpr": tpr_val, "threshold": thr_val, "met_target": met})
    return results


def _eer_only(scores: np.ndarray, labels: np.ndarray, pos_label: int) -> float:
    fpr, fnr, _ = det_curve(labels, scores, pos_label=pos_label)
    diff = np.abs(fnr - fpr)
    eer_idx = int(np.nanargmin(diff))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def _dcf(fnr: float, fpr: float, p_target: float, c_miss: float, c_fa: float) -> float:
    return c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr


def _ppv_npv(fnr: float, fpr: float, p_target: float) -> tuple[float | None, float | None]:
    """
    Positive/negative predictive value under an assumed prior p_target.
    Interpretable as: P(target | accept) and P(non-target | reject).
    """
    if not (0.0 <= p_target <= 1.0) or math.isnan(fnr) or math.isnan(fpr):
        return None, None
    tpr = 1.0 - fnr
    tnr = 1.0 - fpr
    ppv_denom = p_target * tpr + (1.0 - p_target) * fpr
    npv_denom = (1.0 - p_target) * tnr + p_target * fnr
    ppv = (p_target * tpr) / ppv_denom if ppv_denom > 0 else math.nan
    npv = ((1.0 - p_target) * tnr) / npv_denom if npv_denom > 0 else math.nan
    return (None if math.isnan(ppv) else float(ppv)), (None if math.isnan(npv) else float(npv))


def _parse_float_list(spec: str) -> list[float]:
    out: list[float] = []
    for token in spec.replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith("%"):
            out.append(float(token[:-1]) / 100.0)
        else:
            out.append(float(token))
    return out


def _parse_eval_profiles(spec: str | None) -> list[str]:
    if spec is None or not spec.strip():
        # Default: show application-facing presets alongside the historical "primary" setting.
        return ["primary", "forensics", "intel"]
    return [p.strip() for p in spec.split(",") if p.strip()]


def _resolve_profile(name: str, args) -> tuple[float, float, float]:
    if name == "primary":
        return float(args.p_target), float(args.c_miss), float(args.c_fa)
    if name not in DCF_PROFILE_PRESETS:
        known = ", ".join(sorted(["primary", *DCF_PROFILE_PRESETS.keys()]))
        raise ValueError(f"Unknown eval profile '{name}'. Known profiles: {known}")
    prof = DCF_PROFILE_PRESETS[name]
    return float(prof["p_target"]), float(prof["c_miss"]), float(prof["c_fa"])


def _cllr(scores: np.ndarray, labels: np.ndarray, pos_label: int, scores_are_llr: bool = False) -> float:
    """
    Compute C_llr (Brümmer & du Preez) from log-likelihood ratios.

    If scores are in (0,1), they are treated as posterior probabilities under a 0.5 prior
    and converted to log-odds (LLR) via logit().
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels)

    # Treat probability-like scores as posterior under equal priors -> LLR = logit(p)
    if not scores_are_llr and np.all((scores_arr > 0.0) & (scores_arr < 1.0)):
        eps = 1e-6
        s = np.clip(scores_arr, eps, 1.0 - eps)
        llr = np.log(s) - np.log1p(-s)
    else:
        llr = scores_arr

    tar = llr[labels_arr == pos_label]
    non = llr[labels_arr != pos_label]
    if tar.size == 0 or non.size == 0:
        return math.nan

    # softplus(x) = log(1 + exp(x)) implemented stably via logaddexp
    c1 = float(np.mean(np.logaddexp(0.0, -tar)))
    c2 = float(np.mean(np.logaddexp(0.0, non)))
    cllr = (c1 + c2) / (2.0 * math.log(2))
    return float(cllr)


def _scores_domain(scores: np.ndarray, scores_are_llr: bool) -> str:
    if scores_are_llr:
        return "llr"
    if np.all((scores > 0.0) & (scores < 1.0)):
        return "posterior"
    return "raw"


def _act_metrics_from_mode(
    scores: np.ndarray,
    labels: np.ndarray,
    det: dict,
    p_target: float,
    c_miss: float,
    c_fa: float,
    pos_label: int,
    mode: str,
    threshold_override: float | None,
    scores_domain: str,
) -> dict:
    bayes_llr_thr = _bayes_threshold_llr(p_target, c_miss, c_fa)
    bayes_thr = _sigmoid(bayes_llr_thr)

    act_source = mode
    if threshold_override is not None:
        act_thr = float(threshold_override)
        act_source = "manual"
    elif mode == "sweep":
        costs = c_miss * p_target * det["fnr"] + c_fa * (1.0 - p_target) * det["fpr"]
        act_idx = int(np.nanargmin(costs))
        act_thr = float(det["thresholds"][act_idx])
        act_source = "sweep"
    else:
        if scores_domain == "llr":
            act_thr = float(bayes_llr_thr)
            act_source = "bayes_llr"
        else:
            act_thr = float(bayes_thr)
            act_source = "bayes_posterior"

    act_fnr, act_fpr = _rates(scores, labels, act_thr, pos_label=pos_label)
    act_dcf = _dcf(act_fnr, act_fpr, p_target=p_target, c_miss=c_miss, c_fa=c_fa)
    act_dcf_norm = _dcf_norm(act_dcf, p_target, c_miss, c_fa)
    act_ppv, act_npv = _ppv_npv(act_fnr, act_fpr, p_target=p_target)
    return {
        "act_dcf": float(act_dcf),
        "act_dcf_norm": act_dcf_norm,
        "act_threshold": act_thr,
        "act_rates": {
            "fnr": None if math.isnan(act_fnr) else float(act_fnr),
            "fpr": None if math.isnan(act_fpr) else float(act_fpr),
        },
        "act_ppv": act_ppv,
        "act_npv": act_npv,
        "act_threshold_source": act_source,
        "bayes_threshold": bayes_thr,
        "bayes_threshold_llr": bayes_llr_thr,
        "bayes_threshold_lr": float(math.exp(bayes_llr_thr)) if math.isfinite(bayes_llr_thr) else None,
    }


def _bootstrap_eer_confidence_interval(
    scores: np.ndarray,
    labels: np.ndarray,
    pos_label: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    conditions: np.ndarray | None = None,
    random_state: int | None = 0,
) -> tuple[float, float] | None:
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if labels_arr.size == 0:
        return None

    rng = np.random.default_rng(random_state)
    indices = np.arange(labels_arr.shape[0])
    cond_arr = None if conditions is None else np.asarray(conditions)
    eers: list[float] = []
    for _ in range(n_bootstrap):
        if cond_arr is None:
            sample_idx = rng.choice(indices, size=indices.size, replace=True)
        else:
            unique_conditions = np.unique(cond_arr)
            sampled_blocks = []
            for cond in unique_conditions:
                cond_indices = indices[cond_arr == cond]
                if cond_indices.size == 0:
                    continue
                resample = rng.choice(cond_indices, size=cond_indices.size, replace=True)
                sampled_blocks.append(resample)
            if not sampled_blocks:
                continue
            sample_idx = np.concatenate(sampled_blocks)
        sample_labels = labels_arr[sample_idx]
        if len(np.unique(sample_labels)) < 2:
            continue  # skip degenerate resamples
        sample_scores = scores_arr[sample_idx]
        eers.append(_eer_only(sample_scores, sample_labels, pos_label=pos_label))

    if not eers:
        return None

    lower = np.percentile(eers, 100 * (alpha / 2))
    upper = np.percentile(eers, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _eer_confidence_interval(
    scores: np.ndarray,
    labels: np.ndarray,
    pos_label: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    conditions: np.ndarray | None = None,
    random_state: int | None = 0,
) -> tuple[float | None, tuple[float, float] | None]:
    """
    Compute EER and a 95% CI, preferring the Interspeech CI framework when available.
    """
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    cond_arr = None if conditions is None else np.asarray(conditions)

    if evaluate_with_conf_int is not None:
        def _eer_metric(metric_labels, metric_scores):
            return _eer_only(metric_scores, metric_labels, pos_label=pos_label)

        alpha_pct = alpha * 100 if alpha < 1 else alpha
        try:
            center, ci_bounds = evaluate_with_conf_int(
                samples=scores_arr,
                metric=_eer_metric,
                labels=labels_arr,
                conditions=cond_arr,
                num_bootstraps=n_bootstrap,
                alpha=alpha_pct,
            )
            ci_tuple = None if ci_bounds is None else (float(ci_bounds[0]), float(ci_bounds[1]))
            return float(center), ci_tuple
        except Exception:
            # Fallback to bootstrap if the dependency is present but errors out
            pass

    ci = _bootstrap_eer_confidence_interval(
        scores_arr,
        labels_arr,
        pos_label=pos_label,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        conditions=cond_arr,
        random_state=random_state,
    )
    eer = _eer_only(scores_arr, labels_arr, pos_label=pos_label) if ci is not None else None
    return eer, ci


def _build_scenario_map(dataset) -> dict[str, str]:
    """
    Try to build a map from pair_id (pathA|pathB) to scenario using the protocol_df
    (MLAAD pair protocols expose scenario_group).
    """
    scenario_map: dict[str, str] = {}
    proto = getattr(dataset, "protocol_df", None)
    if proto is None:
        return scenario_map
    if "path_A" in proto.columns and "path_B" in proto.columns:
        scenario_col = "scenario_group" if "scenario_group" in proto.columns else None
        if scenario_col:
            for _, row in proto.iterrows():
                pid = f"{row['path_A']}|{row['path_B']}"
                scenario = str(row[scenario_col])
                scenario_map[pid] = scenario
                scenario_map[_canonicalize_pair_id(pid)] = scenario
    return scenario_map


def _load_label_protocol(path: str | Path):
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"Label protocol file not found: {src}")
    return pd.read_csv(src)


def _build_pair_index(protocol_df) -> dict[str, int]:
    if "path_A" not in protocol_df.columns or "path_B" not in protocol_df.columns:
        return {}
    pair_index: dict[str, int] = {}
    dupes = 0
    for idx, (path_a, path_b) in enumerate(zip(protocol_df["path_A"], protocol_df["path_B"])):
        key = _canonicalize_pair_id(f"{path_a}|{path_b}")
        if key in pair_index:
            dupes += 1
        pair_index[key] = idx
    if dupes:
        print(f"[warn] Label protocol has {dupes} duplicate pair_ids; using the last occurrence.")
    return pair_index


def _label_sources_from_protocol(protocol_df) -> dict[str, np.ndarray]:
    label_sources: dict[str, np.ndarray] = {}
    for col in LABEL_COLUMNS:
        if col in protocol_df.columns:
            values = pd.to_numeric(protocol_df[col], errors="coerce").to_numpy(dtype=float)
            label_sources[col] = values

    if DERIVED_ARCH_LABEL in protocol_df.columns:
        values = pd.to_numeric(protocol_df[DERIVED_ARCH_LABEL], errors="coerce").to_numpy(dtype=float)
        label_sources[DERIVED_ARCH_LABEL] = values
    elif "architecture_A" in protocol_df.columns and "architecture_B" in protocol_df.columns:
        arch_a = protocol_df["architecture_A"].replace("", np.nan)
        arch_b = protocol_df["architecture_B"].replace("", np.nan)
        valid = arch_a.notna() & arch_b.notna()
        arch_same = (arch_a == arch_b) & valid
        values = arch_same.astype(float)
        values[~valid] = np.nan
        label_sources[DERIVED_ARCH_LABEL] = values.to_numpy(dtype=float)

    return label_sources


def _align_label_values(
    pair_ids: np.ndarray, pair_index: dict[str, int], values: np.ndarray, label_name: str
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(len(pair_ids), -1, dtype=int)
    missing_pairs = 0
    missing_labels = 0
    values_arr = np.asarray(values, dtype=float)
    for idx, pid in enumerate(pair_ids):
        key = _canonicalize_pair_id(pid)
        row_idx = pair_index.get(key)
        if row_idx is None:
            missing_pairs += 1
            continue
        val = values_arr[row_idx]
        if np.isnan(val):
            missing_labels += 1
            continue
        labels[idx] = int(val)
    if missing_pairs:
        print(f"[warn] {label_name}: {missing_pairs} pairs missing in label protocol; excluding them.")
    if missing_labels:
        print(f"[warn] {label_name}: {missing_labels} pairs with missing labels; excluding them.")
    return labels, labels >= 0


def _extract_label_sets(pair_ids: np.ndarray, protocol_df):
    if protocol_df is None:
        return {}
    if "path_A" not in protocol_df.columns or "path_B" not in protocol_df.columns:
        print("[warn] Label protocol missing path_A/path_B columns; skipping extra label eval.")
        return {}

    pair_index = _build_pair_index(protocol_df)
    if not pair_index:
        print("[warn] Label protocol has no pairs; skipping extra label eval.")
        return {}

    label_sources = _label_sources_from_protocol(protocol_df)
    if not label_sources:
        return {}

    label_sets: dict[str, dict[str, np.ndarray]] = {}
    for name, values in label_sources.items():
        labels, mask = _align_label_values(pair_ids, pair_index, values, name)
        label_sets[name] = {"labels": labels, "mask": mask}
    return label_sets


def _init_csv_writer(path: str | None):
    if path is None:
        return None, None
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    csv_file = dest.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["pathA", "pathB", "score", "label", "scenario_group"])
    csv_file.flush()
    return csv_file, writer


def _score_loader(model, dataloader: DataLoader, device: torch.device, amp_ctx, csv_writer=None, csv_file=None):
    model.eval()
    all_scores = []
    all_labels = []
    all_pair_ids: list[str] = []
    all_scenarios: list[str] = []
    use_embedding_cache = (
        hasattr(model, "forward_from_embeddings")
        and hasattr(model, "extractor")
        and hasattr(model, "feature_processor")
    )
    embedding_cache = None
    use_label_arg = False
    if use_embedding_cache:
        embedding_cache = EmbeddingCache(model.extractor, model.feature_processor, device)
        import inspect

        use_label_arg = "label" in inspect.signature(model.forward_from_embeddings).parameters

    scenario_map = _build_scenario_map(dataloader.dataset)

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Scoring", unit="batch")
        for pair_ids, gt, test, label in iterator:
            label = label.to(device)
            if use_embedding_cache:
                keys_gt, keys_test = build_pair_keys(pair_ids, gt, test)
                emb_gt = embedding_cache.get_embeddings(gt, keys_gt, amp_ctx)
                emb_test = embedding_cache.get_embeddings(test, keys_test, amp_ctx)
                with amp_ctx:
                    if use_label_arg:
                        out = model.forward_from_embeddings(emb_gt, emb_test, label=label)
                    else:
                        out = model.forward_from_embeddings(emb_gt, emb_test)
            else:
                gt = gt.to(device)
                test = test.to(device)
                with amp_ctx:
                    out = model(gt, test)

            # Support both (logits, probs) models and raw-score models that return a single Tensor.
            if isinstance(out, tuple):
                if len(out) == 2:
                    _, probs = out
                elif len(out) == 1:
                    probs = out[0]
                else:
                    raise ValueError(
                        f"Unsupported model output arity={len(out)} for scoring (expected 1 or 2)."
                    )
            else:
                probs = out

            if probs.ndim == 1:
                scores = probs.detach().cpu().numpy()
            else:
                scores = probs[:, 1].detach().cpu().numpy()
            all_scores.extend(scores.tolist())
            label_list = label.cpu().numpy().astype(int).tolist()
            all_labels.extend(label_list)
            all_pair_ids.extend(list(pair_ids))
            batch_scenarios = []
            for pid in pair_ids:
                scenario = _lookup_scenario(scenario_map, pid)
                if scenario is None:
                    a, _ = _split_pair_id(pid)
                    scenario = _extract_scenario(a)
                batch_scenarios.append(scenario if scenario is not None else "unknown")
            all_scenarios.extend(batch_scenarios)
            if csv_writer is not None:
                rows = []
                for pid, s, lbl, scen in zip(pair_ids, scores, label_list, batch_scenarios):
                    a, b = _split_pair_id(pid)
                    rows.append((a, b, float(s), int(lbl), scen))
                csv_writer.writerows(rows)
                if csv_file is not None:
                    csv_file.flush()

    return (
        np.asarray(all_pair_ids),
        np.asarray(all_scores, dtype=float),
        np.asarray(all_labels, dtype=int),
        np.asarray(all_scenarios, dtype=object),
    )


def _split_pair_id(pid: str) -> tuple[str, str]:
    # Common MLAAD pair_id format: "pathA|pathB"
    if "|" in pid:
        a, b = pid.split("|", 1)
        return a, b
    # Fallback: single id, leave pathB empty
    return pid, ""


def _print_scores(
    pair_ids: Iterable[str],
    scores: Iterable[float],
):
    for pid, score in zip(pair_ids, scores):
        path_a, path_b = _split_pair_id(pid)
        print(f"{path_a},{path_b},{score:.6f}")


def _maybe_save(path: str | None, pair_ids: np.ndarray, scores: np.ndarray, labels: np.ndarray, scenarios: np.ndarray):
    if path is None:
        return
    dest = Path(path)
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to save scores; install it or omit --scores-out") from exc
    records = []
    for pid, s, lbl, scen in zip(pair_ids, scores, labels, scenarios):
        a, b = _split_pair_id(pid)
        records.append(
            {
                "pathA": a,
                "pathB": b,
                "score": float(s),
                "label": int(lbl),
                "scenario_group": scen,
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(dest, index=False)
    print(f"[info] Wrote {len(records):,} scores to {dest}")


def _load_scores_csv(path: str | Path, score_column: str = "score"):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to load scores; install it or omit --scores-in") from exc
    df = pd.read_csv(path)
    score_column = str(score_column or "score")
    required_cols = {"pathA", "pathB", score_column, "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Scores file {path} missing required columns: {', '.join(sorted(missing))}")
    scen_col = "scenario_group" if "scenario_group" in df.columns else None
    pair_ids = [f"{a}|{b}" for a, b in zip(df["pathA"], df["pathB"])]
    scores = df[score_column].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)
    if scen_col:
        scenarios = df[scen_col].astype(str).to_numpy()
    else:
        scenarios = np.array(["unknown"] * len(df), dtype=object)
    return np.asarray(pair_ids), scores, labels, scenarios


def _fit_logistic_calibration(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Fit a simple Platt scaling (logistic regression) calibration: prob = sigmoid(a*score + b).
    Returns (a, b).
    """
    x = scores.reshape(-1, 1)
    y = labels.astype(int)
    # balance classes to avoid skew
    class_weight = "balanced"
    lr = LogisticRegression(class_weight=class_weight, solver="lbfgs")
    lr.fit(x, y)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return a, b


def _apply_logistic_calibration(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    logits = a * scores + b
    # stable sigmoid
    return 1.0 / (1.0 + np.exp(-logits))


def _extract_scenario(path_a: str) -> str:
    try:
        parts = Path(path_a).parts
        return parts[0] if parts else "unknown"
    except Exception:
        return "unknown"


def _per_scenario_metrics(
    scenarios: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    act_threshold: float,
    pos_label: int,
    p_target: float,
    c_miss: float,
    c_fa: float,
) -> list[dict]:
    results = []
    for scen in np.unique(scenarios):
        mask = scenarios == scen
        scen_scores = scores[mask]
        scen_labels = labels[mask]
        if scen_scores.size == 0:
            continue
        has_pos = np.any(scen_labels == pos_label)
        has_neg = np.any(scen_labels != pos_label)
        fnr_s, fpr_s = _rates(scen_scores, scen_labels, act_threshold, pos_label=pos_label)
        tpr_s = None if not has_pos else (1.0 - fnr_s if not math.isnan(fnr_s) else None)
        fpr_val = None if not has_neg else (None if math.isnan(fpr_s) else fpr_s)
        eer_s = None
        if has_pos and has_neg:
            det_s = _det_metrics(scen_scores, scen_labels, p_target, c_miss, c_fa, pos_label)
            eer_s = det_s["eer"]
        results.append(
            {
                "scenario": scen,
                "count": int(scen_scores.size),
                "eer": eer_s,
                "fpr": fpr_val,
                "tpr": tpr_s,
            }
        )
    return results


def _empty_overall():
    return {
        "eer": None,
        "eer_ci": None,
        "eer_threshold": None,
        "eer_rates": {"fnr": None, "fpr": None},
        "eer_dcf": None,
        "min_dcf": None,
        "min_dcf_norm": None,
        "min_dcf_threshold": None,
        "min_dcf_rates": {"fnr": None, "fpr": None},
        "act_dcf": None,
        "act_dcf_norm": None,
        "act_threshold": None,
        "act_threshold_source": None,
        "act_rates": {"fnr": None, "fpr": None},
        "act_ppv": None,
        "act_npv": None,
        "cllr": None,
        "tpr_at_fpr": [],
    }


def _compute_label_overall(
    scores: np.ndarray,
    labels: np.ndarray,
    scenarios: np.ndarray,
    args,
    fixed_fprs: list[float],
    scores_domain: str,
    act_threshold_mode: str,
    threshold_override: float | None = None,
    include_ci: bool = False,
    excluded: int = 0,
):
    counts = {
        "total": int(labels.size),
        "positive": int(np.sum(labels == args.pos_label)),
        "negative": int(np.sum(labels != args.pos_label)),
    }
    if excluded:
        counts["excluded"] = int(excluded)

    has_pos = counts["positive"] > 0
    has_neg = counts["negative"] > 0
    if not (has_pos and has_neg):
        return counts, _empty_overall()

    det = _det_metrics(scores, labels, p_target=args.p_target, c_miss=args.c_miss, c_fa=args.c_fa, pos_label=args.pos_label)
    cllr = _cllr(scores, labels, pos_label=args.pos_label, scores_are_llr=(scores_domain == "llr"))
    fixed_operating_points = _tpr_at_fpr_targets(det["fpr"], det["fnr"], det["thresholds"], fixed_fprs)

    if include_ci:
        eer_point, eer_ci = _eer_confidence_interval(
            scores,
            labels,
            pos_label=args.pos_label,
            n_bootstrap=1000,
            conditions=scenarios,
            random_state=args.seed,
        )
        det_eer = eer_point if eer_point is not None else det["eer"]
    else:
        eer_ci = None
        det_eer = det["eer"]

    act_metrics = _act_metrics_from_mode(
        scores,
        labels,
        det,
        p_target=args.p_target,
        c_miss=args.c_miss,
        c_fa=args.c_fa,
        pos_label=args.pos_label,
        mode=act_threshold_mode,
        threshold_override=threshold_override,
        scores_domain=scores_domain,
    )

    eer_fnr, eer_fpr = _rates(scores, labels, det["eer_threshold"], pos_label=args.pos_label)
    min_fnr, min_fpr = _rates(scores, labels, det["min_dcf_threshold"], pos_label=args.pos_label)

    overall = {
        "eer": float(det_eer) if det_eer is not None else None,
        "eer_ci": eer_ci,
        "eer_threshold": det["eer_threshold"],
        "eer_rates": {
            "fnr": None if math.isnan(eer_fnr) else float(eer_fnr),
            "fpr": None if math.isnan(eer_fpr) else float(eer_fpr),
        },
        "eer_dcf": _dcf(eer_fnr, eer_fpr, p_target=args.p_target, c_miss=args.c_miss, c_fa=args.c_fa),
        "min_dcf": float(det["min_dcf"]),
        "min_dcf_norm": float(det.get("min_dcf_norm")) if det.get("min_dcf_norm") is not None else None,
        "min_dcf_threshold": det["min_dcf_threshold"],
        "min_dcf_rates": {
            "fnr": None if math.isnan(min_fnr) else float(min_fnr),
            "fpr": None if math.isnan(min_fpr) else float(min_fpr),
        },
        "act_dcf": float(act_metrics["act_dcf"]),
        "act_dcf_norm": act_metrics["act_dcf_norm"],
        "act_threshold": act_metrics["act_threshold"],
        "act_threshold_source": act_metrics["act_threshold_source"],
        "act_rates": act_metrics["act_rates"],
        "act_ppv": act_metrics["act_ppv"],
        "act_npv": act_metrics["act_npv"],
        "cllr": float(cllr),
        "tpr_at_fpr": fixed_operating_points,
    }
    return counts, overall


def main():
    args = parse_args()
    config = _select_config(args)

    if not args.checkpoint and not args.scores_in:
        raise ValueError("Provide either --checkpoint for inference or --scores-in to reuse precomputed scores.")

    # Light reproducibility for metrics relying on RNG (e.g., bootstrap CI)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev_arg = getattr(args, "device", None)
    device = torch.device(dev_arg) if dev_arg else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _amp_context(args, device)

    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else None
    scores_in_path = Path(args.scores_in).expanduser() if args.scores_in else None
    checkpoint_name = checkpoint_path.stem if checkpoint_path else (scores_in_path.stem if scores_in_path else "scores")
    # Training run identifier derived from checkpoint folder or scores parent
    if checkpoint_path:
        checkpoint_run = checkpoint_path.parent.name or "root"
    elif scores_in_path:
        checkpoint_run = scores_in_path.parent.name or "scores"
    else:
        checkpoint_run = "scores"

    # Script lives under evaluations/scenarios/<scenario>/; repo root is 3 levels up.
    project_root = Path(__file__).resolve().parents[3]
    if args.output_dir and args.output_dir not in (".", ""):
        base_dir = Path(args.output_dir).expanduser()
    else:
        base_dir = project_root / "eval_runs"

    # Avoid duplicating the run identifier if output_dir already points to it
    if base_dir.name == checkpoint_run:
        run_dir = base_dir
    else:
        run_dir = base_dir / checkpoint_run
    run_dir.mkdir(parents=True, exist_ok=True)
    output_tag = args.output_tag.strip() if args.output_tag else None
    if args.scores_out:
        scores_out_path = Path(args.scores_out).expanduser()
    else:
        scores_out_path = _with_tag(run_dir / "scores.csv", output_tag)
    summary_json_path = _with_tag(run_dir / "summary.json", output_tag)
    summary_txt_path = _with_tag(run_dir / "summary.txt", output_tag)

    # Load model FIRST (if not using precomputed scores) to check if calibration is needed
    model = None
    trainer = None
    if not args.scores_in:
        model, trainer = build_model(args)
        if hasattr(trainer, "set_amp_eval"):
            dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
            trainer.set_amp_eval(args.amp_eval, dtype=dtype)
        
        # Load checkpoint (BaseTrainer has been updated to handle missing keys gracefully)
        trainer.load_model(args.checkpoint)
        model.to(device)
        
        if _maybe_disable_cross_attn_weights_for_eval(model, args):
            print(f"[info] SGE: disabling cross-attention weights for {args.classifier} (need_weights=False) to reduce VRAM.")

    calibration_requested = args.calibrate or args.calibrate_from is not None

    # Check if on-the-fly decoupled calibration is needed
    # This happens if model supports it but is uncalibrated (e.g. fresh from training or old checkpoint)
    onthefly_calibration_needed = False
    if model is not None and hasattr(model, "calibrate"):
        # Check buffer. Using getattr with default True to assume calibrated if flag missing
        if not getattr(model, "is_calibrated", True):
             onthefly_calibration_needed = True
             print("[info] Model flagged as uncalibrated. Forcing DEV set loading for on-the-fly calibration.")

    need_dev_scores = (calibration_requested and args.calibrate_from is None) or onthefly_calibration_needed

    if calibration_requested and args.scores_in and args.calibrate_from is None:
        raise ValueError("Calibration requested with --scores-in requires --calibrate-from (cannot score dev without a checkpoint).")
    if need_dev_scores and args.checkpoint is None and not onthefly_calibration_needed:
        # If onthefly is needed, we definitely have a checkpoint (loaded above).
        raise ValueError("Calibration requested without --calibrate-from requires a checkpoint to score the dev set.")

    # Load dataloaders depending on whether dev scoring is needed
    eval_loader = None
    dev_loader = None
    if need_dev_scores:
        loaders = get_dataloaders(
            dataset=args.dataset,
            config=config,
            lstm=True if "LSTM" in args.classifier else False,
            eval_only=False,
            load_eval=True,
            train_batch_size=args.train_batch_size,
            dev_batch_size=args.dev_batch_size,
            train_num_workers=args.train_num_workers,
            dev_num_workers=args.dev_num_workers,
            train_prefetch_factor=args.train_prefetch_factor,
            dev_prefetch_factor=args.dev_prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )
        train_loader, dev_loader, eval_loader = loaders
        
        # Override dev_loader if specific calibration dataset is requested
        if args.calibration_dataset:
            print(f"[info] Loading explicit calibration dataset: {args.calibration_dataset}")
            calib_loaders = get_dataloaders(
                dataset=args.calibration_dataset,
                config=config, # Ensure compatible config is used (might differ if dataset-specific keys exist, but usually fine)
                lstm=True if "LSTM" in args.classifier else False,
                eval_only=False, # We want the dev split
                load_eval=False, # Don't need eval split of calib dataset
                # Use same perf settings
                train_batch_size=args.train_batch_size,
                dev_batch_size=args.dev_batch_size,
                train_num_workers=args.train_num_workers,
                dev_num_workers=args.dev_num_workers,
                train_prefetch_factor=args.train_prefetch_factor,
                dev_prefetch_factor=args.dev_prefetch_factor,
                pin_memory=args.pin_memory,
                persistent_workers=args.persistent_workers,
            )
            # calib_loaders returns (train, dev, eval)
            _, calib_dev_loader, _ = calib_loaders
            if calib_dev_loader is None:
                 raise ValueError(f"Requested calibration dataset {args.calibration_dataset} does not have a validation split.")
            dev_loader = calib_dev_loader
            print(f"[info] Using {type(dev_loader.dataset).__name__} for calibration.")
        
        assert dev_loader is not None, "Dev loader is required for calibration."
    elif not args.scores_in:
        eval_loader = get_dataloaders(
            dataset=args.dataset,
            config=config,
            lstm=True if "LSTM" in args.classifier else False,
            eval_only=True,
            load_eval=True,
            train_batch_size=args.train_batch_size,
            dev_batch_size=args.dev_batch_size,
            train_num_workers=args.train_num_workers,
            dev_num_workers=args.dev_num_workers,
            train_prefetch_factor=args.train_prefetch_factor,
            dev_prefetch_factor=args.dev_prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )
        assert isinstance(eval_loader, DataLoader)

    # Perform On-the-fly Calibration if needed
    if onthefly_calibration_needed and model is not None and dev_loader is not None:
         print(f"[info] Running on-the-fly decoupled calibration on DEV set ({len(dev_loader)} batches)...")
         model.calibrate(dev_loader, device, max_samples=args.calibration_samples)


    if args.scores_in:
        print(f"[info] Loading scores from {args.scores_in}")
        pair_ids, scores, labels, scenarios = _load_scores_csv(
            args.scores_in, score_column=getattr(args, "scores_in_score_column", "score")
        )
        # Copy scores into run_dir for consistent artifacts if not explicitly overridden
        if args.scores_out is None or Path(args.scores_out).expanduser() != Path(args.scores_in).expanduser():
            _maybe_save(scores_out_path, pair_ids, scores, labels, scenarios)
    else:
        pass # Model already loaded


        if dev_loader is not None and need_dev_scores and not onthefly_calibration_needed:
            dev_scores_path = _with_tag(run_dir / "scores_dev.csv", output_tag)
            print(
                f"[info] Scoring dev set for calibration {args.checkpoint} ({type(model).__name__}) on "
                f"{type(dev_loader.dataset).__name__} | batch_size={dev_loader.batch_size}"
            )
            dev_csv_file, dev_csv_writer = _init_csv_writer(str(dev_scores_path))
            dev_pair_ids, dev_scores, dev_labels, dev_scenarios = _score_loader(
                model, dev_loader, device, amp_ctx, csv_writer=dev_csv_writer, csv_file=dev_csv_file
            )
            if dev_csv_file is not None:
                dev_csv_file.close()
            args.calibrate_from = str(dev_scores_path)

        if eval_loader is not None:
            print(
                f"[info] Scoring {args.checkpoint} ({type(model).__name__}) on "
                f"{type(eval_loader.dataset).__name__} | batch_size={eval_loader.batch_size}"
            )

            csv_file = None
            csv_writer = None
            csv_file, csv_writer = _init_csv_writer(str(scores_out_path))

            pair_ids, scores, labels, scenarios = _score_loader(model, eval_loader, device, amp_ctx, csv_writer=csv_writer, csv_file=csv_file)

            if csv_file is not None:
                csv_file.close()

    scores_raw = scores
    scores_are_llr = bool(args.scores_are_llr)

    # Fit calibration after scores are available (from scores_in or from fresh eval/dev)
    calib_info: dict | None = None
    if calibration_requested and args.calibrate_from:
        print(f"[info] Calibration source (dev scores): {args.calibrate_from}")
        dev_pair_ids, dev_scores, dev_labels, _ = _load_scores_csv(
            args.calibrate_from, score_column=getattr(args, "calibrate_from_score_column", "score")
        )
        a_cal, b_cal = _fit_logistic_calibration(dev_scores, dev_labels)
        dev_llr = a_cal * dev_scores + b_cal
        dev_cllr = _cllr(dev_llr, dev_labels, pos_label=args.pos_label, scores_are_llr=True)
        dev_det = _det_metrics(dev_llr, dev_labels, p_target=args.p_target, c_miss=args.c_miss, c_fa=args.c_fa, pos_label=args.pos_label)
        dev_min_dcf_norm = dev_det.get("min_dcf_norm")
        calib_info = {
            "a": a_cal,
            "b": b_cal,
            "dev_cllr": dev_cllr,
            "dev_min_dcf": dev_det["min_dcf"],
            "dev_min_dcf_norm": None if dev_min_dcf_norm is None else float(dev_min_dcf_norm),
            "dev_min_dcf_thr": dev_det["min_dcf_threshold"],
            "calibration_source": args.calibrate_from,
        }

    if calib_info is not None:
        scores = calib_info["a"] * scores_raw + calib_info["b"]
        scores_are_llr = True
    else:
        scores = scores_raw

    scores_domain = _scores_domain(scores, scores_are_llr)
    act_mode = str(args.act_threshold_mode).lower()
    if act_mode == "bayes" and calib_info is None and not scores_are_llr:
        if scores_domain == "posterior":
            print("[warn] act-threshold-mode=bayes assumes calibrated posteriors; consider enabling --calibrate.")
        else:
            print("[warn] act-threshold-mode=bayes with non-posterior scores; threshold may be meaningless.")
    if act_mode == "sweep" and args.act_threshold is not None:
        print("[warn] --act-threshold provided; overriding sweep threshold for the primary profile.")

    det = _det_metrics(scores, labels, p_target=args.p_target, c_miss=args.c_miss, c_fa=args.c_fa, pos_label=args.pos_label)

    cllr = _cllr(scores, labels, pos_label=args.pos_label, scores_are_llr=scores_are_llr)
    fixed_fprs = DEFAULT_FIXED_FPRS if args.fixed_fprs is None else _parse_float_list(args.fixed_fprs)
    fixed_fprs_list = list(fixed_fprs)
    fixed_operating_points = _tpr_at_fpr_targets(det["fpr"], det["fnr"], det["thresholds"], fixed_fprs_list)

    # Confidence intervals for EER (mirror training logic)
    eer_ci_bootstrap = int(getattr(args, "eer_ci_bootstrap", 1000) or 0)
    if eer_ci_bootstrap > 0:
        eer_point, eer_ci = _eer_confidence_interval(
            scores,
            labels,
            pos_label=args.pos_label,
            n_bootstrap=eer_ci_bootstrap,
            conditions=scenarios,
            random_state=args.seed,
        )
        if eer_point is not None:
            det_eer = eer_point
        else:
            det_eer = det["eer"]
    else:
        eer_ci = None
        det_eer = det["eer"]

    # DCF profiles (primary + optional presets)
    profile_names = _parse_eval_profiles(args.eval_profiles)
    if "primary" not in profile_names:
        profile_names = ["primary", *profile_names]

    profile_metrics: dict[str, dict] = {}
    for name in profile_names:
        p_target, c_miss, c_fa = _resolve_profile(name, args)

        # minDCF for the chosen prior/costs using the already-computed DET curve
        costs = c_miss * p_target * det["fnr"] + c_fa * (1.0 - p_target) * det["fpr"]
        min_idx = int(np.nanargmin(costs))
        min_dcf = float(costs[min_idx])
        min_thr = float(det["thresholds"][min_idx])
        min_fnr, min_fpr = _rates(scores, labels, min_thr, pos_label=args.pos_label)
        threshold_override = float(args.act_threshold) if name == "primary" and args.act_threshold is not None else None
        act_metrics = _act_metrics_from_mode(
            scores,
            labels,
            det,
            p_target=p_target,
            c_miss=c_miss,
            c_fa=c_fa,
            pos_label=args.pos_label,
            mode=act_mode,
            threshold_override=threshold_override,
            scores_domain=scores_domain,
        )

        min_dcf_norm = _dcf_norm(min_dcf, p_target, c_miss, c_fa)
        act_dcf_norm = act_metrics["act_dcf_norm"]

        profile_metrics[name] = {
            "p_target": p_target,
            "c_miss": c_miss,
            "c_fa": c_fa,
            "bayes_threshold": act_metrics["bayes_threshold"],
            "bayes_threshold_llr": act_metrics["bayes_threshold_llr"],
            "bayes_threshold_lr": act_metrics["bayes_threshold_lr"],
            "min_dcf": min_dcf,
            "min_dcf_norm": min_dcf_norm,
            "min_dcf_threshold": min_thr,
            "min_dcf_rates": {
                "fnr": None if math.isnan(min_fnr) else float(min_fnr),
                "fpr": None if math.isnan(min_fpr) else float(min_fpr),
            },
            "act_dcf": float(act_metrics["act_dcf"]),
            "act_dcf_norm": act_dcf_norm,
            "act_threshold": act_metrics["act_threshold"],
            "act_threshold_source": act_metrics["act_threshold_source"],
            "act_rates": act_metrics["act_rates"],
            "act_ppv": act_metrics["act_ppv"],
            "act_npv": act_metrics["act_npv"],
        }

    primary_metrics = profile_metrics["primary"]
    act_thr = float(primary_metrics["act_threshold"])
    act_dcf = float(primary_metrics["act_dcf"])
    act_fnr = primary_metrics["act_rates"]["fnr"]
    act_fpr = primary_metrics["act_rates"]["fpr"]

    eer_fnr, eer_fpr = _rates(scores, labels, det["eer_threshold"], pos_label=args.pos_label)
    min_fnr, min_fpr = _rates(scores, labels, det["min_dcf_threshold"], pos_label=args.pos_label)

    scen_metrics = _per_scenario_metrics(
        scenarios, scores, labels, act_threshold=act_thr, pos_label=args.pos_label, p_target=args.p_target, c_miss=args.c_miss, c_fa=args.c_fa
    )

    safe_scenarios = []
    for m in scen_metrics:
        fpr_val = None
        if m["fpr"] is not None:
            fpr_val = None if math.isnan(m["fpr"]) else float(m["fpr"])
        tpr_val = None
        if m["tpr"] is not None:
            tpr_val = None if math.isnan(m["tpr"]) else float(m["tpr"])
        safe_scenarios.append(
            {
                "scenario": str(m["scenario"]),
                "count": int(m["count"]),
                "eer": None if m["eer"] is None else float(m["eer"]),
                "fpr": fpr_val,
                "tpr": tpr_val,
            }
        )

    snorm_summary: dict | None = None
    if args.snorm:
        if args.snorm_cohort_embeddings is None:
            raise ValueError("--snorm-cohort-embeddings is required when --snorm is enabled.")
        if args.scores_in and args.snorm_eval_embeddings is None:
            raise ValueError("--snorm-eval-embeddings is required when using --scores-in with --snorm.")

        cohort_ids, cohort_emb = _load_embedding_table(args.snorm_cohort_embeddings)
        cohort_emb = _normalize_embeddings(cohort_emb.astype(np.float32))
        cohort_total = cohort_emb.shape[0]
        cohort_subsampled = False
        if args.snorm_cohort_max and cohort_total > args.snorm_cohort_max:
            rng = np.random.default_rng(args.snorm_cohort_seed)
            idx = rng.choice(cohort_total, size=args.snorm_cohort_max, replace=False)
            cohort_emb = cohort_emb[idx]
            cohort_subsampled = True

        if args.snorm_eval_embeddings:
            eval_ids, eval_emb = _load_embedding_table(args.snorm_eval_embeddings)
            eval_map = _embedding_map(eval_ids, eval_emb)
        else:
            if model is None or eval_loader is None:
                raise ValueError("S-norm requires a checkpoint and eval loader when --snorm-eval-embeddings is omitted.")
            eval_map = _extract_eval_embeddings(model, eval_loader, device, amp_ctx)

        emb_a, emb_b, snorm_mask = _pair_embeddings_from_map(pair_ids, eval_map)
        excluded = int(snorm_mask.size - snorm_mask.sum())
        if emb_a.size == 0:
            print("[warn] S-norm: no pairs had embeddings; skipping.")
        else:
            emb_a = _normalize_embeddings(emb_a.astype(np.float32))
            emb_b = _normalize_embeddings(emb_b.astype(np.float32))

            scale = 1.0
            bias = 0.0
            score_transform = "cosine"
            if model is not None and hasattr(model, "scale") and hasattr(model, "bias"):
                try:
                    scale = float(model.scale)
                    bias = float(model.bias)
                    score_transform = "cosine_scale_bias"
                except Exception:
                    pass

            _, snorm_scores = _snorm_scores(
                emb_a,
                emb_b,
                cohort_emb,
                scale=scale,
                bias=bias,
                eps=float(args.snorm_eps),
                batch_size=int(args.snorm_batch_size),
            )

            counts_snorm, overall_snorm = _compute_label_overall(
                snorm_scores,
                labels[snorm_mask],
                scenarios[snorm_mask],
                args,
                fixed_fprs=fixed_fprs_list,
                scores_domain="raw",
                act_threshold_mode="sweep",
                threshold_override=args.act_threshold if args.act_threshold is not None else None,
                include_ci=False,
                excluded=excluded,
            )
            snorm_summary = {
                "counts": counts_snorm,
                "overall": overall_snorm,
                "score_transform": score_transform,
                "cohort": {
                    "path": str(Path(args.snorm_cohort_embeddings).expanduser()),
                    "size": int(cohort_emb.shape[0]),
                    "total": int(cohort_total),
                    "subsampled": cohort_subsampled,
                },
                "eval_embeddings": None if args.snorm_eval_embeddings is None else str(Path(args.snorm_eval_embeddings).expanduser()),
            }
            print(
                f"[info] S-norm: cohort={cohort_emb.shape[0]} | score={score_transform} | "
                f"pairs_used={counts_snorm['total']}"
            )

    extra_label_metrics: dict[str, dict] = {}
    label_protocol_df = None
    if args.label_protocol:
        print(f"[info] Loading label protocol from {args.label_protocol}")
        label_protocol_df = _load_label_protocol(args.label_protocol)
    elif eval_loader is not None:
        label_protocol_df = getattr(eval_loader.dataset, "protocol_df", None)

    label_sets = _extract_label_sets(pair_ids, label_protocol_df)
    label_sets.pop("same_model", None)
    for label_name, payload in label_sets.items():
        label_values = payload["labels"]
        label_mask = payload["mask"]
        excluded = int(label_mask.size - label_mask.sum())
        if label_mask.any():
            counts, overall = _compute_label_overall(
                scores[label_mask],
                label_values[label_mask],
                scenarios[label_mask],
                args,
                fixed_fprs=fixed_fprs_list,
                scores_domain=scores_domain,
                act_threshold_mode=act_mode,
                threshold_override=args.act_threshold if args.act_threshold is not None else None,
                include_ci=False,
                excluded=excluded,
            )
        else:
            counts = {"total": 0, "positive": 0, "negative": 0, "excluded": excluded}
            overall = _empty_overall()
        extra_label_metrics[label_name] = {"counts": counts, "overall": overall}
    if label_protocol_df is not None and not extra_label_metrics:
        print("[info] No extra label columns found for multi-level eval.")

    summary = {
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_name": checkpoint_name,
        "checkpoint_run": checkpoint_run,
        "checkpoint_dir": None if checkpoint_path is None else str(checkpoint_path.parent),
        "dataset": args.dataset,
        "extractor": args.extractor,
        "processor": args.processor,
        "classifier": args.classifier,
        "seed": args.seed,
        "pos_label": args.pos_label,
        "p_target": args.p_target,
        "c_miss": args.c_miss,
        "c_fa": args.c_fa,
        "act_threshold_mode": act_mode,
        "counts": {
            "total": int(scores.size),
            "positive": int(np.sum(labels == args.pos_label)),
            "negative": int(np.sum(labels != args.pos_label)),
        },
        "overall": {
            "eer": float(det_eer) if det_eer is not None else None,
            "eer_ci": eer_ci,
            "eer_threshold": det["eer_threshold"],
            "eer_rates": {
                "fnr": None if math.isnan(eer_fnr) else float(eer_fnr),
                "fpr": None if math.isnan(eer_fpr) else float(eer_fpr),
            },
            "eer_dcf": _dcf(eer_fnr, eer_fpr, p_target=args.p_target, c_miss=args.c_miss, c_fa=args.c_fa),
            "min_dcf": float(det["min_dcf"]),
            "min_dcf_norm": float(det.get("min_dcf_norm")) if det.get("min_dcf_norm") is not None else None,
            "min_dcf_threshold": det["min_dcf_threshold"],
            "min_dcf_rates": {
                "fnr": None if math.isnan(min_fnr) else float(min_fnr),
                "fpr": None if math.isnan(min_fpr) else float(min_fpr),
            },
            "act_dcf": float(act_dcf),
            "act_dcf_norm": float(primary_metrics.get("act_dcf_norm")) if primary_metrics.get("act_dcf_norm") is not None else None,
            "act_threshold": act_thr,
            "act_threshold_source": primary_metrics.get("act_threshold_source"),
            "act_rates": {
                "fnr": act_fnr,
                "fpr": act_fpr,
            },
            "act_ppv": primary_metrics.get("act_ppv"),
            "act_npv": primary_metrics.get("act_npv"),
            "cllr": float(cllr),
            "tpr_at_fpr": fixed_operating_points,
        },
        "scenarios": safe_scenarios,
        "dcf_profiles": profile_metrics,
    }
    if calib_info is not None:
        summary["calibration"] = calib_info
    if extra_label_metrics:
        label_metrics = {"same_model": {"counts": summary["counts"], "overall": summary["overall"]}}
        label_metrics.update(extra_label_metrics)
        summary["label_metrics"] = label_metrics
    if snorm_summary is not None:
        summary["snorm"] = snorm_summary

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable summary
    act_source = primary_metrics.get("act_threshold_source")
    if act_source == "sweep":
        act_label = "actDCF (sweep)"
    elif act_source == "manual":
        act_label = "actDCF (manual)"
    elif act_source in {"bayes_llr", "bayes_posterior"}:
        act_label = "actDCF (bayes)"
    else:
        act_label = "actDCF"

    lines = [
        f"Run: {checkpoint_run}",
        f"Checkpoint: {checkpoint_path or args.scores_in}",
        f"Dataset: {args.dataset}",
        "",
        f"EER: {det_eer*100:.2f}% (thr={det['eer_threshold']:.4f})",
        f"EER 95% CI: [{eer_ci[0]*100:.2f}%, {eer_ci[1]*100:.2f}%]" if eer_ci else "EER 95% CI: n/a",
        f"minDCF: {det['min_dcf']:.4f} (thr={det['min_dcf_threshold']:.4f}, norm={det['min_dcf_norm']:.4f})",
        f"{act_label}: {act_dcf:.4f} (thr={act_thr:.4f}, norm={primary_metrics['act_dcf_norm']:.4f})",
        f"C_llr: {cllr:.4f}",
        "Fixed-FPR operating points (TPR@FPR):",
    ]
    for op in fixed_operating_points:
        tpr_str = "n/a" if op["tpr"] is None else f"{op['tpr']*100:.2f}%"
        fpr_str = "n/a" if op["fpr"] is None else f"{op['fpr']*100:.3f}%"
        prefix = "<=" if op.get("met_target") else "≈"
        lines.append(
            f"  FPR{prefix}{op['fpr_target']*100:.2f}% (achieved {fpr_str}) -> TPR={tpr_str} (thr={op['threshold']})"
        )

    extra_profiles = [p for p in profile_names if p != "primary"]
    if extra_profiles:
        lines += [
            "",
            "DCF profiles:",
        ]
        for name in extra_profiles:
            pm = profile_metrics[name]
            act_src = pm.get("act_threshold_source")
            if act_src in {"bayes_llr", "bayes_posterior"}:
                lr = pm.get("bayes_threshold_lr")
                lr_str = "n/a" if lr is None else f"{lr:.2f}"
                act_note = f"(LR={lr_str})"
            elif act_src == "manual":
                act_note = "(manual)"
            elif act_src == "sweep":
                act_note = "(sweep)"
            else:
                act_note = ""
            act_note_str = f" {act_note}" if act_note else ""
            lines.append(
                f"  {name}: p_target={pm['p_target']:.4f} | c_miss={pm['c_miss']:.2f} | c_fa={pm['c_fa']:.2f} | "
                f"act_thr={pm['act_threshold']:.4f}{act_note_str} | actDCF={pm['act_dcf']:.4f} | minDCF={pm['min_dcf']:.4f}"
            )
    if snorm_summary is not None:
        sn_overall = snorm_summary["overall"]
        sn_counts = snorm_summary["counts"]
        sn_cohort = snorm_summary.get("cohort", {})
        sn_size = sn_cohort.get("size")
        sn_size_str = "" if sn_size is None else f" (cohort={sn_size})"
        sn_eer = "n/a" if sn_overall["eer"] is None else f"{sn_overall['eer']*100:.2f}%"
        sn_min = "n/a" if sn_overall["min_dcf"] is None else f"{sn_overall['min_dcf']:.4f}"
        sn_act = "n/a" if sn_overall["act_dcf"] is None else f"{sn_overall['act_dcf']:.4f}"
        sn_excl = sn_counts.get("excluded")
        sn_excl_str = "" if not sn_excl else f", excluded={sn_excl}"
        lines += [
            "",
            f"S-norm (embedding cosine){sn_size_str}:",
            f"  n={sn_counts['total']} (pos={sn_counts['positive']}, neg={sn_counts['negative']}{sn_excl_str}) | "
            f"EER={sn_eer} | minDCF={sn_min} | actDCF={sn_act}",
        ]
    if extra_label_metrics:
        lines += [
            "",
            "Additional label evaluations:",
        ]
        for name, metrics in extra_label_metrics.items():
            counts = metrics["counts"]
            overall = metrics["overall"]
            eer_str = "n/a" if overall["eer"] is None else f"{overall['eer']*100:.2f}%"
            min_dcf_str = "n/a" if overall["min_dcf"] is None else f"{overall['min_dcf']:.4f}"
            act_dcf_str = "n/a" if overall["act_dcf"] is None else f"{overall['act_dcf']:.4f}"
            excluded = counts.get("excluded")
            excluded_str = "" if not excluded else f", excluded={excluded}"
            lines.append(
                f"  {name}: n={counts['total']} (pos={counts['positive']}, neg={counts['negative']}{excluded_str}) | "
                f"EER={eer_str} | minDCF={min_dcf_str} | actDCF={act_dcf_str}"
            )
    lines += [
        "",
        "Per-scenario:",
    ]
    for m in scen_metrics:
        eer_str = f"{m['eer']*100:.2f}%" if m["eer"] is not None else "n/a"
        fpr_str = "n/a" if m["fpr"] is None else f"{m['fpr']*100:.2f}%"
        tpr_str = "n/a" if m["tpr"] is None else f"{m['tpr']*100:.2f}%"
        lines.append(f"  {m['scenario']}: n={m['count']} | EER={eer_str} | FPR={fpr_str} | TPR={tpr_str} @ thr={act_thr:.4f}")
    if calib_info is not None:
        lines += [
            "",
            f"Calibration (dev): a={calib_info['a']:.6f}, b={calib_info['b']:.6f}, dev_C_llr={calib_info['dev_cllr']:.4f}, "
            f"dev_minDCF={calib_info['dev_min_dcf']:.4f} (norm={calib_info.get('dev_min_dcf_norm')}, thr={calib_info['dev_min_dcf_thr']:.4f})",
        ]
    summary_txt = "\n".join(lines)
    summary_txt_path.write_text(summary_txt)

    print(summary_txt)
    print(f"[done] Wrote scores to {scores_out_path}")
    print(f"[done] Summary saved to {summary_json_path} and {summary_txt_path}")


if __name__ == "__main__":
    main()
