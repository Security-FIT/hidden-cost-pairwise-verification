#!/usr/bin/env python3
"""
Evaluate a pairwise model on STOPA (nc-1 reference vs trials) with ATK/AM/VM labels.
Reports pooled metrics plus known-only and unknown-only negative splits.
"""

from __future__ import annotations

import csv
import json
import os
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import det_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import build_model, get_dataloaders
from config import karolina_config, local_config, sge_config
from datasets.STOPA import _coerce_label, _find_attack_id, _normalize_trial_path
from parse_arguments import parse_args


DEFAULT_FIXED_FPRS = (0.0001, 0.001, 0.01, 0.05)


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


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _canonicalize_path(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


def _canonicalize_pair_id(pid: str) -> str:
    if "|" in pid:
        a, b = pid.split("|", 1)
        return f"{_canonicalize_path(a)}|{_canonicalize_path(b)}"
    return _canonicalize_path(pid)


def _split_pair_id(pid: str) -> tuple[str, str]:
    if "|" in pid:
        a, b = pid.split("|", 1)
        return a, b
    return pid, ""


def _det_metrics(scores: np.ndarray, labels: np.ndarray, p_target: float, c_miss: float, c_fa: float, pos_label: int):
    fpr, fnr, thresholds = det_curve(labels, scores, pos_label=pos_label)
    diff = np.abs(fnr - fpr)
    eer_idx = int(np.nanargmin(diff))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_thr = float(thresholds[eer_idx])

    costs = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    min_idx = int(np.nanargmin(costs))
    min_dcf = float(costs[min_idx])
    min_dcf_thr = float(thresholds[min_idx])
    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "min_dcf": min_dcf,
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
            results.append({"fpr_target": target, "fpr": None, "tpr": None, "threshold": None})
            continue

        finite = np.isfinite(fpr) & np.isfinite(tpr) & np.isfinite(thresholds)
        if not np.any(finite):
            results.append({"fpr_target": target, "fpr": None, "tpr": None, "threshold": None})
            continue

        fpr_f = fpr[finite]
        tpr_f = tpr[finite]
        thr_f = thresholds[finite]
        idx = int(np.argmin(np.abs(fpr_f - target)))
        results.append(
            {
                "fpr_target": target,
                "fpr": float(fpr_f[idx]),
                "tpr": float(tpr_f[idx]),
                "threshold": float(thr_f[idx]),
            }
        )
    return results


def _score_loader(model, dataloader: DataLoader, device: torch.device, amp_ctx):
    model.eval()
    all_scores = []
    all_pair_ids: list[str] = []
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Scoring", unit="batch")
        for pair_ids, gt, test, _ in iterator:
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
            all_pair_ids.extend(list(pair_ids))
    return np.asarray(all_pair_ids), np.asarray(all_scores, dtype=float)


def _load_scores_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    pair_ids = []
    scores = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "pair_id" in row:
                pid = row["pair_id"]
            else:
                pid = f"{row.get('pathA','')}|{row.get('pathB','')}"
            pair_ids.append(pid)
            scores.append(float(row["score"]))
    return np.asarray(pair_ids), np.asarray(scores, dtype=float)


def _collect_metadata(
    pair_ids: np.ndarray,
    dataset,
    fallback_meta: dict[str, dict[str, object]] | None = None,
) -> dict[str, np.ndarray]:
    raw_map = getattr(dataset, "pair_metadata_by_id", {})
    meta_map = {_canonicalize_pair_id(k): v for k, v in raw_map.items()}
    if fallback_meta:
        for key, value in fallback_meta.items():
            canon = _canonicalize_pair_id(key)
            if canon not in meta_map:
                meta_map[canon] = value

    labels_atk = []
    labels_am = []
    labels_vm = []
    unknown_flags = []
    hyp_attacks = []
    trial_attacks = []
    hyp_wavs = []
    trial_wavs = []

    for pid in pair_ids:
        key = _canonicalize_pair_id(pid)
        meta = meta_map.get(key)
        if meta is None:
            raise KeyError(f"Missing STOPA metadata for pair_id: {pid}")
        labels_atk.append(int(meta["label_atk"]))
        labels_am.append(int(meta["label_am"]))
        labels_vm.append(int(meta["label_vm"]))
        unknown_flags.append(int(meta["trial_is_unknown"]))
        hyp_attacks.append(str(meta.get("hyp_attack_id", "")))
        trial_attacks.append(str(meta.get("trial_attack_id", "")))
        hyp_wavs.append(str(meta.get("hyp_wav", "")))
        trial_wavs.append(str(meta.get("trial_wav", "")))

    return {
        "label_atk": np.asarray(labels_atk, dtype=int),
        "label_am": np.asarray(labels_am, dtype=int),
        "label_vm": np.asarray(labels_vm, dtype=int),
        "trial_is_unknown": np.asarray(unknown_flags, dtype=int),
        "hyp_attack_id": np.asarray(hyp_attacks, dtype=object),
        "trial_attack_id": np.asarray(trial_attacks, dtype=object),
        "hyp_wav": np.asarray(hyp_wavs, dtype=object),
        "trial_wav": np.asarray(trial_wavs, dtype=object),
    }


def _infer_stopa_protocol_from_scores(scores_in: str | Path, project_root: Path) -> Path | None:
    env_override = os.environ.get("STOPA_TEST_PROTOCOL") or os.environ.get("TEST_PROTOCOL")
    if env_override:
        path = Path(env_override).expanduser()
        return path if path.is_absolute() else (project_root / path)

    name = Path(scores_in).name
    match = re.search(r"_([0-9]+)\.scores\.csv$", name)
    if not match:
        return None
    run_id = match.group(1)
    candidate = project_root / "tmp" / f"stopa_refset_protocols_{run_id}" / "trials.csv"
    return candidate if candidate.exists() else None


def _build_metadata_from_refset_protocol(test_protocol: Path, dataset) -> dict[str, dict[str, object]]:
    df = pd.read_csv(test_protocol)
    cols = {c.lower(): c for c in df.columns}

    def _pick_col(candidates: Iterable[str]) -> str | None:
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    claim_col = _pick_col(("claim_id", "hyp_attack_id", "hyp_attack", "attack_id_hyp"))
    query_col = _pick_col(("query_path", "trial_path", "trial_wav", "path_b", "pathb", "path", "filename"))
    trial_attack_col = _pick_col(("query_model_id", "trial_attack_id", "trial_attack", "attack_id_trial"))
    label_col = _pick_col(("label", "label_atk", "same_atk", "istargetatk"))

    ref_map = getattr(dataset, "ref_map", {})
    attack_metadata = getattr(dataset, "attack_metadata", {})
    known_attacks = set(getattr(dataset, "known_attacks", ()))

    meta_map: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        hyp_attack = str(row[claim_col]) if claim_col else ""
        trial_path_raw = str(row[query_col]) if query_col else ""
        if not hyp_attack or hyp_attack == "nan":
            hyp_attack = _find_attack_id(trial_path_raw) or ""
        if not hyp_attack:
            raise KeyError(f"Missing hypothesis attack id in {test_protocol}.")

        trial_attack = ""
        if trial_attack_col:
            trial_attack = str(row[trial_attack_col])
        if not trial_attack or trial_attack == "nan":
            trial_attack = _find_attack_id(trial_path_raw) or ""
        if not trial_attack:
            trial_attack = "unknown"

        hyp_wav = ref_map.get(hyp_attack)
        if not hyp_wav:
            raise KeyError(f"Missing TEE ref for attack {hyp_attack} while reading {test_protocol}.")

        trial_rel = _normalize_trial_path(trial_path_raw)
        pair_id = f"{_canonicalize_path(hyp_wav)}|{_canonicalize_path(trial_rel)}"

        label_atk = _coerce_label(row[label_col]) if label_col else None
        if label_atk is None and trial_attack:
            label_atk = int(trial_attack == hyp_attack)

        hyp_meta = attack_metadata.get(hyp_attack, {})
        trial_meta = attack_metadata.get(trial_attack, {})
        label_am = None
        label_vm = None
        if hyp_meta and trial_meta:
            hyp_am = hyp_meta.get("am")
            hyp_vm = hyp_meta.get("vm")
            trial_am = trial_meta.get("am")
            trial_vm = trial_meta.get("vm")
            if hyp_am and trial_am:
                label_am = int(trial_am == hyp_am)
            if hyp_vm and trial_vm:
                label_vm = int(trial_vm == hyp_vm)

        if label_atk is None or label_am is None or label_vm is None:
            raise KeyError(
                f"Unable to derive STOPA AM/VM labels for pair {pair_id} from {test_protocol}."
            )

        trial_is_unknown = int(trial_attack not in known_attacks) if trial_attack else 1
        meta_map[pair_id] = {
            "pair_id": pair_id,
            "hyp_attack_id": hyp_attack,
            "hyp_wav": _canonicalize_path(hyp_wav),
            "trial_attack_id": trial_attack,
            "trial_wav": _canonicalize_path(trial_rel),
            "label_atk": int(label_atk),
            "label_am": int(label_am),
            "label_vm": int(label_vm),
            "trial_is_unknown": int(trial_is_unknown),
        }
    return meta_map


def _compute_metrics(scores: np.ndarray, labels: np.ndarray, p_target: float, c_miss: float, c_fa: float, pos_label: int, fixed_fprs: Iterable[float]):
    has_pos = np.any(labels == pos_label)
    has_neg = np.any(labels != pos_label)
    if not (has_pos and has_neg):
        return {
            "eer": None,
            "min_dcf": None,
            "auc": None,
            "tpr_at_fpr": [],
            "num_pos": int(np.sum(labels == pos_label)),
            "num_neg": int(np.sum(labels != pos_label)),
        }
    det = _det_metrics(scores, labels, p_target, c_miss, c_fa, pos_label)
    auc = float(roc_auc_score(labels == pos_label, scores))
    tpr_points = _tpr_at_fpr_targets(det["fpr"], det["fnr"], det["thresholds"], fixed_fprs)
    return {
        "eer": det["eer"],
        "min_dcf": det["min_dcf"],
        "auc": auc,
        "tpr_at_fpr": tpr_points,
        "num_pos": int(np.sum(labels == pos_label)),
        "num_neg": int(np.sum(labels != pos_label)),
    }


def _metrics_by_split(scores: np.ndarray, labels: np.ndarray, unknown_mask: np.ndarray, args, fixed_fprs):
    splits = {
        "all": np.ones_like(labels, dtype=bool),
        "known": unknown_mask == 0,
        "unknown": unknown_mask == 1,
    }
    results = {}
    for name, mask in splits.items():
        if np.sum(mask) == 0:
            results[name] = {
                "eer": None,
                "min_dcf": None,
                "auc": None,
                "tpr_at_fpr": [],
                "num_pos": 0,
                "num_neg": 0,
            }
            continue
        results[name] = _compute_metrics(
            scores[mask],
            labels[mask],
            p_target=args.p_target,
            c_miss=args.c_miss,
            c_fa=args.c_fa,
            pos_label=args.pos_label,
            fixed_fprs=fixed_fprs,
        )
    return results


def _save_scores(path: Path, scores: np.ndarray, metadata: dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "hyp_attack_id",
                "hyp_wav",
                "trial_attack_id",
                "trial_wav",
                "score",
                "label_atk",
                "label_am",
                "label_vm",
                "trial_is_unknown",
            ]
        )
        for idx in range(scores.shape[0]):
            writer.writerow(
                [
                    metadata["hyp_attack_id"][idx],
                    metadata["hyp_wav"][idx],
                    metadata["trial_attack_id"][idx],
                    metadata["trial_wav"][idx],
                    float(scores[idx]),
                    int(metadata["label_atk"][idx]),
                    int(metadata["label_am"][idx]),
                    int(metadata["label_vm"][idx]),
                    int(metadata["trial_is_unknown"][idx]),
                ]
            )


def main():
    args = parse_args()
    config = _select_config(args)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _amp_context(args, device)

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

    dataset = eval_loader.dataset

    checkpoint_name = Path(args.checkpoint).stem if args.checkpoint else f"stopa_eval_{_timestamp()}"
    project_root = Path(__file__).resolve().parents[3]
    base_dir = Path(args.output_dir).expanduser() if args.output_dir not in (".", "") else project_root / "eval_runs"
    run_dir = base_dir / checkpoint_name
    run_dir.mkdir(parents=True, exist_ok=True)
    scores_out = Path(args.scores_out).expanduser() if args.scores_out else run_dir / "scores_stopa.csv"
    summary_json = run_dir / "summary_stopa.json"
    summary_txt = run_dir / "summary_stopa.txt"

    stopa_protocol = None
    stopa_fallback_meta = None
    if args.scores_in:
        pair_ids, scores = _load_scores_csv(args.scores_in)
        if "STOPA" in args.dataset:
            if args.eval_protocol:
                proto_path = Path(args.eval_protocol).expanduser()
                if not proto_path.is_absolute():
                    proto_path = project_root / proto_path
                stopa_protocol = proto_path
            if stopa_protocol is None:
                stopa_protocol = _infer_stopa_protocol_from_scores(args.scores_in, project_root)
            if stopa_protocol and stopa_protocol.exists():
                stopa_fallback_meta = _build_metadata_from_refset_protocol(stopa_protocol, dataset)
    else:
        if args.checkpoint is None:
            raise ValueError("Checkpoint is required when --scores-in is not provided.")
        model, trainer = build_model(args)
        if hasattr(trainer, "set_amp_eval"):
            dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
            trainer.set_amp_eval(args.amp_eval, dtype=dtype)
        trainer.load_model(args.checkpoint)
        model.to(device)
        print(
            f"[info] Scoring {args.checkpoint} ({type(model).__name__}) on "
            f"{type(dataset).__name__} | batch_size={eval_loader.batch_size}"
        )
        pair_ids, scores = _score_loader(model, eval_loader, device, amp_ctx)

    metadata = _collect_metadata(pair_ids, dataset, fallback_meta=stopa_fallback_meta)
    _save_scores(scores_out, scores, metadata)

    fixed_fprs = DEFAULT_FIXED_FPRS if args.fixed_fprs is None else [float(x.strip().rstrip("%")) / (100 if "%" in x else 1) for x in args.fixed_fprs.split(",") if x.strip()]

    summary = {
        "dataset": type(dataset).__name__,
        "num_pairs": int(scores.shape[0]),
        "metrics": {
            "atk": _metrics_by_split(scores, metadata["label_atk"], metadata["trial_is_unknown"], args, fixed_fprs),
            "am": _metrics_by_split(scores, metadata["label_am"], metadata["trial_is_unknown"], args, fixed_fprs),
            "vm": _metrics_by_split(scores, metadata["label_vm"], metadata["trial_is_unknown"], args, fixed_fprs),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        f"STOPA eval summary ({type(dataset).__name__})",
        f"pairs: {summary['num_pairs']}",
    ]
    for label_key, label_metrics in summary["metrics"].items():
        lines.append(f"{label_key.upper()} metrics:")
        for split_name, metrics in label_metrics.items():
            lines.append(
                f"  {split_name}: EER={metrics['eer']} | minDCF={metrics['min_dcf']} | AUC={metrics['auc']} "
                f"| pos={metrics['num_pos']} neg={metrics['num_neg']}"
            )
    summary_txt.write_text("\n".join(lines))
    print(f"[info] Wrote scores to {scores_out}")
    print(f"[info] Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
