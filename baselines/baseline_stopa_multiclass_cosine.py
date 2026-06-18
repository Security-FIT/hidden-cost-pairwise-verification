#!/usr/bin/env python3
"""
STOPA baseline: multiclass FF embeddings + cosine verification.

Scores each trial against R enrollment references for the hypothesized attack
and aggregates with max/mean cosine (R=1 or R=5).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import det_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import build_model
from config import karolina_config, local_config, sge_config
from datasets.utils import custom_single_batch_create
from trainers.utils import calculate_EER


KNOWN_ATTACKS_DEFAULT = ("AA01", "AA03", "AA05", "AA07", "AA10")
_ATTACK_RE = re.compile(r"AA\d+")


@dataclass
class Trial:
    test_path: str
    ref_paths: list[str]
    label_atk: int
    label_am: int
    label_vm: int
    ref_attack: str
    trial_attack: str | None
    trial_is_unknown: int


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

    def __len__(self) -> int:
        return len(self.paths)

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
        description="STOPA baseline: FF multiclass embeddings + cosine scoring."
    )

    env = parser.add_mutually_exclusive_group()
    env.add_argument("--karolina", action="store_true", help="Use Karolina paths/config.")
    env.add_argument("--metacentrum", action="store_true", help="Use MetaCentrum paths/config.")
    env.add_argument("--sge", action="store_true", help="Use SGE paths/config.")
    env.add_argument("--local", action="store_true", help="Use local paths/config (default).")

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

    protocol_group = parser.add_argument_group("STOPA protocol selection")
    protocol_group.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="STOPA root directory (defaults to config data_dir/STOPA).",
    )
    protocol_group.add_argument(
        "--tee-protocols",
        type=Path,
        default=None,
        help="TEE protocols directory or file (default: TEE/TEE_protocols).",
    )
    protocol_group.add_argument(
        "--tee-suffix",
        type=str,
        default="-nc-1.txt",
        help=(
            "Only load TEE protocol files ending with this suffix "
            "(default: -nc-1.txt). Set empty to include all."
        ),
    )
    protocol_group.add_argument(
        "--trials-protocols",
        type=Path,
        default=None,
        help="Trials protocols directory or file (default: Trials/protocols).",
    )
    protocol_group.add_argument(
        "--trials-suffix",
        type=str,
        default="-nc-1_trials.txt",
        help="Only load trial protocol files ending with this suffix (default: -nc-1_trials.txt).",
    )
    protocol_group.add_argument(
        "--known-attacks",
        type=str,
        default=",".join(KNOWN_ATTACKS_DEFAULT),
        help="Comma-separated known attack IDs (default: AA01,AA03,AA05,AA07,AA10).",
    )
    protocol_group.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional attacks.json for AM/VM labels (default: metadata/attacks.json).",
    )
    protocol_group.add_argument(
        "--reference-size",
        type=int,
        default=1,
        help="Number of references per attack (R=1 or R=5).",
    )
    protocol_group.add_argument(
        "--score-reduction",
        choices=("max", "mean"),
        default="max",
        help="Reduction over reference-set cosines (default: max).",
    )
    protocol_group.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    parser.add_argument("--batch-size", type=int, default=12, help="Eval batch size (default: 12).")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers (default: 8).")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor (default: 4).")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Optional fixed segment length (seconds).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for segmenting (default: 16000).",
    )
    parser.add_argument(
        "--amp-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run embedding under torch.cuda.amp autocast.",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=("bf16", "fp16"),
        default="bf16",
        help="Autocast dtype to use when AMP is enabled (bf16 or fp16).",
    )
    parser.add_argument(
        "--skip-bad-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip paths that fail to load with soundfile.",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=None,
        help="Optional path to cache embeddings (npz with utt_ids/embeddings).",
    )
    parser.add_argument(
        "--allow-stale-embeddings-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse an existing embeddings cache even if it was built for a different "
            "checkpoint/config."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV for scores (default: scores_stopa_ref{R}.csv in cwd).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("."),
        help="Directory for summary JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override for evaluation (e.g., cuda:0, cpu). Defaults to auto.",
    )
    parser.add_argument(
        "--p-target",
        type=float,
        default=0.5,
        help="Target prior for minDCF.",
    )
    parser.add_argument(
        "--c-miss",
        type=float,
        default=1.0,
        help="Miss cost for minDCF.",
    )
    parser.add_argument(
        "--c-fa",
        type=float,
        default=1.0,
        help="False-alarm cost for minDCF.",
    )
    parser.add_argument(
        "--fixed-fprs",
        nargs="+",
        type=float,
        default=[0.0001, 0.001, 0.01, 0.05],
        help="Fixed FPR points for TPR@FPR reporting.",
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


def _canonicalize_relpath(path: str) -> str:
    return path.lstrip("./")


def _find_attack_id(text: str) -> str | None:
    match = _ATTACK_RE.search(text)
    return match.group(0) if match else None


def _normalize_trial_path(path: str) -> str:
    if not path:
        return path
    normalized = _canonicalize_relpath(path)
    if "/" not in normalized and "\\" not in normalized:
        return f"Trials/wav/{normalized}"
    return normalized


def _normalize_tee_path(path: str) -> str:
    if not path:
        return path
    normalized = _canonicalize_relpath(path)
    if "/" not in normalized and "\\" not in normalized:
        return f"TEE/wav/{normalized}"
    return normalized


def _coerce_label(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(round(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "same", "target", "pos", "positive"}:
        return 1
    if text in {"0", "false", "no", "n", "different", "nontarget", "neg", "negative"}:
        return 0
    return None


def _read_protocol_df(path: str, expect_header: bool = True) -> tuple[pd.DataFrame, bool]:
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    if not expect_header:
        return df, False
    expected = {
        "attack", "attack_id", "hyp_attack_id", "trial_attack_id",
        "path", "wav", "trial_path", "trial_wav", "path_a", "path_b",
        "same_atk", "same_am", "same_vm", "label_atk", "label_am", "label_vm",
        "abstractmodel", "filename", "atk",
        "istargetatk", "istargetacousticmodel", "istargetvocodermodel",
    }
    cols = {str(c).strip().lower() for c in df.columns}
    if cols & expected:
        return df, True
    df = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
    return df, False


def _iter_protocol_files(path: str, suffix: str | None = None) -> Iterable[str]:
    suffix = suffix or None
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if suffix:
                if "nc-1" in suffix and re.search(r"nc-1\d", name):
                    continue
                if not name.endswith(suffix):
                    continue
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                yield candidate
    elif os.path.isfile(path):
        if suffix:
            name = os.path.basename(path)
            if "nc-1" in suffix and re.search(r"nc-1\d", name):
                return
            if not name.endswith(suffix):
                return
        yield path


def _resolve_tee_suffix(tee_protocols_path: str, tee_suffix: str | None) -> str | None:
    if not tee_suffix:
        return None
    if not os.path.isdir(tee_protocols_path):
        return tee_suffix
    names = sorted(os.listdir(tee_protocols_path))
    if any(name.endswith(tee_suffix) for name in names):
        return tee_suffix
    if tee_suffix == "-nc-1" and any(name.endswith("-nc-1.txt") for name in names):
        print("[warning] No TEE protocol matches '-nc-1'; falling back to '-nc-1.txt'.")
        return "-nc-1.txt"
    return tee_suffix


def _iter_trials_protocol_files(path: str, suffix: str | None) -> Iterable[str]:
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if suffix and not name.endswith(suffix):
                continue
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                yield candidate
    elif os.path.isfile(path):
        yield path


def _load_attack_metadata(path: str) -> dict[str, dict[str, str]]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping: dict[str, dict[str, str]] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                mapping[str(key)] = {
                    "am": str(value.get("AM") or value.get("am") or value.get("acoustic_model") or ""),
                    "vm": str(value.get("VM") or value.get("vm") or value.get("vocoder_model") or ""),
                }
    elif isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            atk = entry.get("attack_id") or entry.get("attack") or entry.get("id")
            if atk is None:
                continue
            mapping[str(atk)] = {
                "am": str(entry.get("AM") or entry.get("am") or entry.get("acoustic_model") or ""),
                "vm": str(entry.get("VM") or entry.get("vm") or entry.get("vocoder_model") or ""),
            }
    return mapping


def _build_reference_sets(
    tee_protocols_path: str,
    known_attacks: Iterable[str],
    reference_size: int,
    seed: int,
    tee_suffix: str | None,
) -> dict[str, list[str]]:
    entries: list[tuple[str, str]] = []
    for proto_path in _iter_protocol_files(tee_protocols_path, tee_suffix):
        df, has_header = _read_protocol_df(proto_path, expect_header=True)
        if has_header:
            cols = {c.lower(): c for c in df.columns}
            path_col = None
            for cand in ("path", "wav", "utt", "audio", "filename"):
                if cand in cols:
                    path_col = cols[cand]
                    break
            attack_col = None
            for cand in ("attack_id", "attack", "hyp_attack_id", "atk"):
                if cand in cols:
                    attack_col = cols[cand]
                    break
            for _, row in df.iterrows():
                path_val = row[path_col] if path_col else None
                attack_val = row[attack_col] if attack_col else None
                path_str = str(path_val) if path_val is not None else ""
                attack_id = str(attack_val) if attack_val is not None else _find_attack_id(path_str or "")
                if not attack_id:
                    attack_id = _find_attack_id(os.path.basename(proto_path))
                if not attack_id or not path_str:
                    continue
                entries.append((attack_id, _normalize_tee_path(path_str)))
        else:
            for _, row in df.iterrows():
                tokens = [str(x) for x in row.to_list() if str(x) != "nan"]
                joined = " ".join(tokens)
                attack_id = _find_attack_id(joined)
                wav_path = next((t for t in tokens if t.lower().endswith((".wav", ".flac"))), None)
                if attack_id and wav_path:
                    entries.append((attack_id, _normalize_tee_path(wav_path)))

    rng = np.random.default_rng(seed)
    ref_sets: dict[str, list[str]] = {}
    for attack_id in known_attacks:
        candidates = [path for atk, path in entries if atk == attack_id]
        if len(candidates) < reference_size:
            raise ValueError(
                f"Missing TEE enrollment for {attack_id}: need {reference_size}, got {len(candidates)}."
            )
        rng.shuffle(candidates)
        ref_sets[attack_id] = candidates[:reference_size]
    return ref_sets


def _pick_col(cols: dict[str, str], candidates: Iterable[str]) -> str | None:
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def _build_trials(
    trials_protocols_path: str,
    trials_suffix: str | None,
    known_attacks: Iterable[str],
    reference_sets: dict[str, list[str]],
    attack_metadata: dict[str, dict[str, str]],
) -> list[Trial]:
    trials: list[Trial] = []
    known_set = set(known_attacks)

    for proto_path in _iter_trials_protocol_files(trials_protocols_path, trials_suffix):
        df, has_header = _read_protocol_df(proto_path, expect_header=True)
        if not has_header:
            raise ValueError(
                f"Trials protocol {proto_path} has no header; provide CSV/TSV with columns."
            )

        cols = {c.lower(): c for c in df.columns}
        hyp_attack_col = _pick_col(cols, ("hyp_attack_id", "hyp_attack", "enroll_attack", "attack_id_hyp"))
        trial_attack_col = _pick_col(cols, ("trial_attack_id", "trial_attack", "attack_id_trial"))
        attack_col = _pick_col(cols, ("attack_id", "attack", "atk"))
        hyp_path_col = _pick_col(cols, ("hyp_path", "hyp_wav", "enroll_path", "ref_path", "path_a", "patha"))
        trial_path_col = _pick_col(cols, ("trial_path", "trial_wav", "path_b", "pathb", "path", "filename"))
        label_atk_col = _pick_col(cols, ("same_atk", "label_atk", "same_attack", "istargetatk"))
        label_am_col = _pick_col(cols, ("same_am", "label_am", "istargetacousticmodel"))
        label_vm_col = _pick_col(cols, ("same_vm", "label_vm", "istargetvocodermodel"))
        unknown_col = _pick_col(cols, ("trial_is_unknown", "unknown_attack", "is_unknown"))

        for _, row in df.iterrows():
            trial_path = str(row[trial_path_col]) if trial_path_col else ""
            hyp_path = str(row[hyp_path_col]) if hyp_path_col else ""

            hyp_attack = None
            if hyp_attack_col:
                hyp_attack = str(row[hyp_attack_col])
            if not hyp_attack and hyp_path:
                hyp_attack = _find_attack_id(hyp_path)
            if not hyp_attack:
                hyp_attack = _find_attack_id(os.path.basename(proto_path))
            if not hyp_attack and attack_col:
                candidate = str(row[attack_col])
                if candidate in known_set:
                    hyp_attack = candidate
            if not hyp_attack:
                raise ValueError(f"Missing hypothesis attack id in {proto_path}.")
            if hyp_attack not in known_set:
                continue

            trial_attack = None
            if trial_attack_col:
                trial_attack = str(row[trial_attack_col])
            if not trial_attack and attack_col and attack_col != hyp_attack_col:
                trial_attack = str(row[attack_col])
            if not trial_attack and trial_path:
                trial_attack = _find_attack_id(trial_path)

            if not trial_path:
                raise ValueError(f"Missing trial wav path in {proto_path}.")

            label_atk = _coerce_label(row[label_atk_col]) if label_atk_col else None
            label_am = _coerce_label(row[label_am_col]) if label_am_col else None
            label_vm = _coerce_label(row[label_vm_col]) if label_vm_col else None

            trial_is_unknown = None
            if unknown_col:
                trial_is_unknown = _coerce_label(row[unknown_col])
            if trial_is_unknown is None and trial_attack:
                trial_is_unknown = int(trial_attack not in known_set)

            if label_atk is None or label_am is None or label_vm is None:
                hyp_meta = attack_metadata.get(hyp_attack, {})
                trial_meta = attack_metadata.get(trial_attack or "", {})
                hyp_am = hyp_meta.get("am")
                hyp_vm = hyp_meta.get("vm")
                trial_am = trial_meta.get("am")
                trial_vm = trial_meta.get("vm")

                if label_atk is None and trial_attack:
                    label_atk = int(trial_attack == hyp_attack)
                if label_am is None and hyp_am and trial_am:
                    label_am = int(trial_am == hyp_am)
                if label_vm is None and hyp_vm and trial_vm:
                    label_vm = int(trial_vm == hyp_vm)

            if label_atk is None or label_am is None or label_vm is None:
                raise ValueError(
                    f"Unable to derive labels for trial {trial_path} (hyp {hyp_attack})."
                )

            trial_rel = _normalize_trial_path(trial_path)
            ref_paths = reference_sets.get(hyp_attack, [])
            if hyp_path:
                hyp_rel = _normalize_tee_path(hyp_path)
                if hyp_rel in ref_paths:
                    ref_paths = [hyp_rel] + [p for p in ref_paths if p != hyp_rel]
                else:
                    ref_paths = [hyp_rel] + ref_paths

            trials.append(
                Trial(
                    test_path=trial_rel,
                    ref_paths=ref_paths,
                    label_atk=int(label_atk),
                    label_am=int(label_am),
                    label_vm=int(label_vm),
                    ref_attack=hyp_attack,
                    trial_attack=trial_attack,
                    trial_is_unknown=int(trial_is_unknown) if trial_is_unknown is not None else 0,
                )
            )

    if not trials:
        raise ValueError("No trials generated; check protocol paths and suffix.")
    return trials


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
    embeddings_cache: Path | None,
    allow_stale_embeddings_cache: bool,
    cache_meta: dict | None,
) -> dict[str, np.ndarray]:
    if not paths:
        return {}
    cache: dict[str, np.ndarray] = {}
    cache_updated = False
    if embeddings_cache is not None and embeddings_cache.exists():
        with np.load(embeddings_cache, allow_pickle=False) as data:
            existing_meta = data.get("cache_meta")
            if isinstance(existing_meta, np.ndarray) and existing_meta.shape == ():
                existing_meta = existing_meta.item()
            if isinstance(existing_meta, (bytes, np.bytes_)):
                existing_meta = existing_meta.decode("utf-8")
            try:
                existing_meta = json.loads(existing_meta) if isinstance(existing_meta, str) else None
            except json.JSONDecodeError:
                existing_meta = None

            compatible = True
            if cache_meta:
                compatible = existing_meta == cache_meta
            if not compatible and not allow_stale_embeddings_cache:
                print(
                    f"[warning] Embeddings cache {embeddings_cache} was built for a different "
                    "checkpoint/config; ignoring and rebuilding."
                )
            else:
                utt_ids = data.get("utt_ids")
                embeddings = data.get("embeddings")
                if utt_ids is None or embeddings is None:
                    raise ValueError(
                        f"Embedding cache missing required arrays: {embeddings_cache}"
                    )
                cache = {str(k): embeddings[i] for i, k in enumerate(utt_ids)}

    missing = [p for p in paths if p not in cache]
    if not missing:
        return cache

    dataset = PathDataset(data_root, missing, segment_samples, skip_bad_paths=skip_bad_paths)
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

    model.eval()
    with torch.no_grad():
        for file_names, waveforms, _ in tqdm(dataloader, desc="Embedding"):
            waveforms = waveforms.to(device)
            with amp_ctx:
                emb = model.extract_embedding(waveforms)
            emb = emb.detach().to(dtype=torch.float32).cpu().numpy()
            for name, vec in zip(file_names, emb):
                cache[name] = vec
                cache_updated = True

    if embeddings_cache is not None and cache_updated:
        embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
        utt_ids = np.array(sorted(cache))
        emb_matrix = np.stack([cache[u] for u in utt_ids], axis=0)
        meta = cache_meta or {}
        np.savez_compressed(
            embeddings_cache,
            embeddings=emb_matrix,
            utt_ids=utt_ids,
            cache_meta=np.array(json.dumps(meta, sort_keys=True)),
        )
    return cache


def _reduce_scores(scores: np.ndarray, mode: str) -> float:
    if scores.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(scores))
    return float(np.max(scores))


def _score_trials(
    embeddings: dict[str, np.ndarray],
    trials: list[Trial],
    reference_size: int,
    reduction: str,
    output_csv: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    rows = []
    for trial in tqdm(trials, desc="Scoring trials"):
        ref_paths = trial.ref_paths[:reference_size]
        if len(ref_paths) < reference_size:
            continue
        test_emb = embeddings[trial.test_path]
        ref_embs = np.stack([embeddings[p] for p in ref_paths], axis=0)
        test_norm = test_emb / (np.linalg.norm(test_emb) + 1e-8)
        ref_norm = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)
        cosines = np.dot(ref_norm, test_norm)
        score = _reduce_scores(cosines, reduction)
        best_ref_idx = int(np.argmax(cosines)) if cosines.size else 0
        best_ref_path = ref_paths[best_ref_idx] if ref_paths else ""
        scores.append(score)
        labels.append(trial.label_atk)
        rows.append(
            {
                "pair_id": f"{_canonicalize_relpath(best_ref_path)}|{_canonicalize_relpath(trial.test_path)}",
                "hyp_attack_id": trial.ref_attack,
                "hyp_wav": _canonicalize_relpath(best_ref_path),
                "trial_attack_id": trial.trial_attack or "",
                "trial_wav": _canonicalize_relpath(trial.test_path),
                "score": score,
                "label_atk": trial.label_atk,
                "label_am": trial.label_am,
                "label_vm": trial.label_vm,
                "trial_is_unknown": trial.trial_is_unknown,
            }
        )

    scores_arr = np.asarray(scores, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    return scores_arr, labels_arr


def _report_metrics(scores: np.ndarray, labels: np.ndarray, label: str):
    labels_arr = np.asarray(labels)
    eer = calculate_EER("STOPA", labels_arr, scores, False, det_subtitle=label)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    print(f"[{label}] EER: {eer * 100:.2f}%  AUC: {auc * 100:.2f}%")
    return {
        "eer": float(eer),
        "auc": float(auc),
    }


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


def _resolve_pin_persist(args: argparse.Namespace, config: dict) -> tuple[bool, bool]:
    pin_memory = config.get("pin_memory", torch.cuda.is_available())
    persistent_workers = config.get("persistent_workers", False)
    return pin_memory and torch.cuda.is_available(), persistent_workers


def main() -> None:
    args = parse_args()
    config = _resolve_config(args)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_state = _load_checkpoint_state_dict(args.checkpoint)

    data_root = str(args.data_root or (Path(config["data_dir"]) / config["stopa"]["train_subdir"]))
    tee_protocols = args.tee_protocols or (Path(data_root) / "TEE" / "TEE_protocols")
    if not tee_protocols.exists():
        alt = Path(data_root) / "TEE" / "protocols"
        if alt.exists():
            tee_protocols = alt
    trials_protocols = args.trials_protocols or (Path(data_root) / "Trials" / "protocols")
    metadata_path = args.metadata_path or (Path(data_root) / "metadata" / "attacks.json")

    known_attacks = [a.strip() for a in args.known_attacks.split(",") if a.strip()]
    if not known_attacks:
        raise ValueError("No known attacks provided.")

    if args.num_classes is None:
        inferred = _infer_num_classes_from_state_dict(checkpoint_state)
        if inferred is not None:
            args.num_classes = inferred
            print(f"Inferred num_classes={args.num_classes} from checkpoint {args.checkpoint}.")
        else:
            raise ValueError("num_classes required when it cannot be inferred from checkpoint.")

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

    cache_meta = None
    if args.embeddings_cache is not None:
        try:
            stat = args.checkpoint.stat()
            mtime_ns = int(stat.st_mtime_ns)
            size = int(stat.st_size)
        except FileNotFoundError:
            mtime_ns = None
            size = None
        cache_meta = {
            "cache_version": 1,
            "checkpoint": str(args.checkpoint),
            "checkpoint_mtime_ns": mtime_ns,
            "checkpoint_size": size,
            "model_class": type(model).__name__,
            "extractor": str(args.extractor),
            "processor": str(args.processor),
            "classifier": str(args.classifier),
            "num_classes": int(args.num_classes),
            "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
            "segment_seconds": float(args.segment_seconds) if args.segment_seconds else 0.0,
            "sample_rate": int(args.sample_rate),
        }

    attack_metadata = _load_attack_metadata(str(metadata_path))
    tee_suffix = _resolve_tee_suffix(str(tee_protocols), args.tee_suffix)
    reference_sets = _build_reference_sets(
        str(tee_protocols), known_attacks, args.reference_size, args.seed, tee_suffix
    )
    trials = _build_trials(
        str(trials_protocols),
        args.trials_suffix,
        known_attacks,
        reference_sets,
        attack_metadata,
    )

    unique_paths = sorted(
        set([t.test_path for t in trials] + [p for refs in reference_sets.values() for p in refs])
    )
    embeddings = _embed_paths(
        model,
        device,
        amp_ctx,
        data_root,
        unique_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        segment_samples=segment_samples,
        skip_bad_paths=args.skip_bad_paths,
        embeddings_cache=args.embeddings_cache,
        allow_stale_embeddings_cache=args.allow_stale_embeddings_cache,
        cache_meta=cache_meta,
    )

    output_csv = args.output_csv or Path(f"scores_stopa_ref{args.reference_size}.csv")
    scores, labels = _score_trials(
        embeddings, trials, args.reference_size, args.score_reduction, output_csv
    )
    metrics = _report_metrics(scores, labels, f"stopa_ref{args.reference_size}")
    det = _det_metrics(
        scores,
        (np.asarray(labels) == 1).astype(int),
        p_target=args.p_target,
        c_miss=args.c_miss,
        c_fa=args.c_fa,
    )
    metrics.update(
        {
            "min_dcf": det["min_dcf"],
            "min_dcf_threshold": det["min_dcf_threshold"],
            "tpr_at_fpr": _tpr_at_fpr(det["fpr"], det["fnr"], args.fixed_fprs),
        }
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.report_dir / f"summary_stopa_ref{args.reference_size}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[info] Wrote scores to {output_csv}")
    print(f"[info] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
