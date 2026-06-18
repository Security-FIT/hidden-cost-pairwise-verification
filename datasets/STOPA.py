import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


KNOWN_ATTACKS_DEFAULT = ("AA01", "AA03", "AA05", "AA07", "AA10")
_ATTACK_RE = re.compile(r"AA\d+")


def _canonicalize_relpath(path: str) -> str:
    return path.lstrip("./")


def _resolve_path(root_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root_dir, _canonicalize_relpath(path))


def _find_attack_id(text: str) -> Optional[str]:
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


def _coerce_label(value) -> Optional[int]:
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


def _load_attack_metadata(path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping: Dict[str, Dict[str, str]] = {}
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


def _read_protocol_df(path: str, expect_header: bool = True) -> Tuple[pd.DataFrame, bool]:
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


def _iter_protocol_files(path: str) -> Iterable[str]:
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                yield candidate
    elif os.path.isfile(path):
        yield path


def _iter_trials_protocol_files(path: str, known_attacks: Iterable[str]) -> Iterable[str]:
    known = set(known_attacks)
    for candidate in _iter_protocol_files(path):
        fname = os.path.basename(candidate)
        if not fname.endswith("_trials.txt"):
            continue
        if not fname.endswith("-nc-1_trials.txt"):
            continue
        attack_id = _find_attack_id(fname)
        if attack_id is None or attack_id not in known:
            continue
        yield candidate


class STOPADataset_pair(Dataset):
    """
    STOPA evaluation dataset for pairwise source verification.

    Builds nc-1 references from TEE_protocols (one utterance per known attack)
    and pairs them with Trials utterances (from Trials_protocols by default).
    """

    def __init__(
        self,
        root_dir: str,
        protocol_file_name: str,
        variant: str = "eval",
        augment: bool = False,
        rir_root: str | None = None,
        known_attacks: Iterable[str] | None = None,
        tee_protocols_path: str | None = None,
        metadata_path: str | None = None,
        segment_seconds: float | None = None,
        sample_rate: int = 16000,
        **_: dict,
    ):
        self.root_dir = root_dir
        self.variant = variant
        self.known_attacks = tuple(known_attacks) if known_attacks else KNOWN_ATTACKS_DEFAULT
        self.segment_samples = None
        if segment_seconds and segment_seconds > 0:
            self.segment_samples = int(segment_seconds * sample_rate)

        self.tee_protocols_path = (
            _resolve_path(root_dir, tee_protocols_path)
            if tee_protocols_path
            else os.path.join(root_dir, "TEE/TEE_protocols")
        )
        if not os.path.exists(self.tee_protocols_path):
            alt_tee = os.path.join(root_dir, "TEE/protocols")
            if os.path.exists(alt_tee):
                self.tee_protocols_path = alt_tee
        self.trials_protocols_path = _resolve_path(root_dir, protocol_file_name) if protocol_file_name else ""
        if self.trials_protocols_path and not os.path.exists(self.trials_protocols_path):
            alt_trials = os.path.join(root_dir, "Trials/protocols")
            if os.path.exists(alt_trials):
                self.trials_protocols_path = alt_trials
        meta_path = _resolve_path(root_dir, metadata_path) if metadata_path else os.path.join(root_dir, "metadata/attacks.json")
        self.attack_metadata = _load_attack_metadata(meta_path)

        self.ref_map = self._build_reference_map()
        self.pairs: List[Dict[str, object]] = []
        self.pair_metadata_by_id: Dict[str, Dict[str, object]] = {}
        self._build_pairs()
        self.protocol_df = pd.DataFrame(self.pairs)

    def _load_waveform(self, abs_path: str) -> torch.Tensor:
        waveform, _ = sf.read(abs_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if self.segment_samples:
            if waveform.shape[0] >= self.segment_samples:
                waveform = waveform[: self.segment_samples]
            else:
                pad = self.segment_samples - waveform.shape[0]
                waveform = np.pad(waveform, (0, pad), mode="constant")
        waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform)

    def _build_reference_map(self) -> Dict[str, str]:
        entries: List[Tuple[str, str]] = []
        for proto_path in _iter_protocol_files(self.tee_protocols_path):
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

        ref_map: Dict[str, str] = {}
        for attack_id in self.known_attacks:
            candidates = sorted(path for atk, path in entries if atk == attack_id)
            if not candidates:
                raise ValueError(f"Missing TEE enrollment for known attack {attack_id}.")
            ref_map[attack_id] = candidates[0]
        return ref_map

    def _build_pairs(self) -> None:
        if not self.trials_protocols_path or not os.path.exists(self.trials_protocols_path):
            print(
                f"Trials protocol not found ({self.trials_protocols_path}); "
                "falling back to full expansion over Trials/wav."
            )
            self._build_pairs_expanded()
            return

        for proto_path in _iter_trials_protocol_files(self.trials_protocols_path, self.known_attacks):
            df, has_header = _read_protocol_df(proto_path, expect_header=True)
            if not has_header:
                raise ValueError(
                    f"Trials protocol {proto_path} has no header; please provide a CSV/TSV with column names."
                )

            cols = {c.lower(): c for c in df.columns}

            def _pick_col(candidates: Iterable[str]) -> Optional[str]:
                for cand in candidates:
                    if cand in cols:
                        return cols[cand]
                return None

            hyp_attack_col = _pick_col(("hyp_attack_id", "hyp_attack", "enroll_attack", "attack_id_hyp"))
            trial_attack_col = _pick_col(("trial_attack_id", "trial_attack", "attack_id_trial"))
            attack_col = _pick_col(("attack_id", "attack", "atk"))
            hyp_path_col = _pick_col(("hyp_path", "hyp_wav", "enroll_path", "ref_path", "path_a", "patha"))
            trial_path_col = _pick_col(("trial_path", "trial_wav", "path_b", "pathb", "path", "filename"))
            label_atk_col = _pick_col(("same_atk", "label_atk", "same_attack", "istargetatk"))
            label_am_col = _pick_col(("same_am", "label_am", "istargetacousticmodel"))
            label_vm_col = _pick_col(("same_vm", "label_vm", "istargetvocodermodel"))
            unknown_col = _pick_col(("trial_is_unknown", "unknown_attack", "is_unknown"))

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
                    if candidate in self.known_attacks:
                        hyp_attack = candidate
                if not hyp_attack:
                    raise ValueError(f"Missing hypothesis attack id in {proto_path}.")
                if hyp_attack not in self.known_attacks:
                    # Skip rows that map to unknown hypotheses (no nc-1 enrollment).
                    continue

                trial_attack = None
                if trial_attack_col:
                    trial_attack = str(row[trial_attack_col])
                if not trial_attack and attack_col and attack_col != hyp_attack_col:
                    trial_attack = str(row[attack_col])
                if not trial_attack and trial_path:
                    trial_attack = _find_attack_id(trial_path)

                hyp_wav = self.ref_map.get(hyp_attack)
                if not hyp_wav:
                    raise ValueError(f"No TEE enrollment found for hypothesis {hyp_attack}.")
                if not trial_path:
                    raise ValueError(f"Missing trial wav path in {proto_path}.")

                label_atk = _coerce_label(row[label_atk_col]) if label_atk_col else None
                label_am = _coerce_label(row[label_am_col]) if label_am_col else None
                label_vm = _coerce_label(row[label_vm_col]) if label_vm_col else None

                trial_is_unknown = None
                if unknown_col:
                    trial_is_unknown = _coerce_label(row[unknown_col])

                if trial_is_unknown is None and trial_attack:
                    trial_is_unknown = int(trial_attack not in self.known_attacks)

                if label_atk is None or label_am is None or label_vm is None:
                    hyp_meta = self.attack_metadata.get(hyp_attack, {})
                    trial_meta = self.attack_metadata.get(trial_attack or "", {})
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
                        f"Unable to derive labels for trial {trial_path} (hyp {hyp_attack}). "
                        "Ensure Trials_protocols provides labels or attacks.json includes AM/VM mappings."
                    )

                trial_rel = _normalize_trial_path(trial_path)
                pair_id = f"{_canonicalize_relpath(hyp_wav)}|{_canonicalize_relpath(trial_rel)}"
                record = {
                    "pair_id": pair_id,
                    "hyp_attack_id": hyp_attack,
                    "hyp_wav": _canonicalize_relpath(hyp_wav),
                    "trial_wav": _canonicalize_relpath(trial_rel),
                    "trial_attack_id": trial_attack,
                    "label_atk": int(label_atk),
                    "label_am": int(label_am),
                    "label_vm": int(label_vm),
                    "trial_is_unknown": int(trial_is_unknown) if trial_is_unknown is not None else 0,
                }
                self.pairs.append(record)
                self.pair_metadata_by_id[pair_id] = record

    def _build_pairs_expanded(self) -> None:
        trials_dir = os.path.join(self.root_dir, "Trials", "wav")
        if not os.path.isdir(trials_dir):
            raise FileNotFoundError(f"Trials wav directory not found: {trials_dir}")

        trial_files = []
        for name in sorted(os.listdir(trials_dir)):
            if name.lower().endswith((".wav", ".flac")):
                trial_files.append(os.path.join(trials_dir, name))
        if not trial_files:
            raise ValueError(f"No trial wav files found in {trials_dir}")

        for trial_path_abs in trial_files:
            trial_rel = _canonicalize_relpath(os.path.relpath(trial_path_abs, self.root_dir))
            trial_attack = _find_attack_id(trial_rel)
            if not trial_attack:
                raise ValueError(f"Unable to infer attack id from trial path: {trial_rel}")

            trial_meta = self.attack_metadata.get(trial_attack)
            if not trial_meta:
                raise ValueError(f"Missing attacks.json metadata for trial attack {trial_attack}")

            for hyp_attack, hyp_wav in self.ref_map.items():
                hyp_meta = self.attack_metadata.get(hyp_attack)
                if not hyp_meta:
                    raise ValueError(f"Missing attacks.json metadata for hypothesis {hyp_attack}")
                label_atk = int(trial_attack == hyp_attack)
                label_am = int(trial_meta.get("am") == hyp_meta.get("am"))
                label_vm = int(trial_meta.get("vm") == hyp_meta.get("vm"))
                trial_is_unknown = int(trial_attack not in self.known_attacks)

                pair_id = f"{_canonicalize_relpath(hyp_wav)}|{trial_rel}"
                record = {
                    "pair_id": pair_id,
                    "hyp_attack_id": hyp_attack,
                    "hyp_wav": _canonicalize_relpath(hyp_wav),
                    "trial_wav": trial_rel,
                    "trial_attack_id": trial_attack,
                    "label_atk": label_atk,
                    "label_am": label_am,
                    "label_vm": label_vm,
                    "trial_is_unknown": trial_is_unknown,
                }
                self.pairs.append(record)
                self.pair_metadata_by_id[pair_id] = record

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        record = self.pairs[idx]
        hyp_path = _resolve_path(self.root_dir, record["hyp_wav"])
        trial_path = _resolve_path(self.root_dir, record["trial_wav"])
        hyp_wav = self._load_waveform(hyp_path)
        trial_wav = self._load_waveform(trial_path)
        return record["pair_id"], hyp_wav, trial_wav, int(record["label_atk"])

    def get_labels(self) -> np.ndarray:
        return np.asarray([r["label_atk"] for r in self.pairs], dtype=np.int32)

    def get_class_weights(self) -> torch.FloatTensor:
        labels = self.get_labels()
        class_counts = np.bincount(labels, minlength=2)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)
