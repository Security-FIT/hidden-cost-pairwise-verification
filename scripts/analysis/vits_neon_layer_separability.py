#!/usr/bin/env python3
import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

from common import EXTRACTORS
from config import karolina_config, local_config, sge_config

try:
    import torchaudio.functional as AF
    import torchaudio.transforms as AT
except Exception:  # pragma: no cover - optional runtime dependency
    AF = None
    AT = None


DEFAULT_TARGET_PAIRS = [
    ("tts_models/en/ljspeech/vits", "tts_models/en/ljspeech/vits--neon"),
    ("tts_models/en/ljspeech/vits--neon", "tts_models/en/ljspeech/vits"),
    ("suno/bark-small", "tts_models/multilingual/multi-dataset/bark"),
    ("parler_tts_mini_v1", "parler_tts_large_v1"),
    ("parler_tts_large_v1", "parler_tts_mini_v1"),
]

DEFAULT_IMPOSTOR_PAIRS = [
    ("suno/bark-small", "tts_models/multilingual/multi-dataset/bark"),
    ("tts_models/en/ljspeech/vits", "tts_models/en/ljspeech/vits--neon"),
    ("tts_models/en/ljspeech/vits--neon", "tts_models/en/ljspeech/vits"),
    ("tts_models/bn/custom/vits-male", "tts_models/bn/custom/vits-female"),
    ("tts_models/bn/custom/vits-female", "tts_models/bn/custom/vits-male"),
    ("tts_models/it/mai_male/glow-tts", "tts_models/fa/custom/glow-tts"),
    ("suno/bark-small", "suno/bark"),
    ("suno/bark", "suno/bark-small"),
    ("tts_models/fa/custom/glow-tts", "tts_models/it/mai_male/glow-tts"),
    ("facebook/mms-tts-swe", "facebook/mms-tts-rus"),
    ("facebook/mms-tts-rus", "facebook/mms-tts-swe"),
    ("facebook/mms-tts-deu", "facebook/mms-tts-swe"),
    ("facebook/mms-tts-swe", "facebook/mms-tts-fra"),
    ("facebook/mms-tts-fra", "facebook/mms-tts-swe"),
    ("facebook/mms-tts-fra", "facebook/mms-tts-rus"),
    ("facebook/mms-tts-deu", "facebook/mms-tts-nld"),
    ("facebook/mms-tts-rus", "facebook/mms-tts-fra"),
    ("facebook/mms-tts-deu", "facebook/mms-tts-rus"),
    ("facebook/mms-tts-hun", "facebook/mms-tts-swe"),
    ("facebook/mms-tts-hun", "facebook/mms-tts-rus"),
    ("facebook/mms-tts-swe", "facebook/mms-tts-hun"),
    ("parler_tts_mini_v1", "parler_tts_large_v1"),
    ("parler_tts_large_v1", "parler_tts_mini_v1"),
]

SPEAKER_COL_CANDIDATES = [
    "speaker_id",
    "speaker",
    "spk_id",
    "speaker_name",
    "speaker_uid",
    "speaker_label",
]

TEXT_COL_CANDIDATES = [
    "text",
    "transcript",
    "prompt",
    "sentence",
]

TEXT_BUCKET_CANDIDATES = [
    "text_bucket",
    "phonetic_bucket",
    "phonetic_group",
    "content_bucket",
]


@dataclass
class PreprocessConfig:
    crop_samples: int | None
    crop_policy: str
    crop_shared: bool
    target_rms_db: float | None
    target_sr: int | None
    pad_value: float


@dataclass
class PairGroup:
    name: str
    same_pairs: list[tuple[Path, Path]]
    cross_pairs: list[tuple[Path, Path]]


def _filter_protocol_pairs(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    require_same: bool | None,
    max_pairs: int | None,
    seed: int,
) -> pd.DataFrame:
    mask_a = df["model_name_A"].astype(str).str.contains(model_a, case=False, regex=False)
    mask_b = df["model_name_B"].astype(str).str.contains(model_b, case=False, regex=False)
    mask = mask_a & mask_b
    if require_same is not None and "same_model" in df.columns:
        same_val = 1 if require_same else 0
        mask &= df["same_model"].astype(int) == same_val
    subset = df[mask]
    if max_pairs is not None and len(subset) > max_pairs:
        subset = subset.sample(n=max_pairs, random_state=seed)
    return subset


@dataclass
class ControlledProbeConfig:
    anchors_per_speaker: int
    same_per_anchor: int
    cross_per_anchor: int
    max_speakers: int | None
    speaker_column: str
    model_column: str
    path_column: str
    text_column: str | None
    match_text: bool
    bidirectional: bool


@dataclass
class ForensicConfig:
    enabled: bool
    anchors_per_model: int


def _select_config(args: argparse.Namespace) -> dict:
    if args.karolina:
        return karolina_config
    if args.sge:
        return sge_config
    return local_config


def _resolve_dataset_config(dataset: str, config: dict) -> dict:
    if "MLAAD" not in dataset:
        raise ValueError("This analysis script currently supports MLAAD datasets only.")
    key_candidates: list[str] = []
    if "CurriculumHardminedFFCosine" in dataset:
        key_candidates = ["mlaad_curriculum_hardmined_ffcosine"]
    elif "CurriculumHardminedFFConcat" in dataset:
        key_candidates = ["mlaad_curriculum_hardmined_ffconcat"]
    elif "CurriculumDirectionalFFCosine" in dataset:
        key_candidates = ["mlaad_curriculum_directional_ffcosine"]
    elif "CurriculumDirectionalFFConcat" in dataset:
        key_candidates = ["mlaad_curriculum_directional_ffconcat"]
    elif "CurriculumRivalFFCosine" in dataset:
        key_candidates = [
            "mlaad_curriculum_rival_ffcosine",
            "mlaad_curriculum_rival_ffmulticlass",
            "mlaad_curriculum_rival_ffconcat",
        ]
    elif "CurriculumRivalFFConcat" in dataset:
        key_candidates = [
            "mlaad_curriculum_rival_ffconcat",
            "mlaad_curriculum_rival_ffcosine",
            "mlaad_curriculum_rival_ffmulticlass",
        ]
    elif "CurriculumRivalFFMulticlass" in dataset:
        key_candidates = [
            "mlaad_curriculum_rival_ffmulticlass",
            "mlaad_curriculum_rival_ffcosine",
            "mlaad_curriculum_rival_ffconcat",
        ]

    for key in key_candidates:
        if key in config:
            if key_candidates and key != key_candidates[0]:
                print(f"[warn] Config missing {key_candidates[0]}; using {key} instead.")
            return config[key]

    if "mlaad" in config:
        if key_candidates:
            print(f"[warn] Config missing {key_candidates[0]}; falling back to mlaad.")
        return config["mlaad"]
    raise KeyError("No MLAAD dataset config found in selected config.")


def _resolve_single_config(config: dict) -> dict:
    if "mlaad_single" in config:
        return config["mlaad_single"]
    raise KeyError("mlaad_single not found in config; pass --utterance-protocol explicitly.")


def _resolve_column(df: pd.DataFrame, provided: str | None, candidates: list[str], label: str) -> str:
    if provided:
        if provided not in df.columns:
            raise ValueError(f"{label} column '{provided}' not in protocol columns: {df.columns.tolist()}")
        return provided
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    available = ", ".join(df.columns.tolist())
    hint = f"Provide --{label.replace(' ', '-')}-column <col>."
    raise ValueError(
        f"Could not infer {label} column. {hint} Available columns: {available}"
    )


def _resolve_path(root_dir: Path, rel_path: str) -> Path:
    return root_dir / rel_path.lstrip("./")


def _load_waveform_raw(abs_path: Path) -> tuple[torch.Tensor, int]:
    waveform, sr = sf.read(str(abs_path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform[np.newaxis, :]
    return torch.from_numpy(waveform), int(sr)


def _resample_if_needed(waveform: torch.Tensor, orig_sr: int, target_sr: int | None) -> torch.Tensor:
    if target_sr is None or orig_sr == target_sr:
        return waveform
    if AF is None:
        raise RuntimeError("torchaudio is required for resampling but is not available.")
    return AF.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)


def _crop_waveform(waveform: torch.Tensor, crop_len: int, start_frac: float) -> torch.Tensor:
    total_len = waveform.shape[1]
    if total_len >= crop_len:
        max_start = total_len - crop_len
        start = int(round(max_start * start_frac))
        return waveform[:, start : start + crop_len]
    pad = crop_len - total_len
    return F.pad(waveform, (0, pad))


def _normalize_rms(waveform: torch.Tensor, target_db: float, max_gain_db: float = 30.0) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    target = 10 ** (target_db / 20.0)
    gain = target / torch.clamp(rms, min=1e-6)
    max_gain = 10 ** (max_gain_db / 20.0)
    gain = torch.clamp(gain, max=max_gain)
    return waveform * gain


def _prepare_pair_waveforms(
    path_a: Path,
    path_b: Path,
    preprocess: PreprocessConfig,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    wav_a, sr_a = _load_waveform_raw(path_a)
    wav_b, sr_b = _load_waveform_raw(path_b)

    if preprocess.target_sr is not None:
        wav_a = _resample_if_needed(wav_a, sr_a, preprocess.target_sr)
        wav_b = _resample_if_needed(wav_b, sr_b, preprocess.target_sr)

    if preprocess.crop_samples is not None:
        if preprocess.crop_policy == "center":
            start_frac = 0.5
        else:
            start_frac = rng.random()
        if preprocess.crop_shared:
            wav_a = _crop_waveform(wav_a, preprocess.crop_samples, start_frac)
            wav_b = _crop_waveform(wav_b, preprocess.crop_samples, start_frac)
        else:
            wav_a = _crop_waveform(wav_a, preprocess.crop_samples, rng.random())
            wav_b = _crop_waveform(wav_b, preprocess.crop_samples, rng.random())

    if preprocess.target_rms_db is not None:
        wav_a = _normalize_rms(wav_a, preprocess.target_rms_db)
        wav_b = _normalize_rms(wav_b, preprocess.target_rms_db)

    return wav_a, wav_b


def _compute_distances(
    extractor: torch.nn.Module,
    pairs: list[tuple[Path, Path]],
    batch_size: int,
    preprocess: PreprocessConfig,
    seed: int,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    if not pairs:
        raise ValueError("No pairs provided for distance computation.")

    all_distances = []
    rng = random.Random(seed)
    for start in tqdm(range(0, len(pairs), batch_size), desc=desc, unit="batch"):
        batch_pairs = pairs[start : start + batch_size]
        wav_a_list = []
        wav_b_list = []
        for path_a, path_b in batch_pairs:
            wav_a, wav_b = _prepare_pair_waveforms(path_a, path_b, preprocess, rng)
            wav_a_list.append(wav_a)
            wav_b_list.append(wav_b)
        max_len_a = max(w.shape[1] for w in wav_a_list)
        max_len_b = max(w.shape[1] for w in wav_b_list)
        wav_a = torch.cat(
            [
                F.pad(w, (0, max_len_a - w.shape[1])) if w.shape[1] < max_len_a else w
                for w in wav_a_list
            ],
            dim=0,
        ).to(device)
        wav_b = torch.cat(
            [
                F.pad(w, (0, max_len_b - w.shape[1])) if w.shape[1] < max_len_b else w
                for w in wav_b_list
            ],
            dim=0,
        ).to(device)

        with torch.no_grad():
            emb_a = extractor.extract_features(wav_a)
            emb_b = extractor.extract_features(wav_b)
            emb_a = emb_a.mean(dim=2)
            emb_b = emb_b.mean(dim=2)
            cos = F.cosine_similarity(emb_a, emb_b, dim=2)
            distances = 1.0 - cos
        all_distances.append(distances.cpu().numpy())

    return np.concatenate(all_distances, axis=1)


def _plot_distance_curves(out_dir: Path, mean_same: np.ndarray, mean_cross: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping plots.")
        return
    layers = np.arange(len(mean_same))
    plt.figure(figsize=(10, 5))
    plt.plot(layers, mean_same, label="same", linewidth=2)
    plt.plot(layers, mean_cross, label="cross", linewidth=2)
    plt.xlabel("Layer")
    plt.ylabel("Cosine distance (1 - cos)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_distance_by_layer.png", dpi=200)
    plt.close()


def _plot_dprime(out_dir: Path, dprime: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping plots.")
        return
    layers = np.arange(len(dprime))
    plt.figure(figsize=(10, 4))
    plt.plot(layers, dprime, label="d-prime", color="#1f77b4", linewidth=2)
    plt.xlabel("Layer")
    plt.ylabel("d-prime (cross - same)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "dprime_by_layer.png", dpi=200)
    plt.close()


def _plot_best_layer_hist(out_dir: Path, best_layer: int, dist_same: np.ndarray, dist_cross: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping plots.")
        return
    plt.figure(figsize=(8, 4))
    plt.hist(dist_same, bins=40, alpha=0.6, label="same", density=True)
    plt.hist(dist_cross, bins=40, alpha=0.6, label="cross", density=True)
    plt.xlabel("Cosine distance (1 - cos)")
    plt.ylabel("Density")
    plt.title(f"Best layer {best_layer} distance distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"distance_hist_layer_{best_layer}.png", dpi=200)
    plt.close()


def _build_model_label(df: pd.DataFrame, model_col: str, model_patterns: list[str]) -> pd.Series:
    labels = pd.Series([None] * len(df), index=df.index)
    lower = df[model_col].astype(str).str.lower()
    ordered = sorted(model_patterns, key=lambda x: len(x), reverse=True)
    for pattern in ordered:
        mask = lower.str.contains(pattern.lower(), regex=False)
        labels = labels.mask(mask & labels.isna(), pattern)
    return labels


def _build_pairs_for_model_pair(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    cfg: ControlledProbeConfig,
    rng: random.Random,
) -> PairGroup:
    if cfg.match_text and cfg.text_column is None:
        print("[warn] --match-text requested but no text column found; ignoring text matching.")

    pair_name = f"{model_a}__vs__{model_b}"
    same_pairs: list[tuple[Path, Path]] = []
    cross_pairs: list[tuple[Path, Path]] = []

    group_cols = [cfg.speaker_column]
    if cfg.match_text and cfg.text_column:
        group_cols.append(cfg.text_column)

    for _, group in df.groupby(group_cols):
        pool_a = group[group["model_label"] == model_a]
        pool_b = group[group["model_label"] == model_b]
        if len(pool_a) < 2 or len(pool_b) < 1:
            continue
        pool_a_records = pool_a.to_dict("records")
        pool_b_records = pool_b.to_dict("records")
        anchors = rng.sample(pool_a_records, min(cfg.anchors_per_speaker, len(pool_a_records)))
        for anchor in anchors:
            same_candidates = [r for r in pool_a_records if r[cfg.path_column] != anchor[cfg.path_column]]
            if not same_candidates:
                continue
            for _ in range(cfg.same_per_anchor):
                same_target = rng.choice(same_candidates)
                same_pairs.append(
                    (Path(anchor[cfg.path_column]), Path(same_target[cfg.path_column]))
                )
            for _ in range(cfg.cross_per_anchor):
                cross_target = rng.choice(pool_b_records)
                cross_pairs.append(
                    (Path(anchor[cfg.path_column]), Path(cross_target[cfg.path_column]))
                )

    if not same_pairs or not cross_pairs:
        print(f"[warn] No pairs built for {pair_name} (same={len(same_pairs)}, cross={len(cross_pairs)}).")

    return PairGroup(pair_name, same_pairs, cross_pairs)


def _build_controlled_pairs(
    df: pd.DataFrame,
    model_pairs: list[tuple[str, str]],
    cfg: ControlledProbeConfig,
    rng: random.Random,
) -> list[PairGroup]:
    groups: list[PairGroup] = []
    for model_a, model_b in model_pairs:
        group = _build_pairs_for_model_pair(df, model_a, model_b, cfg, rng)
        groups.append(group)
        if cfg.bidirectional and (model_b, model_a) not in model_pairs:
            reverse_group = _build_pairs_for_model_pair(df, model_b, model_a, cfg, rng)
            groups.append(reverse_group)
    return groups


def _build_forensic_transforms(sample_rate: int) -> dict[str, callable]:
    if AT is None:
        raise RuntimeError("torchaudio is required for forensic transforms but is not available.")
    mulaw = nn.Sequential(
        AT.MuLawEncoding(quantization_channels=256),
        AT.MuLawDecoding(quantization_channels=256),
    )
    quality = nn.Sequential(
        AT.Resample(orig_freq=sample_rate, new_freq=8000),
        AT.Resample(orig_freq=8000, new_freq=sample_rate),
    )
    resample_target = int(round(sample_rate * 0.75))
    resample_target = min(16000, resample_target)
    if resample_target >= sample_rate:
        resample_target = max(1000, int(round(sample_rate * 0.6)))
    resample = nn.Sequential(
        AT.Resample(orig_freq=sample_rate, new_freq=resample_target),
        AT.Resample(orig_freq=resample_target, new_freq=sample_rate),
    )

    def bitcrush(x: torch.Tensor) -> torch.Tensor:
        return torch.round(x * 512.0) / 512.0

    return {
        "mulaw": mulaw,
        "quality": quality,
        "resample": resample,
        "bitcrush": bitcrush,
    }


def _compute_forensic_distances(
    extractor: torch.nn.Module,
    anchors: list[Path],
    transforms: dict[str, callable],
    batch_size: int,
    preprocess: PreprocessConfig,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> None:
    rng = random.Random(seed)
    for name, transform in transforms.items():
        pairs: list[tuple[Path, Path]] = []
        for anchor in anchors:
            pairs.append((anchor, anchor))
        if not pairs:
            print(f"[warn] No anchors for forensic transform {name}.")
            continue

        all_distances = []
        for start in tqdm(range(0, len(pairs), batch_size), desc=f"forensic-{name}", unit="batch"):
            batch = pairs[start : start + batch_size]
            wav_a_list = []
            wav_b_list = []
            for path_a, _ in batch:
                wav_a, _ = _prepare_pair_waveforms(path_a, path_a, preprocess, rng)
                with torch.no_grad():
                    wav_b = transform(wav_a)
                if preprocess.target_rms_db is not None:
                    wav_b = _normalize_rms(wav_b, preprocess.target_rms_db)
                wav_a_list.append(wav_a)
                wav_b_list.append(wav_b)

            max_len_a = max(w.shape[1] for w in wav_a_list)
            max_len_b = max(w.shape[1] for w in wav_b_list)
            wav_a = torch.cat(
                [
                    F.pad(w, (0, max_len_a - w.shape[1])) if w.shape[1] < max_len_a else w
                    for w in wav_a_list
                ],
                dim=0,
            ).to(device)
            wav_b = torch.cat(
                [
                    F.pad(w, (0, max_len_b - w.shape[1])) if w.shape[1] < max_len_b else w
                    for w in wav_b_list
                ],
                dim=0,
            ).to(device)

            with torch.no_grad():
                emb_a = extractor.extract_features(wav_a)
                emb_b = extractor.extract_features(wav_b)
                emb_a = emb_a.mean(dim=2)
                emb_b = emb_b.mean(dim=2)
                cos = F.cosine_similarity(emb_a, emb_b, dim=2)
                distances = 1.0 - cos
            all_distances.append(distances.cpu().numpy())

        dist = np.concatenate(all_distances, axis=1)
        mean_dist = dist.mean(axis=1)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = pd.DataFrame({"layer": np.arange(len(mean_dist)), "mean_distance": mean_dist})
        metrics.to_csv(out_dir / f"forensic_{name}_metrics.csv", index=False)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib not available; skipping plots.")
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(mean_dist)), mean_dist, linewidth=2)
        plt.xlabel("Layer")
        plt.ylabel("Cosine distance (1 - cos)")
        plt.title(f"Forensic self-impostor ({name})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"forensic_{name}_curve.png", dpi=200)
        plt.close()


def _load_extractor(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    extractor = EXTRACTORS[args.extractor]()
    extractor.to(device)
    extractor.eval()
    if args.checkpoint:
        try:
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(args.checkpoint, map_location="cpu")
        extractor_state = {
            k.replace("extractor.", ""): v for k, v in state.items() if k.startswith("extractor.")
        }
        if extractor_state:
            missing, unexpected = extractor.load_state_dict(extractor_state, strict=False)

            def _is_conv_key(key: str) -> bool:
                return "feature_extractor.conv_layers" in key or "feature_conv_layers" in key

            if missing or unexpected:
                if all(_is_conv_key(k) for k in missing) and all(_is_conv_key(k) for k in unexpected):
                    remapped = {}
                    for key, value in extractor_state.items():
                        if "feature_conv_layers" in key:
                            key = key.replace("feature_conv_layers", "feature_extractor.conv_layers")
                        remapped[key] = value
                    missing, unexpected = extractor.load_state_dict(remapped, strict=False)

            if missing:
                non_conv_missing = [k for k in missing if not _is_conv_key(k)]
                if non_conv_missing:
                    print(f"[warn] Missing extractor keys: {sorted(non_conv_missing)}")
            if unexpected:
                non_conv_unexpected = [k for k in unexpected if not _is_conv_key(k)]
                if non_conv_unexpected:
                    print(f"[warn] Unexpected extractor keys: {sorted(non_conv_unexpected)}")
        else:
            print("[warn] No extractor weights found in checkpoint; using default extractor weights.")
    return extractor


def _parse_pair_args(values: list[str] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not values:
        return pairs
    for value in values:
        if "|" not in value:
            raise ValueError("Pair must be formatted as 'modelA|modelB'.")
        left, right = value.split("|", 1)
        pairs.append((left.strip(), right.strip()))
    return pairs


def _infer_text_column(df: pd.DataFrame, provided: str | None, match_text: bool) -> str | None:
    if provided:
        return provided
    if not match_text:
        return None
    for candidate in TEXT_BUCKET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for candidate in TEXT_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-wise generator separability analysis.")
    cfg_group = parser.add_mutually_exclusive_group(required=True)
    cfg_group.add_argument("--karolina", action="store_true")
    cfg_group.add_argument("--metacentrum", action="store_true")
    cfg_group.add_argument("--sge", action="store_true")
    cfg_group.add_argument("--local", action="store_true")

    parser.add_argument("--checkpoint", type=str, required=True, help="FFMulticlass checkpoint.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MLAADCurriculumRivalFFMulticlassDataset_pair",
        help="Dataset name for resolving protocols.",
    )
    parser.add_argument(
        "--probe-mode",
        type=str,
        choices=("protocol", "controlled", "impostor"),
        default="protocol",
        help="Use pair protocol directly, targeted impostor pairs, or controlled probe pairs.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        help="Override pair protocol CSV path (protocol mode).",
    )
    parser.add_argument(
        "--utterance-protocol",
        type=str,
        default=None,
        help="Single-utterance CSV for controlled mode (defaults to mlaad_single eval).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root for resolving relative paths.",
    )
    parser.add_argument("--extractor", type=str, default="XLSR_300M")
    parser.add_argument("--model-a", type=str, default="vits")
    parser.add_argument("--model-b", type=str, default="neon")
    parser.add_argument("--max-pairs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="analysis_outputs/vits_neon_layers")

    parser.add_argument(
        "--target-pair",
        action="append",
        default=None,
        help="Target generator pair 'modelA|modelB' (repeatable).",
    )
    parser.add_argument(
        "--control-pair",
        action="append",
        default=None,
        help="Control generator pair 'modelA|modelB' (repeatable).",
    )
    parser.add_argument(
        "--impostor-pair",
        action="append",
        default=None,
        help="Impostor generator pair 'modelA|modelB' (repeatable).",
    )
    parser.add_argument("--anchors-per-speaker", type=int, default=2)
    parser.add_argument("--same-per-anchor", type=int, default=1)
    parser.add_argument("--cross-per-anchor", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--speaker-column", type=str, default=None)
    parser.add_argument("--model-column", type=str, default=None)
    parser.add_argument("--path-column", type=str, default=None)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--match-text", action="store_true")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--crop-seconds", type=float, default=None)
    parser.add_argument(
        "--crop-policy",
        type=str,
        choices=("center", "random"),
        default="center",
    )
    parser.add_argument(
        "--crop-shared",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use shared crop start for A/B.",
    )
    parser.add_argument(
        "--target-rms-db",
        type=float,
        default=None,
        help="Target RMS dBFS for loudness normalization.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="Resample to this sample rate before feature extraction.",
    )
    parser.add_argument(
        "--forensic-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run forensic self-impostor probe set.",
    )
    parser.add_argument("--forensic-anchors-per-model", type=int, default=200)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = _select_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.probe_mode == "controlled" and args.target_sr is None:
        args.target_sr = config.get("sample_rate", 16000)

    if args.target_rms_db is None and args.probe_mode == "controlled":
        args.target_rms_db = -25.0
    if args.probe_mode == "controlled" and args.max_speakers is None:
        args.max_speakers = 50

    preprocess = PreprocessConfig(
        crop_samples=None,
        crop_policy=args.crop_policy,
        crop_shared=args.crop_shared,
        target_rms_db=args.target_rms_db,
        target_sr=args.target_sr,
        pad_value=0.0,
    )

    if args.crop_seconds is not None:
        sample_rate = args.target_sr if args.target_sr is not None else config.get("sample_rate", 16000)
        preprocess.crop_samples = int(args.crop_seconds * sample_rate)
    else:
        segment_seconds = config.get("segment_seconds")
        if segment_seconds and segment_seconds > 0:
            sample_rate = args.target_sr if args.target_sr is not None else config.get("sample_rate", 16000)
            preprocess.crop_samples = int(segment_seconds * sample_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = _load_extractor(args, device)

    if args.probe_mode == "protocol":
        dataset_config = _resolve_dataset_config(args.dataset, config)
        if args.protocol:
            protocol_path = Path(args.protocol)
            if not protocol_path.is_absolute():
                protocol_path = Path.cwd() / protocol_path
        else:
            root_dir = Path(config["data_dir"]) / dataset_config["eval_subdir"]
            protocol_path = root_dir / dataset_config["eval_protocol"]

        if args.data_root:
            root_dir = Path(args.data_root)
        else:
            root_dir = Path(config["data_dir"]) / dataset_config["eval_subdir"]

        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        df = pd.read_csv(protocol_path)
        required_cols = {"path_A", "model_name_A", "path_B", "model_name_B"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Protocol missing required columns: {sorted(missing)}")

        a_in_A = df["model_name_A"].astype(str).str.contains(args.model_a, case=False, regex=False)
        a_in_B = df["model_name_B"].astype(str).str.contains(args.model_a, case=False, regex=False)
        b_in_A = df["model_name_A"].astype(str).str.contains(args.model_b, case=False, regex=False)
        b_in_B = df["model_name_B"].astype(str).str.contains(args.model_b, case=False, regex=False)

        same_mask = a_in_A & a_in_B
        cross_mask = (a_in_A & b_in_B) | (b_in_A & a_in_B)

        df_same = df[same_mask]
        df_cross = df[cross_mask]
        if args.max_pairs and len(df_same) > args.max_pairs:
            df_same = df_same.sample(n=args.max_pairs, random_state=args.seed)
        if args.max_pairs and len(df_cross) > args.max_pairs:
            df_cross = df_cross.sample(n=args.max_pairs, random_state=args.seed)

        if df_same.empty or df_cross.empty:
            raise ValueError(
                f"No pairs found for same={len(df_same)} cross={len(df_cross)}. "
                f"Check model name filters: {args.model_a}, {args.model_b}."
            )

        pairs_same = [
            (_resolve_path(root_dir, row["path_A"]), _resolve_path(root_dir, row["path_B"]))
            for _, row in df_same.iterrows()
        ]
        pairs_cross = [
            (_resolve_path(root_dir, row["path_A"]), _resolve_path(root_dir, row["path_B"]))
            for _, row in df_cross.iterrows()
        ]

        dist_same = _compute_distances(
            extractor,
            pairs_same,
            args.batch_size,
            preprocess,
            args.seed,
            device,
            desc="same",
        )
        dist_cross = _compute_distances(
            extractor,
            pairs_cross,
            args.batch_size,
            preprocess,
            args.seed + 1,
            device,
            desc="cross",
        )

        mean_same = dist_same.mean(axis=1)
        mean_cross = dist_cross.mean(axis=1)
        std_same = dist_same.std(axis=1)
        std_cross = dist_cross.std(axis=1)
        pooled = np.sqrt(0.5 * (std_same ** 2 + std_cross ** 2) + 1e-8)
        dprime = (mean_cross - mean_same) / pooled
        diff = mean_cross - mean_same
        best_layer = int(np.nanargmax(dprime))

        metrics = pd.DataFrame(
            {
                "layer": np.arange(len(mean_same)),
                "mean_same": mean_same,
                "mean_cross": mean_cross,
                "std_same": std_same,
                "std_cross": std_cross,
                "dprime": dprime,
                "mean_diff": diff,
            }
        )
        metrics.to_csv(out_dir / "layer_distance_metrics.csv", index=False)
        _plot_distance_curves(out_dir, mean_same, mean_cross, "Generator separability by layer")
        _plot_dprime(out_dir, dprime, "d-prime by layer")
        _plot_best_layer_hist(out_dir, best_layer, dist_same[best_layer, :], dist_cross[best_layer, :])

        print(
            f"Best layer by d-prime: {best_layer} "
            f"(d'={dprime[best_layer]:.3f}, diff={diff[best_layer]:.3f})"
        )
        print(f"Saved metrics and plots to {out_dir}")
        return

    if args.probe_mode == "impostor":
        dataset_config = _resolve_dataset_config(args.dataset, config)
        if args.protocol:
            protocol_path = Path(args.protocol)
            if not protocol_path.is_absolute():
                protocol_path = Path.cwd() / protocol_path
        else:
            root_dir = Path(config["data_dir"]) / dataset_config["eval_subdir"]
            protocol_path = root_dir / dataset_config["eval_protocol"]

        if args.data_root:
            root_dir = Path(args.data_root)
        else:
            root_dir = Path(config["data_dir"]) / dataset_config["eval_subdir"]

        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        df = pd.read_csv(protocol_path)
        required_cols = {"path_A", "model_name_A", "path_B", "model_name_B"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Protocol missing required columns: {sorted(missing)}")

        impostor_pairs = _parse_pair_args(args.impostor_pair) or DEFAULT_IMPOSTOR_PAIRS
        groups: list[PairGroup] = []
        pair_records = []

        for model_a, model_b in impostor_pairs:
            df_same = _filter_protocol_pairs(
                df,
                model_a,
                model_a,
                require_same=True,
                max_pairs=args.max_pairs,
                seed=args.seed,
            )
            df_cross = _filter_protocol_pairs(
                df,
                model_a,
                model_b,
                require_same=False,
                max_pairs=args.max_pairs,
                seed=args.seed,
            )
            same_pairs = [
                (Path(row["path_A"]), Path(row["path_B"])) for _, row in df_same.iterrows()
            ]
            cross_pairs = [
                (Path(row["path_A"]), Path(row["path_B"])) for _, row in df_cross.iterrows()
            ]
            group = PairGroup(f"{model_a}__vs__{model_b}", same_pairs, cross_pairs)
            groups.append(group)
            for path_a, path_b in same_pairs:
                pair_records.append(
                    {
                        "pair_group": group.name,
                        "pair_type": "same",
                        "path_a": str(path_a),
                        "path_b": str(path_b),
                    }
                )
            for path_a, path_b in cross_pairs:
                pair_records.append(
                    {
                        "pair_group": group.name,
                        "pair_type": "cross",
                        "path_a": str(path_a),
                        "path_b": str(path_b),
                    }
                )

        if pair_records:
            pd.DataFrame(pair_records).to_csv(out_dir / "probe_pairs.csv", index=False)

        for group in groups:
            if not group.same_pairs or not group.cross_pairs:
                print(
                    f"[warn] Skipping {group.name} (same={len(group.same_pairs)}, cross={len(group.cross_pairs)})."
                )
                continue
            group_out = out_dir / group.name
            group_out.mkdir(parents=True, exist_ok=True)
            print(
                f"[group] {group.name}: same={len(group.same_pairs)} cross={len(group.cross_pairs)}"
            )

            dist_same = _compute_distances(
                extractor,
                [
                    (_resolve_path(root_dir, p[0].as_posix()), _resolve_path(root_dir, p[1].as_posix()))
                    for p in group.same_pairs
                ],
                args.batch_size,
                preprocess,
                args.seed,
                device,
                desc=f"{group.name}-same",
            )
            dist_cross = _compute_distances(
                extractor,
                [
                    (_resolve_path(root_dir, p[0].as_posix()), _resolve_path(root_dir, p[1].as_posix()))
                    for p in group.cross_pairs
                ],
                args.batch_size,
                preprocess,
                args.seed + 1,
                device,
                desc=f"{group.name}-cross",
            )

            mean_same = dist_same.mean(axis=1)
            mean_cross = dist_cross.mean(axis=1)
            std_same = dist_same.std(axis=1)
            std_cross = dist_cross.std(axis=1)
            pooled = np.sqrt(0.5 * (std_same ** 2 + std_cross ** 2) + 1e-8)
            dprime = (mean_cross - mean_same) / pooled
            diff = mean_cross - mean_same
            best_layer = int(np.nanargmax(dprime))

            metrics = pd.DataFrame(
                {
                    "layer": np.arange(len(mean_same)),
                    "mean_same": mean_same,
                    "mean_cross": mean_cross,
                    "std_same": std_same,
                    "std_cross": std_cross,
                    "dprime": dprime,
                    "mean_diff": diff,
                }
            )
            metrics.to_csv(group_out / "layer_distance_metrics.csv", index=False)
            _plot_distance_curves(
                group_out,
                mean_same,
                mean_cross,
                f"{group.name} separability by layer",
            )
            _plot_dprime(group_out, dprime, f"{group.name} d-prime by layer")
            _plot_best_layer_hist(
                group_out,
                best_layer,
                dist_same[best_layer, :],
                dist_cross[best_layer, :],
            )
            print(
                f"[group] {group.name}: best layer {best_layer} "
                f"(d'={dprime[best_layer]:.3f}, diff={diff[best_layer]:.3f})"
            )

        if args.forensic_probe:
            model_patterns = sorted(
                {m for pair in impostor_pairs for m in pair}
            )
            anchors = []
            for model in model_patterns:
                mask_a = df["model_name_A"].astype(str).str.contains(model, case=False, regex=False)
                mask_b = df["model_name_B"].astype(str).str.contains(model, case=False, regex=False)
                paths = pd.concat(
                    [df.loc[mask_a, "path_A"], df.loc[mask_b, "path_B"]],
                    ignore_index=True,
                ).dropna().unique().tolist()
                if not paths:
                    continue
                take = min(args.forensic_anchors_per_model, len(paths))
                anchors.extend(random.sample(paths, take))
            anchors = list(dict.fromkeys(anchors))
            anchors = [root_dir / Path(p) for p in anchors]
            sample_rate = args.target_sr if args.target_sr is not None else config.get("sample_rate", 16000)
            transforms = _build_forensic_transforms(sample_rate)
            forensic_dir = out_dir / "forensic_probe"
            _compute_forensic_distances(
                extractor,
                anchors,
                transforms,
                args.batch_size,
                preprocess,
                args.seed + 7,
                device,
                forensic_dir,
            )

        print(f"Saved outputs to {out_dir}")
        return

    # Controlled probe mode
    single_config = _resolve_single_config(config)
    if args.utterance_protocol:
        protocol_path = Path(args.utterance_protocol)
        if not protocol_path.is_absolute():
            protocol_path = Path.cwd() / protocol_path
        root_dir = Path(args.data_root) if args.data_root else Path(config["data_dir"]) / single_config["eval_subdir"]
    else:
        root_dir = Path(args.data_root) if args.data_root else Path(config["data_dir"]) / single_config["eval_subdir"]
        protocol_path = root_dir / single_config["eval_protocol"]

    if not protocol_path.exists():
        raise FileNotFoundError(f"Utterance protocol file not found: {protocol_path}")

    df = pd.read_csv(protocol_path)
    path_col = _resolve_column(df, args.path_column, ["path"], "path")
    model_col = _resolve_column(df, args.model_column, ["model_name", "model"], "model")
    speaker_col = _resolve_column(df, args.speaker_column, SPEAKER_COL_CANDIDATES, "speaker")
    text_col = _infer_text_column(df, args.text_column, args.match_text)

    target_pairs = _parse_pair_args(args.target_pair) or DEFAULT_TARGET_PAIRS
    control_pairs = _parse_pair_args(args.control_pair)
    all_pairs = target_pairs + control_pairs
    model_patterns = sorted({m for pair in all_pairs for m in pair})

    df = df.copy()
    df["model_label"] = _build_model_label(df, model_col, model_patterns)
    df = df[df["model_label"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows matched the requested model patterns in the utterance protocol.")

    if args.max_speakers:
        speakers = df[speaker_col].dropna().unique().tolist()
        if len(speakers) > args.max_speakers:
            selected = random.sample(speakers, args.max_speakers)
            df = df[df[speaker_col].isin(selected)].reset_index(drop=True)

    controlled_cfg = ControlledProbeConfig(
        anchors_per_speaker=args.anchors_per_speaker,
        same_per_anchor=args.same_per_anchor,
        cross_per_anchor=args.cross_per_anchor,
        max_speakers=args.max_speakers,
        speaker_column=speaker_col,
        model_column=model_col,
        path_column=path_col,
        text_column=text_col,
        match_text=args.match_text,
        bidirectional=args.bidirectional,
    )

    rng = random.Random(args.seed)
    groups = _build_controlled_pairs(df, all_pairs, controlled_cfg, rng)

    pair_records = []
    for group in groups:
        for path_a, path_b in group.same_pairs:
            pair_records.append(
                {
                    "pair_group": group.name,
                    "pair_type": "same",
                    "path_a": str(path_a),
                    "path_b": str(path_b),
                }
            )
        for path_a, path_b in group.cross_pairs:
            pair_records.append(
                {
                    "pair_group": group.name,
                    "pair_type": "cross",
                    "path_a": str(path_a),
                    "path_b": str(path_b),
                }
            )
    if pair_records:
        pd.DataFrame(pair_records).to_csv(out_dir / "probe_pairs.csv", index=False)

    for group in groups:
        if not group.same_pairs or not group.cross_pairs:
            continue
        group_out = out_dir / group.name
        group_out.mkdir(parents=True, exist_ok=True)
        print(
            f"[group] {group.name}: same={len(group.same_pairs)} cross={len(group.cross_pairs)}"
        )
        dist_same = _compute_distances(
            extractor,
            [(root_dir / p[0], root_dir / p[1]) for p in group.same_pairs],
            args.batch_size,
            preprocess,
            args.seed,
            device,
            desc=f"{group.name}-same",
        )
        dist_cross = _compute_distances(
            extractor,
            [(root_dir / p[0], root_dir / p[1]) for p in group.cross_pairs],
            args.batch_size,
            preprocess,
            args.seed + 1,
            device,
            desc=f"{group.name}-cross",
        )

        mean_same = dist_same.mean(axis=1)
        mean_cross = dist_cross.mean(axis=1)
        std_same = dist_same.std(axis=1)
        std_cross = dist_cross.std(axis=1)
        pooled = np.sqrt(0.5 * (std_same ** 2 + std_cross ** 2) + 1e-8)
        dprime = (mean_cross - mean_same) / pooled
        diff = mean_cross - mean_same
        best_layer = int(np.nanargmax(dprime))

        metrics = pd.DataFrame(
            {
                "layer": np.arange(len(mean_same)),
                "mean_same": mean_same,
                "mean_cross": mean_cross,
                "std_same": std_same,
                "std_cross": std_cross,
                "dprime": dprime,
                "mean_diff": diff,
            }
        )
        metrics.to_csv(group_out / "layer_distance_metrics.csv", index=False)
        _plot_distance_curves(
            group_out,
            mean_same,
            mean_cross,
            f"{group.name} separability by layer",
        )
        _plot_dprime(group_out, dprime, f"{group.name} d-prime by layer")
        _plot_best_layer_hist(group_out, best_layer, dist_same[best_layer, :], dist_cross[best_layer, :])
        print(
            f"[group] {group.name}: best layer {best_layer} "
            f"(d'={dprime[best_layer]:.3f}, diff={diff[best_layer]:.3f})"
        )

    if args.forensic_probe:
        sample_rate = args.target_sr if args.target_sr is not None else config.get("sample_rate", 16000)
        transforms = _build_forensic_transforms(sample_rate)
        anchors = []
        for model in model_patterns:
            candidates = df[df["model_label"] == model][path_col].tolist()
            if not candidates:
                continue
            take = min(args.forensic_anchors_per_model, len(candidates))
            anchors.extend(rng.sample(candidates, take))
        anchors = list(dict.fromkeys(anchors))
        anchors = [root_dir / Path(p) for p in anchors]
        forensic_dir = out_dir / "forensic_probe"
        _compute_forensic_distances(
            extractor,
            anchors,
            transforms,
            args.batch_size,
            preprocess,
            args.seed + 7,
            device,
            forensic_dir,
        )

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
