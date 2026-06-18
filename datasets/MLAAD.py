import os
import random
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf

from augmentation.ForensicAugmenter import ForensicAugmenter


def _normalize_allowed_classes(raw: Iterable[str] | None) -> list[str] | None:
    if raw is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        parts = str(item).split(",")
        for part in parts:
            value = part.strip()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
    return normalized if normalized else None


class MLAADDataset_pair(Dataset):
    """
    Dataset loader for MLAAD source-tracing protocols.

    Each entry contains two utterances (A and B) together with a binary label
    indicating whether both waveforms originate from the same TTS model.
    """

    REQUIRED_COLUMNS = ["path_A", "model_name_A", "path_B", "model_name_B", "same_model"]

    def __init__(
        self,
        root_dir: str,
        protocol_file_name: str,
        variant: str = "eval",
        augment: bool = False,
        rir_root: str | None = None,
        segment_seconds: float | None = None,
        sample_rate: int = 16000,
        allowed_classes: Iterable[str] | None = None,
        **_: dict,
    ):
        self.root_dir = root_dir
        self.augment = augment
        self.rir_root = rir_root
        self.segment_samples = None
        if segment_seconds and segment_seconds > 0:
            self.segment_samples = int(segment_seconds * sample_rate)
        protocol_path = os.path.join(self.root_dir, protocol_file_name)
        if not os.path.isfile(protocol_path):
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        self.protocol_df = pd.read_csv(protocol_path)
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.protocol_df.columns]
        if missing_cols:
            raise ValueError(
                f"Protocol file {protocol_path} missing required columns: {', '.join(missing_cols)}"
            )

        allowed_list = _normalize_allowed_classes(allowed_classes)
        if allowed_list is not None:
            allowed_set = set(allowed_list)
            self.protocol_df["model_name_A"] = self.protocol_df["model_name_A"].astype(str)
            self.protocol_df["model_name_B"] = self.protocol_df["model_name_B"].astype(str)
            present = set(self.protocol_df["model_name_A"]).union(
                set(self.protocol_df["model_name_B"])
            )
            missing = sorted(allowed_set - present)
            before = len(self.protocol_df)
            mask = self.protocol_df["model_name_A"].isin(allowed_set) & self.protocol_df[
                "model_name_B"
            ].isin(allowed_set)
            self.protocol_df = self.protocol_df.loc[mask].reset_index(drop=True)
            after = len(self.protocol_df)
            print(
                f"[MLAAD pair] Applied allowed_classes filter: {before} -> {after} rows "
                f"({len(allowed_set)} classes)."
            )
            if missing:
                print(
                    "[MLAAD pair] Warning: allowed_classes not present in protocol: "
                    + ", ".join(missing)
                )
            if after == 0:
                raise ValueError("allowed_classes filter removed all MLAAD pair samples.")

        # Normalize labels to integers 0/1 for convenience later on
        self.protocol_df["same_model"] = self.protocol_df["same_model"].astype(int)
        self.variant = variant

    def __len__(self) -> int:
        return len(self.protocol_df)

    def _resolve_path(self, rel_path: str) -> str:
        """Return absolute path to an audio file, handling leading './' segments."""
        normalized = rel_path.lstrip("./")
        return os.path.join(self.root_dir, normalized)

    def _load_waveform(self, abs_path: str) -> torch.Tensor:
        """
        Load audio using soundfile to avoid torchcodec dependency.
        Returns a tensor shaped (1, num_samples) regardless of channel count.
        """
        waveform, _ = sf.read(abs_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # convert to mono
        if self.segment_samples:
            if waveform.shape[0] >= self.segment_samples:
                waveform = waveform[: self.segment_samples]
            else:
                pad_width = self.segment_samples - waveform.shape[0]
                waveform = np.pad(waveform, (0, pad_width), mode="constant")
        waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.protocol_df.iloc[idx]
        path_a = row["path_A"]
        path_b = row["path_B"]
        wav_a = self._load_waveform(self._resolve_path(path_a))
        wav_b = self._load_waveform(self._resolve_path(path_b))

        label = int(row["same_model"])
        pair_id = f"{path_a}|{path_b}"
        return pair_id, wav_a, wav_b, label

    def get_labels(self) -> np.ndarray:
        """Return numpy array of classification labels."""
        return self.protocol_df["same_model"].to_numpy(dtype=np.int32)

    def get_class_weights(self) -> torch.FloatTensor:
        """Compute inverse-frequency class weights (0 -> different, 1 -> same)."""
        labels = self.get_labels()
        class_counts = np.bincount(labels, minlength=2)
        class_counts[class_counts == 0] = 1  # avoid div-by-zero if class missing
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)


class MLAADDataset_single(Dataset):
    """
    Dataset loader for MLAAD source-tracing (single-utterance) protocols.

    Each entry contains one utterance with a multiclass label that maps to source_model names.
    Expected columns: path, model_name (configurable via path_column/source_column).
    """

    REQUIRED_COLUMNS = ["path", "model_name"]

    def __init__(
        self,
        root_dir: str,
        protocol_file_name: str,
        variant: str = "train",
        augment: bool = False,
        rir_root: str | None = None,
        path_column: str = "path",
        source_column: str = "model_name",
        label_map: dict[str, int] | None = None,
        allow_unknown: str = "error",
        segment_seconds: float | None = None,
        sample_rate: int = 16000,
        allowed_classes: Iterable[str] | None = None,
        **_: dict,
    ):
        self.root_dir = root_dir
        self.augment = augment
        self.rir_root = rir_root
        self.path_column = path_column
        self.source_column = source_column
        self.segment_samples = None
        if segment_seconds and segment_seconds > 0:
            self.segment_samples = int(segment_seconds * sample_rate)

        protocol_path = os.path.join(self.root_dir, protocol_file_name)
        if not os.path.isfile(protocol_path):
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        self.protocol_df = pd.read_csv(protocol_path)
        missing_cols = [
            col
            for col in (self.path_column, self.source_column)
            if col not in self.protocol_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Protocol file {protocol_path} missing required columns: {', '.join(missing_cols)}"
            )

        allowed_list = _normalize_allowed_classes(allowed_classes)
        self.protocol_df[self.source_column] = self.protocol_df[self.source_column].astype(str)
        if allowed_list is not None:
            allowed_set = set(allowed_list)
            present = set(self.protocol_df[self.source_column])
            missing = sorted(allowed_set - present)
            before = len(self.protocol_df)
            self.protocol_df = self.protocol_df[
                self.protocol_df[self.source_column].isin(allowed_set)
            ].reset_index(drop=True)
            after = len(self.protocol_df)
            print(
                f"[MLAAD single] Applied allowed_classes filter: {before} -> {after} rows "
                f"({len(allowed_set)} classes)."
            )
            if missing:
                print(
                    "[MLAAD single] Warning: allowed_classes not present in protocol: "
                    + ", ".join(missing)
                )
            if after == 0:
                raise ValueError("allowed_classes filter removed all MLAAD single samples.")

        if label_map is None:
            unique_models = sorted(self.protocol_df[self.source_column].dropna().astype(str).unique())
            self.label_map = {name: idx for idx, name in enumerate(unique_models)}
        else:
            self.label_map = dict(label_map)

        self.protocol_df["label"] = self.protocol_df[self.source_column].map(self.label_map)
        if self.protocol_df["label"].isna().any():
            missing = self.protocol_df[self.protocol_df["label"].isna()][self.source_column].unique()
            if allow_unknown == "drop":
                self.protocol_df = self.protocol_df.dropna(subset=["label"]).reset_index(drop=True)
                print(
                    f"Warning: dropping {len(missing)} unseen source labels from {protocol_file_name}."
                )
            else:
                raise ValueError(
                    f"Found unknown source labels not in label_map: {', '.join(map(str, missing))}"
                )

        self.variant = variant

    def __len__(self) -> int:
        return len(self.protocol_df)

    def _resolve_path(self, rel_path: str) -> str:
        normalized = rel_path.lstrip("./")
        return os.path.join(self.root_dir, normalized)

    def _load_waveform(self, abs_path: str) -> torch.Tensor:
        waveform, _ = sf.read(abs_path, dtype="float32")
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

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.protocol_df.iloc[idx]
        rel_path = row[self.path_column]
        wav = self._load_waveform(self._resolve_path(rel_path))
        label = int(row["label"])
        return rel_path, wav, label

    def get_labels(self) -> np.ndarray:
        return self.protocol_df["label"].to_numpy(dtype=np.int32)

    def get_class_weights(self) -> torch.FloatTensor:
        labels = self.get_labels()
        num_classes = len(self.label_map)
        class_counts = np.bincount(labels, minlength=num_classes)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)


class MLAADAugmentedNegativePairDataset(MLAADDataset_pair):
    """
    Dataset that creates A-A_aug negative pairs by applying forensic augmentations to anchor A.
    """

    def __init__(
        self,
        root_dir: str,
        protocol_file_name: str,
        variant: str = "train",
        augment: bool = False,
        rir_root: str | None = None,
        segment_seconds: float | None = None,
        sample_rate: int = 16000,
        anchor_strategy: str = "A",
        augmenter: ForensicAugmenter | None = None,
        **kwargs: dict,
    ):
        super().__init__(
            root_dir=root_dir,
            protocol_file_name=protocol_file_name,
            variant=variant,
            augment=augment,
            rir_root=rir_root,
            segment_seconds=segment_seconds,
            sample_rate=sample_rate,
            **kwargs,
        )
        self.anchor_strategy = anchor_strategy
        self._augmenter = augmenter
        self._augmenter_sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.protocol_df) * 2

    def _get_augmenter(self) -> ForensicAugmenter:
        if self._augmenter is None:
            self._augmenter = ForensicAugmenter(sample_rate=self._augmenter_sample_rate)
        return self._augmenter

    def _select_anchor_path(self, row: pd.Series) -> str:
        if self.anchor_strategy == "A":
            return row["path_A"]
        if self.anchor_strategy == "B":
            return row["path_B"]
        if self.anchor_strategy == "random":
            return row["path_A"] if random.random() < 0.5 else row["path_B"]
        raise ValueError(f"Unknown anchor_strategy: {self.anchor_strategy}")

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_len = len(self.protocol_df)
        reverse_pair = idx >= base_len
        row = self.protocol_df.iloc[idx % base_len]
        anchor_path = self._select_anchor_path(row)
        wav_a = self._load_waveform(self._resolve_path(anchor_path))
        with torch.no_grad():
            wav_b = self._get_augmenter()(wav_a)
        if wav_b.ndim == 1:
            wav_b = wav_b.unsqueeze(0)
        if reverse_pair:
            wav_a, wav_b = wav_b, wav_a
        label = 0
        pair_id = f"{anchor_path}|{anchor_path}|aug{'_rev' if reverse_pair else ''}"
        return pair_id, wav_a, wav_b, label

    def get_labels(self) -> np.ndarray:
        return np.zeros(len(self.protocol_df) * 2, dtype=np.int32)

    def get_class_weights(self) -> torch.FloatTensor:
        labels = self.get_labels()
        class_counts = np.bincount(labels, minlength=2)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)
