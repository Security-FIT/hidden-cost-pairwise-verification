# region Imports
from argparse import Namespace
import math
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

# Classifiers
from classifiers.differential.FFAttn import FFAttn1, FFAttn2, FFAttn3, FFAttn4, FFAttn5
from classifiers.differential.FFConcat import FFLSTM, FFLSTM2, FFConcat1, FFConcat2, FFConcat3
from classifiers.differential.FFCosine import (
    FFCosine,
    FFCosineJoint,
    FFCosine1,
    FFCosine3,
    FFCosineRaw,
    FFCosineRaw2,
    FFMulticlass,
)
from classifiers.differential.FFDiff import FFDiff, FFDiffAbs, FFDiffQuadratic
from classifiers.FFBase import FFBase
from classifiers.single_input.FF import FF

# Config
from config import karolina_config, local_config, sge_config

# Datasets
from datasets.MLAAD import (
    MLAADAugmentedNegativePairDataset,
    MLAADDataset_pair,
    MLAADDataset_single,
)
from datasets.STOPA import STOPADataset_pair
from datasets.utils import (
    PairDatasetWithBenignAugment,
    custom_pair_batch_create,
    custom_single_batch_create,
)

# Extractors
from extractors.XLSR import XLSR_1B, XLSR_2B, XLSR_300M

# Feature processors
from feature_processors.AASIST import AASIST
from feature_processors.MHFA import MHFA

# Trainers
from trainers.BaseTrainer import BaseTrainer
from trainers.FFCosineJointTrainer import FFCosineJointTrainer
from trainers.FFCosineRawTrainer import FFCosineRawTrainer
from trainers.FFPairTrainer import FFPairTrainer
from trainers.FFTrainer import FFTrainer

# endregion


def _build_weighted_sampler(dataset: Dataset, seed: int | None, seed_offset: int = 0) -> WeightedRandomSampler:
    samples_weights = np.vectorize(dataset.get_class_weights().__getitem__)(
        dataset.get_labels()
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed + seed_offset)
    try:
        return WeightedRandomSampler(samples_weights, len(dataset), generator=generator)
    except TypeError:
        # Older torch versions may not accept the generator argument.
        return WeightedRandomSampler(samples_weights, len(dataset))


def _identity_collate(batch: list):
    return batch


def _normalize_allowed_classes(raw: Iterable[str] | None) -> list[str] | None:
    """Normalize allowed-classes inputs from CLI (supports commas and whitespace)."""
    if raw is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if item is None:
            continue
        parts = str(item).split(",")
        for part in parts:
            value = part.strip()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
    return normalized if normalized else None


def _build_label_subset(dataset: Dataset, target_label: int) -> Subset:
    if not hasattr(dataset, "get_labels"):
        raise ValueError("Dataset does not expose get_labels; cannot build label subset.")
    labels = dataset.get_labels()
    idx = np.where(labels == target_label)[0].tolist()
    if not idx:
        raise ValueError(f"Dataset contains no samples with label={target_label}.")
    return Subset(dataset, idx)


def _resolve_curriculum_stream_ratios(
    aug_ratio: float | None,
    easy_ratio: float | None,
    hard_ratio: float | None,
) -> dict[str, float]:
    if aug_ratio is None:
        raise ValueError("curriculum_aug_ratio must be provided for augmented curriculum mixing.")
    aug_ratio = float(aug_ratio)
    if not (0.0 < aug_ratio < 1.0):
        raise ValueError("curriculum_aug_ratio must be between 0 and 1 (exclusive).")
    if easy_ratio is None and hard_ratio is None:
        remaining = 1.0 - aug_ratio
        easy_ratio = remaining / 2.0
        hard_ratio = remaining / 2.0
    elif easy_ratio is None:
        hard_ratio = float(hard_ratio)
        easy_ratio = 1.0 - aug_ratio - hard_ratio
    elif hard_ratio is None:
        easy_ratio = float(easy_ratio)
        hard_ratio = 1.0 - aug_ratio - easy_ratio
    easy_ratio = float(easy_ratio)
    hard_ratio = float(hard_ratio)
    total = easy_ratio + hard_ratio + aug_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Curriculum stream ratios must sum to 1.0.")
    if easy_ratio < 0 or hard_ratio < 0:
        raise ValueError("curriculum_easy_ratio and curriculum_hard_ratio must be non-negative.")
    return {"easy": easy_ratio, "hard": hard_ratio, "aug": aug_ratio}


def _allocate_stream_batch_sizes(total_size: int, ratios: dict[str, float]) -> dict[str, int]:
    active = [name for name, ratio in ratios.items() if ratio > 0]
    if total_size < len(active):
        raise ValueError("train_batch_size too small for requested curriculum stream ratios.")
    raw = {name: ratios[name] * total_size for name in ratios}
    counts = {name: int(math.floor(raw[name])) for name in ratios}
    for name in active:
        if counts[name] == 0:
            counts[name] = 1
    current = sum(counts.values())
    if current < total_size:
        order = sorted(
            ratios.keys(),
            key=lambda k: (raw[k] - math.floor(raw[k])),
            reverse=True,
        )
        idx = 0
        while current < total_size:
            name = order[idx % len(order)]
            counts[name] += 1
            current += 1
            idx += 1
    elif current > total_size:
        min_counts = {name: (1 if ratios[name] > 0 else 0) for name in ratios}
        order = sorted(
            ratios.keys(),
            key=lambda k: (raw[k] - math.floor(raw[k])),
        )
        idx = 0
        while current > total_size:
            name = order[idx % len(order)]
            if counts[name] > min_counts[name]:
                counts[name] -= 1
                current -= 1
            idx += 1
            if idx > total_size * len(order):
                raise ValueError("Unable to satisfy batch size allocation with given ratios.")
    return counts


class CurriculumTriMixLoader:
    def __init__(
        self,
        pos_loader: DataLoader,
        easy_neg_loader: DataLoader,
        hard_neg_loader: DataLoader,
        steps_per_epoch: int,
        seed: int = 0,
        collate_fn: Callable[[list], tuple] = custom_pair_batch_create,
        pos_per_batch: int = 0,
        easy_neg_per_batch: int = 0,
        hard_neg_per_batch: int = 0,
    ) -> None:
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1 for curriculum loader.")
        self.pos_loader = pos_loader
        self.easy_neg_loader = easy_neg_loader
        self.hard_neg_loader = hard_neg_loader
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 1
        self.collate_fn = collate_fn
        self.dataset = hard_neg_loader.dataset
        self._last_epoch_stats: dict[str, float | int] | None = None
        self._pos_per_batch = pos_per_batch
        self._easy_neg_per_batch = easy_neg_per_batch
        self._hard_neg_per_batch = hard_neg_per_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable:
        epoch = self.epoch
        rng = np.random.default_rng(self.seed + epoch)
        pos_iter = iter(self.pos_loader)
        easy_neg_iter = iter(self.easy_neg_loader)
        hard_neg_iter = iter(self.hard_neg_loader)
        pos_batches = 0
        easy_batches = 0
        hard_batches = 0

        for _ in range(self.steps_per_epoch):
            try:
                pos_items = next(pos_iter)
            except StopIteration:
                pos_iter = iter(self.pos_loader)
                pos_items = next(pos_iter)
            pos_batches += 1

            try:
                easy_items = next(easy_neg_iter)
            except StopIteration:
                easy_neg_iter = iter(self.easy_neg_loader)
                easy_items = next(easy_neg_iter)
            easy_batches += 1

            try:
                hard_items = next(hard_neg_iter)
            except StopIteration:
                hard_neg_iter = iter(self.hard_neg_loader)
                hard_items = next(hard_neg_iter)
            hard_batches += 1

            combined = list(pos_items) + list(easy_items) + list(hard_items)
            if len(combined) > 1:
                order = rng.permutation(len(combined))
                combined = [combined[idx] for idx in order]
            yield self.collate_fn(combined)

        neg_total = max(1, self._easy_neg_per_batch + self._hard_neg_per_batch)
        p_hard = self._hard_neg_per_batch / neg_total
        self._last_epoch_stats = {
            "epoch": epoch,
            "p_hard": float(p_hard),
            "hard_batches": hard_batches,
            "easy_batches": easy_batches,
            "pos_batches": pos_batches,
            "steps_per_epoch": self.steps_per_epoch,
            "pos_per_batch": self._pos_per_batch,
            "neg_per_batch": self._easy_neg_per_batch + self._hard_neg_per_batch,
            "easy_neg_per_batch": self._easy_neg_per_batch,
            "hard_neg_per_batch": self._hard_neg_per_batch,
        }

    def get_last_epoch_stats(self) -> dict[str, float | int] | None:
        return self._last_epoch_stats


class CurriculumQuadMixLoader:
    def __init__(
        self,
        pos_loader: DataLoader,
        easy_neg_loader: DataLoader,
        hard_neg_loader: DataLoader,
        aug_neg_loader: DataLoader,
        steps_per_epoch: int,
        seed: int = 0,
        collate_fn: Callable[[list], tuple] = custom_pair_batch_create,
        pos_per_batch: int = 0,
        easy_neg_per_batch: int = 0,
        hard_neg_per_batch: int = 0,
        aug_neg_per_batch: int = 0,
    ) -> None:
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1 for curriculum loader.")
        self.pos_loader = pos_loader
        self.easy_neg_loader = easy_neg_loader
        self.hard_neg_loader = hard_neg_loader
        self.aug_neg_loader = aug_neg_loader
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 1
        self.collate_fn = collate_fn
        self.dataset = hard_neg_loader.dataset
        self._last_epoch_stats: dict[str, float | int] | None = None
        self._pos_per_batch = pos_per_batch
        self._easy_neg_per_batch = easy_neg_per_batch
        self._hard_neg_per_batch = hard_neg_per_batch
        self._aug_neg_per_batch = aug_neg_per_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable:
        epoch = self.epoch
        rng = np.random.default_rng(self.seed + epoch)
        pos_iter = iter(self.pos_loader)
        easy_neg_iter = iter(self.easy_neg_loader)
        hard_neg_iter = iter(self.hard_neg_loader)
        aug_neg_iter = iter(self.aug_neg_loader)
        pos_batches = 0
        easy_batches = 0
        hard_batches = 0
        aug_batches = 0

        for _ in range(self.steps_per_epoch):
            try:
                pos_items = next(pos_iter)
            except StopIteration:
                pos_iter = iter(self.pos_loader)
                pos_items = next(pos_iter)
            pos_batches += 1

            try:
                easy_items = next(easy_neg_iter)
            except StopIteration:
                easy_neg_iter = iter(self.easy_neg_loader)
                easy_items = next(easy_neg_iter)
            easy_batches += 1

            try:
                hard_items = next(hard_neg_iter)
            except StopIteration:
                hard_neg_iter = iter(self.hard_neg_loader)
                hard_items = next(hard_neg_iter)
            hard_batches += 1

            try:
                aug_items = next(aug_neg_iter)
            except StopIteration:
                aug_neg_iter = iter(self.aug_neg_loader)
                aug_items = next(aug_neg_iter)
            aug_batches += 1

            combined = list(pos_items) + list(easy_items) + list(hard_items) + list(aug_items)
            if len(combined) > 1:
                order = rng.permutation(len(combined))
                combined = [combined[idx] for idx in order]
            yield self.collate_fn(combined)

        neg_total = max(
            1, self._easy_neg_per_batch + self._hard_neg_per_batch + self._aug_neg_per_batch
        )
        p_hard = self._hard_neg_per_batch / neg_total
        p_aug = self._aug_neg_per_batch / neg_total
        self._last_epoch_stats = {
            "epoch": epoch,
            "p_hard": float(p_hard),
            "p_aug": float(p_aug),
            "hard_batches": hard_batches,
            "easy_batches": easy_batches,
            "aug_batches": aug_batches,
            "pos_batches": pos_batches,
            "steps_per_epoch": self.steps_per_epoch,
            "pos_per_batch": self._pos_per_batch,
            "neg_per_batch": neg_total,
            "easy_neg_per_batch": self._easy_neg_per_batch,
            "hard_neg_per_batch": self._hard_neg_per_batch,
            "aug_neg_per_batch": self._aug_neg_per_batch,
        }

    def get_last_epoch_stats(self) -> dict[str, float | int] | None:
        return self._last_epoch_stats


class CurriculumTripleMixLoader:
    def __init__(
        self,
        easy_loader: DataLoader,
        hard_loader: DataLoader,
        aug_loader: DataLoader,
        steps_per_epoch: int,
        seed: int = 0,
        collate_fn: Callable[[list], tuple] = custom_pair_batch_create,
        easy_per_batch: int = 0,
        hard_per_batch: int = 0,
        aug_per_batch: int = 0,
        stream_names: tuple[str, str, str] = ("easy", "hard", "aug"),
    ) -> None:
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1 for curriculum loader.")
        self.easy_loader = easy_loader
        self.hard_loader = hard_loader
        self.aug_loader = aug_loader
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 1
        self.collate_fn = collate_fn
        self.dataset = easy_loader.dataset
        self._last_epoch_stats: dict[str, float | int] | None = None
        self._easy_per_batch = easy_per_batch
        self._hard_per_batch = hard_per_batch
        self._aug_per_batch = aug_per_batch
        self._stream_names = stream_names

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable:
        epoch = self.epoch
        rng = np.random.default_rng(self.seed + epoch)
        easy_iter = iter(self.easy_loader)
        hard_iter = iter(self.hard_loader)
        aug_iter = iter(self.aug_loader)
        easy_batches = 0
        hard_batches = 0
        aug_batches = 0

        for _ in range(self.steps_per_epoch):
            try:
                easy_items = next(easy_iter)
            except StopIteration:
                easy_iter = iter(self.easy_loader)
                easy_items = next(easy_iter)
            easy_batches += 1

            try:
                hard_items = next(hard_iter)
            except StopIteration:
                hard_iter = iter(self.hard_loader)
                hard_items = next(hard_iter)
            hard_batches += 1

            try:
                aug_items = next(aug_iter)
            except StopIteration:
                aug_iter = iter(self.aug_loader)
                aug_items = next(aug_iter)
            aug_batches += 1

            combined = list(easy_items) + list(hard_items) + list(aug_items)
            if len(combined) > 1:
                order = rng.permutation(len(combined))
                combined = [combined[idx] for idx in order]
            yield self.collate_fn(combined)

        total_per_batch = max(1, self._easy_per_batch + self._hard_per_batch + self._aug_per_batch)
        ratios = {
            self._stream_names[0]: self._easy_per_batch / total_per_batch,
            self._stream_names[1]: self._hard_per_batch / total_per_batch,
            self._stream_names[2]: self._aug_per_batch / total_per_batch,
        }
        self._last_epoch_stats = {
            "epoch": epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "stream_ratios": ratios,
            "stream_per_batch": {
                self._stream_names[0]: self._easy_per_batch,
                self._stream_names[1]: self._hard_per_batch,
                self._stream_names[2]: self._aug_per_batch,
            },
            "stream_batches": {
                self._stream_names[0]: easy_batches,
                self._stream_names[1]: hard_batches,
                self._stream_names[2]: aug_batches,
            },
        }

    def get_last_epoch_stats(self) -> dict[str, float | int] | None:
        return self._last_epoch_stats


# map of argument names to the classes
EXTRACTORS: dict[str, type] = {
    "XLSR_300M": XLSR_300M,
    "XLSR_1B": XLSR_1B,
    "XLSR_2B": XLSR_2B,
}
CLASSIFIERS: Dict[str, Tuple[type, Dict[str, type]]] = {
    # Maps the classifier to tuples of the corresponding class and the initializable arguments
    "FF": (FF, {"num_classes": int, "embedding_dim": int}),
    "FFAttn1": (FFAttn1, {}),
    "FFAttn2": (FFAttn2, {}),
    "FFAttn3": (FFAttn3, {}),
    "FFAttn4": (FFAttn4, {}),
    "FFAttn5": (FFAttn5, {}),
    "FFConcat1": (FFConcat1, {}),
    "FFConcat2": (FFConcat2, {}),
    "FFConcat3": (FFConcat3, {}),
    "FFCosine": (FFCosine, {}),
    "FFCosineJoint": (FFCosineJoint, {"num_classes": int, "pair_loss_weight": float}),
    "FFCosine1": (FFCosine1, {"num_classes": int}),
    "FFCosine3": (FFCosine3, {}),
    "FFCosineRaw": (FFCosineRaw, {"l2_normalize": bool}),
    "FFCosineRaw2": (FFCosineRaw2, {"l2_normalize": bool, "num_classes": int}),
    "FFMulticlass": (FFMulticlass, {"num_classes": int}),
    "FFDiff": (FFDiff, {}),
    "FFDiffAbs": (FFDiffAbs, {}),
    "FFDiffQuadratic": (FFDiffQuadratic, {}),
    "FFLSTM": (FFLSTM, {}),
    "FFLSTM2": (FFLSTM2, {}),
}
TRAINERS = {  # Maps the classifier to the trainer
    "FF": FFTrainer,
    "FFAttn1": FFPairTrainer,
    "FFAttn2": FFPairTrainer,
    "FFAttn3": FFPairTrainer,
    "FFAttn4": FFPairTrainer,
    "FFAttn5": FFPairTrainer,
    "FFConcat1": FFPairTrainer,
    "FFConcat2": FFPairTrainer,
    "FFConcat3": FFPairTrainer,
    "FFCosine": FFPairTrainer,
    "FFCosineJoint": FFCosineJointTrainer,
    "FFCosine1": FFPairTrainer,
    "FFCosine3": FFPairTrainer,
    "FFCosineRaw": FFCosineRawTrainer,
    "FFCosineRaw2": FFCosineRawTrainer,
    "FFMulticlass": FFTrainer,
    "FFDiff": FFPairTrainer,
    "FFDiffAbs": FFPairTrainer,
    "FFDiffQuadratic": FFPairTrainer,
    "FFLSTM": FFPairTrainer,
    "FFLSTM2": FFPairTrainer,
}
DATASETS = {  # map the dataset name to the dataset class
    "MLAADSampleDataset_pair": MLAADDataset_pair,
    "MLAADMinimalDataset_pair": MLAADDataset_pair,
    "MLAADIntermediateDataset_pair": MLAADDataset_pair,
    "MLAADCuratedDataset_pair": MLAADDataset_pair,
    "MLAADCuratedBalancedDataset_pair": MLAADDataset_pair,
    "MLAADHardminedFFCosineDataset_pair": MLAADDataset_pair,
    "MLAADHardminedFFConcatDataset_pair": MLAADDataset_pair,
    "MLAADDirectionalFFCosineDataset_pair": MLAADDataset_pair,
    "MLAADDirectionalFFConcatDataset_pair": MLAADDataset_pair,
    "MLAADCurriculumHardminedFFCosineDataset_pair": MLAADDataset_pair,
    "MLAADCurriculumHardminedFFCosineDataset_pair_s123": MLAADDataset_pair,
    "MLAADCurriculumHardminedFFCosineDataset_pair_s222": MLAADDataset_pair,
    "MLAADCurriculumHardminedFFConcatDataset_pair": MLAADDataset_pair,
    "MLAADCurriculumDirectionalFFCosineDataset_pair": MLAADDataset_pair,
    "MLAADCurriculumDirectionalFFCosineDataset_pair_s123": MLAADDataset_pair,
    "MLAADCurriculumDirectionalFFCosineDataset_pair_s222": MLAADDataset_pair,
    "MLAADCurriculumDirectionalFFConcatDataset_pair": MLAADDataset_pair,
    "MLAADCurriculumRivalFFCosineDataset_pair": MLAADDataset_pair,
    "MLAADCurriculumRivalFFCosineDataset_pair_s123": MLAADDataset_pair,
    "MLAADCurriculumRivalFFCosineDataset_pair_s222": MLAADDataset_pair,
    "MLAADCurriculumRivalFFConcatDataset_pair": MLAADDataset_pair,
    "MLAADCurriculumRivalFFMulticlassDataset_pair": MLAADDataset_pair,
    "MLAADDataset_single": MLAADDataset_single,
    "STOPADataset_pair": STOPADataset_pair,
}


def get_dataloaders(
    dataset: str = "MLAADIntermediateDataset_pair",
    config: dict = local_config,
    lstm: bool = False,
    augment: bool = False,
    eval_only: bool = False,
    load_eval: bool = True,
    train_protocol: str | None = None,
    dev_protocol: str | None = None,
    eval_protocol: str | None = None,
    allowed_classes: Iterable[str] | None = None,
    curriculum_easy_csv: str | None = None,
    curriculum_hard_csv: str | None = None,
    curriculum_steps_per_epoch: int | None = None,
    curriculum_three_stream: bool = False,
    curriculum_hard_neg_ratio: float | None = None,
    curriculum_aug_csv: str | None = None,
    curriculum_aug_ratio: float | None = None,
    curriculum_easy_ratio: float | None = None,
    curriculum_hard_ratio: float | None = None,
    curriculum_pairs_per_epoch: int | None = None,
    curriculum_total_epochs: int | None = None,
    benign_aug_prob_min: float | None = None,
    benign_aug_prob_max: float | None = None,
    seed: int | None = None,
    train_batch_size: int | None = None,
    dev_batch_size: int | None = None,
    train_num_workers: int | None = None,
    dev_num_workers: int | None = None,
    train_prefetch_factor: int | None = None,
    dev_prefetch_factor: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader | None] | DataLoader:

    # Get the dataset class and config
    # Always train on ASVspoof2019LA, evaluate on the specified dataset (except ASVspoof5)
    allowed_classes_list = _normalize_allowed_classes(allowed_classes)
    dataset_config = {}
    t = "pair" if "pair" in dataset else "single"
    if "MLAAD" in dataset:
        train_dataset_class = DATASETS[dataset]
        eval_dataset_class = DATASETS[dataset]
        if "single" in dataset:
            dataset_config = config.get("mlaad_single", config["mlaad"])
        elif "CurriculumHardminedFFCosineDataset_pair_s123" in dataset:
            dataset_config = config["mlaad_curriculum_hardmined_ffcosine_s123"]
        elif "CurriculumHardminedFFCosineDataset_pair_s222" in dataset:
            dataset_config = config["mlaad_curriculum_hardmined_ffcosine_s222"]
        elif "CurriculumHardminedFFCosine" in dataset:
            dataset_config = config["mlaad_curriculum_hardmined_ffcosine"]
        elif "CurriculumHardminedFFConcat" in dataset:
            dataset_config = config["mlaad_curriculum_hardmined_ffconcat"]
        elif "CurriculumDirectionalFFCosineDataset_pair_s123" in dataset:
            dataset_config = config["mlaad_curriculum_directional_ffcosine_s123"]
        elif "CurriculumDirectionalFFCosineDataset_pair_s222" in dataset:
            dataset_config = config["mlaad_curriculum_directional_ffcosine_s222"]
        elif "CurriculumDirectionalFFCosine" in dataset:
            dataset_config = config["mlaad_curriculum_directional_ffcosine"]
        elif "CurriculumDirectionalFFConcat" in dataset:
            dataset_config = config["mlaad_curriculum_directional_ffconcat"]
        elif "CurriculumRivalFFCosine_s123" in dataset:
            dataset_config = config["mlaad_curriculum_rival_ffcosine_s123"]
        elif "CurriculumRivalFFCosine_s222" in dataset:
            dataset_config = config["mlaad_curriculum_rival_ffcosine_s222"]
        elif "CurriculumRivalFFCosine" in dataset:
            dataset_config = config["mlaad_curriculum_rival_ffcosine"]
        elif "CurriculumRivalFFConcat" in dataset:
            dataset_config = config["mlaad_curriculum_rival_ffconcat"]
        elif "CurriculumRivalFFMulticlass" in dataset:
            dataset_config = config["mlaad_curriculum_rival_ffmulticlass"]
        elif "HardminedFFCosine" in dataset:
            dataset_config = config["mlaad_hardmined_ffcosine"]
        elif "HardminedFFConcat" in dataset:
            dataset_config = config["mlaad_hardmined_ffconcat"]
        elif "DirectionalFFCosine" in dataset:
            dataset_config = config["mlaad_directional_ffcosine"]
        elif "DirectionalFFConcat" in dataset:
            dataset_config = config["mlaad_directional_ffconcat"]
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
    elif "STOPA" in dataset:
        train_dataset_class = DATASETS[dataset]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["stopa"]
    # elif "MLDF" in dataset:
    #     return mldf_dataloader
    else:
        raise ValueError("Invalid dataset name.")

    dataset_config = dict(dataset_config)
    if train_protocol is not None:
        dataset_config["train_protocol"] = train_protocol
    if dev_protocol is not None:
        dataset_config["dev_protocol"] = dev_protocol
    if eval_protocol is not None:
        dataset_config["eval_protocol"] = eval_protocol

    if "curriculum_easy_protocol" in dataset_config:
        if curriculum_easy_csv is None:
            curriculum_easy_csv = dataset_config["curriculum_easy_protocol"]
        if curriculum_hard_csv is None:
            curriculum_hard_csv = dataset_config["curriculum_hard_protocol"]
        if curriculum_steps_per_epoch is None:
            curriculum_steps_per_epoch = dataset_config.get("curriculum_steps_per_epoch")
        curriculum_three_stream = dataset_config.get(
            "curriculum_three_stream", curriculum_three_stream
        )
        curriculum_hard_neg_ratio = dataset_config.get(
            "curriculum_hard_neg_ratio", curriculum_hard_neg_ratio
        )
        if curriculum_aug_csv is None:
            curriculum_aug_csv = dataset_config.get("curriculum_aug_protocol")
        if curriculum_aug_ratio is None:
            curriculum_aug_ratio = dataset_config.get("curriculum_aug_ratio")
        if curriculum_easy_ratio is None:
            curriculum_easy_ratio = dataset_config.get("curriculum_easy_ratio")
        if curriculum_hard_ratio is None:
            curriculum_hard_ratio = dataset_config.get("curriculum_hard_ratio")
        curriculum_pairs_per_epoch = dataset_config.get(
            "curriculum_pairs_per_epoch", curriculum_pairs_per_epoch
        )

    # Common parameters
    collate_func = custom_single_batch_create if "single" in dataset else custom_pair_batch_create
    base_train_bs = config["lstm_batch_size"] if lstm else config["batch_size"]
    train_bs = train_batch_size if train_batch_size else base_train_bs
    default_dev_bs = config.get("dev_batch_size")
    if default_dev_bs is None:
        multiplier = config.get("eval_batch_size_multiplier", 1)
        default_dev_bs = max(1, int(train_bs * multiplier))
    dev_bs = dev_batch_size if dev_batch_size else default_dev_bs

    base_train_workers = config.get("train_num_workers", 0)
    base_dev_workers = config.get("dev_num_workers", base_train_workers)
    train_workers = train_num_workers if train_num_workers is not None else base_train_workers
    dev_workers = dev_num_workers if dev_num_workers is not None else base_dev_workers

    base_train_prefetch = config.get("train_prefetch_factor")
    base_dev_prefetch = config.get("dev_prefetch_factor", base_train_prefetch)
    train_prefetch = train_prefetch_factor if train_prefetch_factor is not None else base_train_prefetch
    dev_prefetch = dev_prefetch_factor if dev_prefetch_factor is not None else base_dev_prefetch

    base_pin_memory = config.get("pin_memory", torch.cuda.is_available())
    pin_memory_flag = pin_memory if pin_memory is not None else base_pin_memory
    pin_memory_flag = pin_memory_flag and torch.cuda.is_available()

    persistent_flag = persistent_workers if persistent_workers is not None else config.get("persistent_workers", False)

    # Load the datasets
    train_dataloader = DataLoader(Dataset())  # dummy dataloader for type hinting compliance
    val_dataloader = DataLoader(Dataset())  # dummy dataloader for type hinting compliance
    if curriculum_total_epochs is None:
        curriculum_total_epochs = config.get("num_epochs")
    curriculum_enabled = curriculum_easy_csv is not None or curriculum_hard_csv is not None
    curriculum_aug_enabled = curriculum_aug_ratio is not None and float(curriculum_aug_ratio) > 0
    if curriculum_aug_ratio is not None and float(curriculum_aug_ratio) < 0:
        raise ValueError("curriculum_aug_ratio must be >= 0.")
    if curriculum_enabled and (curriculum_easy_csv is None or curriculum_hard_csv is None):
        raise ValueError("Both curriculum_easy_csv and curriculum_hard_csv must be provided.")
    if curriculum_aug_enabled and not curriculum_enabled:
        raise ValueError("Augmented curriculum requires curriculum_easy_csv and curriculum_hard_csv.")
    if not curriculum_aug_enabled and any(
        v is not None for v in (curriculum_aug_csv, curriculum_easy_ratio, curriculum_hard_ratio)
    ):
        raise ValueError(
            "Augmented curriculum options require --curriculum-aug-ratio to be set."
        )
    if curriculum_enabled:
        if "MLAAD" not in dataset or "pair" not in dataset:
            raise ValueError("Curriculum sampling is currently supported for MLAAD pair datasets.")
        if curriculum_total_epochs is None:
            raise ValueError("curriculum_total_epochs must be provided when curriculum is enabled.")

    benign_prob_min = 0.25 if benign_aug_prob_min is None else float(benign_aug_prob_min)
    benign_prob_max = 0.375 if benign_aug_prob_max is None else float(benign_aug_prob_max)
    if benign_prob_min < 0.0 or benign_prob_max > 1.0 or benign_prob_min > benign_prob_max:
        raise ValueError("benign_aug_prob_min/max must satisfy 0 <= min <= max <= 1.")

    if not eval_only:
        print("Loading training datasets...")
        train_kwargs = {
            "root_dir": config["data_dir"] + dataset_config["train_subdir"],
            "protocol_file_name": dataset_config["train_protocol"],
            "variant": "train",
            "augment": augment,
            "rir_root": config["rir_root"],
        }
        if "MLAAD" in dataset:
            train_kwargs["segment_seconds"] = config.get("segment_seconds")
            train_kwargs["sample_rate"] = config.get("sample_rate", 16000)
            if allowed_classes_list is not None:
                train_kwargs["allowed_classes"] = allowed_classes_list
        easy_dataset = None
        hard_dataset = None
        if curriculum_enabled:
            easy_kwargs = dict(train_kwargs)
            easy_kwargs["protocol_file_name"] = curriculum_easy_csv
            hard_kwargs = dict(train_kwargs)
            hard_kwargs["protocol_file_name"] = curriculum_hard_csv
            easy_dataset = train_dataset_class(**easy_kwargs)
            hard_dataset = train_dataset_class(**hard_kwargs)
            train_dataset = hard_dataset
        else:
            train_dataset = train_dataset_class(**train_kwargs)

        use_benign_pair_aug = augment and "MLAAD" in dataset and "pair" in dataset
        if use_benign_pair_aug and not curriculum_aug_enabled:
            if curriculum_enabled:
                if easy_dataset is None:
                    raise ValueError("Easy dataset is required for benign pair augmentation.")
                easy_dataset = PairDatasetWithBenignAugment(
                    easy_dataset,
                    prob_min=benign_prob_min,
                    prob_max=benign_prob_max,
                    sample_rate=config.get("sample_rate", 16000),
                )
            else:
                train_dataset = PairDatasetWithBenignAugment(
                    train_dataset,
                    prob_min=benign_prob_min,
                    prob_max=benign_prob_max,
                    sample_rate=config.get("sample_rate", 16000),
                )

        dev_kwargs = {  # kwargs for the dataset class
            "root_dir": config["data_dir"] + dataset_config["dev_subdir"],
            "protocol_file_name": dataset_config["dev_protocol"],
            "variant": "dev",
        }
        if "MLAAD" in dataset:
            dev_kwargs["segment_seconds"] = config.get("segment_seconds")
            dev_kwargs["sample_rate"] = config.get("sample_rate", 16000)
            dev_kwargs["allow_unknown"] = "drop"
            if allowed_classes_list is not None:
                dev_kwargs["allowed_classes"] = allowed_classes_list
        label_map = getattr(train_dataset, "label_map", None)
        if label_map is not None:
            dev_kwargs["label_map"] = label_map
        if "2021DF" in dataset:  # 2021DF has a local variant
            dev_kwargs["local"] = True if "--local" in config["argv"] else False
            dev_kwargs["variant"] = "progress"
            val_dataset = eval_dataset_class(**dev_kwargs)
        else:
            # Create the dataset based on dynamically created dev_kwargs
            val_dataset = train_dataset_class(**dev_kwargs)

        # create dataloader, use custom collate_fn to pad the data to the longest recording in batch
        train_loader_kwargs = {
            "batch_size": train_bs,
            "collate_fn": collate_func,
            "drop_last": True,
            "num_workers": train_workers,
            "pin_memory": pin_memory_flag,
        }
        if train_workers > 0:
            train_loader_kwargs["persistent_workers"] = persistent_flag
            if train_prefetch:
                train_loader_kwargs["prefetch_factor"] = train_prefetch

        if curriculum_enabled:
            seed_value = seed if seed is not None else 0
            if not curriculum_three_stream:
                raise ValueError(
                    "Only deterministic curriculum mixing is supported; set --curriculum-three-stream."
                )

            if curriculum_aug_enabled:
                if easy_dataset is None:
                    raise ValueError("Easy dataset is required for augmented curriculum mixing.")
                if use_benign_pair_aug:
                    easy_dataset = PairDatasetWithBenignAugment(
                        easy_dataset,
                        prob_min=benign_prob_min,
                        prob_max=benign_prob_max,
                        sample_rate=config.get("sample_rate", 16000),
                    )
                ratios = _resolve_curriculum_stream_ratios(
                    curriculum_aug_ratio,
                    curriculum_easy_ratio,
                    curriculum_hard_ratio,
                )
                pos_bs = max(1, train_bs // 2)
                neg_bs = train_bs - pos_bs
                if neg_bs < 1:
                    raise ValueError("Balanced curriculum requires at least 1 negative per batch.")
                stream_counts = _allocate_stream_batch_sizes(neg_bs, ratios)
                mix_batch_size = pos_bs + sum(stream_counts.values())

                steps_default = max(1, len(easy_dataset) // mix_batch_size)
                if curriculum_steps_per_epoch is not None:
                    steps_per_epoch = curriculum_steps_per_epoch
                elif curriculum_pairs_per_epoch is not None:
                    steps_per_epoch = max(1, int(math.ceil(curriculum_pairs_per_epoch / mix_batch_size)))
                else:
                    steps_per_epoch = steps_default

                aug_protocol = curriculum_aug_csv if curriculum_aug_csv is not None else curriculum_easy_csv
                if aug_protocol is None:
                    raise ValueError("curriculum_aug_csv must be provided for augmented curriculum mixing.")
                aug_kwargs = dict(train_kwargs)
                aug_kwargs["protocol_file_name"] = aug_protocol
                if "MLAAD" in dataset:
                    aug_kwargs["segment_seconds"] = config.get("segment_seconds")
                    aug_kwargs["sample_rate"] = config.get("sample_rate", 16000)
                    if allowed_classes_list is not None:
                        aug_kwargs["allowed_classes"] = allowed_classes_list
                aug_dataset = MLAADAugmentedNegativePairDataset(**aug_kwargs)

                sub_loader_kwargs = dict(train_loader_kwargs)
                sub_loader_kwargs.pop("batch_size", None)
                sub_loader_kwargs.pop("sampler", None)
                sub_loader_kwargs["collate_fn"] = _identity_collate
                sub_loader_kwargs["shuffle"] = True

                pos_dataset = _build_label_subset(easy_dataset, 1)
                easy_neg_dataset = _build_label_subset(easy_dataset, 0)
                hard_neg_dataset = _build_label_subset(hard_dataset, 0)

                pos_loader = DataLoader(pos_dataset, batch_size=pos_bs, **sub_loader_kwargs)
                easy_neg_loader = DataLoader(
                    easy_neg_dataset, batch_size=stream_counts["easy"], **sub_loader_kwargs
                )
                hard_neg_loader = DataLoader(
                    hard_neg_dataset, batch_size=stream_counts["hard"], **sub_loader_kwargs
                )
                aug_neg_loader = DataLoader(
                    aug_dataset, batch_size=stream_counts["aug"], **sub_loader_kwargs
                )

                train_dataloader = CurriculumQuadMixLoader(
                    pos_loader,
                    easy_neg_loader,
                    hard_neg_loader,
                    aug_neg_loader,
                    steps_per_epoch=steps_per_epoch,
                    seed=seed_value,
                    collate_fn=collate_func,
                    pos_per_batch=pos_bs,
                    easy_neg_per_batch=stream_counts["easy"],
                    hard_neg_per_batch=stream_counts["hard"],
                    aug_neg_per_batch=stream_counts["aug"],
                )
                mode_note = "deterministic 4-stream (pos/easy-neg/hard-neg/aug-neg)"
                batch_note = (
                    f"batch_size={mix_batch_size} "
                    f"(pos:{pos_bs}, "
                    f"easy_neg:{stream_counts['easy']}, "
                    f"hard_neg:{stream_counts['hard']}, "
                    f"aug_neg:{stream_counts['aug']})"
                )
                pairs_note = (
                    f", pairs_per_epoch={curriculum_pairs_per_epoch}"
                    if curriculum_pairs_per_epoch is not None
                    else ""
                )
                ratio_note = (
                    f" neg_ratios=({ratios['easy']:.2f},"
                    f"{ratios['hard']:.2f},"
                    f"{ratios['aug']:.2f})"
                )
                print(
                    "Curriculum mixing enabled: "
                    f"easy={curriculum_easy_csv}, hard={curriculum_hard_csv}, "
                    f"aug={aug_protocol}, steps_per_epoch={steps_per_epoch}, "
                    f"mode={mode_note}, {batch_note}{pairs_note}{ratio_note}"
                )
            else:
                if train_bs < 4:
                    raise ValueError(
                        "Three-stream curriculum requires train_batch_size >= 4 for 2:1:1 mixing."
                    )
                hard_ratio = (
                    0.5 if curriculum_hard_neg_ratio is None else float(curriculum_hard_neg_ratio)
                )
                if not (0.0 < hard_ratio < 1.0):
                    raise ValueError("curriculum_hard_neg_ratio must be between 0 and 1 (exclusive).")

                pos_bs = max(1, train_bs // 2)
                pos_bs = min(train_bs - 2, pos_bs)
                neg_bs = train_bs - pos_bs
                if neg_bs < 2:
                    raise ValueError("Three-stream curriculum requires at least 2 negatives per batch.")

                hard_neg_bs = int(round(neg_bs * hard_ratio))
                hard_neg_bs = max(1, min(neg_bs - 1, hard_neg_bs))
                easy_neg_bs = neg_bs - hard_neg_bs

                mix_batch_size = pos_bs + easy_neg_bs + hard_neg_bs
                steps_default = max(1, len(easy_dataset) // mix_batch_size)
                if curriculum_steps_per_epoch is not None:
                    steps_per_epoch = curriculum_steps_per_epoch
                elif curriculum_pairs_per_epoch is not None:
                    steps_per_epoch = max(1, int(math.ceil(curriculum_pairs_per_epoch / mix_batch_size)))
                else:
                    steps_per_epoch = steps_default

                pos_dataset = _build_label_subset(easy_dataset, 1)
                easy_neg_dataset = _build_label_subset(easy_dataset, 0)
                hard_neg_dataset = _build_label_subset(hard_dataset, 0)

                sub_loader_kwargs = dict(train_loader_kwargs)
                sub_loader_kwargs.pop("batch_size", None)
                sub_loader_kwargs.pop("sampler", None)
                sub_loader_kwargs["collate_fn"] = _identity_collate
                sub_loader_kwargs["shuffle"] = True

                pos_loader = DataLoader(pos_dataset, batch_size=pos_bs, **sub_loader_kwargs)
                easy_neg_loader = DataLoader(
                    easy_neg_dataset, batch_size=easy_neg_bs, **sub_loader_kwargs
                )
                hard_neg_loader = DataLoader(
                    hard_neg_dataset, batch_size=hard_neg_bs, **sub_loader_kwargs
                )

                train_dataloader = CurriculumTriMixLoader(
                    pos_loader,
                    easy_neg_loader,
                    hard_neg_loader,
                    steps_per_epoch=steps_per_epoch,
                    seed=seed_value,
                    collate_fn=collate_func,
                    pos_per_batch=pos_bs,
                    easy_neg_per_batch=easy_neg_bs,
                    hard_neg_per_batch=hard_neg_bs,
                )
                mode_note = "deterministic 3-stream (pos/easy-neg/hard-neg)"
                batch_note = (
                    f"batch_size={mix_batch_size}, hard_neg_ratio={hard_ratio:.2f} "
                    f"(pos:{pos_bs}, easy_neg:{easy_neg_bs}, hard_neg:{hard_neg_bs})"
                )

                pairs_note = (
                    f", pairs_per_epoch={curriculum_pairs_per_epoch}"
                    if curriculum_pairs_per_epoch is not None
                    else ""
                )
                print(
                    "Curriculum mixing enabled: "
                    f"easy={curriculum_easy_csv}, hard={curriculum_hard_csv}, "
                    f"steps_per_epoch={steps_per_epoch}, mode={mode_note}, {batch_note}{pairs_note}"
                )
        else:
            # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
            weighted_sampler = _build_weighted_sampler(train_dataset, seed=seed)
            train_dataloader = DataLoader(train_dataset, sampler=weighted_sampler, **train_loader_kwargs)
        val_loader_kwargs = {
            "batch_size": dev_bs,
            "collate_fn": collate_func,
            "shuffle": True,
            "num_workers": dev_workers,
            "pin_memory": pin_memory_flag,
        }
        if dev_workers > 0:
            val_loader_kwargs["persistent_workers"] = persistent_flag
            if dev_prefetch:
                val_loader_kwargs["prefetch_factor"] = dev_prefetch
        val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)

    eval_dataloader: DataLoader | None = None
    if eval_only:
        load_eval = True

    if load_eval:
        print("Loading eval dataset...")
        eval_kwargs = {  # kwargs for the dataset class
            "root_dir": config["data_dir"] + dataset_config["eval_subdir"],
            "protocol_file_name": dataset_config["eval_protocol"],
            "variant": "eval",
        }
        if "2021DF" in dataset:  # 2021DF has a local variant
            eval_kwargs["local"] = True if "--local" in config["argv"] else False
        if "MLAAD" in dataset:
            eval_kwargs["segment_seconds"] = config.get("segment_seconds")
            eval_kwargs["sample_rate"] = config.get("sample_rate", 16000)
            eval_kwargs["allow_unknown"] = "drop"
            if allowed_classes_list is not None:
                eval_kwargs["allowed_classes"] = allowed_classes_list
        label_map = None if eval_only else getattr(train_dataloader.dataset, "label_map", None)
        if label_map is not None:
            eval_kwargs["label_map"] = label_map

        eval_dataset = eval_dataset_class(**eval_kwargs)
        eval_loader_kwargs = {
            "batch_size": dev_bs,
            "collate_fn": collate_func,
            "shuffle": True,
            "num_workers": dev_workers,
            "pin_memory": pin_memory_flag,
        }
        if dev_workers > 0:
            eval_loader_kwargs["persistent_workers"] = persistent_flag
            if dev_prefetch:
                eval_loader_kwargs["prefetch_factor"] = dev_prefetch
        eval_dataloader = DataLoader(eval_dataset, **eval_loader_kwargs)
    else:
        print("Skipping eval dataset loading.")

    if eval_only:
        if eval_dataloader is None:
            raise RuntimeError("Eval dataset was not loaded despite eval_only=True.")
        return eval_dataloader
    else:
        return train_dataloader, val_dataloader, eval_dataloader


def build_model(args: Namespace) -> Tuple[FFBase, BaseTrainer]:
    # region Extractor
    extractor_cls = EXTRACTORS[args.extractor]
    extractor_kwargs: dict[str, object] = {}
    if getattr(args, "finetune_ssl", False):
        extractor_kwargs["finetune"] = True
    try:
        extractor = extractor_cls(**extractor_kwargs)
    except TypeError:
        # Backward compatibility for extractors without finetune kwarg.
        extractor = extractor_cls()
    if getattr(args, "finetune_ssl", False) and hasattr(extractor, "finetune"):
        extractor.finetune = True
    # endregion

    # region Processor (pooling)
    processor = None
    if args.processor == "MHFA":
        input_transformer_nb = extractor.transformer_layers
        input_dim = extractor.feature_size

        processor_output_dim = (
            input_dim  # Output the same dimension as input - might want to play around with this
        )
        compression_dim = processor_output_dim // 8
        head_nb = round(
            input_transformer_nb * 4 / 3
        )  # Half random guess number, half based on the paper and testing

        processor = MHFA(
            head_nb=head_nb,
            input_transformer_nb=input_transformer_nb,
            inputs_dim=input_dim,
            compression_dim=compression_dim,
            outputs_dim=processor_output_dim,
        )
    elif args.processor == "AASIST":
        processor = AASIST(
            inputs_dim=extractor.feature_size,
            # compression_dim=extractor.feature_size // 8,  # compression dim is hardcoded at the moment
            outputs_dim=extractor.feature_size,  # Output the same dimension as input, might want to play around with this
        )
    else:
        raise ValueError("Only AASIST and MHFA processors are supported in this release.")
    # endregion

    # region Model and trainer
    model: FFBase
    trainer = None
    match args.classifier:
        case _:
            try:
                cls_input_dim = extractor.feature_size
                ff_kwargs = {}
                if args.classifier in {"FF", "FFCosine1", "FFCosineRaw2", "FFCosineJoint"} and getattr(args, "num_classes", None):
                    ff_kwargs["num_classes"] = args.num_classes
                if args.classifier == "FF":
                    embedding_dim = getattr(args, "embedding_dim", None)
                    if embedding_dim is not None:
                        ff_kwargs["embedding_dim"] = int(embedding_dim)
                if args.classifier in {"FFCosineRaw", "FFCosineRaw2"}:
                    l2_normalize = getattr(args, "l2_normalize", None)
                    if l2_normalize is not None:
                        ff_kwargs["l2_normalize"] = bool(l2_normalize)
                if args.classifier == "FFCosineJoint":
                    pair_loss_weight = getattr(args, "pair_loss_weight", None)
                    if pair_loss_weight is not None:
                        ff_kwargs["pair_loss_weight"] = float(pair_loss_weight)
                
                # Optional classifier-specific init args
                if args.classifier == "FFCosine3":
                    ff_kwargs["bottleneck_dim"] = getattr(args, "bottleneck_dim", 50)
                elif args.classifier == "FFMulticlass":
                    ff_kwargs["bottleneck_dim"] = getattr(args, "bottleneck_dim", 50)
                    # Only pass num_classes if explicitly provided (truthy).
                    # Otherwise, let FFMulticlass.__init__ use its default (2).
                    nc = getattr(args, "num_classes", None)
                    if nc:
                        ff_kwargs["num_classes"] = nc
                # FFCosine and FFConcat3 (original) do not need these args in their constructor

                model = CLASSIFIERS[str(args.classifier)][0](
                    extractor, processor, in_dim=cls_input_dim, **ff_kwargs
                )
                if hasattr(model, "get_param_groups"):
                    model._ssl_finetune = bool(getattr(args, "finetune_ssl", False))
                    model._extractor_lr = getattr(args, "extractor_lr", None)
                trainer = TRAINERS[str(args.classifier)](model)
            except KeyError:
                raise ValueError(f"Invalid classifier, should be one of: {list(CLASSIFIERS.keys())}")
    # endregion

    # Print model info
    print(f"Building {type(model).__name__} model with {type(model.extractor).__name__} extractor", end="")
    if isinstance(model, FFBase):
        print(f" and {type(model.feature_processor).__name__} processor.")
    else:
        print(".")

    return model, trainer
