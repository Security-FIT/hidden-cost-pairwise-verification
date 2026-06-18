import random

import numpy as np
import torch

from augmentation.BenignAugmenter import BenignAugmenter


class PairDatasetWithBenignAugment(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        prob_min: float = 0.1,
        prob_max: float = 0.2,
        sample_rate: int = 16000,
        augmenter: BenignAugmenter | None = None,
    ):
        self.base_dataset = base_dataset
        self.prob_min = prob_min
        self.prob_max = prob_max
        self._augmenter = augmenter
        self._augmenter_sample_rate = sample_rate

    def _get_augmenter(self) -> BenignAugmenter:
        if self._augmenter is None:
            self._augmenter = BenignAugmenter(sample_rate=self._augmenter_sample_rate)
        return self._augmenter

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str):
        return getattr(self.base_dataset, name)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        if len(item) < 4:
            return item
        pair_id, wav_a, wav_b, label = item
        if int(label) == 1:
            threshold = random.uniform(self.prob_min, self.prob_max)
            if random.random() < threshold:
                augmenter = self._get_augmenter()
                with torch.no_grad():
                    mode = random.choice(["a", "b", "both"])
                    if mode in ("a", "both"):
                        wav_a = augmenter(wav_a)
                    if mode in ("b", "both"):
                        wav_b = augmenter(wav_b)
        return pair_id, wav_a, wav_b, label

def custom_pair_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for batch training.

    Creates batches of pairs of genuine and spoofing speech for differential-based detection.
    Shorter waveforms are padded with zeros to match the length of the longest waveform in the batch.
    """

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    lengths_gt = torch.tensor([item[1].size(1) for item in batch])
    lengths_test = torch.tensor([item[2].size(1) for item in batch])

    # Find the maximum length
    max_length_gt = int(torch.max(lengths_gt))
    max_length_test = int(torch.max(lengths_test))

    # Pad the tensors to have the maximum length
    file_names = []
    padded_gts = torch.zeros(batch_size, max_length_gt)
    padded_tests = torch.zeros(batch_size, max_length_test)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        file_names.append(item[0])
        waveform_gt = item[1]
        waveform_test = item[2]
        padded_waveform_gt = torch.nn.functional.pad(
            waveform_gt, (0, max_length_gt - waveform_gt.size(1))
        ).squeeze(0)
        padded_waveform_test = torch.nn.functional.pad(
            waveform_test, (0, max_length_test - waveform_test.size(1))
        ).squeeze(0)
        try:  # If the label is not available (or is None), set it to np.nan
            label = torch.tensor(item[3])
        except:
            label = np.nan

        padded_gts[i] = padded_waveform_gt
        padded_tests[i] = padded_waveform_test
        labels[i] = label

    return file_names, padded_gts, padded_tests, labels

def custom_single_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for batch training.

    Creates batches of single recordings for "normal" detection.
    Shorter waveforms are padded with zeros to match the length of the longest waveform in the batch.
    """

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    lengths = torch.tensor([item[1].size(1) for item in batch])

    # Find the maximum length
    max_length = int(torch.max(lengths))

    # Pad the tensors to have the maximum length
    file_names = []
    padded_waveforms = torch.zeros(batch_size, max_length)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        file_names.append(item[0])
        waveform = item[1]
        padded_waveform = torch.nn.functional.pad(
            waveform, (0, max_length - waveform.size(1))
        ).squeeze(0)
        try:  # If the label is not available (or is None), set it to np.nan
            label = torch.tensor(item[2])
        except:
            label = np.nan

        padded_waveforms[i] = padded_waveform
        labels[i] = label

    return file_names, padded_waveforms, labels
