from __future__ import annotations

import hashlib
from typing import Iterable, Tuple

import torch


def _hash_waveform(waveform: torch.Tensor) -> str:
    wav = waveform.detach()
    if wav.dim() > 1:
        wav = wav.view(-1)
    wav = wav.contiguous()
    nonzero = torch.nonzero(wav, as_tuple=False)
    if nonzero.numel() == 0:
        end = wav.numel()
    else:
        end = int(nonzero[-1]) + 1
    data = wav[:end].cpu().numpy().tobytes()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def build_pair_keys(
    pair_ids: Iterable[str],
    gt: torch.Tensor,
    test: torch.Tensor,
) -> Tuple[list[str], list[str]]:
    keys_gt: list[str] = []
    keys_test: list[str] = []
    for idx, pair_id in enumerate(pair_ids):
        pid = "" if pair_id is None else str(pair_id)
        if "|" in pid:
            left, right = pid.split("|", 1)
            keys_gt.append(left)
            keys_test.append(right)
        else:
            if not pid:
                pid = f"__test__{_hash_waveform(test[idx])}"
            keys_test.append(pid)
            keys_gt.append(f"__gt__{_hash_waveform(gt[idx])}")
    return keys_gt, keys_test


class EmbeddingCache:
    def __init__(self, extractor, feature_processor, device: torch.device):
        self.extractor = extractor
        self.feature_processor = feature_processor
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}

    def get_embeddings(
        self,
        waveforms: torch.Tensor,
        keys: list[str],
        amp_ctx,
    ) -> torch.Tensor:
        if len(keys) != waveforms.shape[0]:
            raise ValueError("Embedding cache: number of keys does not match batch size.")
        missing_indices = [i for i, key in enumerate(keys) if key not in self._cache]
        if missing_indices:
            missing_waveforms = waveforms[missing_indices].to(self.device)
            with amp_ctx:
                feats = self.extractor.extract_features(missing_waveforms)
                emb = self.feature_processor(feats)
            emb_cpu = emb.detach().to(dtype=torch.float32).cpu()
            for offset, idx in enumerate(missing_indices):
                self._cache[keys[idx]] = emb_cpu[offset]
        batch_emb = torch.stack([self._cache[key] for key in keys], dim=0)
        return batch_emb.to(self.device)
