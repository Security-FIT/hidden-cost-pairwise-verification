import random

import torch
import torch.nn as nn
import torchaudio.transforms as T


class ForensicAugmenter(nn.Module):
    def __init__(self, sample_rate: int = 22050):
        super().__init__()
        # 1. VITS vs NEON (Precision/Quantization)
        self.mulaw = nn.Sequential(
            T.MuLawEncoding(quantization_channels=256),
            T.MuLawDecoding(quantization_channels=256),
        )
        # 2. BARK vs BARK-SMALL (Fidelity/Bandwidth)
        self.quality_crush = nn.Sequential(
            T.Resample(orig_freq=sample_rate, new_freq=8000),
            T.Resample(orig_freq=8000, new_freq=sample_rate),
        )
        # 3. RUNTIME ARTIFACTS (Resampling/Aliasing)
        resample_target = int(round(sample_rate * 0.75))
        resample_target = min(16000, resample_target)
        if resample_target >= sample_rate:
            resample_target = max(1000, int(round(sample_rate * 0.6)))
        self.resample_glitch = nn.Sequential(
            T.Resample(orig_freq=sample_rate, new_freq=resample_target),
            T.Resample(orig_freq=resample_target, new_freq=sample_rate),
        )
        # 4. CODEC (MP3/Vorbis simulation via bit-depth truncation)
        self._bit_crush_scale = 512.0  # 10-bit

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        method = random.choice(["mulaw", "quality", "resample", "bitcrush"])
        if method == "mulaw":
            return self.mulaw(waveform)
        if method == "quality":
            return self.quality_crush(waveform)
        if method == "resample":
            return self.resample_glitch(waveform)
        if method == "bitcrush":
            return torch.round(waveform * self._bit_crush_scale) / self._bit_crush_scale
        return waveform
