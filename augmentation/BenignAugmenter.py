import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class BenignAugmenter(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_snr_db: tuple[float, float] = (20.0, 35.0),
        reverb_ms: tuple[float, float] = (20.0, 60.0),
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.noise_snr_db = noise_snr_db
        self.reverb_ms = reverb_ms

    def _apply_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
        snr_db = random.uniform(*self.noise_snr_db)
        noise_std = rms / (10 ** (snr_db / 20.0))
        noise = torch.randn_like(waveform) * noise_std
        return waveform + noise

    def _apply_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        max_len = max(1, int(self.sample_rate * (random.uniform(*self.reverb_ms) / 1000.0)))
        t = torch.linspace(0.0, 1.0, max_len, device=waveform.device)
        decay = torch.exp(-t * random.uniform(3.0, 8.0))
        for _ in range(random.randint(2, 4)):
            idx = random.randint(0, max_len - 1)
            decay[idx] += random.uniform(0.1, 0.3)
        kernel = decay / (decay.sum() + 1e-8)
        kernel = kernel.view(1, 1, -1)
        wf = waveform.unsqueeze(0)
        out = F.conv1d(wf, kernel, padding=kernel.size(-1) - 1)
        out = out[:, :, : waveform.size(-1)].squeeze(0)
        return out

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        method = random.choice(["gaussian", "reverb"])
        if method == "gaussian":
            augmented = self._apply_gaussian_noise(waveform)
        else:
            augmented = self._apply_reverb(waveform)
        return torch.clamp(augmented, min=-1.0, max=1.0)
