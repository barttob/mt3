"""Waveform-level data augmentation for MT3 training.

Applies random gain perturbation, Gaussian noise injection, and time masking
to raw waveforms during training. Each augmentation is applied independently
with probability ``p``.
"""

from __future__ import annotations

import random

import torch


class WaveformAugmenter:
    """Apply random augmentations to a waveform tensor.

    Each augmentation is applied independently with probability ``p``.

    Args:
        gain_range: Min and max gain multiplier.
        noise_std: Standard deviation of additive Gaussian noise.
        time_mask_max_ratio: Maximum fraction of the waveform to zero out.
        p: Probability of applying each individual augmentation.
    """

    def __init__(
        self,
        gain_range: tuple[float, float] = (0.5, 1.5),
        noise_std: float = 0.005,
        time_mask_max_ratio: float = 0.1,
        p: float = 0.5,
    ) -> None:
        self.gain_range = gain_range
        self.noise_std = noise_std
        self.time_mask_max_ratio = time_mask_max_ratio
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Augment a waveform tensor in-place (or return a new tensor).

        Args:
            waveform: 1-D float tensor of audio samples.

        Returns:
            Augmented waveform tensor (same shape).
        """
        # Gain perturbation
        if random.random() < self.p:
            gain = random.uniform(*self.gain_range)
            waveform = waveform * gain

        # Additive Gaussian noise
        if random.random() < self.p:
            noise = torch.randn_like(waveform) * self.noise_std
            waveform = waveform + noise

        # Time masking (zero out a random contiguous region)
        if random.random() < self.p:
            length = waveform.shape[-1]
            mask_len = int(length * random.uniform(0, self.time_mask_max_ratio))
            if mask_len > 0 and length > mask_len:
                start = random.randint(0, length - mask_len)
                waveform = waveform.clone()
                waveform[..., start : start + mask_len] = 0.0

        return waveform
