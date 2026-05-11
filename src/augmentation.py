"""Waveform-level and spectrogram-level data augmentation for MT3 training.

WaveformAugmenter: gain perturbation, Gaussian noise, and multiple time masks
applied to raw waveforms. Each augmentation is applied independently with
probability ``p``.

SpecAugment: frequency masking and time masking applied to log-mel spectrograms
(Google SpecAugment, Park et al. 2019). Applied inside MT3Model.forward() only
when the model is in training mode.
"""

from __future__ import annotations

import random

import torch
import torch.compiler


class WaveformAugmenter:
    """Apply random augmentations to a waveform tensor.

    Each augmentation is applied independently with probability ``p``.

    Args:
        gain_range: Min and max gain multiplier.
        noise_std: Standard deviation of additive Gaussian noise.
        time_mask_max_ratio: Maximum fraction of the waveform to zero out per mask.
        num_time_masks: Number of independent time masks to apply.
        p: Probability of applying each individual augmentation.
    """

    def __init__(
        self,
        gain_range: tuple[float, float] = (0.5, 1.5),
        noise_std: float = 0.01,
        time_mask_max_ratio: float = 0.2,
        num_time_masks: int = 2,
        p: float = 0.5,
    ) -> None:
        self.gain_range = gain_range
        self.noise_std = noise_std
        self.time_mask_max_ratio = time_mask_max_ratio
        self.num_time_masks = num_time_masks
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Augment a waveform tensor.

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

        # Multiple time masks
        if random.random() < self.p:
            length = waveform.shape[-1]
            waveform = waveform.clone()
            for _ in range(self.num_time_masks):
                mask_len = int(length * random.uniform(0, self.time_mask_max_ratio))
                if mask_len > 0 and length > mask_len:
                    start = random.randint(0, length - mask_len)
                    waveform[..., start : start + mask_len] = 0.0

        return waveform


class SpecAugment:
    """SpecAugment: frequency and time masking on log-mel spectrograms.

    Applied to (B, n_mels, T) tensors inside MT3Model.forward() when the model
    is in training mode. Frequency masking zeros out random bands of mel bins;
    time masking zeros out random contiguous frame windows.

    Args:
        freq_mask_param: Maximum number of mel bins to mask per frequency mask.
        time_mask_param: Maximum number of frames to mask per time mask.
        num_freq_masks: Number of independent frequency masks.
        num_time_masks: Number of independent time masks.
        p: Probability of applying the full transform to each sample in the batch.
    """

    def __init__(
        self,
        freq_mask_param: int = 48,
        time_mask_param: int = 64,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        p: float = 0.8,
    ) -> None:
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    @torch.compiler.disable
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency and time masking to a log-mel spectrogram batch.

        Args:
            spec: Log-mel spectrogram of shape (B, n_mels, T).

        Returns:
            Augmented spectrogram of the same shape.
        """
        spec = spec.clone()
        n_mels = spec.shape[-2]
        T = spec.shape[-1]

        for b in range(spec.shape[0]):
            if random.random() >= self.p:
                continue

            for _ in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                f0 = random.randint(0, max(n_mels - f, 0))
                spec[b, f0 : f0 + f, :] = 0.0

            for _ in range(self.num_time_masks):
                t = random.randint(0, min(self.time_mask_param, T))
                t0 = random.randint(0, max(T - t, 0))
                spec[b, :, t0 : t0 + t] = 0.0

        return spec
