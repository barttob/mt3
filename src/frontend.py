"""SpectrogramFrontend: converts raw audio waveforms to log-mel spectrograms."""

import torch
import torch.nn as nn
import torchaudio


class SpectrogramFrontend(nn.Module):
    """Convert raw audio waveform to log-mel spectrogram.

    Args:
        sample_rate: Audio sample rate in Hz. Default: 16000.
        n_fft: FFT size. Default: 2048.
        hop_length: Hop length between frames. Default: 128.
        n_mels: Number of mel filterbanks. Default: 512.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 128,
        n_mels: int = 512,
    ) -> None:
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    @torch.compiler.disable
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram from waveform.

        Args:
            waveform: Raw audio tensor of shape (B, num_samples).

        Returns:
            log_mel: Log-mel spectrogram of shape (B, n_mels, num_frames).
        """
        mel = self.mel_spec(waveform)          # (B, n_mels, T)
        log_mel = torch.log(mel + 1e-6)        # log scale
        return log_mel


if __name__ == "__main__":
    frontend = SpectrogramFrontend(
        sample_rate=16000,
        n_fft=2048,
        hop_length=128,
        n_mels=512,
    )

    # Random waveform: batch=2, 16 seconds at 16 kHz
    waveform = torch.randn(2, 256000)
    log_mel = frontend(waveform)

    print(f"Input shape:  {waveform.shape}")
    print(f"Output shape: {log_mel.shape}")

    # Expected: (2, 512, ~2001)
    assert log_mel.ndim == 3, "Output must be 3-dimensional"
    assert log_mel.shape[0] == 2, f"Batch size mismatch: {log_mel.shape[0]}"
    assert log_mel.shape[1] == 512, f"n_mels mismatch: {log_mel.shape[1]}"
    assert 1990 <= log_mel.shape[2] <= 2010, f"Unexpected frame count: {log_mel.shape[2]}"

    print("Smoke test passed.")
