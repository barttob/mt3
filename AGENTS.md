# AGENTS.md — Building an Autoregressive MT3-like Piano Transcription Model

> **Goal**: Train a sequence-to-sequence Transformer that takes audio spectrograms as input and
> autoregressively decodes MIDI-like event tokens, targeting piano transcription on
> **MAESTRO v3.0.0** and multi-instrument transcription on **Slakh2100-FLAC-Redux**.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Token Vocabulary](#2-token-vocabulary)
3. [Audio Frontend — Spectrogram Encoder Input](#3-audio-frontend)
4. [Encoder](#4-encoder)
5. [Decoder](#5-decoder)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference & Decoding](#8-inference--decoding)
9. [Evaluation](#9-evaluation)
10. [Project Structure](#10-project-structure)
11. [Dependency List](#11-dependency-list)
12. [Hyperparameter Reference](#12-hyperparameter-reference)
13. [Common Pitfalls & Tips](#13-common-pitfalls--tips)
14. [References](#14-references)

---

## 1. Architecture Overview

The model follows the original MT3 design: a **T5-style encoder-decoder Transformer** that
frames music transcription as a **conditional language modeling** task.

```
Audio waveform
    │
    ▼
Log-Mel Spectrogram  (B, F=512, T=256)
    │
    ▼
┌──────────────────────────┐
│     Spectrogram Encoder  │  ← Transformer encoder (no causal mask)
│     (bidirectional)      │     Input: spectrogram frames as "tokens"
└──────────┬───────────────┘
           │  encoder hidden states  (B, T, D)
           ▼
┌──────────────────────────┐
│   Autoregressive Decoder │  ← Transformer decoder (causal mask)
│   (token-by-token)       │     Cross-attends to encoder states
└──────────┬───────────────┘
           │
           ▼
   Event token sequence   →  Post-process → MIDI file
```

**Key design points:**

- The encoder consumes spectrogram *frames* (each frame = one "token" for the encoder).
  There is NO convolution frontend in vanilla MT3 — frames are linearly projected.
  However, YourMT3+ showed benefits from a small ResNet pre-encoder; this is optional.
- The decoder is a standard autoregressive Transformer decoder with causal self-attention
  and cross-attention into the encoder outputs.
- The model is trained with **teacher forcing** and standard **cross-entropy loss** over the
  token vocabulary.
- At inference time, tokens are decoded greedily or with temperature sampling.

---

## 2. Token Vocabulary

MT3 uses a compact MIDI-like event vocabulary. For piano-only transcription (MAESTRO),
the vocabulary is smaller; for multi-instrument (Slakh), program tokens are added.

### 2.1 Token Types

| Token Type       | Range / Count | Description                                       |
|------------------|---------------|----------------------------------------------------|
| `<pad>`          | 0             | Padding token                                      |
| `<sos>`          | 1             | Start-of-sequence                                  |
| `<eos>`          | 2             | End-of-sequence                                    |
| `<tie>`          | 3             | Tie section separator (notes held from prev segment)|
| `time_shift`     | 4–603         | 600 bins: time offsets in 8 ms steps (0–4.792 s)   |
| `velocity`       | 604–731       | 128 velocity values (0–127)                        |
| `note_on`        | 732–859       | 128 MIDI pitches for note onset                    |
| `note_off`       | 860–987       | 128 MIDI pitches for note offset                   |
| `program`        | 988–1115      | 128 MIDI programs (multi-instrument only)          |

**Total vocabulary size:**
- Piano-only (MAESTRO): ~988 tokens (no program tokens needed, or keep 1 piano program)
- Multi-instrument (Slakh): ~1116 tokens

### 2.2 Event Sequence Structure

For each audio segment, the target token sequence is ordered as:

```
<sos> [TIE SECTION] <tie> [EVENT SECTION] <eos>
```

**Tie section:** Lists notes that are still sounding from the previous segment.

```
<sos> program velocity note_on note_on ... <tie>
```

**Event section:** Time-ordered events within the current segment.

```
time_shift program velocity note_on
time_shift note_off
time_shift program velocity note_on note_on
...
<eos>
```

Within a single time step, events are ordered: `program → velocity → note_on/note_off`.
Multiple notes at the same time share the same `time_shift`.

### 2.3 Tokenizer Implementation

```python
class MidiTokenizer:
    """Converts MIDI note events to/from token sequences."""

    def __init__(self, time_step_ms=8, max_time_steps=600,
                 num_velocities=128, num_pitches=128, num_programs=128,
                 multi_instrument=False):
        self.time_step_ms = time_step_ms
        self.max_time_steps = max_time_steps
        self.multi_instrument = multi_instrument

        # Build vocabulary offsets
        self.special = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<tie>': 3}
        offset = 4
        self.time_offset = offset;          offset += max_time_steps    # 4..603
        self.velocity_offset = offset;      offset += num_velocities    # 604..731
        self.note_on_offset = offset;       offset += num_pitches       # 732..859
        self.note_off_offset = offset;      offset += num_pitches       # 860..987
        if multi_instrument:
            self.program_offset = offset;   offset += num_programs      # 988..1115
        self.vocab_size = offset

    def notes_to_tokens(self, notes, segment_start_s, segment_end_s,
                        prev_active_notes=None):
        """
        Args:
            notes: list of (onset_s, offset_s, pitch, velocity, program)
            segment_start_s: start time of this audio segment in seconds
            segment_end_s: end time of this audio segment
            prev_active_notes: notes still sounding from previous segment
        Returns:
            list[int]: token ids
        """
        tokens = [self.special['<sos>']]

        # --- Tie section ---
        if prev_active_notes:
            for (pitch, velocity, program) in prev_active_notes:
                if self.multi_instrument:
                    tokens.append(self.program_offset + program)
                tokens.append(self.velocity_offset + velocity)
                tokens.append(self.note_on_offset + pitch)
        tokens.append(self.special['<tie>'])

        # --- Event section ---
        events = []
        for (onset, offset, pitch, vel, prog) in notes:
            if onset >= segment_start_s and onset < segment_end_s:
                t = onset - segment_start_s
                events.append((t, 'on', pitch, vel, prog))
            if offset >= segment_start_s and offset < segment_end_s:
                t = offset - segment_start_s
                events.append((t, 'off', pitch, 0, prog))

        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1, e[2]))

        prev_time_bin = -1
        for (t, etype, pitch, vel, prog) in events:
            time_bin = min(int(t * 1000 / self.time_step_ms), self.max_time_steps - 1)
            if time_bin != prev_time_bin:
                tokens.append(self.time_offset + time_bin)
                prev_time_bin = time_bin
            if etype == 'on':
                if self.multi_instrument:
                    tokens.append(self.program_offset + prog)
                tokens.append(self.velocity_offset + vel)
                tokens.append(self.note_on_offset + pitch)
            else:
                tokens.append(self.note_off_offset + pitch)

        tokens.append(self.special['<eos>'])
        return tokens
```

---

## 3. Audio Frontend

### 3.1 Spectrogram Computation

```python
import torch
import torchaudio

class SpectrogramFrontend(torch.nn.Module):
    """Convert raw audio waveform to log-mel spectrogram."""

    def __init__(self, sample_rate=16000, n_fft=2048, hop_length=128,
                 n_mels=512):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, waveform):
        """
        Args:
            waveform: (B, num_samples)
        Returns:
            log_mel: (B, n_mels, num_frames)
        """
        mel = self.mel_spec(waveform)              # (B, n_mels, T)
        log_mel = torch.log(mel + 1e-6)            # log scale
        return log_mel
```

### 3.2 Segment Parameters

| Parameter          | Value   | Notes                                       |
|--------------------|---------|----------------------------------------------|
| Sample rate        | 16 kHz  | Downsample from 44.1/48 kHz                 |
| Segment samples    | 256,000 | = 16 seconds at 16 kHz                      |
| FFT size           | 2048    | ~128 ms window at 16 kHz                    |
| Hop length         | 128     | 8 ms hop → 125 Hz frame rate                |
| Mel bins           | 512     | High resolution for polyphonic content       |
| Frames per segment | 2000    | 256000 / 128 = 2000 total frames            |
| Sampled frames     | 256     | Random contiguous crop of 256 frames for training |

During training, you randomly sample 256 contiguous frames from the 2000-frame spectrogram.
This 256-frame window corresponds to ~2.048 seconds of audio, which is the input to the encoder.

---

## 4. Encoder

The encoder is a standard Transformer encoder operating on spectrogram frames.
Each frame (a 512-dim mel vector) is linearly projected to the model dimension.

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SpectrogramEncoder(nn.Module):
    """
    Transformer encoder for spectrogram frames.
    Alternatively, use T5-style relative position biases instead of
    sinusoidal encoding for better generalization.
    """

    def __init__(self, n_mels=512, d_model=512, nhead=8, num_layers=8,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: (B, n_mels, T) — log-mel spectrogram
        Returns:
            enc_out: (B, T, d_model) — encoder hidden states
        """
        x = spectrogram.transpose(1, 2)          # (B, T, n_mels)
        x = self.input_proj(x)                    # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)
        enc_out = self.encoder(x)                 # (B, T, d_model)
        enc_out = self.layer_norm(enc_out)
        return enc_out
```

**Encoder sizing guide:**

| Size    | d_model | nhead | layers | d_ff  | Params (enc) |
|---------|---------|-------|--------|-------|--------------|
| Small   | 256     | 4     | 6      | 1024  | ~8M          |
| Base    | 512     | 8     | 8      | 2048  | ~33M         |
| Large   | 768     | 12    | 12     | 3072  | ~85M         |

---

## 5. Decoder

The decoder autoregressively generates event tokens, cross-attending to encoder outputs.

```python
class EventDecoder(nn.Module):
    """
    Autoregressive Transformer decoder for MIDI event tokens.
    """

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=8,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Tie output projection weights with embedding
        self.output_proj.weight = self.token_embedding.weight

    def forward(self, tgt_tokens, enc_out, tgt_mask=None, tgt_padding_mask=None):
        """
        Args:
            tgt_tokens:       (B, S)        — target token ids (shifted right)
            enc_out:          (B, T, d_model) — encoder hidden states
            tgt_mask:         (S, S)        — causal attention mask
            tgt_padding_mask: (B, S)        — True where padded
        Returns:
            logits: (B, S, vocab_size)
        """
        x = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        x = self.decoder(
            tgt=x,
            memory=enc_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        x = self.layer_norm(x)
        logits = self.output_proj(x)    # (B, S, vocab_size)
        return logits
```

### 5.1 Full Model Wrapper

```python
class MT3Model(nn.Module):
    def __init__(self, frontend, encoder, decoder, tokenizer):
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    def forward(self, waveform, tgt_tokens, tgt_mask, tgt_padding_mask):
        spec = self.frontend(waveform)                     # (B, F, T)
        enc_out = self.encoder(spec)                       # (B, T, D)
        logits = self.decoder(tgt_tokens, enc_out,
                              tgt_mask, tgt_padding_mask)  # (B, S, V)
        return logits

    @torch.no_grad()
    def transcribe(self, waveform, max_len=1024, temperature=0.0):
        """Greedy / temperature-based autoregressive decoding."""
        spec = self.frontend(waveform)
        enc_out = self.encoder(spec)
        B = enc_out.size(0)
        device = enc_out.device

        generated = torch.full((B, 1), self.tokenizer.special['<sos>'],
                               dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            S = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                S, device=device
            )
            logits = self.decoder(generated, enc_out, tgt_mask=tgt_mask)
            next_logits = logits[:, -1, :]  # (B, V)

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced <eos>
            if (next_token == self.tokenizer.special['<eos>']).all():
                break

        return generated
```

---

## 6. Dataset Preparation

### 6.1 MAESTRO v3.0.0 (Piano)

**Download:**
```bash
# Full dataset (~103 GB)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip maestro-v3.0.0.zip -d data/maestro/

# Or MIDI-only for prototyping (~57 MB)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
```

**Structure:**
```
data/maestro/maestro-v3.0.0/
├── maestro-v3.0.0.csv          # metadata: split, composer, title, filenames
├── maestro-v3.0.0.json         # same metadata in JSON
├── 2004/
│   ├── MIDI-Unprocessed_..._wav--1.midi
│   └── MIDI-Unprocessed_..._wav--1.wav
├── 2006/
│   └── ...
└── 2018/
    └── ...
```

**Metadata CSV fields:** `canonical_composer`, `canonical_title`, `split` (train/validation/test),
`year`, `midi_filename`, `audio_filename`, `duration`.

**Pre-split:** ~1,184 train / ~137 validation / ~177 test pieces.
All audio is 44.1 or 48 kHz 16-bit PCM stereo. MIDI has key velocities + sustain/sostenuto/una corda pedal.

**Preprocessing pipeline for MAESTRO:**

```python
import os
import csv
import pretty_midi
import librosa
import numpy as np
from pathlib import Path

def preprocess_maestro(maestro_root, output_dir, sample_rate=16000):
    """
    Resample audio to 16 kHz mono and extract note events from MIDI.
    Save as numpy arrays for fast loading during training.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(maestro_root, 'maestro-v3.0.0.csv')

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row['split']
            audio_path = os.path.join(maestro_root, row['audio_filename'])
            midi_path = os.path.join(maestro_root, row['midi_filename'])

            # Resample audio
            waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

            # Extract notes from MIDI
            midi = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                prog = instrument.program  # 0 for piano
                for note in instrument.notes:
                    notes.append({
                        'onset': note.start,
                        'offset': note.end,
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'program': prog,
                    })
            notes.sort(key=lambda n: (n['onset'], n['pitch']))

            # Save
            stem = Path(row['audio_filename']).stem
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            np.save(os.path.join(split_dir, f'{stem}_audio.npy'), waveform)
            np.save(os.path.join(split_dir, f'{stem}_notes.npy'), notes)

    print("MAESTRO preprocessing complete.")
```

### 6.2 Slakh2100-FLAC-Redux (Multi-instrument)

**Download:**
```bash
# From Zenodo (~105 GB)
wget "https://zenodo.org/record/4599666/files/slakh2100_flac_redux.tar.gz?download=1" \
     -O slakh2100_flac_redux.tar.gz
tar xzf slakh2100_flac_redux.tar.gz -C data/
```

**Structure:**
```
data/slakh2100_flac_redux/
├── train/
│   ├── Track00001/
│   │   ├── mix.flac                # full mixture audio (44.1 kHz mono FLAC)
│   │   ├── all_src.mid             # complete MIDI
│   │   ├── metadata.yaml           # instrument info per stem
│   │   ├── MIDI/
│   │   │   ├── S01.mid             # per-stem MIDI
│   │   │   └── S02.mid
│   │   └── stems/
│   │       ├── S01.flac            # per-stem audio
│   │       └── S02.flac
│   └── Track00002/
│       └── ...
├── validation/
│   └── ...
├── test/
│   └── ...
└── omitted/                        # duplicated tracks — DO NOT USE for training
    └── ...
```

**Redux splits:** 1,289 train / 270 validation / 151 test tracks (duplicates moved to `omitted/`).

**metadata.yaml per track:**
```yaml
UUID: 1a81ae09...
stems:
  S00:
    audio_rendered: true
    inst_class: Guitar          # high-level instrument class
    integrated_loudness: -18.5
    is_drum: false
    midi_program_name: Acoustic Guitar (nylon)
    program_num: 24             # MIDI program number
  S01:
    ...
```

**Preprocessing pipeline for Slakh:**

```python
import yaml
import soundfile as sf
import librosa
import pretty_midi

def preprocess_slakh(slakh_root, output_dir, sample_rate=16000):
    """
    Resample mix.flac to 16 kHz and collect per-stem MIDI notes
    with their program numbers.
    """
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(slakh_root, split)
        if not os.path.exists(split_dir):
            continue

        for track_name in sorted(os.listdir(split_dir)):
            track_dir = os.path.join(split_dir, track_name)
            mix_path = os.path.join(track_dir, 'mix.flac')
            meta_path = os.path.join(track_dir, 'metadata.yaml')

            if not os.path.exists(mix_path):
                continue

            # Load and resample mixture audio
            audio, sr = sf.read(mix_path)
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            # Load metadata for program mapping
            with open(meta_path) as f:
                metadata = yaml.safe_load(f)

            # Collect notes from all stems
            notes = []
            midi_dir = os.path.join(track_dir, 'MIDI')
            for stem_id, stem_info in metadata.get('stems', {}).items():
                if not stem_info.get('audio_rendered', False):
                    continue
                midi_path = os.path.join(midi_dir, f'{stem_id}.mid')
                if not os.path.exists(midi_path):
                    continue

                program = stem_info.get('program_num', 0)
                is_drum = stem_info.get('is_drum', False)
                midi = pretty_midi.PrettyMIDI(midi_path)

                for instrument in midi.instruments:
                    for note in instrument.notes:
                        notes.append({
                            'onset': note.start,
                            'offset': note.end,
                            'pitch': note.pitch,
                            'velocity': note.velocity,
                            'program': 128 if is_drum else program,
                        })

            notes.sort(key=lambda n: (n['onset'], n['pitch']))

            # Save
            out_split = os.path.join(output_dir, split)
            os.makedirs(out_split, exist_ok=True)
            np.save(os.path.join(out_split, f'{track_name}_audio.npy'), audio)
            np.save(os.path.join(out_split, f'{track_name}_notes.npy'), notes)

    print("Slakh preprocessing complete.")
```

### 6.3 PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class TranscriptionDataset(Dataset):
    """
    Loads preprocessed audio and note files.
    Returns random audio segments + corresponding tokenized events.
    """

    def __init__(self, data_dir, tokenizer, sample_rate=16000,
                 segment_samples=256000, n_frames=256, hop_length=128,
                 max_token_len=1024):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.n_frames = n_frames
        self.hop_length = hop_length
        self.max_token_len = max_token_len

        # Find all audio/note file pairs
        self.samples = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith('_audio.npy'):
                stem = fname.replace('_audio.npy', '')
                notes_file = os.path.join(data_dir, f'{stem}_notes.npy')
                if os.path.exists(notes_file):
                    self.samples.append((
                        os.path.join(data_dir, fname),
                        notes_file
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, notes_path = self.samples[idx]
        audio = np.load(audio_path)
        notes = np.load(notes_path, allow_pickle=True)

        # Random segment start
        max_start = max(0, len(audio) - self.segment_samples)
        start_sample = random.randint(0, max_start)
        end_sample = start_sample + self.segment_samples
        segment = audio[start_sample:end_sample]

        # Zero-pad if shorter than segment_samples
        if len(segment) < self.segment_samples:
            segment = np.pad(segment, (0, self.segment_samples - len(segment)))

        start_s = start_sample / self.sample_rate
        end_s = end_sample / self.sample_rate

        # Extract notes within segment and convert to token tuple format
        note_tuples = [
            (n['onset'], n['offset'], n['pitch'], n['velocity'], n['program'])
            for n in notes
            if n['onset'] < end_s and n['offset'] > start_s
        ]

        # Identify tied notes (active before segment start)
        prev_active = [
            (n['pitch'], n['velocity'], n['program'])
            for n in notes
            if n['onset'] < start_s and n['offset'] > start_s
        ]

        # Tokenize
        tokens = self.tokenizer.notes_to_tokens(
            note_tuples, start_s, end_s, prev_active
        )

        # Truncate / pad tokens
        if len(tokens) > self.max_token_len:
            tokens = tokens[:self.max_token_len - 1] + [self.tokenizer.special['<eos>']]
        else:
            tokens = tokens + [self.tokenizer.special['<pad>']] * (
                self.max_token_len - len(tokens)
            )

        waveform = torch.tensor(segment, dtype=torch.float32)
        token_ids = torch.tensor(tokens, dtype=torch.long)

        return waveform, token_ids


def collate_fn(batch):
    waveforms = torch.stack([b[0] for b in batch])
    tokens = torch.stack([b[1] for b in batch])
    return waveforms, tokens
```

---

## 7. Training Pipeline

### 7.1 Training Script Skeleton

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Tokenizer ---
    tokenizer = MidiTokenizer(
        multi_instrument=config['multi_instrument']
    )

    # --- Model ---
    frontend = SpectrogramFrontend(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels'],
    )
    encoder = SpectrogramEncoder(
        n_mels=config['n_mels'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['enc_layers'],
        dim_feedforward=config['d_ff'],
        dropout=config['dropout'],
    )
    decoder = EventDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['dec_layers'],
        dim_feedforward=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['max_token_len'],
    )
    model = MT3Model(frontend, encoder, decoder, tokenizer).to(device)

    # --- Data ---
    train_ds = TranscriptionDataset(
        data_dir=config['train_dir'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        max_token_len=config['max_token_len'],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config['weight_decay'],
    )

    # Linear warmup + inverse sqrt decay (T5 style)
    warmup_steps = config['warmup_steps']
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps / step) ** 0.5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Loss ---
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.special['<pad>'],
        label_smoothing=0.1,
    )

    # --- Mixed precision ---
    scaler = GradScaler()

    # --- Training loop ---
    global_step = 0
    for epoch in range(config['max_epochs']):
        model.train()
        epoch_loss = 0.0

        for waveform, tokens in train_loader:
            waveform = waveform.to(device)
            tokens = tokens.to(device)

            # Teacher forcing: input = tokens[:-1], target = tokens[1:]
            tgt_input = tokens[:, :-1]
            tgt_output = tokens[:, 1:]

            # Causal mask
            S = tgt_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                S, device=device
            )
            # Padding mask
            tgt_padding_mask = (tgt_input == tokenizer.special['<pad>'])

            optimizer.zero_grad()

            with autocast():
                logits = model(waveform, tgt_input, tgt_mask, tgt_padding_mask)
                loss = criterion(
                    logits.reshape(-1, tokenizer.vocab_size),
                    tgt_output.reshape(-1)
                )

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % config['log_every'] == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            if global_step % config['save_every'] == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': global_step,
                }, f"checkpoints/step_{global_step}.pt")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
```

### 7.2 Data Augmentation Strategies

For better generalization, apply during training:

1. **Pitch shifting** (±3 semitones): Shift both audio (via resampling trick or librosa) and
   MIDI pitch values. Clamp to valid MIDI range [0, 127].

2. **Time stretching** (0.95–1.05x): Use `torchaudio.functional.speed` or phase vocoder.
   Adjust MIDI onset/offset times proportionally.

3. **Random gain** (−6 dB to +6 dB): Scale waveform amplitude. Helps with volume robustness.

4. **Intra-stem augmentation** (Slakh only): Randomly mute some stems during training to
   create diverse instrument combinations from the same track. For each stem, include it with
   probability p ∈ [0.6, 0.8].

5. **Spectrogram SpecAugment**: Apply frequency and time masking on the log-mel spectrogram
   (borrowed from ASR). Mask 1–2 frequency bands (width 10–30 bins) and 1–2 time bands
   (width 5–20 frames).

### 7.3 Curriculum Strategy

1. **Phase 1 — Piano only (MAESTRO):** Train for 100K–200K steps on MAESTRO alone.
   This gives the model a strong foundation for pitch/timing.

2. **Phase 2 — Multi-instrument (Slakh):** Fine-tune on Slakh2100-redux with program tokens
   enabled. Alternatively, train jointly from the start with dataset sampling.

3. **Joint training (alternative):** Sample batches from both datasets with a temperature-based
   mixing ratio. Use higher sampling probability for the smaller/harder dataset.

---

## 8. Inference & Decoding

### 8.1 Sliding Window Inference

Full-length audio must be split into overlapping segments:

```python
def transcribe_full_audio(model, audio, tokenizer, sample_rate=16000,
                          segment_samples=256000, hop_samples=128000):
    """
    Sliding window transcription with tie-note handling.
    """
    device = next(model.parameters()).device
    model.eval()

    all_notes = []
    active_notes = set()  # (pitch, program) currently sounding

    num_segments = max(1, (len(audio) - segment_samples) // hop_samples + 1)

    for i in range(num_segments):
        start = i * hop_samples
        end = start + segment_samples
        segment = audio[start:end]

        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))

        waveform = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)

        token_ids = model.transcribe(waveform, max_len=1024)
        tokens = token_ids[0].cpu().tolist()

        # Decode tokens back to note events
        segment_notes = tokenizer.tokens_to_notes(
            tokens,
            segment_start_s=start / sample_rate,
        )

        all_notes.extend(segment_notes)

    # De-duplicate overlapping notes
    all_notes = deduplicate_notes(all_notes)
    return all_notes
```

### 8.2 Token-to-MIDI Conversion

```python
def tokens_to_midi(tokens, tokenizer, output_path='output.mid'):
    """Convert decoded token sequence to a MIDI file."""
    notes = tokenizer.tokens_to_notes(tokens, segment_start_s=0.0)

    midi = pretty_midi.PrettyMIDI()
    programs = {}

    for note in notes:
        prog = note['program']
        if prog not in programs:
            is_drum = (prog == 128)
            instrument = pretty_midi.Instrument(
                program=prog if not is_drum else 0,
                is_drum=is_drum,
            )
            programs[prog] = instrument

        midi_note = pretty_midi.Note(
            velocity=note['velocity'],
            pitch=note['pitch'],
            start=note['onset'],
            end=note['offset'],
        )
        programs[prog].notes.append(midi_note)

    for inst in programs.values():
        midi.instruments.append(inst)

    midi.write(output_path)
    return midi
```

---

## 9. Evaluation

### 9.1 Metrics

Use `mir_eval` for standard transcription metrics:

```python
import mir_eval

def evaluate_transcription(ref_notes, est_notes, onset_tolerance=0.05):
    """
    Args:
        ref_notes: list of (onset, offset, pitch, velocity)
        est_notes: list of (onset, offset, pitch, velocity)
    Returns:
        dict with precision, recall, F1 for onset, onset+offset, onset+offset+velocity
    """
    ref_intervals = np.array([(n[0], n[1]) for n in ref_notes])
    ref_pitches = np.array([librosa.midi_to_hz(n[2]) for n in ref_notes])
    est_intervals = np.array([(n[0], n[1]) for n in est_notes])
    est_pitches = np.array([librosa.midi_to_hz(n[2]) for n in est_notes])

    # Onset-only F1
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
    )

    # Onset + Offset F1
    p_off, r_off, f1_off, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=0.2,       # offset within 20% of note duration or 50ms
        offset_min_tolerance=0.05,
    )

    return {
        'onset_P': p, 'onset_R': r, 'onset_F1': f1,
        'onset_offset_P': p_off, 'onset_offset_R': r_off, 'onset_offset_F1': f1_off,
    }
```

### 9.2 Expected Baselines

| Model        | Dataset   | Onset F1 | Onset+Offset F1 |
|--------------|-----------|----------|------------------|
| MT3 (piano)  | MAESTRO   | ~96–97%  | ~82–85%          |
| MT3 (multi)  | Slakh     | ~74–78%  | ~55–62%          |
| MR-MT3       | Slakh     | ~79–82%  | ~60–65%          |

### 9.3 Multi-instrument Metrics (Slakh)

For multi-instrument evaluation, compute per-program F1 and macro-average.
Also measure **instrument detection F1** (did the model correctly identify which
programs are present?) and **instrument leakage ratio** (fraction of notes assigned to
wrong instruments).

---

## 10. Project Structure

```
mt3-piano/
├── AGENTS.md                        # ← this file
├── README.md
├── requirements.txt
├── configs/
│   ├── maestro_piano.yaml           # piano-only config
│   └── slakh_multi.yaml             # multi-instrument config
├── src/
│   ├── __init__.py
│   ├── tokenizer.py                 # MidiTokenizer class
│   ├── frontend.py                  # SpectrogramFrontend
│   ├── encoder.py                   # SpectrogramEncoder
│   ├── decoder.py                   # EventDecoder
│   ├── model.py                     # MT3Model wrapper
│   ├── dataset.py                   # TranscriptionDataset + collate
│   └── metrics.py                   # Evaluation functions
├── scripts/
│   ├── preprocess_maestro.py        # MAESTRO preprocessing
│   ├── preprocess_slakh.py          # Slakh preprocessing
│   ├── train.py                     # Training entry point
│   ├── evaluate.py                  # Evaluation script
│   └── transcribe.py               # Inference script (audio → MIDI)
├── checkpoints/                     # Saved model weights
├── data/
│   ├── maestro/                     # Raw MAESTRO download
│   ├── slakh2100_flac_redux/        # Raw Slakh download
│   └── processed/                   # Preprocessed numpy files
│       ├── maestro/
│       │   ├── train/
│       │   ├── validation/
│       │   └── test/
│       └── slakh/
│           ├── train/
│           ├── validation/
│           └── test/
└── notebooks/
    ├── 01_explore_data.ipynb
    ├── 02_debug_tokenizer.ipynb
    └── 03_visualize_results.ipynb
```

---

## 11. Dependency List

```
# requirements.txt
torch>=2.1
torchaudio>=2.1
numpy>=1.24
librosa>=0.10
pretty_midi>=0.2.10
mir_eval>=0.7
soundfile>=0.12
pyyaml>=6.0
tqdm
tensorboard
matplotlib
```

Install:
```bash
pip install -r requirements.txt
```

---

## 12. Hyperparameter Reference

### 12.1 Base Configuration (Piano / MAESTRO)

```yaml
# configs/maestro_piano.yaml
data:
  dataset: maestro
  train_dir: data/processed/maestro/train
  val_dir: data/processed/maestro/validation
  sample_rate: 16000
  segment_samples: 256000          # 16 seconds → 2000 mel frames
  n_frames: 256                    # sampled frames per training example

audio:
  n_fft: 2048
  hop_length: 128
  n_mels: 512

model:
  d_model: 512
  nhead: 8
  enc_layers: 8
  dec_layers: 8
  d_ff: 2048
  dropout: 0.1
  max_token_len: 1024

tokenizer:
  multi_instrument: false
  time_step_ms: 8
  max_time_steps: 600

training:
  batch_size: 8                    # per GPU; use gradient accumulation for larger
  grad_accum_steps: 4              # effective batch = 32
  lr: 0.0003                       # peak learning rate
  warmup_steps: 4000
  max_steps: 500000
  max_epochs: 200
  weight_decay: 0.01
  grad_clip: 1.0
  label_smoothing: 0.1
  log_every: 100
  save_every: 10000
  eval_every: 10000
  fp16: true
```

### 12.2 Multi-instrument Configuration (Slakh)

```yaml
# configs/slakh_multi.yaml
# Inherits most settings from piano config, overrides:

data:
  dataset: slakh
  train_dir: data/processed/slakh/train
  val_dir: data/processed/slakh/validation

tokenizer:
  multi_instrument: true

training:
  batch_size: 4                    # multi-instrument sequences are longer
  grad_accum_steps: 8              # effective batch = 32
  lr: 0.0001                       # slightly lower for fine-tuning from piano
  max_steps: 300000
```

### 12.3 Compute Requirements

| Config                  | GPU Memory | Training Time (est.)     |
|-------------------------|------------|--------------------------|
| Base (d=512, 8L) 1×A100 | ~20 GB     | ~3–5 days MAESTRO        |
| Base 1×RTX 4090          | ~18 GB     | ~5–7 days MAESTRO        |
| Small (d=256, 6L) 1×3090 | ~8 GB     | ~2–3 days MAESTRO        |
| Base 4×A100 (DDP)       | ~20 GB/GPU | ~1 day MAESTRO           |

---

## 13. Common Pitfalls & Tips

1. **Token sequence too long**: If many notes occur in one segment, tokens get truncated at
   `max_token_len=1024`. Monitor truncation rate. If >5% of samples truncate, either increase
   max length (costs memory) or shorten segment duration.

2. **Tie section matters**: Skipping tie tokens causes systematic offset errors at segment
   boundaries. Always implement tie handling, even for piano-only.

3. **Spectrogram normalization**: Apply per-example z-normalization or global mean/std
   normalization on the log-mel. Without this, training can be unstable.

4. **Velocity quantization**: Original MT3 uses 128 velocity levels. For faster convergence,
   consider quantizing to 32 or 64 bins (loses some dynamics but reduces vocabulary).

5. **Note-off alignment**: For piano, note-offs derived from MIDI can be noisy (pedaling
   extends notes). Consider using onset-only evaluation initially, then add offset metrics.

6. **Drums in Slakh**: Drum notes use program 128 (outside standard 0–127). Map all drum
   tracks to a single "drum" program token. Drum note "pitch" maps to GM drum kit mappings.

7. **Gradient accumulation**: With 512-dim models and 1024-length sequences, batch size is
   limited by GPU memory. Use gradient accumulation to reach effective batch ≥32.

8. **Decoding temperature**: At inference, temperature=0 (greedy) works well for transcription.
   Small temperature (0.05–0.1) can help with rare instruments but may introduce errors.

9. **Weight tying**: Tying decoder input embedding with output projection saves parameters
   and often improves performance. Already included in the decoder code above.

10. **Pre-LN vs Post-LN**: Use pre-layer-norm (`norm_first=True`) for training stability,
    especially at larger model sizes. This is the T5 convention.

---

## 14. References

**Core Papers:**

- Gardner et al., "MT3: Multi-Task Multitrack Music Transcription" (ICLR 2022).
  Original MT3 architecture and token vocabulary design.

- Hawthorne et al., "Sequence-to-Sequence Piano Transcription with Transformers" (ISMIR 2021).
  Single-instrument predecessor to MT3.

- Tan et al., "MR-MT3: Memory Retaining Multi-Track Music Transcription to Mitigate
  Instrument Leakage" (arXiv 2024). Memory retention mechanism for segment continuity.

- Chang et al., "YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer
  Architectures and Cross-dataset Stem Augmentation" (2024). PerceiverTF encoder, MoE,
  multi-channel decoder, cross-dataset augmentation.

**Datasets:**

- Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO
  Dataset" (ICLR 2019). ~200 hours of piano, 3 ms MIDI-audio alignment.
  URL: https://magenta.withgoogle.com/datasets/maestro

- Manilow et al., "Cutting Music Source Separation Some Slakh" (WASPAA 2019). 2100 tracks,
  145 hours, multi-instrument with aligned MIDI.
  URL: https://zenodo.org/records/4599666

**Code Repositories:**

- Official MT3 (JAX/T5X): https://github.com/magenta/mt3
- MT3-PyTorch (kunato): https://github.com/kunato/mt3-pytorch
- MR-MT3 (PyTorch): https://github.com/gudgud96/MR-MT3
- YourMT3 (PyTorch): https://github.com/mimbres/YourMT3
- Slakh utilities: https://github.com/ethman/slakh-utils
- Slakh PyTorch loader: https://github.com/KinWaiCheuk/slakh_loader
