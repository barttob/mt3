# MT3 Piano Transcription — Deep Technical Reference

> Comprehensive reference for writing a research paper on automatic music transcription (AMT)
> using sequence-to-sequence Transformer models.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [System Overview](#2-system-overview)
3. [Audio Frontend — Spectrogram Extraction](#3-audio-frontend--spectrogram-extraction)
4. [Token Vocabulary](#4-token-vocabulary)
5. [Spectrogram Encoder](#5-spectrogram-encoder)
6. [Autoregressive Event Decoder](#6-autoregressive-event-decoder)
7. [Full Model (MT3Model)](#7-full-model-mt3model)
8. [Dataset Pipeline](#8-dataset-pipeline)
9. [Data Augmentation](#9-data-augmentation)
10. [Training Procedure](#10-training-procedure)
11. [Inference & Decoding](#11-inference--decoding)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Hyperparameter Reference](#13-hyperparameter-reference)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Vocabulary Size Derivation](#15-vocabulary-size-derivation)
16. [Segment Boundary Handling (Tie Section)](#16-segment-boundary-handling-tie-section)
17. [Novel Architectural Contributions](#17-novel-architectural-contributions)

---

## 1. Problem Framing

**Automatic Music Transcription (AMT)** is the task of converting an audio recording into a symbolic musical representation (typically MIDI). This project frames AMT as a **conditional sequence-to-sequence language modeling** problem:

- **Input**: raw audio waveform → log-mel spectrogram
- **Output**: a sequence of discrete MIDI event tokens
- **Model family**: encoder-decoder Transformer (T5-style)

This framing, originally introduced by Gardner et al. (2022) in the MT3 paper (*"MT3: Multi-Task Multitrack Music Transcription"*), unifies piano and multi-instrument transcription under a single autoregressive model. The key insight is that MIDI event streams can be treated as a "language" over a structured, finite vocabulary, enabling standard cross-entropy training.

### Why Sequence-to-Sequence?

Traditional AMT approaches use frame-level classification (e.g., piano roll prediction per frame) or HMM-based methods. A seq2seq approach allows:
- Implicit modeling of note duration (onset/offset pairing in the token stream)
- Handling of polyphony naturally through per-time-step token groups
- Multi-instrument transcription via program tokens in the same sequence
- Transfer of architectural improvements from NLP (pre-norm, weight tying, etc.)

---

## 2. System Overview

```
Raw waveform  (B, num_samples)
        │
        ▼
┌─────────────────────────────┐
│   SpectrogramFrontend       │  log-mel spectrogram
│   MelSpectrogram + log(·)   │  (B, 512, T_frames)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│   SpectrogramEncoder                        │  bidirectional Transformer encoder
│   [default] Linear proj + 1D SinPE          │  8 layers, d_model=512, nhead=8
│   [novel]   PatchEmbedding2D + 2D SinPE     │  output: (B, N, 512)
│   + TransformerEncoderLayer × 8             │  N = T_frames or n_patches
└─────────────────────┬───────────────────────┘
                      │  encoder hidden states
                      ▼
┌─────────────────────────────────────────────┐
│   EventDecoder                              │  causal (autoregressive) decoder
│   Token Embed + 1D SinPE                    │  8 layers, d_model=512, nhead=8
│   [default] TransformerDecoderLayer × 8     │  cross-attends to encoder states
│   [novel]   PitchAwareDecoderLayer × 8      │  output: (B, S, vocab_size)
│   + output_proj (weight-tied)               │
└─────────────────────────────────────────────┘
              │
              ▼
        MIDI event tokens  →  MidiTokenizer.tokens_to_notes()  →  note list
```

**Data flow summary:**

| Stage | Input shape | Output shape | Module |
|---|---|---|---|
| Waveform → Spectrogram | `(B, N)` | `(B, 512, T)` | `SpectrogramFrontend` |
| Spectrogram → Encoder states | `(B, 512, T)` | `(B, T, 512)` | `SpectrogramEncoder` |
| Encoder states + tokens → logits | `(B, T, 512)` + `(B, S)` | `(B, S, vocab)` | `EventDecoder` |
| Logits → token IDs (inference) | `(B, S, vocab)` | `(B, S)` | argmax / multinomial |
| Token IDs → note events | `(B, S)` | list of dicts | `MidiTokenizer` |

---

## 3. Audio Frontend — Spectrogram Extraction

**Module**: `src/frontend.py` → `SpectrogramFrontend`

### Parameters

| Parameter | Value | Notes |
|---|---|---|
| `sample_rate` | 16,000 Hz | Audio resampled to 16 kHz during preprocessing |
| `n_fft` | 2048 | FFT window size → frequency resolution ≈ 7.8 Hz |
| `hop_length` | 128 | Frame shift → time resolution = 128/16000 = 8 ms |
| `n_mels` | 512 | Number of mel filterbank channels |
| `power` | 2.0 | Power spectrogram (magnitude²) before mel |

### Computation

```
waveform (B, N)
    → torchaudio MelSpectrogram (STFT → mel filterbank → power)
    → mel (B, 512, T)
    → log(mel + 1e-6)           # log scale; 1e-6 floor avoids log(0)
    → log_mel (B, 512, T)
```

### Time–Frequency Properties

- **Frequency axis**: 512 mel-spaced bands from 0 Hz to `sample_rate/2` = 8 kHz
- **Time axis**: for a 2.048 s segment with hop=128 → `T = 32768/128 = 256` frames
- **Frame rate**: `16000/128 = 125 frames/s`
- **Segment duration**: `n_frames × hop / sample_rate = 256 × 128 / 16000 = 2.048 s`

The 512-mel configuration is higher resolution than standard speech models (typically 80–128 mels), appropriate for music where fine frequency discrimination matters for pitch accuracy.

---

## 4. Token Vocabulary

**Module**: `src/tokenizer.py` → `MidiTokenizer`

Two vocabulary layouts are supported, selected by `tokenizer.use_hierarchical_time` in config.

### Layout A — Flat Time (default, `use_hierarchical_time: false`)

The vocabulary is flat and contiguous. All offsets are computed at construction time:

| Token ID range | Type | Count | Description |
|---|---|---|---|
| 0 | `<pad>` | 1 | Padding token (ignored in loss) |
| 1 | `<sos>` | 1 | Start-of-sequence |
| 2 | `<eos>` | 1 | End-of-sequence |
| 3 | `<tie>` | 1 | Separates tie section from event section |
| 4 – 603 | `time_shift(k)` | 600 | Time advance: `k × 8 ms` |
| 604 – 731 | `velocity(v)` | 128 | MIDI velocity 0–127 |
| 732 – 859 | `note_on(p)` | 128 | Note-on for MIDI pitch 0–127 |
| 860 – 987 | `note_off(p)` | 128 | Note-off for MIDI pitch 0–127 |
| 988 – 1116 | `program(g)` | 129 | MIDI program 0–127 + drum ID 128 *(multi-instrument only)* |

**Vocabulary sizes — flat:**

| Mode | Formula | Total |
|---|---|---|
| Piano-only | `4 + 600 + 128 + 128 + 128` | **988** |
| Multi-instrument | `988 + 129` | **1,117** |

### Layout B — Hierarchical Time (`use_hierarchical_time: true`) *(Novel contribution)*

Each timestamp is encoded as **two consecutive tokens** — a coarse bucket and a fine offset — instead of one flat token. Default parameters: `coarse_step_ms=64`, `num_coarse=75`.

| Token ID range | Type | Count | Description |
|---|---|---|---|
| 0 – 3 | special | 4 | Unchanged |
| 4 – 78 | `coarse_time_shift(c)` | 75 | Coarse bucket: `c × 64 ms` |
| 79 – 86 | `fine_time_shift(f)` | 8 | Fine offset: `f × 8 ms` within bucket |
| 87 – 214 | `velocity(v)` | 128 | MIDI velocity 0–127 |
| 215 – 342 | `note_on(p)` | 128 | Note-on for MIDI pitch 0–127 |
| 343 – 470 | `note_off(p)` | 128 | Note-off for MIDI pitch 0–127 |
| 471 – 599 | `program(g)` | 129 | MIDI program 0–127 + drum ID 128 *(multi-instrument only)* |

A time value of `Δt ms` is encoded as:
```
coarse = Δt // 64       (range 0–74)
fine   = (Δt % 64) // 8 (range 0–7)
→ tokens: coarse_time_shift(coarse), fine_time_shift(fine)
```

**Vocabulary sizes — hierarchical:**

| Mode | Formula | Total | vs. flat |
|---|---|---|---|
| Piano-only | `4 + 75 + 8 + 128 + 128 + 128` | **471** | −517 (−52%) |
| Multi-instrument | `471 + 129` | **600** | −517 (−46%) |

**Coverage**: `74 × 64 ms + 7 × 8 ms = 4,792 ms` — identical to the flat layout.
**Precision**: 8 ms — identical to the flat layout.
**Sequence length**: +1 token per distinct timestamp (2 tokens per time event vs. 1).

### Time Resolution (both layouts)

- Fine resolution = 8 ms (`time_step_ms = 8`)
- Coverage ≥ 4.8 s (longer than the 2.048 s training segment)
- Time is quantized: onset/offset times are rounded to the nearest 8 ms bin

### Event Sequence Grammar

Within each segment the token sequence follows a strict grammar:

```
<sos>
[TIE SECTION: for each held note from previous segment]
    [program(g)]?    ← only in multi-instrument mode
    velocity(v)
    note_on(p)
<tie>
[EVENT SECTION: time-ordered note events]
    time_shift(k)    ← emitted once per distinct time step
    for each note_on event at this time:
        [program(g)]?
        velocity(v)
        note_on(p)
    for each note_off event at this time:
        note_off(p)
<eos>
[<pad> ...]          ← right-padded to max_token_len=1024
```

**Ordering within a time step**: note_off events are placed before note_on events (sort key: `(time, 0 if off else 1, pitch)`). This convention ensures that a retrigger of the same pitch (note_off + note_on at the same time) is unambiguous.

---

## 5. Spectrogram Encoder

**Module**: `src/encoder.py` → `SpectrogramEncoder`, `PatchEmbedding2D`

Two input embedding modes are supported, selected by `model.use_2d_patches` in config. The Transformer stack (8 pre-LN layers) is identical in both modes.

### Mode A — Per-Frame Projection (default, `use_2d_patches: false`)

```
log_mel (B, 512, T)
    → transpose → (B, T, 512)
    → Linear(512, d_model=512)       # per-frame linear projection
    → SinusoidalPositionalEncoding   # 1D time PE
    → Dropout(0.1)
    → TransformerEncoder (8 layers, pre-LN)
    → LayerNorm(d_model)
    → enc_out (B, T, 512)
```

### Mode B — 2D Frequency-Time Patch Encoder (`use_2d_patches: true`) *(Novel contribution)*

Inspired by Vision Transformers (Dosovitskiy et al., 2021), the spectrogram is divided into non-overlapping 2D patches of size `(patch_f × patch_t)`. Each patch spans multiple mel bins **and** multiple time frames, giving the encoder explicit pitch-band positional awareness through a 2D positional encoding.

```
log_mel (B, 512, T)
    → PatchEmbedding2D(patch_f=64, patch_t=8)
        ├─ reshape into (B, n_fp=8, n_tp=32, patch_f×patch_t=512)
        ├─ Linear(patch_f×patch_t, d_model)    # patch projection
        └─ add 2D PE:
             time_PE[i_t]: sinusoidal, dim d_model//2, encodes time patch index
             freq_PE[i_f]: sinusoidal, dim d_model//2, encodes frequency band index
             PE(i_f, i_t) = concat(time_PE[i_t], freq_PE[i_f])
    → flatten to (B, n_fp × n_tp = 256, d_model)
    → Dropout(0.1)
    → TransformerEncoder (8 layers, pre-LN)   # unchanged from Mode A
    → LayerNorm(d_model)
    → enc_out (B, 256, 512)
```

**Default patch dimensions** (`n_mels=512`, `n_frames=256`):

| Parameter | Value | Result |
|---|---|---|
| `patch_f` | 64 | `512/64 = 8` frequency patches |
| `patch_t` | 8 | `256/8 = 32` time patches |
| Total patches | 256 | Identical sequence length to Mode A |

Each frequency patch covers approximately one octave of the piano range. The 2D PE encodes both *when* (time patch index) and *at what pitch band* (frequency patch index) each patch originates, information unavailable in the flat 1D encoding.

**Note on torchaudio framing**: `torchaudio.MelSpectrogram` with `center=True` (default) produces `T = N/hop + 1` frames. The encoder trims `spec[:, :, :T_trimmed]` where `T_trimmed = (T // patch_t) × patch_t` before patch extraction to handle the extra frame.

### Per-Layer Structure (Pre-LayerNorm = T5 convention, both modes)

Each `TransformerEncoderLayer` with `norm_first=True`:

```
x → LayerNorm → MultiHeadSelfAttention(nhead=8) → Dropout → residual
  → LayerNorm → FFN(d_model→d_ff=2048→d_model) → Dropout → residual
```

### Positional Encoding — 1D (Mode A)

Fixed sinusoidal encoding (Vaswani et al., 2017):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Positional Encoding — 2D (Mode B)

Two independent sinusoidal encodings of dimension `d_model//2` each:
```
time_PE(t, 2i)   = sin(t / 10000^(2i / (d_model/2)))
freq_PE(f, 2i)   = sin(f / 10000^(2i / (d_model/2)))
PE(t, f) = concat(time_PE(t), freq_PE(f))   → d_model
```

This factored design ensures the model can independently generalise to unseen time positions (longer audio) and unseen frequency band configurations.

### Key Properties (both modes)

- **No causal mask**: bidirectional (full self-attention over all N tokens)
- **Activation**: GELU
- **Pre-norm**: applied before each sub-layer (T5 convention)
- **Output shape**: `(B, N, d_model)` where N = T (Mode A) or n_fp × n_tp (Mode B)

### Parameter Count (approximate)

| Component | Mode A | Mode B |
|---|---|---|
| Input projection | 262,656 (512→512) | 262,144 (512→512) |
| Encoder layers × 8 | ~12.6 M | ~12.6 M |
| **Total encoder** | **~13 M** | **~13 M** |

The parameter counts are nearly identical because both modes use the same `d_model` and the same Transformer stack.

---

## 6. Autoregressive Event Decoder

**Module**: `src/decoder.py` → `EventDecoder`, `PitchAwareDecoderLayer`

Two decoder layer variants are supported, selected by `model.use_pitch_aware_attention` in config. The embedding, positional encoding, output projection, and weight tying are identical in both modes.

### Architecture (both modes)

```
tgt_tokens (B, S)
    → Embedding(vocab_size, d_model) × sqrt(d_model)   # scaled embedding
    → SinusoidalPositionalEncoding
    → Dropout(0.1)
    → [8 decoder layers — see below]
    → LayerNorm(d_model)
    → Linear(d_model, vocab_size, bias=False)           # output projection (weight-tied)
    → logits (B, S, vocab_size)
```

### Mode A — Standard Decoder (default, `use_pitch_aware_attention: false`)

Each `TransformerDecoderLayer` with `norm_first=True`:

```
x → LayerNorm → CausalSelfAttention(nhead=8, tgt_mask=causal) → Dropout → residual
  → LayerNorm → CrossAttention(nhead=8, Q=x, K/V=enc_out)     → Dropout → residual
  → LayerNorm → FFN(d_model→2048→d_model)                      → Dropout → residual
```

### Mode B — Pitch-Aware Decoder (`use_pitch_aware_attention: true`) *(Novel contribution)*

Each `PitchAwareDecoderLayer` is architecturally identical to Mode A **except** that a learned pitch-context embedding is added to the cross-attention query input before the cross-attention sublayer:

```
x → LayerNorm → CausalSelfAttention(nhead=8, tgt_mask=causal) → Dropout → residual
  → LayerNorm → x_q = x + pitch_embedding(pitch_ids)           # pitch bias on query
              → CrossAttention(nhead=8, Q=x_q, K/V=enc_out)   → Dropout → residual
  → LayerNorm → FFN(d_model→2048→d_model)                      → Dropout → residual
```

**`pitch_embedding`**: `nn.Embedding(129, d_model)` — one `d_model`-dimensional vector per MIDI pitch (0–127) plus one null embedding at index 128 (initialised to zero, acts as a no-op when no pitch context is available). The null embedding means that positions with no prior `note_on` in the sequence contribute zero bias, making the feature fully backwards-compatible.

**`pitch_ids` tensor** `(B, S)`: at each decoder position `s`, the value is the MIDI pitch of the most recently emitted `note_on` token in the sequence up to position `s`, or 128 if no `note_on` has been seen. Computed by `compute_pitch_context()` in `src/model.py`:

```python
# Vectorized forward-fill of last note_on pitch
is_note_on = (tokens >= note_on_offset) & (tokens < note_on_offset + 128)
pitch_ids = where(is_note_on, tokens - note_on_offset, 128)
for s in range(1, S):
    pitch_ids[:, s] = where(pitch_ids[:, s] == 128, pitch_ids[:, s-1], pitch_ids[:, s])
```

**Intuition**: when the decoder is generating notes around pitch P (e.g., after emitting `velocity(v)` and preparing to emit `note_on(P)` or `note_off(P)`), the cross-attention query is biased toward encoder representations that co-occur with energy at the mel frequency bands corresponding to P. This provides an explicit pitch-frequency inductive bias grounded in music acoustics.

**Extra parameters**: `129 × d_model × 8 layers = 129 × 512 × 8 ≈ 528 K`.

### Weight Tying

The output projection weight matrix is shared with the token embedding matrix:

```python
self.output_proj.weight = self.token_embedding.weight
```

This reduces the parameter count by `vocab_size × d_model` (≈508 K for piano mode) and is standard practice in language models. The embedding weights are initialized with `N(0, d_model^{-0.5})` (scaled Xavier) to prevent excessively large initial logits.

### Causal Mask

During training, position `i` in the target sequence may only attend to positions `j ≤ i`:

```python
tgt_mask = nn.Transformer.generate_square_subsequent_mask(S)
# Returns upper-triangular float mask with -inf above diagonal
```

### Teacher Forcing (Training)

The decoder input is the token sequence shifted right by one:

```
tgt_input  = tokens[:, :-1]   # <sos>, t1, t2, ..., t_{S-1}
tgt_output = tokens[:, 1:]    # t1, t2, ..., t_S (including <eos>)
```

The model predicts `tgt_output[i]` given `tgt_input[:i+1]` (the prefix) and all encoder states.

---

## 7. Full Model (MT3Model)

**Module**: `src/model.py` → `MT3Model`, `build_model`

### Training Forward Pass

```python
logits = model(waveform, tgt_input, tgt_mask, tgt_padding_mask)
# waveform:         (B, N)
# tgt_input:        (B, S)   — right-shifted token IDs
# tgt_mask:         (S, S)   — causal float mask
# tgt_padding_mask: (B, S)   — bool True at <pad> positions
# logits:           (B, S, vocab_size)
```

Internally: `frontend → encoder → [compute_pitch_context] → decoder`.

When `use_pitch_aware_attention=True`, `compute_pitch_context(tgt_input, note_on_offset)` is called before the decoder to produce `pitch_ids (B, S)`, which is passed to each `PitchAwareDecoderLayer`.

### Inference (`transcribe`)

```python
generated = model.transcribe(waveform, max_len=1024, temperature=0.0)
```

1. Compute encoder hidden states once (`frontend + encoder`).
2. Initialize `generated = [[<sos>]]` (or a tie-section prompt).
3. At each step `t`:
   - If `use_pitch_aware_attention`: compute `pitch_ids` from `generated`
   - Run decoder on `generated[:t]` with optional `pitch_ids` → `logits[:, -1, :]`
   - If `temperature=0.0`: next token = `argmax(logits)` (greedy)
   - If `temperature>0`: next token ~ `Softmax(logits / T)`
   - Append to `generated`
4. Stop when all sequences emit `<eos>` or `max_len` is reached.

### Factory Function

```python
model = build_model("configs/maestro_piano.yaml")
```

Reads YAML sections `audio`, `model`, `tokenizer`, `data` and constructs all sub-modules with consistent hyperparameters.

### Total Parameter Count

With the default `maestro_piano.yaml` configuration (piano-only, vocab=988):

| Sub-module | Approximate params |
|---|---|
| Frontend | ~0 (torchaudio ops, no learnable params) |
| Encoder | ~13 M |
| Decoder (unique, weight-tied) | ~13 M |
| **Total unique** | **~26 M** |

---

## 8. Dataset Pipeline

**Module**: `src/dataset.py` → `TranscriptionDataset`, `collate_fn`

### Data Format (on Disk)

Preprocessed by `scripts/preprocess_maestro.py` / `scripts/preprocess_slakh.py`:

```
data/processed/{maestro|slakh}/{train|validation}/
    {stem}_audio.npy    # float32 1-D array, 16 kHz mono
    {stem}_notes.npy    # object array of dicts: onset, offset, pitch, velocity, program
```

### Segment Sampling

At each `__getitem__`:

1. Select file: `file_idx = idx % num_files`
2. **Random crop** (training): `start_sample ~ Uniform(0, len(audio) - segment_samples)`
3. **Center crop** (validation): `start_sample = (len(audio) - segment_samples) // 2`
4. Extract waveform window: `audio[start:start+segment_samples]`
5. Zero-pad if the file is shorter than one segment

This produces `segments_per_file=10` virtual samples per file per epoch (controlled by `__len__`), maximizing data utilization from long audio files.

### Note Extraction

For a segment `[start_s, end_s)`:
- **Overlapping notes**: `onset < end_s AND offset > start_s` — included in `notes`
- **Previously active**: `onset < start_s AND offset > start_s` — added to `prev_active` (go into the tie section)

### Tokenization

```python
tokens = tokenizer.notes_to_tokens(notes, start_s, end_s, prev_active)
```

### Padding / Truncation

```
if len(tokens) > max_token_len:
    tokens = tokens[:max_token_len-1] + [<eos>]   # hard truncation, preserve <eos>
else:
    tokens = tokens + [<pad>] * (max_token_len - len(tokens))
```

Result: every sample has exactly `max_token_len=1024` tokens.

### DataLoader

```python
DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn,
           pin_memory=True, drop_last=True, persistent_workers=True, num_workers=4)
```

`collate_fn` simply stacks pre-padded tensors: `waveforms (B, N)`, `tokens (B, 1024)`.

### Datasets

| Dataset | Domain | Size | Notes |
|---|---|---|---|
| **MAESTRO v3.0.0** | Solo piano | ~200 h | Professional recordings, MusicXML-aligned |
| **Slakh2100-FLAC-Redux** | Multi-instrument | ~145 h | Synthesized from MIDI, 34 instrument families |

---

## 9. Data Augmentation

**Module**: `src/augmentation.py` → `WaveformAugmenter`

Applied to training waveforms only (not validation). Each augmentation is applied independently with probability `p=0.5`:

| Augmentation | Parameter | Effect |
|---|---|---|
| Gain perturbation | `gain ~ Uniform(0.5, 1.5)` | Scale waveform amplitude |
| Additive Gaussian noise | `σ=0.005` | Low-level background noise injection |
| Time masking | `mask_len = Uniform(0, 0.1) × N` | Zero out a random contiguous region |

**Rationale**:
- Gain perturbation increases robustness to recording level variation
- Noise injection prevents overfitting to clean studio recordings
- Time masking (SpecAugment-inspired, applied at the waveform level) encourages the model to use broader temporal context

No pitch shifting or time stretching is applied, as these would require corresponding changes to the MIDI labels.

---

## 10. Training Procedure

**Script**: `scripts/train.py`

### Loss Function

Cross-entropy with label smoothing and padding masking:

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.special["<pad>"],  # don't penalize padding positions
    label_smoothing=0.1,
)
loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
```

**Label smoothing** (`ε=0.1`): replaces the one-hot target distribution with `(1-ε)·one_hot + ε/V`, reducing overconfidence and improving calibration.

### Optimizer

**AdamW** with the following hyperparameters:

```python
AdamW(
    lr=3e-4,          # peak learning rate (piano); 1e-4 (slakh fine-tune)
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01,
)
```

The `eps=1e-9` and `beta2=0.98` match the original Transformer paper settings and are standard for attention-based models.

### Learning Rate Schedule

Linear warmup followed by cosine decay:

```
lr(step) = peak_lr × {
    step / warmup_steps,                         if step < warmup_steps
    0.5 × (1 + cos(π × progress)),              otherwise
}
where progress = (step - warmup_steps) / (max_steps - warmup_steps)
```

| Phase | Duration |
|---|---|
| Linear warmup | 4,000 steps |
| Cosine decay | remaining steps to 500,000 |

### Gradient Accumulation

Effective batch size is much larger than the per-GPU batch:

```
effective_batch = batch_size_per_gpu × grad_accum_steps
               = 4 × 16 = 64  (piano)
               = 4 × 8  = 32  (slakh)
```

Loss is scaled by `1/grad_accum_steps` before backward so accumulated gradients are equivalent to one step with the full effective batch.

### Mixed Precision

| Config | Mode | Notes |
|---|---|---|
| Piano (`maestro_piano.yaml`) | **BFloat16** (bf16) | No GradScaler needed; wider dynamic range than fp16 |
| Slakh (`slakh_multi.yaml`) | **Float16** (fp16) | GradScaler used; skip update on non-finite gradients |

### Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Applied after `scaler.unscale_()`. Non-finite gradient norms trigger a skipped update (logged as a warning).

### Early Stopping

Validation loss is checked every `eval_every=10,000` steps. If it does not improve for `patience=10` consecutive evaluations (100,000 steps), training stops. The best checkpoint is saved separately as `checkpoints/best.pt`.

### Checkpointing

Saved every `save_every=10,000` steps and on every validation improvement:

```python
{
    "model":     model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler":    scaler.state_dict(),
    "step":      global_step,
    "config":    config,
}
```

Checkpoints are stored under `checkpoints/step_{N}.pt` and `checkpoints/best.pt`.

### Training Budget

| Configuration | Max steps | Effective batch | Expected time |
|---|---|---|---|
| Piano (MAESTRO) | 500,000 | 64 | ~days on 1 GPU |
| Multi-inst (Slakh) | 300,000 | 32 | ~days on 1 GPU |

---

## 11. Inference & Decoding

### Single-Segment Transcription

```python
generated_ids = model.transcribe(waveform_segment, max_len=1024, temperature=0.0)
notes = tokenizer.tokens_to_notes(generated_ids[0].tolist(), segment_start_s)
```

### Sliding-Window Long-Audio Inference

For files longer than 2.048 s, a sliding window is applied:

1. Divide audio into overlapping segments (segment size = 2.048 s)
2. For each segment, build a tie-section prompt from notes held from the previous segment
3. Pass the prompt as `prompt_tokens` to `model.transcribe()`
4. Collect all decoded note lists, offset by each segment's `start_s`
5. Deduplicate near-duplicate notes at segment boundaries

### Note Deduplication

`src/metrics.py` → `deduplicate_notes(notes, overlap_s=0.05)`:

Two notes are duplicates if:
- Same pitch and program
- Onset difference ≤ 50 ms

The note with longer duration is kept. This handles overlapping window artifacts.

### Greedy vs. Sampled Decoding

| `temperature` | Strategy | Use case |
|---|---|---|
| `0.0` (default) | Greedy argmax | Deterministic; best for evaluation |
| `> 0.0` | Softmax sampling | Diverse hypotheses; useful for ensembling |

---

## 12. Evaluation Metrics

**Module**: `src/metrics.py`

Uses the `mir_eval` library's `transcription` module, which is the standard for AMT evaluation in MIR research.

### Note Matching

Each estimated note is matched to a reference note if:
1. Pitches match (in Hz, via `librosa.midi_to_hz`)
2. Onsets differ by ≤ `onset_tolerance = 50 ms`
3. *(for onset+offset)* Offsets differ by ≤ `max(offset_ratio × duration, offset_min_tolerance)` where `offset_ratio=0.2`, `offset_min_tolerance=50 ms`

### Reported Metrics

```python
{
    "onset_P":         float,   # onset-only Precision
    "onset_R":         float,   # onset-only Recall
    "onset_F1":        float,   # onset-only F1 (primary metric for piano)
    "onset_offset_P":  float,   # onset+offset Precision
    "onset_offset_R":  float,   # onset+offset Recall
    "onset_offset_F1": float,   # onset+offset F1
}
```

### Per-Program Metrics (Multi-Instrument)

`per_program_metrics()` computes the same six metrics separately for each MIDI program present in reference or estimate. `macro_average_metrics()` computes the unweighted mean across all programs.

### Instrument Detection

`instrument_detection_f1()` measures whether each instrument (program) was detected at all (at least one note assigned to it). Gives a set-level P/R/F1 independent of note count.

---

## 13. Hyperparameter Reference

### Shared Architecture (Both Configs)

| Parameter | Value | Location |
|---|---|---|
| `d_model` | 512 | `model.d_model` |
| `nhead` | 8 | `model.nhead` |
| `enc_layers` | 8 | `model.enc_layers` |
| `dec_layers` | 8 | `model.dec_layers` |
| `d_ff` | 2048 | `model.d_ff` |
| `dropout` | 0.1 | `model.dropout` |
| `max_token_len` | 1024 | `model.max_token_len` |
| `n_fft` | 2048 | `audio.n_fft` |
| `hop_length` | 128 | `audio.hop_length` |
| `n_mels` | 512 | `audio.n_mels` |
| `sample_rate` | 16,000 Hz | `data.sample_rate` |
| `segment_samples` | 32,768 | `data.segment_samples` (= 256 frames × 128) |
| `n_frames` | 256 | `data.n_frames` |
| `time_step_ms` | 8 ms | `tokenizer.time_step_ms` |
| `max_time_steps` | 600 | `tokenizer.max_time_steps` |
| `use_hierarchical_time` | `false` | `tokenizer.use_hierarchical_time` |
| `coarse_step_ms` | 64 ms | `tokenizer.coarse_step_ms` |
| `num_coarse` | 75 | `tokenizer.num_coarse` |
| `use_2d_patches` | `false` | `model.use_2d_patches` |
| `patch_f` | 64 | `model.patch_f` |
| `patch_t` | 8 | `model.patch_t` |
| `use_pitch_aware_attention` | `false` | `model.use_pitch_aware_attention` |

### Piano Config (`maestro_piano.yaml`)

| Parameter | Value |
|---|---|
| Dataset | MAESTRO |
| `multi_instrument` | `false` |
| `vocab_size` | 988 |
| `batch_size` | 4 |
| `grad_accum_steps` | 16 (effective batch = 64) |
| `lr` | 3×10⁻⁴ |
| `warmup_steps` | 4,000 |
| `max_steps` | 500,000 |
| `weight_decay` | 0.01 |
| `label_smoothing` | 0.1 |
| `precision` | bf16 |

### Multi-Instrument Config (`slakh_multi.yaml`)

| Parameter | Value |
|---|---|
| Dataset | Slakh2100 |
| `multi_instrument` | `true` |
| `num_programs` | 129 (MIDI 0–127 + drum ID 128) |
| `vocab_size` | 1,117 |
| `batch_size` | 4 |
| `grad_accum_steps` | 8 (effective batch = 32) |
| `lr` | 1×10⁻⁴ |
| `warmup_steps` | 4,000 |
| `max_steps` | 300,000 |
| `weight_decay` | 0.01 |
| `label_smoothing` | 0.1 |
| `precision` | fp16 |

---

## 14. Key Design Decisions

### 1. Pre-LayerNorm (T5 Convention)

Standard (post-LN) Transformers apply LayerNorm after the residual addition. Pre-LN applies it *before* each sub-layer:

```
pre-LN:  x = x + sublayer(LayerNorm(x))
post-LN: x = LayerNorm(x + sublayer(x))
```

Pre-LN provides more stable gradients during early training (no vanishing gradient through long residual chains) and is the default in T5 and GPT-style models.

### 2. Sinusoidal (Fixed) vs. Learned Positional Encoding

Fixed sinusoidal encoding is used for both encoder and decoder. Benefits:
- Zero additional parameters
- Better extrapolation to sequence lengths not seen in training
- Identical behavior at inference vs. training

### 3. Weight Tying

Sharing embedding and output-projection weights is justified because:
- The model must both *understand* tokens (embedding) and *predict* them (output), making shared representations coherent
- Reduces parameter count by ~7–9% depending on vocab size
- Typically improves perplexity in language models

### 4. Velocity as a Discrete Token

MIDI velocity (0–127) is encoded as 128 discrete tokens rather than as a continuous regression target. This:
- Keeps the model purely generative (single cross-entropy loss)
- Avoids mixed discrete/continuous output heads
- Quantizes velocity to MIDI's native 128-level resolution

### 5. Tie Section for Segment Continuity

Notes held across segment boundaries are encoded in a special "tie section" (`<sos>…<tie>`) before the regular event tokens. This:
- Gives the decoder initial state information about held notes
- Avoids erroneously re-triggering notes that were started in previous segments
- Allows accurate decoding of long sustained notes (piano sustain pedal, etc.)

### 6. Note-off Before Note-on at the Same Timestep

When note_off and note_on events occur at the same quantized time bin, note_off tokens appear first. This is critical for re-triggered notes on the same pitch (e.g., piano staccato to legato): the model sees the release before the new attack, preventing it from merging two notes into one.

### 7. No Beam Search

Greedy decoding (`temperature=0.0`) is used in the reference implementation. For music transcription:
- The output vocabulary has strong structural constraints (the grammar described in Section 4) that limit prediction uncertainty
- Beam search adds significant compute cost at inference time
- The primary evaluation target is F1, not perplexity or log-likelihood

### 8. Hierarchical Time Tokenization — Why Two Tokens?

The flat 600-token time vocabulary is 61% of the piano vocabulary (600/988). Hierarchical encoding decomposes a timestamp into an independently learnable coarse scale (musical measure/beat level) and fine scale (individual onset timing). Arguments for this design:

- **Vocabulary efficiency**: 83 tokens achieve identical precision and coverage as 600, freeing embedding space for musical structure
- **Compositional generalisation**: coarse and fine embeddings can each specialise, potentially improving timing precision at inference
- **Sequence length tradeoff**: each time event costs 2 tokens instead of 1; for typical piano densities (~5 events per 100 ms) this overhead is ~8% of total sequence length

The decomposition mirrors how musicians conceptualise rhythm: at a beat level and a sub-beat subdivision level.

### 9. Pitch-Aware Cross-Attention — Query Bias vs. Other Approaches

The pitch bias is added to the **query** of cross-attention, not to keys or values. This choice is deliberate:

- **Query-side**: changes *what the decoder looks for* in the encoder output — appropriate since pitch determines the frequency band of interest
- **Key-side**: would change how the encoder output is indexed, but encoder outputs are pitch-agnostic
- **Value-side**: would change what information is retrieved, but values should remain the full encoder context
- **Input-side** (before the full layer): would affect both self-attention and cross-attention; less targeted

The null embedding (index 128 = zero vector) means positions before any `note_on` token behave exactly like the standard decoder, ensuring the feature degrades gracefully on non-note tokens.

### 10. 2D Patch Encoder — Sequence Length Preservation

The patch parameters are chosen so that the total number of patches equals the standard encoder sequence length:

```
n_patches = (n_mels / patch_f) × (T / patch_t)
          = (512 / 64) × (256 / 8)
          = 8 × 32 = 256
```

This means 2D patch mode can be dropped in as a replacement without changing decoder capacity, memory budget, or training schedules. A 2× patch_t (e.g., 16) would halve the sequence length for faster attention at the cost of temporal resolution.

---

## 15. Vocabulary Size Derivation

### Flat layout (default)

```
Piano-only:
    special tokens:  4   (pad, sos, eos, tie)
    time_shift:    600   (0–4.792 s in 8 ms bins)
    velocity:      128   (MIDI 0–127)
    note_on:       128   (MIDI pitch 0–127)
    note_off:      128   (MIDI pitch 0–127)
    ─────────────────────
    total:         988

Multi-instrument adds:
    program:       129   (MIDI 0–127 + drum ID 128)
    ─────────────────────
    total:       1,117
```

### Hierarchical layout (`use_hierarchical_time: true`)

```
Piano-only:
    special tokens:    4   (pad, sos, eos, tie)
    coarse_time_shift: 75  (0–4.736 s in 64 ms buckets)
    fine_time_shift:    8  (0–56 ms fine offset in 8 ms steps)
    velocity:         128   (MIDI 0–127)
    note_on:          128   (MIDI pitch 0–127)
    note_off:         128   (MIDI pitch 0–127)
    ─────────────────────
    total:            471

Multi-instrument adds:
    program:          129   (MIDI 0–127 + drum ID 128)
    ─────────────────────
    total:            600
```

Vocabulary reduction: **−517 tokens** regardless of instrument mode (86% reduction in the time sub-vocabulary).

### General formula (hierarchical)

```
V = 4 + num_coarse + (coarse_step_ms // time_step_ms)
      + num_velocities + num_pitches + num_pitches
      [+ num_programs  if multi_instrument]
```

The drum ID `128` is a synthetic addition outside the standard MIDI program range (0–127). All drum notes are assigned `program=128` during preprocessing, enabling the model to differentiate pitched instruments from percussion without any special-casing in the decoding logic.

---

## 16. Segment Boundary Handling (Tie Section)

This is one of the more subtle aspects of the MT3 tokenization scheme, important for long-form transcription.

### Problem

Audio is processed in fixed-length 2.048 s windows. Some notes (e.g., sustained piano chords) span multiple windows. Naively, the decoder would have no information about notes already sounding at the start of each window.

### Solution: Tie Section

At the start of each decoded sequence, before the `<tie>` delimiter, the decoder is given the set of notes that were active (held over) from the previous segment:

```
<sos>  velocity(80) note_on(60)  velocity(70) note_on(48)  <tie>  ...events...  <eos>
       │─── C4 held from prev ─────────────────────────────│      │─── new ──────│
```

These tie-section tokens inform the decoder:
- Which pitches are already sounding (so it can correctly emit note_off when they end)
- At what velocities (for accurate reproduction)
- For which program (multi-instrument mode)

### Decoding the Tie Section

In `tokens_to_notes()`, tokens between `<sos>` and `<tie>` are treated as already-open notes with `onset = segment_start_s`. Their `offset` fields are filled in when the corresponding `note_off` token appears in the event section.

### Training-Time Tie Section

During dataset construction:
```python
prev_active = [(pitch, velocity, program)
               for note in all_notes
               if note.onset < start_s and note.offset > start_s]
```

These tuples are passed to `notes_to_tokens()` to build the tie prefix, ensuring the model learns to use the tie section during teacher-forced training.

---

---

## 17. Novel Architectural Contributions

This section summarises the three architectural innovations introduced beyond the baseline MT3 implementation. All are individually togglable via YAML config for ablation studies.

### Overview

| Feature | Config flag | Files | Baseline comparison |
|---|---|---|---|
| Hierarchical Time Tokenization | `tokenizer.use_hierarchical_time` | `src/tokenizer.py` | Flat 600-token time vocab → coarse+fine 83-token scheme |
| Pitch-Aware Cross-Attention | `model.use_pitch_aware_attention` | `src/decoder.py`, `src/model.py` | Standard cross-attention → pitch-conditioned query bias |
| 2D Frequency-Time Patch Encoder | `model.use_2d_patches` | `src/encoder.py` | Per-frame 1D projection → 2D patch embedding with frequency PE |

### Feature 1 — Hierarchical Time Tokenization

**Motivation**: The flat time vocabulary is the single largest contributor to vocabulary size (600/988 = 61% of piano vocab). Reducing it should improve the embedding space's capacity for musical structure without losing temporal precision.

**Method**: A two-level coarse/fine decomposition of each timestamp. Coarse tokens encode 64 ms bucket indices; fine tokens encode 8 ms sub-bucket offsets. Both are learned embeddings summed into the decoder's initial representation.

**Expected thesis findings**:
- Lower perplexity per token (smaller vocab, easier to predict)
- No degradation in note-level F1 (identical 8 ms precision)
- Possible improvement in onset precision if fine embedding specialises

**Config to enable**:
```yaml
tokenizer:
  use_hierarchical_time: true
  coarse_step_ms: 64
  num_coarse: 75
```

---

### Feature 2 — Pitch-Aware Cross-Attention

**Motivation**: The decoder's cross-attention has no prior knowledge about which part of the audio spectrum is relevant for the note currently being generated. An explicit pitch-frequency inductive bias could improve note detection precision, particularly for closely-spaced pitches in polyphonic music.

**Method**: A learned `pitch_embedding(129, d_model)` is looked up from the last emitted `note_on` pitch and added to cross-attention queries in every decoder layer at every step. The null embedding (index 128) initialised to zero ensures no bias before the first note.

**Expected thesis findings**:
- Improved onset-only F1, especially on high-density polyphonic passages
- Better discrimination between adjacent semitones (e.g., C vs C#)
- Interpretable attention maps: higher attention on spectrogram frames near the relevant mel frequency

**Config to enable**:
```yaml
model:
  use_pitch_aware_attention: true
```

---

### Feature 3 — 2D Frequency-Time Patch Encoder

**Motivation**: The standard encoder treats each mel frame as a single `d_model`-dimensional vector, discarding all spatial (frequency-axis) structure within the frame. A 2D patch representation lets the encoder attend across both time and frequency dimensions simultaneously, capturing harmonic relationships that co-occur within a patch.

**Method**: ViT-style 2D patch extraction from the spectrogram. Patches of size `(patch_f=64 × patch_t=8)` are projected to `d_model` with a factored 2D sinusoidal positional encoding (separate time and frequency axes, concatenated).

**Expected thesis findings**:
- Better multi-pitch detection (harmonic structure visible within patches)
- Potential improvement on the Slakh multi-instrument task (denser frequency content)
- Attention pattern analysis: each attention head may specialise to frequency bands

**Config to enable**:
```yaml
model:
  use_2d_patches: true
  patch_f: 64
  patch_t: 8
```

---

### Ablation Experiment Design

To isolate each contribution, run the following configurations against the baseline:

| Experiment | `use_hierarchical_time` | `use_2d_patches` | `use_pitch_aware_attention` |
|---|---|---|---|
| Baseline | false | false | false |
| +F1 only | **true** | false | false |
| +F2 only | false | false | **true** |
| +F3 only | false | **true** | false |
| +F1+F2 | **true** | false | **true** |
| +F1+F3 | **true** | **true** | false |
| +F2+F3 | false | **true** | **true** |
| All features | **true** | **true** | **true** |

Evaluate each on MAESTRO validation using `scripts/evaluate.py`. Report `onset_F1` and `onset_offset_F1`.

---

## References

1. Gardner, J., Simon, I., Manilow, E., Hawthorne, C., & Engel, J. (2022). **MT3: Multi-Task Multitrack Music Transcription**. *ICLR 2022*.

2. Hawthorne, C., et al. (2021). **Sequence-to-Sequence Piano Transcription with Transformers**. *ISMIR 2021*.

3. Vaswani, A., et al. (2017). **Attention Is All You Need**. *NeurIPS 2017*.

4. Raffel, C., et al. (2020). **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)**. *JMLR 2020*.

5. Hawthorne, C., et al. (2019). **Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset**. *ICLR 2019*.

6. Manilow, E., et al. (2020). **Cutting Through the Noise on Guitar String Detection with Slakh2100**. *ISMIR 2020*.

7. Park, D. S., et al. (2019). **SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition**. *Interspeech 2019*.

8. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). **Layer Normalization**. *arXiv:1607.06450*.

9. Press, O., & Wolf, L. (2017). **Using the Output Embedding to Improve Language Models**. *EACL 2017*.

10. Loshchilov, I., & Hutter, F. (2019). **Decoupled Weight Decay Regularization (AdamW)**. *ICLR 2019*.

11. Dosovitskiy, A., et al. (2021). **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT)**. *ICLR 2021*. *(Inspiration for 2D patch embedding in Feature 3.)*

12. Gong, Y., et al. (2021). **AST: Audio Spectrogram Transformer**. *Interspeech 2021*. *(Application of ViT-style patch encoding to audio spectrograms.)*
