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
┌─────────────────────────────┐
│   SpectrogramEncoder        │  bidirectional Transformer encoder
│   Linear proj + SinPE       │  8 layers, d_model=512, nhead=8
│   + TransformerEncoderLayer │  output: (B, T_frames, 512)
└─────────────┬───────────────┘
              │  encoder hidden states
              ▼
┌─────────────────────────────┐
│   EventDecoder              │  causal (autoregressive) decoder
│   Token Embed + SinPE       │  8 layers, d_model=512, nhead=8
│   + TransformerDecoderLayer │  cross-attends to encoder states
│   + output_proj (tied)      │  output: (B, S, vocab_size)
└─────────────────────────────┘
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

### Vocabulary Layout

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

### Vocabulary Sizes

| Mode | Formula | Total |
|---|---|---|
| Piano-only | `4 + 600 + 128 + 128 + 128` | **988** |
| Multi-instrument | `988 + 129` | **1,117** |

### Time Resolution

- 1 time_shift bin = 8 ms (`time_step_ms = 8`)
- 600 bins cover up to `600 × 8 ms = 4.8 s` (more than the 2.048 s segment)
- Time is quantized: `bin = floor(t_rel_ms / 8)`, clamped to 599

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

**Module**: `src/encoder.py` → `SpectrogramEncoder`

### Architecture

```
log_mel (B, 512, T)
    → transpose → (B, T, 512)
    → Linear(512, d_model=512)     # per-frame projection
    → SinusoidalPositionalEncoding
    → Dropout(0.1)
    → TransformerEncoder (8 layers, pre-LN)
    → LayerNorm(d_model)
    → enc_out (B, T, 512)
```

### Per-Layer Structure (Pre-LayerNorm = T5 convention)

Each `TransformerEncoderLayer` with `norm_first=True`:

```
x → LayerNorm → MultiHeadSelfAttention(nhead=8) → Dropout → residual
  → LayerNorm → FFN(d_model→d_ff=2048→d_model) → Dropout → residual
```

### Positional Encoding

Fixed sinusoidal encoding (Vaswani et al., 2017):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Precomputed up to `max_len=2048`, stored as a non-trainable buffer. Added directly to the projected frame embeddings.

### Key Properties

- **No causal mask**: the encoder is bidirectional (full self-attention over all T frames)
- **Activation**: GELU (smoother gradient than ReLU, empirically preferred in Transformer LM)
- **Pre-norm**: LayerNorm applied *before* each sub-layer (as in T5) rather than after, which improves training stability at the cost of slight implementation complexity
- **Output**: contextual representation of every spectrogram frame, shape `(B, T, d_model)`

### Parameter Count (approximate)

| Component | Parameters |
|---|---|
| Input projection (512→512) | 262,656 |
| Encoder layer × 8 (attn + FFN) | ~12.6 M |
| Layer norm | ~1,024 |
| **Total encoder** | **~13 M** |

---

## 6. Autoregressive Event Decoder

**Module**: `src/decoder.py` → `EventDecoder`

### Architecture

```
tgt_tokens (B, S)
    → Embedding(vocab_size, d_model) × sqrt(d_model)   # scaled embedding
    → SinusoidalPositionalEncoding
    → Dropout(0.1)
    → TransformerDecoder (8 layers, pre-LN, causal mask)
        [cross-attends to enc_out (B, T, d_model)]
    → LayerNorm(d_model)
    → Linear(d_model, vocab_size, bias=False)           # output projection
    → logits (B, S, vocab_size)
```

### Per-Layer Structure

Each `TransformerDecoderLayer` with `norm_first=True`:

```
x → LayerNorm → CausalSelfAttention(nhead=8, tgt_mask=causal) → Dropout → residual
  → LayerNorm → CrossAttention(nhead=8, key/value=enc_out)     → Dropout → residual
  → LayerNorm → FFN(d_model→2048→d_model)                      → Dropout → residual
```

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

Internally: `frontend → encoder → decoder`.

### Inference (`transcribe`)

```python
generated = model.transcribe(waveform, max_len=1024, temperature=0.0)
```

1. Compute encoder hidden states once (`frontend + encoder`).
2. Initialize `generated = [[<sos>]]` (or a tie-section prompt).
3. At each step `t`:
   - Run decoder on `generated[:t]` → `logits[:, -1, :]`
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

---

## 15. Vocabulary Size Derivation

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
