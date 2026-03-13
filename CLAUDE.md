# MT3 Piano Transcription

## Project Overview
Autoregressive encoder-decoder Transformer for music transcription.
Audio spectrograms → MIDI event tokens. Targeting piano (MAESTRO) and
multi-instrument (Slakh2100) datasets.

## Architecture
- T5-style encoder-decoder Transformer (PyTorch)
- Encoder: spectrogram frames → bidirectional transformer
- Decoder: autoregressive MIDI token generation with cross-attention
- Full spec: see AGENTS.md

## Tech Stack
- Python 3.11+, PyTorch 2.1+, torchaudio
- librosa, pretty_midi, mir_eval, soundfile
- Training: mixed precision (AMP), AdamW, linear warmup + inv sqrt decay

## Project Structure
src/
├── tokenizer.py    — MidiTokenizer (MIDI events ↔ token IDs)
├── frontend.py     — SpectrogramFrontend (waveform → log-mel)
├── encoder.py      — SpectrogramEncoder (transformer encoder)
├── decoder.py      — EventDecoder (autoregressive transformer decoder)
├── model.py        — MT3Model wrapper + transcribe() method
├── dataset.py      — TranscriptionDataset + collate_fn
└── metrics.py      — mir_eval evaluation wrappers

scripts/
├── preprocess_maestro.py
├── preprocess_slakh.py
├── train.py
├── evaluate.py
└── transcribe.py

## Commands
```bash
# Install deps
pip install -r requirements.txt

# Preprocess
python scripts/preprocess_maestro.py --input data/maestro --output data/processed/maestro
python scripts/preprocess_slakh.py --input data/slakh2100_flac_redux --output data/processed/slakh

# Train
python scripts/train.py --config configs/maestro_piano.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/maestro_piano.yaml

# Transcribe
python scripts/transcribe.py --audio input.wav --output output.mid --checkpoint checkpoints/best.pt
```

## Coding Conventions
- Type hints on all function signatures
- Docstrings (Google style) on all public functions
- No wildcard imports
- Use pathlib.Path over os.path where practical
- Follow the architecture in AGENTS.md exactly

## Key Reference
- Read AGENTS.md for full architecture, token vocabulary, hyperparameters
- Read configs/*.yaml for training configurations