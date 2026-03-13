# MT3 Piano Transcription

Autoregressive sequence-to-sequence Transformer for music transcription, targeting:
- **Piano transcription** on MAESTRO v3.0.0
- **Multi-instrument transcription** on Slakh2100-FLAC-Redux

Based on [MT3 (Gardner et al., ICLR 2022)](https://github.com/magenta/mt3).

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download datasets and run preprocessing scripts before training:

```bash
# MAESTRO
python scripts/preprocess_maestro.py

# Slakh2100
python scripts/preprocess_slakh.py
```

## Training

```bash
# Piano (MAESTRO)
python scripts/train.py --config configs/maestro_piano.yaml

# Multi-instrument (Slakh)
python scripts/train.py --config configs/slakh_multi.yaml
```

## Inference

```bash
python scripts/transcribe.py --audio path/to/audio.wav --output output.mid
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/maestro_piano.yaml --checkpoint checkpoints/step_N.pt
```

## Project Structure

See [AGENTS.md](AGENTS.md) for full architecture documentation.
