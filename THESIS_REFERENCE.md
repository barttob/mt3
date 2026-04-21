# MT3 — Kompleksowy opis techniczny projektu
## Referencja dla pracy magisterskiej

---

## Spis treści

1. [Przegląd projektu](#1-przegląd-projektu)
2. [Stos technologiczny i wymagania](#2-stos-technologiczny-i-wymagania)
3. [Struktura repozytorium](#3-struktura-repozytorium)
4. [Architektura systemu](#4-architektura-systemu)
5. [Tokenizacja MIDI — `src/tokenizer.py`](#5-tokenizacja-midi--srctokenizerpy)
6. [Frontend spektrogram — `src/frontend.py`](#6-frontend-spektrogram--srcfrontendpy)
7. [Enkoder — `src/encoder.py`](#7-enkoder--srcencoderpy)
8. [Dekoder — `src/decoder.py`](#8-dekoder--srcdecoderpy)
9. [Model główny — `src/model.py`](#9-model-główny--srcmodelpy)
10. [Dataset — `src/dataset.py`](#10-dataset--srcdatasetpy)
11. [Augmentacja — `src/augmentation.py`](#11-augmentacja--srcaugmentationpy)
12. [Metryki — `src/metrics.py`](#12-metryki--srcmetricspy)
13. [Trening — `scripts/train.py`](#13-trening--scriptstrainpy)
14. [Ewaluacja — `scripts/evaluate.py`](#14-ewaluacja--scriptsevaluatepy)
15. [Transkrypcja — `scripts/transcribe.py`](#15-transkrypcja--scriptstranskribepy)
16. [Preprocessing danych](#16-preprocessing-danych)
17. [Konfiguracje — `configs/*.yaml`](#17-konfiguracje--configsyaml)
18. [Pięć nowatorskich rozszerzeń (F1–F5)](#18-pięć-nowatorskich-rozszerzeń-f1f5)
19. [Plan eksperymentów (TRAINING_PLAN.md)](#19-plan-eksperymentów-training_planmd)
20. [Kompletny przepływ danych](#20-kompletny-przepływ-danych)
21. [Kluczowe decyzje projektowe](#21-kluczowe-decyzje-projektowe)
22. [Rozbieżności dokumentacja–kod](#22-rozbieżności-dokumentacjakod)
23. [Oczekiwane wyniki i baselines](#23-oczekiwane-wyniki-i-baselines)

---

## 1. Przegląd projektu

Projekt jest implementacją **MT3** (Music Transcription with Transformers, Gardner et al., ICLR 2022) w PyTorchu, rozszerzoną o pięć nowatorskich modyfikacji architektonicznych. System traktuje automatyczną transkrypcję muzyczną (AMT) jako zadanie **sekwencja→sekwencja**: spektrogram log-mel → strumień tokenów zdarzeń MIDI.

**Cel badawczy pracy magisterskiej**: porównanie z bazowym CNN+BiLSTM oraz ablacja pięciu rozszerzeń (F1–F5) na dwóch zbiorach danych (MAESTRO v3.0.0, Slakh2100).

**Kluczowe cechy architektury:**
- Transformer enkoder-dekoder w stylu T5 (pre-LayerNorm, weight tying)
- Enkoder: ramki spektrogramu → dwukierunkowy Transformer
- Dekoder: autoregresywne generowanie tokenów MIDI z cross-attention
- Obsługa ciągłości między segmentami poprzez mechanizm „sekcji tie"

---

## 2. Stos technologiczny i wymagania

```
Python 3.11+
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

Trening: mixed precision (AMP), optymalizator AdamW, warmup liniowy + cosine decay.

---

## 3. Struktura repozytorium

```
mt3/
├── AGENTS.md                  Pełna specyfikacja architektury (44 KB)
├── CLAUDE.md                  Skrócony opis projektu dla Claude Code
├── TRAINING_PLAN.md           Matryca eksperymentów C0–C5, M0–M3, S1–S6, N1–N8
├── PROJECT_REFERENCE.md       Głęboka referencja techniczna (56 KB)
├── integration_test.py        Test end-to-end (syntetyczne dane, 20 kroków, MIDI)
├── requirements.txt
│
├── src/
│   ├── tokenizer.py           MidiTokenizer (płaski + hierarchiczny)
│   ├── frontend.py            SpectrogramFrontend (log-mel)
│   ├── encoder.py             SpectrogramEncoder + RoPE + 2D patches + Conv frontend
│   ├── decoder.py             EventDecoder + PitchAwareDecoderLayer + RoPE decoder
│   ├── model.py               MT3Model + build_model + compute_pitch_context
│   ├── dataset.py             TranscriptionDataset + collate_fn
│   ├── augmentation.py        WaveformAugmenter (gain, noise, time-mask)
│   └── metrics.py             Wrappery mir_eval, filter_notes, deduplicate_notes
│
├── scripts/
│   ├── preprocess_maestro.py  WAV+MIDI → *_audio.npy / *_notes.npy
│   ├── preprocess_slakh.py    mix.flac+per-stem MIDI → *_audio.npy / *_notes.npy
│   ├── train.py               Punkt wejścia treningu
│   ├── evaluate.py            Okienkowa ewaluacja na splitcie
│   └── transcribe.py          Plik audio → .mid (okna z tie-continuation)
│
├── configs/
│   ├── maestro_piano.yaml     Bazowy M0 (piano, 26M params)
│   ├── slakh_multi.yaml       N8 (Slakh, wszystkie F1–F5 włączone)
│   ├── slakh_small.yaml       S1 (d_model=256, ~4M params)
│   ├── slakh_long_context.yaml  S3 (4.096 s segmenty)
│   ├── slakh_no_drums.yaml    S4 (bez perkusji)
│   ├── slakh_from_piano.yaml  S2 (transfer piano→multi)
│   ├── finetune_slakh_to_maestro.yaml  S5 (cross-dataset fine-tune)
│   ├── finetune_slakh_maestro_mix.yaml S6 (mixed dataset)
│   ├── colab_maestro.yaml     Colab T4/V100 (fp16)
│   ├── colab_slakh_multi.yaml Colab A100 Pro (bf16, batch=16)
│   └── colab_slakh_small.yaml Colab mały model
│
└── data/                      (gitignored) raw + processed
```

---

## 4. Architektura systemu

### Schemat wysokopoziomowy

```
Waveform (16 kHz, mono)
        │
        ▼
┌─────────────────────┐
│  SpectrogramFrontend │  log-mel, (B, 512, T)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  SpectrogramEncoder  │  8 warstw pre-LN Transformer → (B, T, 512)
│  (T5-style, bidi)   │
└─────────────────────┘
        │  cross-attention
        ▼
┌─────────────────────┐
│    EventDecoder      │  8 warstw kauzalnych + cross-attn → (B, S, vocab)
│  (autoregresywny)   │
└─────────────────────┘
        │
        ▼
  Tokeny MIDI → pretty_midi → .mid
```

### Parametry bazowego modelu (maestro_piano.yaml)

| Składnik | Parametry |
|----------|-----------|
| Frontend | ~0 (transformacje torchaudio) |
| Enkoder | ~13 M |
| Dekoder (z weight tying) | ~13 M |
| **Razem** | **~26 M** |

### Rozmiary modeli według AGENTS.md

| Wariant | d_model | nhead | warstwy | d_ff | params |
|---------|---------|-------|---------|------|--------|
| Small | 256 | 4 | 6 | 1024 | ~8 M |
| Base | 512 | 8 | 8 | 2048 | ~26 M |
| Large | 768 | 12 | 12 | 3072 | ~85 M |

---

## 5. Tokenizacja MIDI — `src/tokenizer.py`

### Klasa `MidiTokenizer`

```python
MidiTokenizer(
    time_step_ms: int = 8,
    max_time_steps: int = 600,
    num_velocities: int = 128,
    num_pitches: int = 128,
    num_programs: int = 129,        # 0-127 MIDI + 128 = perkusja
    multi_instrument: bool = False,
    use_hierarchical_time: bool = False,  # F1
    coarse_step_ms: int = 64,
    num_coarse: int = 75,
)
```

### Słownik tokenów — układ płaski (domyślny)

| Zakres ID | Typ tokenu | Liczba | Opis |
|-----------|-----------|--------|------|
| 0 | `<pad>` | 1 | Wypełnienie |
| 1 | `<sos>` | 1 | Początek sekwencji |
| 2 | `<eos>` | 1 | Koniec sekwencji |
| 3 | `<tie>` | 1 | Separator sekcji tie |
| 4–603 | `time_shift` | 600 | Przesunięcia czasowe × 8 ms (0–4,792 s) |
| 604–731 | `velocity` | 128 | Siła uderzenia MIDI 0–127 |
| 732–859 | `note_on` | 128 | Atak nuty (dźwięk MIDI 0–127) |
| 860–987 | `note_off` | 128 | Koniec nuty (dźwięk MIDI 0–127) |
| 988–1116 | `program` | 129 | Instrument MIDI (tylko multi-instrument) |

**Rozmiary słownika:**
- Piano (mono-instrument): **988 tokenów**
- Multi-instrument: **1117 tokenów** (`num_programs=129`)

### Słownik tokenów — układ hierarchiczny (F1)

Zastępuje 600 płaskich time_shift tokenów kombinacją 75 coarse + 8 fine:
- `coarse_time_shift(Δt // 64 ms)` + `fine_time_shift((Δt % 64) // 8 ms)`
- Zachowuje rozdzielczość 8 ms przy mniejszym słowniku
- Koszt: +1 token na każdy unikalny timestamp

**Rozmiary słownika (hierarchiczny):**
- Piano: **471 tokenów** (−52 % względem płaskiego)
- Multi-instrument: **600 tokenów** (−46 %)

### Gramatyka sekwencji tokenów

```
<sos>
[SEKCJA TIE: dla każdej nuty trzymanej z poprzedniego segmentu]
  program? velocity note_on(pitch)
<tie>
[SEKCJA ZDARZEŃ:
  time_shift(k)                         # jeden na unikalny bin czasowy
  dla każdego note_off w tym czasie: note_off(pitch)
  dla każdego note_on w tym czasie: program? velocity note_on(pitch)
]
<eos>
[<pad> <pad> ...]                       # prawostronne dopełnienie do max_token_len=1024
```

**Porządek sortowania** przy tym samym czasie: `(time, 0 jeśli off else 1, pitch)` — note_off przed note_on, by retrigger tej samej nuty był jednoznaczny.

### Kluczowe metody

| Metoda | Wejście | Wyjście | Opis |
|--------|---------|---------|------|
| `build_tie_prefix(prev_active)` | lista aktywnych nut | `list[int]` | Buduje prompt dla dekodera (sekcja tie) |
| `notes_to_tokens(notes, start_s, end_s, prev_active)` | lista nut, czasy | `list[int]` | Koduje nuty na tokeny |
| `tokens_to_notes(tokens, segment_start_s)` | `list[int]` | `list[dict]` | Dekoduje tokeny na nuty |
| `token_type(token_id)` | `int` | `str` | Etykieta czytelna dla człowieka |

### Obsługa nut przez `tokens_to_notes`

- Utrzymuje słownik `open_notes[(pitch, program)] → list[dict]`
- Nuty w sekcji tie mają onset ucinany do `segment_start_s`
- Niezamknięte nuty są zamykane w ostatnim zdekodowanym czasie
- Fallback dla note_off: jeśli brak pasującego `(pitch, program)`, dopasowuje do dowolnej otwartej nuty o tym dźwięku (wybiera najstarszą)

---

## 6. Frontend spektrogram — `src/frontend.py`

### Klasa `SpectrogramFrontend(nn.Module)`

```python
SpectrogramFrontend(sample_rate=16000, n_fft=2048, hop_length=128, n_mels=512)
```

**Brak parametrów trenowalnych.** Opakowuje `torchaudio.transforms.MelSpectrogram(power=2.0)` i stosuje `log(mel + 1e-6)`.

**Właściwości czasowo-częstotliwościowe:**
- Częstotliwość ramek: 16000/128 = **125 Hz** (hop = 8 ms)
- Dla segmentu 32768 próbek (2,048 s): T ≈ 256–257 ramek
- n_mels=512 (celowo wyższe niż standardowe 80–128 dla mowy — muzyka wymaga większej rozdzielczości częstotliwościowej)

**I/O:** `waveform (B, N)` → `log_mel (B, 512, T)`

---

## 7. Enkoder — `src/encoder.py`

### Klasa `SpectrogramEncoder(nn.Module)`

```python
SpectrogramEncoder(
    n_mels=512, d_model=512, nhead=8, num_layers=8,
    dim_feedforward=2048, dropout=0.1,
    use_2d_patches=False, patch_f=64, patch_t=8,   # F3
    use_rope=False,                                 # F4
    use_conv_frontend=False, conv_layers=2,         # F5
)
```

### Klasy pomocnicze

**`SinusoidalPositionalEncoding(d_model, max_len=2048)`**
- Standardowe PE Vaswani, bufor nietrenovalny

**`RotaryEmbedding(head_dim, base=10000)`** (F4)
- Implementacja RoPE; leniwie buduje cache cos/sin kształtu `(1, 1, T, head_dim)`
- `apply(q, k)` — obraca Q i K: `x·cos + rotate_half(x)·sin`
- Zero parametrów trenowalnych

**`RoPEMultiheadAttention(d_model, nhead, dropout)`** (F4)
- Drop-in replacement dla `nn.MultiheadAttention(batch_first=True)`
- RoPE stosowane tylko do Q i K (nie V, nie cross-attention)

**`ConvFrontend(n_mels, d_model, num_layers=2, kernel_size=3)`** (F5)
- Stos Conv1d (stride=1, padding=1) zamiast liniowej projekcji per-ramka
- Receptive field: ~24 ms przy 2 warstwach × 512 kanałów
- Dodatkowe parametry: ~1,57 M

**`PatchEmbedding2D(n_mels, patch_f, patch_t, d_model, ...)`** (F3)
- ViT-style 2D embedding; każdy patch: 64 mel biny × 8 ramek (≈ 1 oktawa × 64 ms)
- Reshape: `(B, n_mels, T) → (B, n_fp, n_tp, patch_f·patch_t)`
- Liniowa projekcja każdego patcha do `d_model`
- 2D sinusoidalne PE: `concat(time_PE[d_model/2], freq_PE[d_model/2])`
- Default `patch_f=64, patch_t=8`, `n_mels=512, T=256` → **256 patchy** (ta sama długość sekwencji)

### Cztery tryby operacyjne

| Tryb | Konfiguracja | Embedding | PE | Atencja |
|------|-------------|-----------|-----|---------|
| A (domyślny) | oba flagi OFF | `Linear(n_mels, d_model)` | 1D sinusoidalne | `nn.TransformerEncoder` |
| B (2D patches) | `use_2d_patches=True` | `PatchEmbedding2D` (baked 2D PE) | w PatchEmbedding2D | `nn.TransformerEncoder` |
| C (conv) | `use_conv_frontend=True` | `ConvFrontend` | 1D sinusoidalne (lub brak z RoPE) | `nn.TransformerEncoder` |
| D (RoPE) | `use_rope=True` | jak A/C | brak (RoPE w Q/K) | `RoPETransformerEncoderLayer` |

Wszystkie tryby kończą się `LayerNorm(d_model)`. Dekoder nie widzi różnicy między trybami.

---

## 8. Dekoder — `src/decoder.py`

### Klasa `EventDecoder(nn.Module)`

```python
EventDecoder(
    vocab_size: int,
    d_model=512, nhead=8, num_layers=8,
    dim_feedforward=2048, dropout=0.1,
    max_seq_len=1024,
    use_pitch_aware_attention=False,   # F2
    use_rope=False,                    # F4
)
```

### `PitchAwareDecoderLayer` (F2)

Uczone `pitch_embedding: nn.Embedding(129, d_model)` (128 MIDI + 1 null = idx 128, zerowane przy inicjalizacji) dodawane do zapytania (query) cross-attention:

```
x2 = norm2(x)
x2_q = x2 + pitch_embedding(pitch_ids)    # query uwarunkowany na dźwięk
x2, _ = cross_attn(x2_q, memory, memory)  # klucze/wartości niezmienione
```

Dodatkowe parametry: `129 × 512 × 8 warstw = ~528 K`

### Architektura dekodera

```
tgt_tokens (B, S)
  → Embedding(vocab_size, d_model) × sqrt(d_model)
  → SinusoidalPositionalEncoding       [pominięte jeśli use_rope]
  → Dropout
  → 8 warstw dekodera
  → LayerNorm
  → Linear(d_model, vocab_size, bias=False)   [wagi współdzielone z Embedding]
  → logity (B, S, vocab_size)
```

### Weight tying (związanie wag)

`self.output_proj.weight = self.token_embedding.weight`
- Oszczędność: ~508 K params dla piano (~988 × 512)
- Standardowa praktyka dla LM, poprawia generalizację

### Macierz trybów dekodera

| `use_pitch_aware` | `use_rope` | Stos warstw |
|-------------------|-----------|-------------|
| False | False | `nn.TransformerDecoder` (domyślny) |
| False | True | `RoPETransformerDecoderLayer` × 8 |
| True | False | `PitchAwareDecoderLayer` × 8 |
| True | True | `PitchAwareDecoderLayer(use_rope=True)` × 8 |

**Uwaga projektowa**: RoPE stosowane tylko do self-attention (nie cross-attention), ponieważ pozycje enkodera i dekodera są niezależnymi sekwencjami.

---

## 9. Model główny — `src/model.py`

### `compute_pitch_context(tokens, note_on_offset)`

Forward-fill (pętla Python): dla każdej pozycji S — MIDI dźwięku ostatniego tokenu `note_on`, lub 128 (null) jeśli brak. Wynik: `pitch_ids (B, S)` → wejście do `PitchAwareDecoderLayer`.

### `class MT3Model(nn.Module)`

**`forward(waveform, tgt_tokens, tgt_mask=None, tgt_padding_mask=None)`**
- Teacher-forced: `frontend → encoder → [compute_pitch_context] → decoder`
- Zwraca logity `(B, S, vocab_size)`

**`transcribe(waveform, max_len=1024, temperature=0.0, prompt_tokens=None, return_confidences=False)`**
- Autoregresywne dekodowanie: enkoduje raz, generuje token po tokenie
- Temperature=0 → zachłanne; >0 → próbkowanie multinomialne
- `prompt_tokens` = prefiks (typowo: output `tokenizer.build_tie_prefix(prev_active)`)
- Opcjonalne raportowanie pewności (max softmax probability)
- Wczesne zakończenie gdy wszystkie sekwencje wyemitują `<eos>`

### `build_model(config)` — fabryka modelu

Czyta sekcje YAML (`audio`, `model`, `tokenizer`, `data`), tworzy wszystkie podmoduły. Przekazuje flagi F1–F5 do odpowiednich konstruktorów.

---

## 10. Dataset — `src/dataset.py`

### `class TranscriptionDataset(Dataset)`

```python
TranscriptionDataset(
    data_dir: str | Path,
    tokenizer: MidiTokenizer,
    sample_rate=16000,
    segment_samples=32768,       # 2,048 s przy 16 kHz
    max_token_len=1024,
    segments_per_file=10,        # wirtualne próbki na plik na epokę
    augmenter: WaveformAugmenter | None = None,
    random_crop=True,            # False dla walidacji → deterministyczne centrum
)
```

**`__len__`** = `len(files) × segments_per_file`

### Przepływ `__getitem__`

1. `file_idx = idx % len(files)` → wczytaj `*_audio.npy` i `*_notes.npy`
2. Wybór okna: losowy start (`random.randint`) lub centrum (`max_start // 2`)
3. Zero-padding jeśli plik krótszy niż segment
4. **Filtr nakładania**: nuta włączona jeśli `onset < end_s AND offset > start_s`
5. **Filtr prev_active (tie)**: nuta jeśli `onset < start_s AND offset > start_s`
6. `tokenizer.notes_to_tokens(notes, start_s, end_s, prev_active)`
7. Ucięcie do `max_token_len-1` lub dopełnienie `<pad>`
8. Opcjonalny `augmenter(waveform)`
9. Zwraca `(waveform: float32 (segment_samples,), tokens: int64 (max_token_len,))`

---

## 11. Augmentacja — `src/augmentation.py`

### `class WaveformAugmenter`

Stosowana tylko podczas treningu (nie walidacji). Każda augmentacja niezależnie z prawdopodobieństwem p=0.5:

| Augmentacja | Parametr | Domyślna wartość |
|-------------|----------|-----------------|
| Perturbacja wzmocnienia | mnożnik `~ Uniform(*gain_range)` | `(0.5, 1.5)` |
| Szum Gaussowski | odchylenie σ | `0.005` |
| Time masking | zerowanie ciągłego regionu `~ Uniform(0, max_ratio) × N` | `time_mask_max_ratio=0.1` |

**Brak pitch shift / time stretch** — wymagałoby jednoczesnego retimingu etykiet MIDI.

---

## 12. Metryki — `src/metrics.py`

Opakowuje `mir_eval.transcription.precision_recall_f1_overlap`.

### Funkcje

| Funkcja | Opis |
|---------|------|
| `filter_notes(notes, min_duration_s=0.03)` | Usuwa zbyt krótkie nuty (10 ms dla perkusji) |
| `deduplicate_notes(notes, overlap_s=0.05)` | Usuwa duplikaty z nakładających się okien; wygrywa dłuższa nuta |
| `evaluate_transcription(ref, est, ...)` | Zwraca onset P/R/F1 i onset+offset P/R/F1 |
| `per_program_metrics(ref, est, ...)` | `{program → metryki}` |
| `macro_average_metrics(per_prog)` | Nieważona średnia po programach |
| `instrument_detection_f1(ref, est)` | Precyzja/recall/F1 wykrywania instrumentów |

### Parametry tolerancji (domyślne)

| Parametr | Wartość | Opis |
|----------|---------|------|
| `onset_tolerance` | 0.05 s (50 ms) | Dopuszczalna różnica ataku |
| `offset_ratio` | 0.2 | Offset do `max(0.2 × czas_trwania, 50 ms)` |
| `offset_min_tolerance` | 0.05 s | Minimalna tolerancja końca nuty |

---

## 13. Trening — `scripts/train.py`

### Wywołanie

```bash
python scripts/train.py --config configs/maestro_piano.yaml
python scripts/train.py --config configs/... --resume checkpoints/step_N.pt
python scripts/train.py --config configs/... --dry-run    # 2 syntetyczne kroki
```

### Pipeline treningu

1. Wczytanie konfig (`yaml.safe_load`)
2. `build_model(config)` → `MT3Model` na device (CUDA jeśli dostępne)
3. Złożenie datasetu:
   - Pojedynczy `train_dir` lub wiele `train_dirs` → `ConcatDataset`
   - `train_weights` → `WeightedRandomSampler` (nadpróbkowanie zbiorów)
4. **Optymalizator**: `AdamW(lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)`
5. **Harmonogram LR**: warmup liniowy → **cosine decay** do 0 *(uwaga: AGENTS.md mówi inverse-sqrt, ale kod implementuje cosine)*
6. **Strata**: `CrossEntropyLoss(ignore_index=<pad>, label_smoothing=0.1)` na `logits[:,:-1]` vs `tokens[:,1:]`
7. **Mixed precision**: bf16 (bez GradScaler) lub fp16 (z GradScaler)
8. **Gradient accumulation**: efektywny batch = `batch_size × grad_accum_steps`; grad clip = 1.0; skip NaN/Inf gradientów
9. **Walidacja**: co `eval_every` kroków, max 200 batchy; wczesne zatrzymanie po `patience=10` epokach bez poprawy
10. **Logging**: konsola + TensorBoard (`runs/train/`) + CSV (`runs/train/metrics.csv`)
11. **Checkpointing**: co `save_every` kroków zapisuje `{model, optimizer, scheduler, scaler, step, config}`; `best.pt` dla najlepszej val_loss

### Konfiguracja bazowa (maestro_piano.yaml)

| Parametr | Wartość |
|----------|---------|
| batch_size | 4 |
| grad_accum_steps | 8 (efektywny batch = 32) |
| lr | 3e-4 |
| warmup_steps | 4000 |
| max_steps | 500 000 |
| weight_decay | 0.01 |
| grad_clip | 1.0 |
| label_smoothing | 0.1 |
| patience | 10 |
| num_workers | 4 |
| precision | bf16 |

---

## 14. Ewaluacja — `scripts/evaluate.py`

### Wywołanie

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt \
    --config configs/maestro_piano.yaml \
    --split {train,validation,test} \
    [--per-program] [--temperature 0.0] [--max-len 1024]
```

### Przepływ

1. Wczytanie modelu z checkpointu (`torch.load(..., weights_only=True)`)
2. Rozwiązanie rozmiaru okna z parametrów lub konfiguracji
3. Dla każdego pliku w splicie:
   - `transcribe_full_audio` (okienkowa transkrypcja z tie-continuation)
   - `evaluate_transcription(ref, est)`
   - Opcjonalnie: `per_program_metrics`, `instrument_detection_f1`
4. **Uśrednianie per-plik** (nie per-nuta) — zgodnie z konwencją artykułu MT3
5. Wydruk tabel: overall, macro-avg, per-program, instrument detection

---

## 15. Transkrypcja — `scripts/transcribe.py`

### Wywołanie

```bash
python scripts/transcribe.py --audio input.wav --output output.mid \
    --checkpoint checkpoints/best.pt --config configs/maestro_piano.yaml \
    [--segment-seconds X] [--hop-seconds X | --overlap 0.5] \
    [--min-duration-ms 30.0]
```

### `transcribe_full_audio` — główna funkcja

```python
transcribe_full_audio(
    model, audio, sample_rate=16000,
    segment_samples=256000, hop_samples=128000,
    max_len=1024, temperature=0.0,
    device=None, min_duration_ms=30.0
)
```

**Algorytm okienkowy:**

1. `num_segments = max(1, (total - segment + hop) // hop)`
2. Dla każdego segmentu (zero-padding do `segment_samples`):
   a. Zbuduj prompt: `tokenizer.build_tie_prefix(prev_active)`
   b. `model.transcribe(waveform, max_len, temperature, prompt_tokens=prompt)`
   c. Dekoduj tokeny → nuty z `segment_start_s`
   d. `prev_active` = nuty aktywne w momencie `next_start_s`
3. Po wszystkich oknach: `deduplicate_notes` + `filter_notes`

### Wyjście MIDI (`notes_to_midi`)

- Używa `pretty_midi`
- Program 128 → `is_drum=True`
- Wymusza `offset > onset + 1 ms`
- Velocity: zakres [1, 127]

---

## 16. Preprocessing danych

### MAESTRO — `scripts/preprocess_maestro.py`

- Wczytuje `maestro-v3.0.0.csv`, grupuje według splitu
- Multiprocessing (`Pool`, domyślnie 4 workery)
- `librosa.load(sr=16000, mono=True)` → float32
- `pretty_midi.PrettyMIDI` → iteracja po non-drum instrumentach
- Zapis: `{stem}_audio.npy` i `{stem}_notes.npy`
- Opcja `--dry-run` (5 ścieżek, bez zapisu)

### Slakh2100 — `scripts/preprocess_slakh.py`

- Dla każdego tracku w `train/`, `validation/`, `test/`
- Wczytuje `mix.flac` (mono, resampling do 16 kHz)
- `metadata.yaml` → `program_num` i `is_drum` dla każdego stemu
- **Mapowanie perkusji**: program 128 (poza standardowym zakresem 0–127)
- Zbiera nuty ze wszystkich `audio_rendered=True` stemów

**Rozmiary zbiorów Slakh2100-FLAC-Redux:**
- Train: 1289 ścieżek
- Validation: 270 ścieżek
- Test: 151 ścieżek

---

## 17. Konfiguracje — `configs/*.yaml`

Wszystkie konfigi mają wspólną strukturę: `data`, `audio`, `model`, `tokenizer`, `training`.

### Główne konfiguracje

| Plik | Eksperyment | Kluczowe różnice |
|------|-------------|-----------------|
| `maestro_piano.yaml` | **M0** — baseline piano | d_model=512, 26M params, 500K steps, bf16, wszystkie F1-F5 OFF |
| `slakh_multi.yaml` | **N8** — wszystkie features | multi_instrument, num_programs=129, **wszystkie F1-F5 ON**, 300K steps |
| `slakh_small.yaml` | **S1** — mały model | d_model=256, nhead=4, 6 warstw, ~4M params, 200K steps |
| `slakh_long_context.yaml` | **S3** — długi kontekst | segment=65536 (4,096 s), batch=2, grad_accum=16 |
| `slakh_no_drums.yaml` | **S4** — bez perkusji | num_programs=128, exclude_drums=true |
| `slakh_from_piano.yaml` | **S2** — transfer piano→multi | lr=3e-5, warmup=1000, 150K steps, multi_instrument=true |
| `finetune_slakh_to_maestro.yaml` | **S5** — cross-dataset | lr=3e-5, 100K steps, multi_instrument=true (kompatybilność vocab) |
| `finetune_slakh_maestro_mix.yaml` | **S6** — mixed dataset | train_dirs: [slakh, maestro], train_weights: [1.0, 2.0] |
| `colab_maestro.yaml` | Colab T4/V100 | fp16, batch=2, grad_accum=16, save/eval co 2000 kroków |
| `colab_slakh_multi.yaml` | Colab A100 Pro | bf16, batch=16, grad_accum=2 |

---

## 18. Pięć nowatorskich rozszerzeń (F1–F5)

Każda cecha jest niezależną flagą ON/OFF. Flagi przekazywane przez `build_model(config)`.

### F1 — Hierarchiczna tokenizacja czasowa

**Flaga**: `tokenizer.use_hierarchical_time: true`

**Opis**: Zastąpienie 600 płaskich tokenów `time_shift` parą tokenów coarse+fine. Czas emitowany jako 2 kolejne tokeny: `coarse(Δt // 64 ms)` + `fine((Δt % 64) // 8 ms)`.

**Efekt na słownik:**
- Piano: 988 → 471 (−52 %)
- Multi: 1117 → 600 (−46 %)

**Pytanie badawcze**: Czy mniejszy słownik przekłada się na lepszą efektywność próbek przy tym samym budżecie kroków?

**Parametry:** `coarse_step_ms=64`, `num_coarse=75`, `num_fine=8`

---

### F2 — Pitch-aware cross-attention

**Flaga**: `model.use_pitch_aware_attention: true`

**Opis**: Wyuczona tablica `pitch_embedding: Embedding(129, d_model)` dodawana do query cross-attention w dekoderze. Null embedding (idx 128) zero-inicjowany. Klucze i wartości enkodera niezmienione.

**Intuicja**: Gdy dekoder emituje `note_on(P)`, query powinno preferencyjnie uwzględniać pozycje enkodera z energią w pasmach mel odpowiadających dźwiękowi P.

**Dodatkowe parametry**: ~528 K (129 × 512 × 8 warstw)

---

### F3 — 2D frequency-time patch encoder

**Flaga**: `model.use_2d_patches: true`, `model.patch_f: 64`, `model.patch_t: 8`

**Opis**: Inspirowane ViT, każdy patch obejmuje 64 mel biny × 8 ramek (≈ 1 oktawa × 64 ms). 2D PE faktoryzowane na komponenty osi czasu i osi częstotliwości (każda d_model/2).

**Wymiary**: `n_mels=512, T=256, patch_f=64, patch_t=8` → **8 × 32 = 256 patchy** (ta sama długość sekwencji enkodera)

**Uzasadnienie**: Enkoder zyskuje jawną świadomość położenia w osi tonalnej przez PE częstotliwościowe.

---

### F4 — Rotary Position Embeddings (RoPE)

**Flaga**: `model.use_rope: true`

**Opis**: Zastąpienie addytywnego PE sinusoidalnego rotacją Q i K wewnątrz każdej głowicy atencji. Daje wrażliwość na pozycje względne. Zero dodatkowych parametrów.

**Stosowany**: tylko w self-attention (nie w cross-attention — enkoder i dekoder mają niezależne sekwencje pozycji).

**Implementacja**: `rotate_half([x1, x2]) = [-x2, x1]`, `x·cos + rotate_half(x)·sin`

---

### F5 — Konwolucyjny frontend enkodera

**Flaga**: `model.use_conv_frontend: true`, `model.conv_layers: 2`

**Opis**: Zastąpienie per-ramkowej projekcji `Linear(n_mels, d_model)` stosem Conv1d (k=3, stride=1, padding=1) zachowującym długość T.

**Receptive field**: ~24 ms przy 2 warstwach (2 × 3 ramki × 8 ms)

**Uzasadnienie**: Lokalny kontekst temporalny przydatny dla ostrych transjentów ataków nut.

**Dodatkowe parametry**: ~1,57 M (2 warstwy × 512 kanałów)

---

## 19. Plan eksperymentów (TRAINING_PLAN.md)

### Macierz przebiegów treningowych

| ID | Opis | Zbiór | Kroki | Parametry | Notatka |
|----|------|-------|-------|-----------|---------|
| **C0–C5** | CNN+BiLSTM baselines | — | — | — | Zewnętrzna implementacja |
| **M0** | Baseline piano | MAESTRO | 500K | 26M, d_model=512 | Cel: Onset F1 > 90% |
| **M1** | Mały model | MAESTRO | 100K | d_model=256 | Ablacja rozmiaru |
| **M2** | Bez augmentacji | MAESTRO | 100K | d_model=512 | Ablacja augmentacji |
| **M3** | Baseline multi | Slakh | 300K | d_model=512 | Bazowy Slakh |
| **S1** | Mały Slakh | Slakh | 200K | d_model=256, 6 warstw | ~4M params |
| **S2** | Transfer piano→multi | MAESTRO→Slakh | 150K | lr=3e-5 | Fine-tune M0 |
| **S3** | Długi kontekst | Slakh | — | segment=65536 | 4,096 s |
| **S4** | Bez perkusji | Slakh | — | num_programs=128 | |
| **S5** | Cross-dataset | Slakh→MAESTRO | 100K | lr=3e-5 | Fine-tune Slakh→MAESTRO |
| **S6** | Mixed dataset | Slakh+MAESTRO | 100K | weights=[1,2] | Oversampling MAESTRO |
| **N1** | F1 (hier. czas) | Slakh | — | — | Izolowana ablacja |
| **N2** | F2 (pitch-aware) | Slakh | — | — | Izolowana ablacja |
| **N3** | F3 (2D patches) | Slakh | — | — | Izolowana ablacja |
| **N4** | F4 (RoPE) | Slakh | — | — | Izolowana ablacja |
| **N5** | F5 (conv frontend) | Slakh | — | — | Izolowana ablacja |
| **N6** | F1+F2 | Slakh | — | — | Parowe kombinacje |
| **N7** | F4+F5 | Slakh | — | — | Parowe kombinacje |
| **N8** | F1+F2+F3+F4+F5 | Slakh | 300K | — | Wszystkie features |

**Minimalna wersja tezy**: C0 + M0 + jedna ablacja C
**Ablacja novel features**: M0@200K jako referencja, następnie N1–N5 izolowane + N8 łączone (~7 przebiegów)

---

## 20. Kompletny przepływ danych

```
Surowe audio (WAV 44.1 kHz stereo)
  │  librosa.load(sr=16000, mono=True)               [preprocess_*.py]
  ▼
float32 waveform + lista {onset, offset, pitch, velocity, program}
  │  np.save(*_audio.npy, *_notes.npy)
  ▼
TranscriptionDataset.__getitem__                     [src/dataset.py]
  │  Losowe wycięcie segment_samples=32768 (2,048 s)
  │  Ekstrakcja nakładających się + prev_active nut
  │  WaveformAugmenter (gain, noise, time-mask)      [src/augmentation.py]
  │  MidiTokenizer.notes_to_tokens(... prev_active)  [src/tokenizer.py]
  │  Dopełnienie do max_token_len=1024
  ▼
DataLoader: (waveform (B, 32768), tokens (B, 1024))
  │  Teacher forcing: tgt_input=tokens[:,:-1], tgt_output=tokens[:,1:]
  ▼
MT3Model.forward                                     [src/model.py]
  ├─ SpectrogramFrontend → log-mel (B, 512, 257)     [src/frontend.py]
  ├─ SpectrogramEncoder → (B, 256, 512)              [src/encoder.py]
  └─ [compute_pitch_context jeśli F2]
     EventDecoder → logity (B, 1023, vocab)          [src/decoder.py]
  ▼
CrossEntropyLoss(ignore_index=<pad>, label_smoothing=0.1)
  │  AdamW(lr=3e-4 peak, β=(0.9,0.98), wd=0.01)
  │  Warmup liniowy 4000 kroków + cosine decay
  │  AMP (bf16), grad_accum × batch efektywny batch 32, clip=1.0
  │  Early stopping: patience=10 ewaluacji
  ▼
Checkpoint best.pt (najlepsza val_loss)

INFERENCJA [transcribe.py]:
  Okno 32768 próbek, hop=16384 (50% overlap domyślnie)
  │  Dla każdego segmentu:
  │    model.transcribe(waveform, prompt=build_tie_prefix(prev_active))
  │    tokens → notes (segment_start_s)
  │    prev_active ← nuty aktywne przy next_start_s
  │  deduplicate_notes(overlap_s=0.05)
  │  filter_notes(min_dur=30ms, 10ms dla drums)
  │  pretty_midi: program 128 → is_drum=True → zapis .mid
  ▼
EWALUACJA [evaluate.py]:
  Per-plik: onset F1 + onset+offset F1 (mir_eval)
  Opcjonalnie: per-program + macro-average + instrument detection F1
  Uśrednianie z równą wagą per-plik (nie per-nuta)
```

---

## 21. Kluczowe decyzje projektowe

### 1. Sekwencja→Sekwencja vs klasyfikacja per-ramka
MT3 unifikuje onset/offset/velocity/polifonię/multi-instrument pod jednym autoregresywnym celem językowym. Umożliwia transfer całego aparatu Transformer z NLP. Koszt: inferencja jest O(S) sekwencyjnych kroków na segment (wolniej niż CNN one-shot piano-roll).

### 2. Mechanizm „sekcji tie" dla ciągłości segmentów
Zamiast utrzymywania ukrytego stanu dekodera (podejście MR-MT3), nuty wciąż brzmiące na początku segmentu są preparowane jako prompt `<sos>…<tie>`. Dekoder uczy się traktować te note_on jako już otwarte od `segment_start_s`. Każdy segment transkrybowany jest niezależnie, ale nuty przekraczające granicę segmentów są poprawnie rekonstruowane.

### 3. Hierarchiczny czas (F1)
Redukcja słownika o ~52% (988→471 piano) kosztem +1 tokenu na timestamp (+~8% długości sekwencji). Pytanie badawcze: czy mniejszy słownik = lepsza efektywność próbek?

### 4. Pitch-aware cross-attention (F2)
Uczony embedding dźwięku (129 wpisów) dodawany do query cross-attention. Intuicja: przy emisji `note_on(P)` query powinno preferencyjnie uwzględniać pozycje enkodera z energią w pasmach mel dźwięku P.

### 5. 2D frequency-time patches (F3)
Każdy patch ≈ 1 oktawa × 64 ms. PE faktoryzowane na czas+częstotliwość, dając enkoderkowi jawną świadomość tonalną.

### 6. RoPE (F4)
Relative-position sensitivity bez dodatkowych parametrów. Nie stosowany do cross-attention (pozycje enkodera i dekodera są niezależne).

### 7. Conv frontend (F5)
Local temporal context (~24 ms) dla ostrych transjentów ataków, które globalny Transformer może przegapić.

### 8. Weight tying
Oszczędza ~508 K params, standardowe dla LM.

### 9. Pre-LayerNorm (konwencja T5)
Lepsza stabilność treningu dla głębokich Transformerów niż Post-LN.

### 10. Label smoothing ε=0.1
Redukuje nadmierne zaufanie, standard dla seq2seq.

### 11. Deterministyczne wycięcia walidacyjne
Val dataset używa centrum (`random_crop=False`) → porównywalna val-loss między przebiegami.

### 12. Uśrednianie metryk per-plik
`evaluate.py` uśrednia słowniki metryk per-plik (równa waga per-utwór, nie per-nuta). Zgodnie z konwencją artykułu MT3.

### 13. Deduplikacja z preferencją dla dłuższej nuty
Nakładające się okna mogą dekodować tę samą nutę dwukrotnie. `deduplicate_notes` identyfikuje pary z tym samym `(pitch, program)` i atakiem w zakresie 50 ms, zachowuje nutę o dłuższym czasie trwania.

---

## 22. Rozbieżności dokumentacja–kod

| # | Gdzie | AGENTS.md mówi | Kod robi | Implikacja dla tezy |
|---|-------|----------------|----------|---------------------|
| 1 | `scripts/train.py` | Harmonogram LR: inverse-sqrt (T5) | Cosine decay do 0 | Czynnik wpływający na konwergencję — należy zanotować |
| 2 | `src/tokenizer.py` | Vocab ~1116 | Faktycznie 1117 (`num_programs=129`) | Tabele rozmiarów słownika: używać 1117 |
| 3 | `configs/slakh_multi.yaml` | "Baseline Slakh" | **Wszystkie F1–F5 włączone** | Ten config NIE jest czystym baseline'em — wymaga ręcznego wyłączenia flag |
| 4 | `configs/slakh_no_drums.yaml` | `exclude_drums: true` | `TranscriptionDataset` nie czyta tej flagi | Wymagana implementacja lub preprocessing |
| 5 | AGENTS.md | Segmenty 256 000 próbek (16 s), losowanie 256 ramek z 2000 | Faktycznie 32 768 próbek (2,048 s), bezpośrednio 256 ramek | Opis architektury w tezie powinien odzwierciedlać faktyczne wartości |
| 6 | `src/model.py` | — | `compute_pitch_context` używa pętli Python (O(S)) | Potencjalne wąskie gardło przy F2 (~1023 iteracje per forward pass) |
| 7 | `src/encoder.py` | — | Wyjście enkodera: 256 lub 257 ramek zależnie od trybu (center=True) | `PatchEmbedding2D` przycina do wielokrotności `patch_t`; tryb domyślny nie |

---

## 23. Oczekiwane wyniki i baselines

### Wyniki MT3 z artykułu (Gardner et al., ICLR 2022)

| Model | Zbiór | Onset F1 | Onset+Offset F1 |
|-------|-------|----------|-----------------|
| MT3 (piano) | MAESTRO | ~96–97 % | ~82–85 % |
| MT3 (multi) | Slakh | ~74–78 % | ~55–62 % |
| MR-MT3 | Slakh | ~79–82 % | ~60–65 % |

### Cel eksperymentów M0

- MAESTRO, piano-only
- 500 000 kroków, ~26M params
- **Oczekiwane**: Onset F1 > 90%

### Kluczowe referencje literaturowe

1. **Gardner et al. (2022)** — MT3: Multi-Task Multitrack Music Transcription. ICLR 2022.
2. **Hawthorne et al. (2021)** — Sequence-to-Sequence Piano Transcription with Transformers. ISMIR 2021.
3. **MR-MT3** — Multi-Resolution MT3 (segment-spanning state mechanism).
4. **YourMT3+** — Rozszerzenie MT3 z dodatkowymi technikamui.
5. **Vaswani et al. (2017)** — Attention Is All You Need.
6. **Su & Lu (2021)** — Roformer: Enhanced Transformer with Rotary Position Embedding.
7. **Dosovitskiy et al. (2021)** — ViT: An Image is Worth 16×16 Words. (inspiracja F3)

### Zbiory danych

- **MAESTRO v3.0.0**: ~200 h solowego fortepianu
- **Slakh2100-FLAC-Redux**: 2100 ścieżek, ~145 h, multi-instrument
  - Train: 1289 | Val: 270 | Test: 151

---

*Plik wygenerowany automatycznie na podstawie analizy całości kodu źródłowego projektu MT3. Stan: kwiecień 2026.*
