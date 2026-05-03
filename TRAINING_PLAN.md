# Plan treningu modeli — AMT Praca Magisterska

> **Cel:** Wytrenowanie modeli do porównania skuteczności i wydajności w Rozdziale 6 pracy.
> Wszystkie modele ewaluowane na **MAESTRO v3.0.0 test set (177 nagrań)** za pomocą `mir_eval`.

---

## Podsumowanie runów

| ID | Model | Konfiguracja | Priorytet | Szac. czas | Status |
|----|-------|-------------|-----------|-----------|--------|
| C0 | CNN+RNN | **Baseline** | 🔴 Krytyczny | ~8–16 h | ⬜ |
| C1 | CNN+RNN | n_mels=128 | 🟡 Ważny | ~6–12 h | ⬜ |
| C2 | CNN+RNN | n_mels=512 | 🟡 Ważny | ~10–20 h | ⬜ |
| C3 | CNN+RNN | pos_weight onset=2.0 | 🟢 Opcjonalny | ~8–16 h | ⬜ |
| C4 | CNN+RNN | pos_weight onset=10.0 | 🟢 Opcjonalny | ~8–16 h | ⬜ |
| C5 | CNN+RNN | BiLSTM 1 warstwa | 🟢 Opcjonalny | ~6–12 h | ⬜ |
| M0 | MT3-like | **Baseline Piano** (500k steps) | 🔴 Krytyczny | ~3–5 dni | ⬜ |
| M1 | MT3-like | d_model=256 (100k steps) | 🟡 Ważny | ~1–2 dni | ⬜ |
| M2 | MT3-like | bez augmentacji (100k steps) | 🟢 Opcjonalny | ~1–2 dni | ⬜ |
| M3 | MT3-like | **Baseline Slakh** (300k steps) | 🟡 Ważny | ~2–4 dni | ⬜ |
| S1 | MT3-like | Slakh d_model=256 (200k steps) | 🟢 Opcjonalny | ~1–2 dni | ⬜+ |
| S2 | MT3-like | Slakh fine-tune z M0 (150k steps) | 🟢 Opcjonalny | ~1–2 dni | ⬜ |
| S3 | MT3-like | Slakh dłuższy kontekst 4s (300k steps) | 🟢 Opcjonalny | ~2–4 dni | + |
| S4 | MT3-like | Slakh bez perkusji (300k steps) | 🟢 Opcjonalny | ~2–4 dni | ⬜/stopped |
| S5 | MT3-like | Slakh→MAESTRO fine-tune (100k steps) | 🟡 Ważny | ~12–24 h | ⬜ |
| S6 | MT3-like | Slakh+MAESTRO mix fine-tune (100k steps) | 🟡 Ważny | ~12–24 h | ⬜ |
| N1 | MT3-like | +F1: hierarchical time (200k steps) | 🟡 Ważny | ~2–3 dni | ⬜ |
| N2 | MT3-like | +F2: pitch-aware attention (200k steps) | 🟡 Ważny | ~2–3 dni | ⬜ |
| N3 | MT3-like | +F3: 2D patch encoder (200k steps) | 🟡 Ważny | ~2–3 dni | ⬜ |
| N4 | MT3-like | +F4: RoPE (200k steps) | 🟡 Ważny | ~2–3 dni | ⬜ |
| N5 | MT3-like | +F5: conv frontend (200k steps) | 🟡 Ważny | ~2–3 dni | ⬜ |
| N6 | MT3-like | +F2+F4: pitch-aware + RoPE (200k steps) | 🟢 Opcjonalny | ~2–3 dni | ⬜ |
| N7 | MT3-like | +F4+F5: RoPE + conv (200k steps) | 🟢 Opcjonalny | ~2–3 dni | ⬜ |
| N8 | MT3-like | **Wszystkie 5 cech** (300k steps) | 🔴 Krytyczny | ~3–5 dni | ⬜ |

**Minimalne wymaganie do obrony:** C0 + M0 (porównanie główne)
**Pełna ablacja:** C0–C2 + M0–M1
**Ablacja Slakh:** M3 + S5 + S6 (cross-dataset generalizacja)
**Ablacja nowatorskich cech:** M0 (baseline 200k) + N1–N5 (izolacja) + N8 (wszystkie)

---

## CNN+RNN — Szczegóły runów

### C0 — Baseline (OBOWIĄZKOWY)

> Model referencyjny. Wszystkie inne runy CNN porównywane względem tego.

```
Dataset:        MAESTRO v3.0.0
n_mels:         229
fmin/fmax:      30 / 8000 Hz
hop_length:     512 (32 ms)
n_fft:          2048

Optimizer:      AdamW
lr:             6e-4
weight_decay:   1e-5
batch_size:     8
chunk_frames:   640
max_epochs:     100
grad_clip:      3.0

onset pos_weight:   5.0
frame pos_weight:   2.0
onset multiplier:   2.0×

lr_scheduler:   ReduceLROnPlateau (patience=5, factor=0.5)
early_stopping: patience=10 epok (val Frame F1)
```

**Checkpoint:** `checkpoints/cnn_rnn_C0_baseline/best.pt`
**Oczekiwane wyniki:** Frame F1 ~85–90%, Onset F1 ~90–95%, Onset+Offset F1 ~75–85%

---

### C1 — Mniej pasm mel (n_mels=128)

> Sprawdza wpływ rozdzielczości częstotliwościowej na jakość transkrypcji.
> Tańszy obliczeniowo — mniejsza macierz wejściowa do CNN.

```
Różni się od C0:
  n_mels: 128  (zamiast 229)
  
Uwaga: warstwa CNN FC ma inny rozmiar wejścia:
  Po Block 2: 96 × (128//4) = 96 × 32 = 3072
  Linear(3072 → 768)  (zamiast Linear(5472 → 768))
```

**Checkpoint:** `checkpoints/cnn_rnn_C1_mels128/best.pt`
**Pytanie badawcze:** Czy redukcja z 229 do 128 pasm istotnie obniża F1?

---

### C2 — Więcej pasm mel (n_mels=512)

> Sprawdza czy wyższa rozdzielczość (jak w MT3) pomaga CNN+RNN.

```
Różni się od C0:
  n_mels: 512  (zamiast 229)
  
Uwaga: warstwa CNN FC:
  Po Block 2: 96 × (512//4) = 96 × 128 = 12288
  Linear(12288 → 768)  — więcej parametrów w tej warstwie (~9.4M zamiast ~4.2M)
```

**Checkpoint:** `checkpoints/cnn_rnn_C2_mels512/best.pt`
**Pytanie badawcze:** Czy CNN+RNN z 512 pasmami zbliża się do MT3-like?

---

### C3 — Niższy pos_weight dla onsetów (pw=2.0)

> Sprawdza wpływ balansu klasy pozytywnej na precision/recall.
> pw=2.0 → model będzie bardziej "ostrożny" w detekcji onsetów (wyższa precyzja, niższy recall).

```
Różni się od C0:
  onset pos_weight: 2.0  (zamiast 5.0)
```

**Checkpoint:** `checkpoints/cnn_rnn_C3_pw2/best.pt`

---

### C4 — Wyższy pos_weight dla onsetów (pw=10.0)

> pw=10.0 → model bardziej agresywny w detekcji (wyższy recall, niższa precyzja).

```
Różni się od C0:
  onset pos_weight: 10.0  (zamiast 5.0)
```

**Checkpoint:** `checkpoints/cnn_rnn_C4_pw10/best.pt`

---

### C5 — Płytszy RNN (1 warstwa BiLSTM)

> Sprawdza czy dwie warstwy BiLSTM są konieczne, czy wystarczy jedna.
> Mniej parametrów → szybsza inferencja.

```
Różni się od C0:
  onset_rnn: BiLSTM(768→256, num_layers=1)
  frame_rnn: BiLSTM(856→256, num_layers=1)
  
Parametry: ~8.7M (zamiast ~11.89M)
```

**Checkpoint:** `checkpoints/cnn_rnn_C5_lstm1layer/best.pt`

---

## MT3-like — Szczegóły runów

### M0 — Baseline Piano (OBOWIĄZKOWY)

> Model referencyjny Transformer. Pełny trening do zbieżności.

```yaml
# maestro_piano.yaml
dataset:          MAESTRO v3.0.0
multi_instrument: false
vocab_size:       988

d_model:          512
nhead:            8
enc_layers:       8
dec_layers:       8
d_ff:             2048
dropout:          0.1
max_token_len:    1024

n_fft:            2048
hop_length:       128
n_mels:           512
sample_rate:      16000
segment_samples:  32768  (= 2.048 s)

optimizer:        AdamW (β1=0.9, β2=0.98, eps=1e-9)
lr:               3e-4
weight_decay:     0.01
batch_size:       4
grad_accum_steps: 16  → efektywny batch = 64
warmup_steps:     4000
max_steps:        500000
grad_clip:        1.0
label_smoothing:  0.1
precision:        bf16

augmentacja:
  gain:        Uniform(0.5, 1.5), p=0.5
  noise:       σ=0.005, p=0.5
  time_mask:   Uniform(0, 10%), p=0.5
```

**Checkpoint:** `checkpoints/mt3_M0_baseline_piano/best.pt`
**Oczekiwane wyniki:** Onset F1 > 90% (per literatura Gardner et al., 2022)

---

### M1 — Mniejszy model (d_model=256, skrócony trening)

> Sprawdza kompromis rozmiar↔jakość. Trening skrócony do 100k steps —
> wystarczający do oceny względnej jakości między konfiguracjami.

```yaml
Różni się od M0:
  d_model:          256  (zamiast 512)
  d_ff:             1024 (zamiast 2048)
  max_steps:        100000  (zamiast 500000)
  
Parametry: ~7M (zamiast ~26M)
```

**Checkpoint:** `checkpoints/mt3_M1_dmodel256/best.pt`
**Pytanie badawcze:** Jak bardzo spada jakość przy 3.7× redukcji liczby parametrów?

---

### M2 — Bez augmentacji (skrócony trening)

> Izoluje wpływ augmentacji waveformu na generalizację modelu.

```yaml
Różni się od M0:
  augmentacja:   WYŁĄCZONA (gain=off, noise=off, time_mask=off)
  max_steps:     100000
```

**Checkpoint:** `checkpoints/mt3_M2_no_aug/best.pt`

---

### M3 — Baseline Multi-instrument na Slakh (opcjonalny)

> Trenowanie na danych wieloinstrumentalnych. Ewaluacja na Slakh test set.
> Uwaga: porównanie z C0/M0 jest utrudnione (inny dataset) — opisywane osobno.

```yaml
# slakh_multi.yaml
dataset:          Slakh2100-FLAC-Redux
multi_instrument: true
vocab_size:       1117

lr:               1e-4
grad_accum_steps: 8  → efektywny batch = 32
max_steps:        300000
precision:        fp16  (z GradScaler)
```

**Checkpoint:** `checkpoints/mt3_M3_slakh_multi/best.pt`

---

## MT3-like — Slakh (ablacje)

> Wszystkie runy S* ewaluowane na **Slakh2100 test set** (nie MAESTRO).
> Pytanie badawcze wspólne: co decyduje o jakości transkrypcji wieloinstrumentalnej?

---

### S1 — Slakh mały model (d_model=256)

> Odpowiednik M1 na danych Slakh. Sprawdza, czy redukcja rozmiaru modelu bardziej
> boli w zadaniu wieloinstrumentalnym niż w pianistycznym (M1 vs S1).

```yaml
# slakh_small.yaml
d_model:          256
nhead:            4
enc_layers:       6
dec_layers:       6
d_ff:             1024
max_steps:        200000  (dłużej niż M1 bo trudniejsze zadanie)
lr:               1e-4
```

**Checkpoint:** `checkpoints/mt3_S1_slakh_small/best.pt`
**Pytanie badawcze:** Czy mały model bardziej traci na multi-instrument niż na piano?
**Params:** ~4M (zamiast ~26M w M3)

---

### S2 — Transfer Piano→Slakh (fine-tuning M0)

> Fine-tuning pretrenowanego M0 (MAESTRO piano) na Slakh.
> Testuje hipotezę, że wiedza o strukturze dźwięków pianistycznych
> ułatwia naukę wieloinstrumentalności (szybsza konwergencja, lepsze F1).
>
> **WAŻNE:** przy ładowaniu checkpointu vocab rośnie z 988→1117 tokenów —
> nowe wiersze embeddingu (program tokens) inicjalizowane losowo.

```yaml
# slakh_from_piano.yaml
# uruchomienie: python scripts/train.py --config configs/slakh_from_piano.yaml \
#                 --resume checkpoints/mt3_M0_baseline_piano/best.pt
lr:               3e-5   (10× niższe niż od zera)
warmup_steps:     1000
max_steps:        150000
multi_instrument: true
```

**Checkpoint:** `checkpoints/mt3_S2_slakh_from_piano/best.pt`
**Pytanie badawcze:** Ile kroków treningowych oszczędza pretraining na pianinie?

---

### S3 — Slakh dłuższy kontekst (4 s)

> segment_samples=65536 — okno 4.096 s zamiast 2.048 s.
> Hipoteza: gęsta polifonia wymaga dłuższego kontekstu, żeby enkoder
> widział pełne nakładające się nuty wielu instrumentów.

```yaml
# slakh_long_context.yaml
segment_samples:  65536  (4.096 s)
n_frames:         512    (zamiast 256)
batch_size:       2      (zmniejszony z powodu 2× dłuższych sekwencji)
grad_accum_steps: 16     → efektywny batch = 32
max_steps:        300000
```

**Checkpoint:** `checkpoints/mt3_S3_slakh_long_ctx/best.pt`
**Pytanie badawcze:** Czy 4 s kontekstu poprawia F1 dla instrumentów z długimi nutami (smyczki)?

---

### S4 — Slakh bez perkusji

> Trening i ewaluacja na Slakh z wykluczonymi ścieżkami perkusyjnymi.
> Perkusja (program ID 128) generuje dużo krótkich, atonalnych onsetów —
> hipoteza: ich usunięcie poprawia metryki dla instrumentów melodycznych.

```yaml
# slakh_no_drums.yaml
num_programs:   128     (0-127, bez drum ID 128)
exclude_drums:  true    # sygnał dla loadera danych
max_steps:      300000
```

**Checkpoint:** `checkpoints/mt3_S4_slakh_no_drums/best.pt`
**Pytanie badawcze:** O ile perkusja obniża F1 na instrumentach melodycznych?

---

### S5 — Slakh→MAESTRO fine-tune (cross-dataset)

> Fine-tuning M3 (Slakh baseline) na danych MAESTRO (solo fortepian).
> Testuje generalizację: czy model wieloinstrumentalny może stać się
> równie dobry jak dedykowany model pianistyczny?

```yaml
# finetune_slakh_to_maestro.yaml
dataset:        maestro
lr:             3e-5
warmup_steps:   1000
max_steps:      100000
```

**Checkpoint:** `checkpoints/mt3_S5_slakh_to_maestro/best.pt`
**Ewaluacja:** MAESTRO v3.0.0 test set (bezpośrednie porównanie z M0)

---

### S6 — Slakh+MAESTRO mix fine-tune

> Fine-tuning z mieszanego datasetu (1× Slakh + 2× MAESTRO oversampling).
> Testuje czy joint training daje lepszy model niż finetuning sekwencyjny (S5).

```yaml
# finetune_slakh_maestro_mix.yaml
dataset:        mixed
train_dirs:     [slakh/train, maestro/train]
train_weights:  [1.0, 2.0]
lr:             3e-5
max_steps:      100000
```

**Checkpoint:** `checkpoints/mt3_S6_slakh_maestro_mix/best.pt`
**Ewaluacja:** MAESTRO v3.0.0 test set (porównanie z M0 i S5)

---

## MT3-like — Ablacja nowatorskich cech

> Wszystkie runy N* oparte na konfiguracji **maestro_piano.yaml** (MAESTRO v3.0.0, piano-only).
> Trenowane 200k kroków (N1–N7) lub 300k kroków (N8) — krótszy budżet niż M0 (500k),
> wystarczający do rzetelnego porównania względnego między konfiguracjami.
>
> **Wspólny punkt odniesienia:** M0 wytrenowany 200k kroków (checkpoint `mt3_M0_200k/best.pt`).
> Dzięki temu wszystkie N* porównywane przy identycznym budżecie treningowym.
>
> Cechy (F1–F5) odpowiadają Sekcji 17 PROJECT_REFERENCE.md.

---

### N1 — +F1: Hierarchiczna tokenizacja czasu

> Izoluje wpływ zamiany 600 płaskich tokenów czasu na schemat coarse+fine (83 tokeny).
> Vocab zmniejsza się: 988 → 471 (piano). Sekwencja wydłuża się ok. ~8% (2 tokeny/zdarzenie).

```yaml
# Na bazie maestro_piano.yaml z modyfikacją:
tokenizer:
  use_hierarchical_time: true
  coarse_step_ms: 64
  num_coarse: 75

training:
  max_steps: 200000
```

**Checkpoint:** `checkpoints/mt3_N1_hierarchical_time/best.pt`
**Pytanie badawcze:** Czy mniejszy vocab poprawia jakość przy identycznym budżecie kroków?
**Uwaga:** vocab_size zmienia się → nie można ładować checkpoint M0 bez resizingu embeddingu.

---

### N2 — +F2: Pitch-Aware Cross-Attention

> Izoluje wpływ embeddingu wysokości nutowej dodawanego do zapytań (query) cross-attention.
> +528K parametrów (8 warstw × 129 embeddings × 512 d_model).

```yaml
model:
  use_pitch_aware_attention: true

training:
  max_steps: 200000
```

**Checkpoint:** `checkpoints/mt3_N2_pitch_aware/best.pt`
**Pytanie badawcze:** Czy bias wysokości nutowej poprawia F1 na gęstej polifonii (akordy)?

---

### N3 — +F3: Enkoder 2D patch

> Izoluje wpływ podziału spektrogramu na 2D patche (patch_f=64, patch_t=8).
> Sekwencja enkodera: 256 patchy (identyczna długość jak tryb A).

```yaml
model:
  use_2d_patches: true
  patch_f: 64
  patch_t: 8

training:
  max_steps: 200000
```

**Checkpoint:** `checkpoints/mt3_N3_2d_patches/best.pt`
**Pytanie badawcze:** Czy 2D PE poprawia detekcję harmonicznych współwystępujących w patchu?

---

### N4 — +F4: Rotary Position Embeddings (RoPE)

> Izoluje wpływ zamiany addytywnego PE na RoPE (rotacja Q/K w każdej warstwie self-attention).
> Zero dodatkowych parametrów. Sinusoidalny PE pomijany w enkoderze i dekoderze.

```yaml
model:
  use_rope: true

training:
  max_steps: 200000
```

**Checkpoint:** `checkpoints/mt3_N4_rope/best.pt`
**Pytanie badawcze:** Czy kodowanie względnej pozycji poprawia precyzję timingu onsetów?

---

### N5 — +F5: Konwolucyjny front-end

> Izoluje wpływ zastąpienia projekcji Linear(n_mels, d_model) stosem Conv1d(k=3).
> +1.57M parametrów (2 bloki Conv1d, n_mels=512→d_model=512).

```yaml
model:
  use_conv_frontend: true
  conv_layers: 2

training:
  max_steps: 200000
```

**Checkpoint:** `checkpoints/mt3_N5_conv_frontend/best.pt`
**Pytanie badawcze:** Czy lokalna receptive field (3 ramki = 24 ms) poprawia F1 dla krótkich nut?

---

### N6 — +F2+F4: Pitch-Aware + RoPE (kombinacja)

> Sprawdza synergię cech po stronie dekodera: RoPE w self-attention + bias wysokości w cross-attention.
> Oba działają ortogonalnie — RoPE dotyczy self-attention, bias pitch dotyczy cross-attention.

```yaml
model:
  use_pitch_aware_attention: true
  use_rope: true

training:
  max_steps: 200000
```

**Checkpoint:** `checkpoints/mt3_N6_pitch_rope/best.pt`
**Pytanie badawcze:** Czy RoPE + pitch-aware dają efekt addytywny czy synergiczny?

---

### N7 — +F4+F5: RoPE + Konwolucyjny front-end (kombinacja)

> Sprawdza synergię cech po stronie enkodera: lokalny kontekst Conv (F5) + relatywna pozycja RoPE (F4).

```yaml
model:
  use_rope: true
  use_conv_frontend: true
  conv_layers: 2

training:
  max_steps: 200000
```

**Checkpoint:** `checkpoints/mt3_N7_rope_conv/best.pt`
**Pytanie badawcze:** Czy lokalne wzorce konwolucyjne i relatywna pozycja RoPE uzupełniają się?

---

### N8 — Wszystkie 5 cech (OBOWIĄZKOWY dla rozdziału ablacyjnego)

> Model z włączonymi wszystkimi pięcioma nowatorskimi cechami.
> Stanowi główny argument pracy — porównanie z M0 pokazuje łączny wkład innowacji.
> Trening 300k kroków (dłuższy niż N1–N7 bo model ma więcej parametrów i mniejszy vocab).

```yaml
# Pełna konfiguracja — na bazie maestro_piano.yaml:
model:
  use_2d_patches: true
  patch_f: 64
  patch_t: 8
  use_pitch_aware_attention: true
  use_rope: true
  use_conv_frontend: true
  conv_layers: 2

tokenizer:
  use_hierarchical_time: true
  coarse_step_ms: 64
  num_coarse: 75

training:
  max_steps: 300000
```

**Checkpoint:** `checkpoints/mt3_N8_all_features/best.pt`
**Oczekiwane wyniki:** Onset F1 ≥ M0 (cel: ≥ 1–2 pp poprawa nad baseline)
**Uwaga:** vocab_size=471 (hierarchical piano) — nie można ładować checkpoint M0.
**Parametry (approx.):** ~28M (+Conv ~1.57M, +PitchEmb ~0.53M, +RoPE ±0)

---

## Tabela ewaluacyjna (do wypełnienia po treningu)

### MAESTRO test set (piano, 177 nagrań)

| ID | Model | Params (M) | Frame F1 | Onset F1 | Onset+Off F1 | RTF | VRAM (MB) |
|----|-------|-----------|---------|---------|-------------|-----|----------|
| C0 | CNN+RNN baseline | 11.89 | — | — | — | — | — |
| C1 | CNN+RNN n_mels=128 | ~8.5 | — | — | — | — | — |
| C2 | CNN+RNN n_mels=512 | ~16.5 | — | — | — | — | — |
| C3 | CNN+RNN pw=2.0 | 11.89 | — | — | — | — | — |
| C4 | CNN+RNN pw=10.0 | 11.89 | — | — | — | — | — |
| C5 | CNN+RNN 1×BiLSTM | ~8.7 | — | — | — | — | — |
| M0 | MT3 baseline (piano) | 26 | — | — | — | — | — |
| M1 | MT3 d_model=256 | ~7 | — | — | — | — | — |
| M2 | MT3 no aug | 26 | — | — | — | — | — |
| S5 | MT3 Slakh→MAESTRO | 26 | — | — | — | — | — |
| S6 | MT3 Slakh+MAESTRO mix | 26 | — | — | — | — | — |

### MAESTRO test set — ablacja nowatorskich cech (200k kroków)

> Punkt odniesienia: M0@200k (ten sam config co M0, ale max_steps=200000).

| ID | Cechy włączone | Vocab | Params (M) | Onset F1 | Onset+Off F1 | ΔOnset vs M0@200k |
|----|---------------|-------|-----------|---------|-------------|-------------------|
| M0@200k | brak (baseline) | 988 | 26 | — | — | 0 (ref) |
| N1 | F1: hierarchical time | 471 | ~26 | — | — | — |
| N2 | F2: pitch-aware attn | 988 | ~26.5 | — | — | — |
| N3 | F3: 2D patches | 988 | ~26 | — | — | — |
| N4 | F4: RoPE | 988 | ~26 | — | — | — |
| N5 | F5: conv frontend | 988 | ~27.6 | — | — | — |
| N6 | F2+F4 | 988 | ~26.5 | — | — | — |
| N7 | F4+F5 | 988 | ~27.6 | — | — | — |
| N8 | F1+F2+F3+F4+F5 (300k) | 471 | ~28 | — | — | — |

### Slakh test set (multi-instrument)

| ID | Model | Params (M) | Frame F1 | Onset F1 | Onset+Off F1 | RTF | VRAM (MB) |
|----|-------|-----------|---------|---------|-------------|-----|----------|
| M3 | MT3 Slakh baseline | 26 | — | — | — | — | — |
| S1 | MT3 Slakh d_model=256 | ~4 | — | — | — | — | — |
| S2 | MT3 Piano→Slakh transfer | 26 | — | — | — | — | — |
| S3 | MT3 Slakh 4s kontekst | 26 | — | — | — | — | — |
| S4 | MT3 Slakh bez perkusji | 26 | — | — | — | — | — |

> RTF = czas inferencji / czas trwania audio (mierzony na CPU i GPU osobno)

---

## Kolejność uruchamiania (zalecana)

```
1.  C0          ← najważniejszy, uruchom pierwszy
2.  M0          ← uruchom równolegle z C0 jeśli masz 2 GPU, lub po C0
3.  M0@200k     ← krótki run M0 (200k steps) — punkt odniesienia dla ablacji N*
4.  M3          ← Slakh baseline; uruchom równolegle z C0/M0 jeśli dostępne GPU
5.  C1          ← szybki, dobry punkt odniesienia dla C2
6.  C2          ← po C1
7.  M1          ← po M0 (do porównania architektury)
8.  S5          ← po M3 (krótki, ważny wynik cross-dataset)
9.  S6          ← po S5 lub równolegle
──── Ablacje nowatorskich cech (wszystkie po M0@200k) ────
10. N4          ← RoPE: najtańszy (0 param extra), dobry punkt wyjścia
11. N5          ← Conv: szybki, izoluje wpływ local context
12. N2          ← Pitch-aware: izoluje wpływ bias wysokości
13. N3          ← 2D patches: izoluje wpływ 2D PE
14. N1          ← Hierarch. time: izoluje wpływ mniejszego vocab (inny vocab_size)
15. N6, N7      ← pairwise: po N2+N4 i N4+N5
16. N8          ← wszystkie cechy (300k): ostatni, najdłuższy
──── Opcjonalne ────
17. C5          ← opcjonalny CNN
18. C3, C4      ← opcjonalne (razem, bo tanie)
19. M2          ← opcjonalny
20. S2          ← po M0 i M3 (wymaga checkpoint M0)
21. S1, S3, S4  ← opcjonalne Slakh ablacje (od najszybszego)
```

---

## Minimalne eksperymenty do obrony

Jeśli czas/zasoby są ograniczone, **wystarczą 3 runy**:

```
C0 + M0  →  porównanie główne (CNN+RNN vs Transformer, MAESTRO)
C1       →  jedna ablacja CNN (wpływ reprezentacji wejścia)
```

Daje to: tabelę wyników, wykres F1 vs. RTF, odpowiedź na PB1 i PB2 z Rozdziału 3.

**Jeśli chcesz uwzględnić Slakh w pracy** (Rozdział 6 multi-instrument):

```
M3  →  Slakh baseline
S5  →  generalizacja cross-dataset (Slakh→MAESTRO)
S6  →  joint training (Slakh+MAESTRO mix)
```

**Jeśli chcesz uwzględnić ablację nowatorskich cech** (Rozdział nowatorski):

```
M0@200k  →  wspólny punkt odniesienia (identyczny budżet jak N*)
N1–N5    →  izolacja każdej cechy osobno (5 runów × ~2–3 dni)
N8       →  wszystkie cechy naraz (wynik finalny, ~3–5 dni)
```

Minimalne 7 runów dla pełnej ablacji cech: `M0@200k + N1 + N2 + N3 + N4 + N5 + N8`.

---

## Notatki techniczne

- Checkpointy zapisywać z pełnymi metadanymi (epoch/step, config, metryki)
- Po każdym runie zmierzyć RTF na **tym samym** nagraniu testowym (np. pierwsze nagranie z test set)
- VRAM mierzyć przez `torch.cuda.max_memory_allocated()` podczas inferencji
- Wyniki wpisywać do tabeli ewaluacyjnej powyżej na bieżąco
- **N1 i N8**: vocab_size=471 (hierarchical) — nie można restartować od checkpoint M0; trening od zera
- **N2–N7**: vocab_size=988 — checkpoint M0 można użyć jako warm start (opcjonalnie, skraca trening)
