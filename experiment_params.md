# Parametry do zebrania — eksperymenty AMT

> Dokument na podstawie rozdziałów 5 i 6 pracy magisterskiej.  
> Wszystkie pola `[??]` w rozdziale 6 wymagają uzupełnienia danymi eksperymentalnymi.

---

## 1. Konfiguracja środowiska (Tab. 5.1)

Dane sprzętowe i wersje bibliotek do wpisania w tabeli `tab:hardware`:

| Pole | Wartość |
|------|---------|
| CPU — model i liczba rdzeni | `[uzupełnij]` |
| RAM — pojemność [GB] | `[uzupełnij]` |
| System operacyjny i wersja | `[uzupełnij]` |
| Sterownik CUDA — wersja | `[uzupełnij]` |
| PyTorch — wersja | `[uzupełnij]` |
| Python — wersja | `[uzupełnij]` |
| librosa — wersja | `[uzupełnij]` |
| torchaudio — wersja | `[uzupełnij]` |

GPU jest już wpisane: NVIDIA GeForce RTX 4070 Ti (12 GB VRAM), mir_eval: 0.7.

---

## 2. Metryki skuteczności (Tab. 6.1)

Obliczane przez `mir_eval` na podzbiorze testowym **MAESTRO v3.0.0 (177 nagrań)**.

### 2.1 Modele własne (do zmierzenia)

| Model | Onset F1 [%] | Note F1 [%] | Frame F1 [%] |
|-------|-------------|-------------|--------------|
| CNN+RNN własny (mel, 229 pasm) | `[??]` | `[??]` | `[??]` |
| MT3-like własny M0 (bazowy) | `[??]` | `[??]` | `[??]` |

### 2.2 Modele referencyjne (z publikacji, oznaczone †)

| Model | Onset F1 [%] | Note F1 [%] | Frame F1 [%] |
|-------|-------------|-------------|--------------|
| Onsets and Frames† | `[??]` | `[??]` | `[??]` |
| Basic Pitch† | `[??]` | `[??]` | `[??]` |
| MT3† | `[??]` | `[??]` | `[??]` |
| YourMT3+† | `[??]` | `[??]` | `[??]` |

> **Definicje tolerancji mir_eval:**  
> - Onset F1: onset w oknie ±50 ms  
> - Note F1 (z offsetem): onset ±50 ms AND offset ±20% czasu trwania lub ±50 ms  
> - Frame F1: porównanie binarnych piano-roll (ramka po ramce, 88 klawiszy)

---

## 3. Metryki wydajności obliczeniowej (Tab. 6.2)

Pomiary na tym samym sprzęcie co w Tab. 5.1. Warunki pomiaru:  
- tryb `model.eval()`, `torch.no_grad()`  
- VRAM: mediana z 5 pomiarów, fragment 20 s, `torch.cuda.max_memory_allocated()`  
- Czas inferencji: `torch.cuda.Event` z cudaSynchronize, średnia z ≥10 nagrań  
- RTF_CPU: próbka 10 nagrań, łączny czas ≥60 min

### 3.1 Modele własne (do zmierzenia)

| Model | Params [M] | VRAM [MB] | RTF_GPU | RTF_CPU |
|-------|-----------|-----------|---------|---------|
| CNN+RNN własny (mel) | ≈15.80 | `[??]` | `[??]` | `[??]` |
| MT3-like własny M0 | ≈26.0 | `[??]` | `[??]` | `[??]` |

### 3.2 Modele zewnętrzne (z publikacji lub zmierzone ‡)

| Model | Params [M] | VRAM [MB] | RTF_GPU | RTF_CPU |
|-------|-----------|-----------|---------|---------|
| Onsets and Frames† | ≈27.0 | `[??]` | `[??]` | `[??]` |
| Basic Pitch† | ≈4.5 | `[??]` | `[??]` | `[??]` |
| MT3† | ≈26.0 | `[??]` | `[??]` | `[??]` |
| YourMT3+† | `[??]` | `[??]` | `[??]` | `[??]` |

> Wartości zewnętrzne oznaczone † zaczerpnięte z publikacji; te zmierzone własnoręcznie przy identycznej konfiguracji sprzętowej — oznaczone ‡.

---

## 4. Skalowalność w funkcji długości nagrania (Tab. 6.3)

Pomiar dla obu modeli własnych przy długościach: **5 s, 10 s, 20 s, 60 s, 120 s**.

### CNN+RNN własny

| Czas audio | Czas inf. [s] | VRAM [MB] | RTF |
|-----------|--------------|-----------|-----|
| 5 s | `[??]` | `[??]` | `[??]` |
| 10 s | `[??]` | `[??]` | `[??]` |
| 20 s | `[??]` | `[??]` | `[??]` |
| 60 s | `[??]` | `[??]` | `[??]` |
| 120 s | `[??]` | `[??]` | `[??]` |

### MT3-like własny M0

| Czas audio | Czas inf. [s] | VRAM [MB] | RTF |
|-----------|--------------|-----------|-----|
| 5 s | `[??]` | `[??]` | `[??]` |
| 10 s | `[??]` | `[??]` | `[??]` |
| 20 s | `[??]` | `[??]` | `[??]` |
| 60 s | `[??]` | `[??]` | `[??]` |
| 120 s | `[??]` | `[??]` | `[??]` |

> Oczekiwana złożoność: CNN+RNN → O(T) liniowa; MT3-like → O(S²) kwadratowa względem liczby tokenów S.

---

## 5. Ablacja — wpływ reprezentacji wejściowej CNN+RNN (Tab. 6.4, PB3)

Ocena na MAESTRO v3.0.0. Architektura modelu i wszystkie hiperparametry poza modułem ekstrakcji cech pozostają stałe.

| Reprezentacja | Params [M] | Onset F1 [%] | Note F1 [%] | Frame F1 [%] | VRAM [MB] | RTF_GPU |
|--------------|-----------|-------------|-------------|--------------|-----------|---------|
| Mel (bazowa, 229 pasm, 30–8000 Hz) | ≈15.80 | `[??]` | `[??]` | `[??]` | `[??]` | `[??]` |
| CQT (252 biny, 36 bin/okt, 27.5–3520 Hz) | ≈16.24 | `[??]` | `[??]` | `[??]` | `[??]` | `[??]` |
| HCQT (6 × 252 biny, 27.5–3520 Hz) | ≈16.24 | `[??]` | `[??]` | `[??]` | `[??]` | `[??]` |

---

## 6. Ablacja — skalowalność głębokości BiLSTM w CNN+RNN (sekcja 5.5.3)

| Wariant | Onset F1 [%] | Frame F1 [%] | Params [M] | RTF |
|---------|-------------|--------------|-----------|-----|
| 1 warstwa BiLSTM | `[??]` | `[??]` | `[??]` | `[??]` |
| 2 warstwy BiLSTM (bazowa) | `[??]` | `[??]` | ≈15.80 | `[??]` |
| 3 warstwy BiLSTM | `[??]` | `[??]` | `[??]` | `[??]` |

---

## 7. Ablacja — rozszerzenia MT3-like N1–N5 vs M0 (Tab. 6.5, PB4)

Trening na Slakh2100, 300 000 kroków. Ocena na MAESTRO (piano) i Slakh (uśrednione po instrumentach).

| ID | Rozszerzenie | MAESTRO Note F1 [%] | Δ vs M0 [pp] | Slakh Note F1 [%] | RTF_GPU |
|----|-------------|--------------------|--------------|--------------------|---------|
| M0 | Brak (bazowy) | `[??]` | — | `[??]` | `[??]` |
| N1 | F1 — hierarchiczna tokenizacja czasowa (Δparams=0) | `[??]` | `[??]` | `[??]` | `[??]` |
| N2 | F2 — pitch-aware cross-attention (+528K params) | `[??]` | `[??]` | `[??]` | `[??]` |
| N3 | F3 — dwuwymiarowe 2D patches (Δparams≈0) | `[??]` | `[??]` | `[??]` | `[??]` |
| N4 | F4 — rotacyjne osadzenia pozycyjne RoPE (Δparams=0) | `[??]` | `[??]` | `[??]` | `[??]` |
| N5 | F5 — konwolucyjny frontend enkodera (+1.57M params) | `[??]` | `[??]` | `[??]` | `[??]` |
| N8 | F1+F2+F3+F4+F5 (wszystkie, +≈2.1M params) | `[??]` | `[??]` | `[??]` | `[??]` |

> Konfiguracja M0 trenowana na MAESTRO piano-only; N1–N8 trenowane na Slakh2100.  
> Sprawdź addytywność: `Δ_N8` vs `Σ(Δ_N1..N5)` — efekt addytywny / synergistyczny / subaddytywny.

---

## 8. Dodatkowe obserwacje do zebrania w opisie wyników

Poniższe wartości są potrzebne do uzupełnienia tekstu narracyjnego w rozdziale 6:

- **CNN+RNN inferencja**: która warstwa dominuje czasowo (FC enkodera / BiLSTM / inna)?
- **MT3-like zależność od polifonii**: średnia liczba tokenów na sekundę dla nagrań rzadkich vs gęstych
- **Kompromis skuteczność–wydajność**: Note F1 na jednostkę RTF dla CNN+RNN i MT3-like
- **Basic Pitch na CPU**: RTF_CPU i Note F1 względem CNN+RNN (krotność przyspieszenia i różnica pp)
- **N8 addytywność**: suma Δ z N1–N5 vs rzeczywisty Δ_N8

---

## Podsumowanie — liczba brakujących wartości

| Sekcja | Liczba pól `[??]` |
|--------|------------------|
| Środowisko sprzętowe/softwarowe | 8 |
| Skuteczność — modele własne | 6 |
| Skuteczność — modele referencyjne | 12 |
| Wydajność — modele własne | 6 |
| Wydajność — modele zewnętrzne | 12 |
| Skalowalność CNN+RNN | 15 |
| Skalowalność MT3-like | 15 |
| Ablacja reprezentacji wejściowej | 12 |
| Ablacja BiLSTM depth | 8 |
| Ablacja rozszerzeń MT3-like N1–N8 | 28 |
| **Łącznie** | **~122** |
