# MLB First-Pitch Velocity Model

Predict whether the **first pitch a starting pitcher throws in a game** will
exceed **89.95 mph**, using historical pitch-tracking data
(`mlb_pitch_velo_assessment.csv`, 2021–2022 seasons).

## Problem framing

- **Prediction unit:** one row per (game, starting pitcher) — that pitcher's
  first pitch of the game. Starters are identified as pitchers whose first pitch
  occurs in the 1st inning (later first-appearances are relievers and are excluded).
- **Target:** binary, `release_speed > 89.95` on that first pitch.
- **Base rate:** ~79.5% of first pitches clear the threshold, so the problem is
  mildly imbalanced toward the positive class.

## Avoiding data leakage

This is the central modeling concern:

1. **No same-event features.** The first pitch's own speed/type is never used as input.
2. **Temporal safety.** Every per-pitcher historical feature is computed from that
   pitcher's *prior* games only (`shift(1)` before any expanding/rolling aggregate),
   ordered by date.
3. **Time-based split.** Train on 2021, test on 2022 — mimicking real forecasting
   and preventing a pitcher's later-season tendencies from leaking into training.

## Project structure

| File | Responsibility |
|------|----------------|
| `data_prep.py` | Load raw pitches, parse dates, drop null-velocity rows, build the (game × starting pitcher) first-pitch dataset and target. |
| `features.py` | Leakage-safe per-pitcher historical features (expanding velocity, prior over-rate, rest days, rolling means). |
| `model.py` | Time split, baselines, logistic regression + gradient-boosted trees, evaluation and calibration plot. |
| `pitch_projection.py` | Orchestrates the pipeline and prints the report. |
| `calibration.png` | Calibration curves on the 2022 test set (generated). |

## Setup & run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pitch_projection.py
```

## Features

All features are known *before* the pitch is thrown:

- `hist_first_velo_mean`, `hist_first_velo_std`, `prev3_first_velo`, `last_first_velo` — prior first-pitch velocity signals
- `hist_over_rate` — pitcher's prior rate of clearing 89.95
- `hist_starts` — number of prior starts (confidence)
- `hist_game_velo_mean` — mean velocity across all pitches in prior games
- `days_rest` — days since previous start
- `is_home_pitcher`, `season`, `month` — context

## Results (2022 test set)

| Model | AUC | Log loss | Brier | Accuracy |
|-------|-----|----------|-------|----------|
| Baseline – majority class | 0.500 | 0.490 | 0.155 | 0.809 |
| Baseline – pitcher historical rate | 0.921 | 0.268 | 0.055 | 0.931 |
| Logistic regression | 0.904 | 0.258 | 0.067 | 0.928 |
| Gradient-boosted trees | 0.904 | 0.255 | 0.063 | 0.928 |

## Key takeaways

- A pitcher's own velocity history is the dominant signal. With only 60 pitchers
  in the data, a simple "predict the pitcher's prior over-rate" baseline already
  reaches 0.92 AUC.
- Trained models match the baseline on log loss/calibration but do not
  meaningfully beat it on ranking — expected given the strength of the per-pitcher signal.
- No model scores a suspiciously high AUC (~0.99), which is consistent with a
  leakage-free setup.
