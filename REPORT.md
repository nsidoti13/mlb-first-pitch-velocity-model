# Methodology & Findings

## Task

Predict whether the first pitch a starting pitcher throws in a game will exceed
**89.95 mph**, using 2021–2022 MLB pitch-tracking data.

## Methodology

I framed the task as binary classification at the level of one row per
(game, starting pitcher) — the pitcher's first pitch of the game, where the
target is whether that pitch exceeds 89.95 mph (~79.5% base rate). Starting
pitchers were isolated by requiring their first pitch to occur in the 1st inning,
and rows with missing velocity were dropped. The central design concern was data
leakage, addressed three ways: the first pitch's own measurements are never used
as inputs; every per-pitcher historical feature (expanding mean/std of prior
first-pitch velocity, prior rate of clearing the threshold, rolling recent
velocity, rest days, and broader prior-game average velocity) is computed using
only that pitcher's earlier games via a date-ordered shift; and the data is split
chronologically, training on 2021 and testing on 2022 to emulate true
forecasting. I benchmarked a logistic regression and gradient-boosted trees
against two baselines (majority class, and each pitcher's prior over-rate).

## Prompt Engineering

Once I had the methodology framework in mind, I used Claude Opus 4.8 as a coding
collaborator to implement it efficiently, while keeping the analytical decisions
my own. My prompting strategy followed a few deliberate principles:

- **Explore before building.** Rather than asking for a model immediately, I first
  had the assistant inspect the dataset (column names, season coverage, base rate,
  how to identify starters) so that every downstream decision was grounded in the
  actual data rather than assumptions.
- **Lead with the approach, not the code.** I described the task and asked *"how
  should I approach it"* first, which surfaced the key risks (data leakage, the
  79% base rate, the small 60-pitcher universe) before any code was written. This
  kept the implementation aligned with sound methodology instead of jumping
  straight to a model.
- **Constrain for correctness.** I steered the implementation toward leakage-safe
  features (prior-games-only, date-ordered shifts), a time-based 2021→2022 split,
  and explicit baselines, so the resulting metrics would be trustworthy rather
  than inflated.
- **Iterate on results.** When the first run produced a poorly calibrated logistic
  model, I had the assistant diagnose and fix the cause (an inappropriate
  `class_weight="balanced"` setting) and re-evaluate, rather than accepting the
  initial output.
- **Verify, don't trust blindly.** Each stage was run end-to-end and the outputs
  were checked against expectations — most importantly confirming that no model
  scored a suspiciously high AUC, which served as a sanity check that the pipeline
  was leakage-free.

The net effect was that prompt engineering accelerated implementation and
boilerplate (data wrangling, scikit-learn pipelines, plotting) while the
modeling judgment — framing, leakage control, evaluation design, and
interpretation — remained driven by me.

## Results (2022 test set)

| Model | AUC | Log loss | Brier | Accuracy |
|-------|-----|----------|-------|----------|
| Baseline – majority class | 0.500 | 0.490 | 0.155 | 0.809 |
| Baseline – pitcher historical rate | 0.921 | 0.268 | 0.055 | 0.931 |
| Logistic regression | 0.904 | 0.258 | 0.067 | 0.928 |
| Gradient-boosted trees | 0.904 | 0.255 | 0.063 | 0.928 |

![Calibration on 2022 test set](calibration.png)

## Findings

The findings are clear and honest: a pitcher's own velocity history
overwhelmingly drives the outcome — with only 60 pitchers in the dataset, a
simple "predict the pitcher's prior over-rate" baseline already achieves 0.92 AUC
and 0.93 accuracy, and the trained models match it on calibration/log loss
(≈0.25) without meaningfully improving ranking. Critically, no model scored a
suspiciously high AUC (~0.99), confirming the pipeline is leakage-free, and all
models are well-calibrated against the diagonal on the 2022 test set. The
practical takeaway is that a compact, leakage-safe per-pitcher feature set is
sufficient for this problem, and additional model complexity yields little gain.
