"""End-to-end pipeline: predict whether a starting pitcher's first pitch of the
game exceeds 89.95 mph.

Run with:  python pitch_projection.py
"""

import pandas as pd

from data_prep import build_first_pitch_dataset, load_raw
from features import add_features
from model import evaluate

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


def main() -> None:
    raw = load_raw()
    starters = build_first_pitch_dataset(raw)
    data = add_features(starters, raw)

    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Starter-game rows : {len(data)}")
    print(f"Unique pitchers   : {data['pitcher_id'].nunique()}")
    print(f"Seasons           : {sorted(data['season'].unique())}")
    print(f"Overall >89.95 rate: {data['target'].mean():.3f}")

    report = evaluate(data)

    print()
    print("=" * 60)
    print("SPLIT")
    print("=" * 60)
    print(f"Train (2021): {report['n_train']} rows, base rate {report['train_base_rate']:.3f}")
    print(f"Test  (2022): {report['n_test']} rows, base rate {report['test_base_rate']:.3f}")

    print()
    print("=" * 60)
    print("RESULTS (2022 test set)")
    print("=" * 60)
    metrics_df = pd.DataFrame(report["metrics"]).T
    metrics_df = metrics_df[["auc", "log_loss", "brier", "accuracy"]]
    print(metrics_df.round(4).to_string())

    print()
    print("Confusion matrix - HistGBM @0.5 (rows=actual, cols=pred):")
    print(report["confusion_gbm"])

    print()
    print("Calibration plot saved to calibration.png")


if __name__ == "__main__":
    main()
