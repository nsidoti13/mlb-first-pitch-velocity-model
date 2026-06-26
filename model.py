"""Train and evaluate models for the first-pitch >89.95 mph task.

Split is time-based (train on 2021, test on 2022) to mimic forecasting future
games and to avoid leaking a pitcher's later-season tendencies into training.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS

TRAIN_SEASON = 2021
TEST_SEASON = 2022


def time_split(df: pd.DataFrame):
    """Split into an earlier training season and a later test season."""
    train = df[df["season"] == TRAIN_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()
    return train, test


def _metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
    }


def evaluate(df: pd.DataFrame) -> dict:
    """Train baselines and models, returning a metrics report keyed by name."""
    train, test = time_split(df)
    y_train, y_test = train["target"].values, test["target"].values
    base_rate = y_train.mean()

    results = {}

    # Baseline 1: always predict the majority outcome (positive here).
    results["baseline_majority"] = _metrics(y_test, np.full(len(y_test), base_rate))

    # Baseline 2: predict each pitcher's prior over-rate (global fallback for cold start).
    over_rate = test["hist_over_rate"].fillna(base_rate).clip(1e-6, 1 - 1e-6).values
    results["baseline_pitcher_rate"] = _metrics(y_test, over_rate)

    # Model 1: logistic regression (impute + scale; add a NaN flag for cold start).
    logit = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    logit.fit(train[FEATURE_COLS], y_train)
    logit_proba = logit.predict_proba(test[FEATURE_COLS])[:, 1]
    results["logistic_regression"] = _metrics(y_test, logit_proba)

    # Model 2: gradient-boosted trees (handles NaNs natively).
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    gbm.fit(train[FEATURE_COLS], y_train)
    gbm_proba = gbm.predict_proba(test[FEATURE_COLS])[:, 1]
    results["hist_gbm"] = _metrics(y_test, gbm_proba)

    _plot_calibration(
        y_test,
        {"logistic": logit_proba, "hist_gbm": gbm_proba, "pitcher_rate": over_rate},
    )

    return {
        "n_train": len(train),
        "n_test": len(test),
        "train_base_rate": base_rate,
        "test_base_rate": y_test.mean(),
        "metrics": results,
        "confusion_gbm": confusion_matrix(y_test, (gbm_proba >= 0.5).astype(int)),
    }


def _plot_calibration(y_true, proba_by_model: dict, path: str = "calibration.png") -> None:
    from sklearn.calibration import calibration_curve

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for name, proba in proba_by_model.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title("Calibration (2022 test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
