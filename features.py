"""Leakage-safe feature engineering.

Every per-pitcher historical feature is computed from that pitcher's PRIOR
games only (via shift before expanding/rolling), ordered by date. The current
game's own first pitch never contributes to its own features, which prevents
both target leakage and temporal leakage.
"""

import pandas as pd

from data_prep import SPEED_THRESHOLD, game_level_velocity

# Columns fed to the models.
FEATURE_COLS = [
    "hist_first_velo_mean",
    "hist_first_velo_std",
    "hist_over_rate",
    "hist_starts",
    "last_first_velo",
    "prev3_first_velo",
    "days_rest",
    "hist_game_velo_mean",
    "is_home_pitcher",
    "season",
    "month",
]


def add_features(starters: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe predictive features to the starter-game rows."""
    df = starters.sort_values(["pitcher_id", "date", "game_id"]).copy()
    by_pitcher = df.groupby("pitcher_id", sort=False)

    # Prior first-pitch velocity statistics (shift excludes the current game).
    shifted_velo = by_pitcher["release_speed"].shift(1)
    df["last_first_velo"] = shifted_velo
    df["hist_first_velo_mean"] = shifted_velo.groupby(df["pitcher_id"]).expanding().mean().reset_index(level=0, drop=True)
    df["hist_first_velo_std"] = shifted_velo.groupby(df["pitcher_id"]).expanding().std().reset_index(level=0, drop=True)
    df["prev3_first_velo"] = shifted_velo.groupby(df["pitcher_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Prior rate of clearing the threshold for this pitcher.
    over = (df["release_speed"] > SPEED_THRESHOLD).astype(float)
    shifted_over = over.groupby(df["pitcher_id"]).shift(1)
    df["hist_over_rate"] = shifted_over.groupby(df["pitcher_id"]).expanding().mean().reset_index(level=0, drop=True)
    df["hist_starts"] = shifted_over.groupby(df["pitcher_id"]).expanding().count().reset_index(level=0, drop=True)

    # Days of rest since the pitcher's previous start.
    prev_date = by_pitcher["date"].shift(1)
    df["days_rest"] = (df["date"] - prev_date).dt.days

    # Broader velocity signal: mean speed across all pitches in prior games.
    game_velo = game_level_velocity(raw)
    df = df.merge(game_velo, on=["game_id", "pitcher_id"], how="left")
    df = df.sort_values(["pitcher_id", "date", "game_id"])
    shifted_game_velo = df.groupby("pitcher_id")["game_velo_mean"].shift(1)
    df["hist_game_velo_mean"] = shifted_game_velo.groupby(df["pitcher_id"]).expanding().mean().reset_index(level=0, drop=True)

    return df.reset_index(drop=True)
