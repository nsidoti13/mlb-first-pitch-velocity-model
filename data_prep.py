"""Load the raw pitch data and build the modeling dataset.

The prediction unit is one row per (game, starting pitcher): the first pitch
that starting pitcher throws in that game. The target is whether that first
pitch exceeds the 89.95 mph threshold.
"""

import pandas as pd

CSV_PATH = "mlb_pitch_velo_assessment.csv"
SPEED_THRESHOLD = 89.95


def load_raw(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load raw pitches, parse dates, and drop rows with no recorded speed."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    # Rows without a release speed cannot be labeled or used for velo features.
    df = df.dropna(subset=["release_speed"]).reset_index(drop=True)
    return df


def build_first_pitch_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (game, starting pitcher) for their first pitch.

    A starter is identified by their first pitch of the game occurring in the
    1st inning; pitchers whose first appearance is in a later inning are
    relievers and are excluded.
    """
    ordered = df.sort_values(["game_id", "pitcher_id", "pitch_number"])
    first = ordered.groupby(["game_id", "pitcher_id"], as_index=False).first()

    # Keep starters only (first pitch thrown in the 1st inning).
    starters = first[first["pre_pitch_inning"] == 1].copy()

    # is_top_half == 1 means the home team is pitching, so this is the home starter.
    starters["is_home_pitcher"] = starters["is_top_half"].astype(int)
    starters["month"] = starters["date"].dt.month
    starters["target"] = (starters["release_speed"] > SPEED_THRESHOLD).astype(int)

    return starters.reset_index(drop=True)


def game_level_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Mean release speed per (game, pitcher) across all of that game's pitches.

    Used later (shifted to prior games only) as a broader velocity signal than
    the first pitch alone.
    """
    return (
        df.groupby(["game_id", "pitcher_id"], as_index=False)["release_speed"]
        .mean()
        .rename(columns={"release_speed": "game_velo_mean"})
    )
