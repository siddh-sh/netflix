"""Data loading and cleaning for the Netflix titles dataset."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "netflix_titles.csv"


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["director", "cast", "country", "rating", "listed_in", "description"]:
        df[col] = df[col].fillna("Unknown")
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    # Some rows have the duration misplaced in the rating column
    misplaced = df["rating"].str.contains("min", na=False)
    df.loc[misplaced, "duration"] = df.loc[misplaced, "rating"]
    df.loc[misplaced, "rating"] = "Unknown"

    df["duration_value"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df["primary_country"] = df["country"].str.split(",").str[0].str.strip()
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()
    return df.drop_duplicates(subset="show_id")


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
