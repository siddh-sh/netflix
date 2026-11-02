import json

import pandas as pd

from src.data import clean, load_clean
from src.train import build_pipeline, train


def test_clean_handles_missing_and_misplaced_duration():
    raw = pd.DataFrame({
        "show_id": ["s1", "s2"],
        "type": ["Movie", "TV Show"],
        "title": ["A", "B"],
        "director": [None, "X"],
        "cast": [None, None],
        "country": ["India, United States", None],
        "date_added": [" January 1, 2020", None],
        "release_year": [2019, 2020],
        "rating": ["74 min", "TV-MA"],
        "duration": [None, "2 Seasons"],
        "listed_in": ["Dramas", "TV Comedies"],
        "description": ["d1", None],
    })
    df = clean(raw)
    assert df.loc[0, "duration"] == "74 min"
    assert df.loc[0, "rating"] == "Unknown"
    assert df.loc[0, "primary_country"] == "India"
    assert df.loc[0, "year_added"] == 2020
    assert df.loc[1, "duration_value"] == 2
    assert df["description"].notna().all()


def test_dataset_loads():
    df = load_clean()
    assert len(df) > 8000
    assert set(df["type"].unique()) == {"Movie", "TV Show"}


def test_pipeline_predicts():
    pipe = build_pipeline()
    pipe.fit(["a movie film story", "season episodes series show"] * 5, ["Movie", "TV Show"] * 5)
    assert pipe.predict(["episodes of a series"])[0] == "TV Show"


def test_train_writes_artifacts(tmp_path):
    metrics = train(model_dir=tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["accuracy"] == metrics["accuracy"]
    assert metrics["accuracy"] > 0.7
