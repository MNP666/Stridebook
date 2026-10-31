"""The toy source must look like Stridelab's output, so swapping it in changes nothing else."""

import numpy as np
import pandas as pd
import pytest

from stridebook.sources import toy
from tests.conftest import START

# Columns and dtypes Stridelab's records_df() gives for a Coros run with running dynamics
STRIDELAB_RECORD_DTYPES = {
    "distance": "float64",
    "heart_rate": "float64",
    "speed": "float64",
    "altitude": "float64",
    "cadence": "float64",
    "stance_time": "float64",
    "vertical_oscillation": "float64",
    "vertical_ratio": "float64",
    "step_length": "float64",
    "position_lat": "float64",
    "position_long": "float64",
    "elapsed_s": "float64",
    "lap_index": "Int64",
}
STRIDELAB_LAP_COLUMNS = {
    "message_index",
    "timestamp",
    "start_time",
    "total_timer_time",
    "total_elapsed_time",
    "total_distance",
    "avg_speed",
    "max_speed",
    "avg_heart_rate",
    "max_heart_rate",
    "avg_cadence",
    "avg_step_length",
}


@pytest.mark.parametrize("name", list(toy.WORKOUTS))
def test_records_match_stridelab_shape(name):
    p = toy.generate(name, start=START, seed=3)
    r = p.records
    assert isinstance(r.index, pd.DatetimeIndex)
    assert str(r.index.dtype) == "datetime64[us, UTC]"
    assert {c: str(r[c].dtype) for c in STRIDELAB_RECORD_DTYPES} == STRIDELAB_RECORD_DTYPES
    assert STRIDELAB_LAP_COLUMNS <= set(p.laps.columns)
    assert p.summary["start"] == START
    assert p.summary["records"] == len(r)


@pytest.mark.parametrize("name", list(toy.WORKOUTS))
def test_laps_tile_the_timeline(name):
    laps = toy.generate(name, start=START, seed=3).laps
    assert (laps["timestamp"].iloc[:-1].to_numpy() == laps["start_time"].iloc[1:].to_numpy()).all()
    assert laps["message_index"].tolist() == list(range(len(laps)))


def test_physics_relations_hold():
    """Stridelab's sample-file checks: step_length ≈ speed / (2 · cadence / 60) and
    vertical_ratio ≈ 100 · vertical_oscillation / step_length."""
    r = toy.generate("long_run", start=START, seed=3).records
    run = r[r["speed"] > 2]
    implied = run["speed"] / (2 * run["cadence"] / 60) * 1000
    assert np.median((implied - run["step_length"]).abs() / run["step_length"]) < 0.01
    ratio = 100 * run["vertical_oscillation"] / run["step_length"]
    assert np.median((ratio - run["vertical_ratio"]).abs()) < 0.05


def test_distance_reps_end_on_the_distance():
    laps = toy.generate("intervals_6x1000m", start=START, seed=3).laps
    reps = laps[laps["avg_speed"] > 3.5]
    assert len(reps) == 6
    assert reps["total_distance"].between(1000, 1010).all()


def test_same_bytes_give_the_same_run():
    a = toy.ToySource().parse(b"some file", "a.fit")
    b = toy.ToySource().parse(b"some file", "b.fit")
    pd.testing.assert_frame_equal(
        a.laps.drop(columns=["timestamp", "start_time"]),
        b.laps.drop(columns=["timestamp", "start_time"]),
    )


def test_demo_history_has_every_category():
    runs = toy.demo_history(weeks=8)
    assert {r.workout.category for r in runs} == set(toy.RunCategory)
