import numpy as np
import pandas as pd
import pytest

from stridebook.analysis import hr_fit
from stridebook.analysis.metrics import METRIC_COLUMNS, METRICS
from stridebook.analysis.segments import blocks, propose_segments
from stridebook.analysis.structure import derive_structure, format_duration, race_distance_label
from stridebook.domain import RunCategory, SegmentRole
from stridebook.sources import toy
from tests.conftest import START, make


@pytest.mark.parametrize(
    ("name", "structure", "category"),
    [
        ("intervals_5x6min", "5x6:00 r1:30", RunCategory.INTERVAL),
        ("intervals_6x1000m", "6x1000m r2:30", RunCategory.INTERVAL),
        ("tempo_2x15min", "2x15:00 r3:00", RunCategory.TEMPO),
        ("tempo_25min", "1x25:00", RunCategory.TEMPO),
        ("easy", None, RunCategory.EASY),
        ("long_run", None, RunCategory.LONG_RUN),
    ],
)
def test_new_run_suggests_structure_and_category(name, structure, category, settings):
    p = make(name, settings)
    assert p.run.meta.structure == structure
    assert p.run.meta.category == category


def test_proposed_roles_match_the_workout_steps():
    w = toy.workout("intervals_5x6min")
    laps = toy.simulate(w, start=START, seed=4).laps
    assert propose_segments(laps)["role"].tolist() == [s.role.value for s in w.steps]


def test_consecutive_active_segments_form_one_block():
    seg = propose_segments(toy.generate("tempo_25min", start=START).laps)
    seg.loc[1, "role"] = SegmentRole.ACTIVE.value  # e.g. a rep split by an autolap
    seg.loc[0, "role"] = SegmentRole.ACTIVE.value
    assert len(blocks(seg, SegmentRole.ACTIVE)) == 1


def test_race_distance_labels():
    assert race_distance_label(10_050) == "10k"
    assert race_distance_label(21_300) == "half"
    assert race_distance_label(15_000) == "15.0 km"
    assert derive_structure(pd.DataFrame(), RunCategory.RACE, 9_990) == "10k"


def test_format_duration():
    assert format_duration(90) == "1:30"
    assert format_duration(3725) == "1:02:05"


def test_interval_reps_and_recovery(settings):
    r = make("intervals_5x6min", settings).results["intervals"]
    assert len(r.reps) == 5
    assert len(r.recoveries) == 4
    assert r.recoveries["rest_no"].tolist() == [1, 2, 3, 4]
    assert (r.recoveries["hr_drop_60s"] > 20).all()
    assert r.reps["pace_s_per_km"].between(230, 260).all()
    # The slow component makes later reps harder
    assert r.reps["avg_hr"].iloc[-1] > r.reps["avg_hr"].iloc[1]


def test_tempo_blocks_drift_upwards(settings):
    r = make("tempo_2x15min", settings).results["tempo"]
    assert len(r.blocks) == 2
    assert (r.blocks["hr_drift_bpm_per_min"] > 0).all()


@pytest.mark.parametrize(
    ("model", "truth", "expected"),
    [
        ("exp", lambda t: 120 + 50 * (1 - np.exp(-t / 30)), {"tau_s": 30, "plateau_bpm": 170}),
        (
            "exp_drift",
            lambda t: 120 + 50 * (1 - np.exp(-t / 30)) + 1.2 / 60 * t,
            {"tau_s": 30, "plateau_bpm": 170, "drift_bpm_per_min": 1.2},
        ),
        (
            "stretched",
            lambda t: 120 + 50 * (1 - np.exp(-((t / 40) ** 0.6))),
            {"tau_s": 40, "beta": 0.6},
        ),
    ],
)
def test_hr_fit_recovers_known_parameters(model, truth, expected):
    t = np.arange(0, 600, 2.0)
    hr = truth(t) + np.random.default_rng(0).normal(0, 1, len(t))
    p = hr_fit.fit(t, hr, model)
    for name, value in expected.items():
        assert p[name] == pytest.approx(value, rel=0.08), name
    assert p["rmse_bpm"] < 1.2


def test_hr_fit_needs_enough_samples():
    t = np.arange(10.0)
    assert all(np.isnan(v) for v in hr_fit.fit(t, 150 + t, "exp").values())


def test_hr_fits_per_rep_and_block(settings):
    fits = make("intervals_5x6min", settings).results["intervals"].hr_fits
    assert len(fits.params) == 5 * len(hr_fit.MODELS)
    assert fits.params["tau_s"].between(20, 60).all()  # the toy HR rises with τ = 35 s
    # A 25 min block keeps drifting, so drift beats a plain plateau
    tempo = make("tempo_25min", settings).results["tempo"].hr_fits.params.set_index("model")
    assert tempo.loc["exp_drift", "bic"] < tempo.loc["exp", "bic"] - 10


def test_hr_fit_metrics_per_rep(settings):
    m = make("intervals_5x6min", settings).metrics
    tau = m[m["metric"] == hr_fit.metric_name("rep", "exp_drift", "tau_s")]
    assert tau["part_no"].tolist() == [1, 2, 3, 4, 5]
    assert tau["value"].between(20, 60).all()


def test_metric_labels_are_unique_but_for_the_model():
    # The trends page lists metrics by label, and fitted values by label plus model
    keys = [(i.label, i.model) for i in METRICS.values()]
    assert len(keys) == len(set(keys))


def test_long_run_bins_and_fatigue(settings):
    r = make("long_run", settings, distance_km=28).results["long_run"]
    assert len(r.bins) == 28
    assert r.changes_pct["step_length"] < 0
    assert r.changes_pct["vertical_oscillation"] > 0


def test_race_splits(settings):
    p = make("race_10k", settings)
    r = p.results["race"]
    assert len(r.splits) == 10
    assert r.finish_time_s == pytest.approx(r.splits["duration_s"].sum())


def test_every_metric_is_registered(demo):
    for p in demo:
        assert list(p.metrics.columns) == METRIC_COLUMNS
        assert set(p.metrics["metric"]) <= set(METRICS)
        per_part = p.metrics["metric"].map(lambda m: METRICS[m].per_part)
        assert (p.metrics["part_no"].notna() == per_part).all()
