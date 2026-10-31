"""Long runs: running mechanics and cardiac drift over distance."""

from dataclasses import dataclass

import pandas as pd

from stridebook.analysis.common import bin_by_distance, decoupling_pct
from stridebook.analysis.metrics import metric_frame
from stridebook.config import Settings

# Columns other vendors may lack; missing ones are simply left out
MECHANICS = ["cadence_spm", "step_length", "vertical_oscillation", "vertical_ratio", "stance_time"]
CHANGE_METRICS = {
    "cadence_spm": "cadence_change_pct",
    "step_length": "step_length_change_pct",
    "vertical_oscillation": "vertical_oscillation_change_pct",
    "stance_time": "stance_time_change_pct",
}
MIN_BIN_M = 500  # a shorter last bin is too noisy to compare


@dataclass
class LongRunResult:
    bins: pd.DataFrame  # per km: bin_no, start_km, pace_s_per_km, heart_rate, ef, mechanics
    decoupling_pct: float
    changes_pct: dict[str, float]  # column -> % change from the first to the last third


def compute(
    records: pd.DataFrame, segments: pd.DataFrame, settings: Settings, bin_m: float = 1000
) -> LongRunResult:
    bins = bin_by_distance(records, ["heart_rate", "speed", *MECHANICS], bin_m=bin_m)
    bins["ef"] = bins["speed"] * 60 / bins["heart_rate"]
    full = bins[bins["distance_m"] >= MIN_BIN_M]
    third = max(1, len(full) // 3)
    first, last = full.iloc[:third], full.iloc[-third:]
    changes = {
        c: float((last[c].mean() - first[c].mean()) / first[c].mean() * 100)
        for c in CHANGE_METRICS
        if c in full and first[c].mean()
    }
    return LongRunResult(bins=bins, decoupling_pct=decoupling_pct(records), changes_pct=changes)


def metrics(r: LongRunResult) -> pd.DataFrame:
    return metric_frame(
        {"decoupling_pct": r.decoupling_pct}
        | {CHANGE_METRICS[c]: v for c, v in r.changes_pct.items()}
    )
