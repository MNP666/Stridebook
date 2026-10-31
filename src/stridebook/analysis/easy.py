"""Easy runs: aerobic efficiency, the trend that matters most for easy running."""

from dataclasses import dataclass

import pandas as pd

from stridebook.analysis.common import decoupling_pct, efficiency, running_mask
from stridebook.analysis.metrics import metric_frame
from stridebook.config import Settings

SETTLE_S = 600  # HR is still settling in the first minutes


@dataclass
class EasyResult:
    ef: float
    hr_at_ref_pace: float  # HR the efficiency factor implies at the settings' reference pace
    decoupling_pct: float
    ef_trace: pd.DataFrame  # elapsed_min, ef: 5-min rolling efficiency factor
    hr_vs_pace: pd.DataFrame  # one row per 30 s: pace_s_per_km, heart_rate


def compute(records: pd.DataFrame, segments: pd.DataFrame, settings: Settings) -> EasyResult:
    run = records[running_mask(records)]
    steady = run[run["elapsed_s"] >= SETTLE_S]
    ef = efficiency(steady["speed"], steady["heart_rate"])
    ref_speed = 1000 / settings.reference_pace_s_per_km
    rolling = run[["speed", "heart_rate"]].rolling("5min", min_periods=60).mean()
    ef_trace = pd.DataFrame(
        {"elapsed_min": run["elapsed_s"] / 60, "ef": rolling["speed"] * 60 / rolling["heart_rate"]}
    ).iloc[::15]
    binned = steady[["speed", "heart_rate"]].resample("30s").mean().dropna()
    hr_vs_pace = pd.DataFrame(
        {"pace_s_per_km": 1000 / binned["speed"], "heart_rate": binned["heart_rate"]}
    )
    return EasyResult(
        ef=ef,
        hr_at_ref_pace=ref_speed * 60 / ef if ef else float("nan"),
        decoupling_pct=decoupling_pct(records, skip_s=SETTLE_S),
        ef_trace=ef_trace.dropna().reset_index(drop=True),
        hr_vs_pace=hr_vs_pace.reset_index(drop=True),
    )


def metrics(r: EasyResult) -> pd.DataFrame:
    return metric_frame({"hr_at_ref_pace": r.hr_at_ref_pace, "decoupling_pct": r.decoupling_pct})
