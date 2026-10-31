"""Headline numbers for every run, whatever its category."""

from dataclasses import dataclass

import pandas as pd

from stridebook.analysis.common import dt_seconds, efficiency, running_mask, time_in_zones
from stridebook.analysis.metrics import metric_frame
from stridebook.config import Settings


@dataclass
class OverviewResult:
    distance_m: float
    duration_s: float  # timer time: pauses left out, standing rests kept
    avg_pace_s_per_km: float
    avg_hr: float
    max_hr: float
    ef: float  # over running records
    zones: pd.Series  # seconds per zone name


def compute(records: pd.DataFrame, segments: pd.DataFrame, settings: Settings) -> OverviewResult:
    duration = float(dt_seconds(records.index).sum())
    distance = float(records["distance"].iloc[-1] - records["distance"].iloc[0])
    run = records[running_mask(records)]
    return OverviewResult(
        distance_m=distance,
        duration_s=duration,
        avg_pace_s_per_km=duration / distance * 1000 if distance else float("nan"),
        avg_hr=float(records["heart_rate"].mean()),
        max_hr=float(records["heart_rate"].max()),
        ef=efficiency(run["speed"], run["heart_rate"]),
        zones=time_in_zones(records, settings.zones),
    )


def metrics(r: OverviewResult) -> pd.DataFrame:
    return metric_frame(
        {
            "distance_km": r.distance_m / 1000,
            "duration_s": r.duration_s,
            "avg_pace_s_per_km": r.avg_pace_s_per_km,
            "avg_hr": r.avg_hr,
            "max_hr": r.max_hr,
            "ef": r.ef,
        }
    )
