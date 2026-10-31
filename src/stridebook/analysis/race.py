"""Races: splits, pacing evenness and HR."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stridebook.analysis.common import bin_by_distance, dt_seconds
from stridebook.analysis.metrics import metric_frame
from stridebook.config import Settings


@dataclass
class RaceResult:
    splits: pd.DataFrame  # per km: bin_no, start_km, distance_m, duration_s, pace_s_per_km,
    #                       heart_rate, cumulative_s
    distance_m: float
    finish_time_s: float
    avg_pace_s_per_km: float
    split_ratio: float  # time for the second half of the distance / the first; >1 = slowed
    pace_cv_pct: float  # variation of full-km split paces
    avg_hr: float


def compute(records: pd.DataFrame, segments: pd.DataFrame, settings: Settings) -> RaceResult:
    splits = bin_by_distance(records, ["heart_rate"], bin_m=1000)
    splits["cumulative_s"] = splits["duration_s"].cumsum()
    distance = float(records["distance"].iloc[-1] - records["distance"].iloc[0])
    finish = float(dt_seconds(records.index).sum())
    elapsed = dt_seconds(records.index).cumsum().shift(fill_value=0.0)
    half_time = float(
        np.interp(distance / 2, records["distance"] - records["distance"].iloc[0], elapsed)
    )
    full = splits.loc[splits["distance_m"] >= 900, "pace_s_per_km"]
    return RaceResult(
        splits=splits,
        distance_m=distance,
        finish_time_s=finish,
        avg_pace_s_per_km=finish / distance * 1000,
        split_ratio=(finish - half_time) / half_time,
        pace_cv_pct=float(full.std() / full.mean() * 100),
        avg_hr=float(records["heart_rate"].mean()),
    )


def metrics(r: RaceResult) -> pd.DataFrame:
    return metric_frame(
        {
            "finish_time_s": r.finish_time_s,
            "split_ratio": r.split_ratio,
            "pace_cv_pct": r.pace_cv_pct,
        }
    )
