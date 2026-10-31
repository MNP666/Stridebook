"""Tempo runs: how HR and pace develop over each threshold block."""

from dataclasses import dataclass

import pandas as pd

from stridebook.analysis.common import decoupling_pct, mean_over, slope_per_min, smooth
from stridebook.analysis.metrics import metric_frame
from stridebook.analysis.segments import blocks
from stridebook.config import Settings
from stridebook.domain import SegmentRole

SETTLE_S = 180  # HR is still rising towards steady state in a block's first minutes
TRACE_STEP_S = 5


@dataclass
class TempoResult:
    blocks: pd.DataFrame  # block_no, start, end, duration_s, distance_m, pace_s_per_km, avg_hr,
    #                       hr_drift_bpm_per_min, pace_drift_s_per_km_per_min, decoupling_pct
    traces: pd.DataFrame  # block_no, t_min, heart_rate, pace_s_per_km (30 s smoothing)


def compute(records: pd.DataFrame, segments: pd.DataFrame, settings: Settings) -> TempoResult:
    b = blocks(segments, SegmentRole.ACTIVE).drop(columns="segment_nos")
    hr_s = smooth(records["heart_rate"], "30s")
    pace_s = 1000 / smooth(records["speed"], "30s").where(records["speed"] >= 0.5)
    hr_drift, pace_drift, decoupling, traces = [], [], [], []
    for no, s, e in zip(b["block_no"], b["start"], b["end"], strict=True):
        block = records[(records.index >= s) & (records.index < e)]
        settled = block[block.index >= s + pd.Timedelta(seconds=SETTLE_S)]
        hr_drift.append(slope_per_min(settled["heart_rate"]))
        pace_drift.append(slope_per_min(1000 / settled["speed"].where(settled["speed"] >= 0.5)))
        decoupling.append(decoupling_pct(settled.assign(elapsed_s=0.0), skip_s=0))
        traces.append(
            pd.DataFrame(
                {
                    "block_no": no,
                    "t_min": (block.index - s).total_seconds() / 60,
                    "heart_rate": hr_s[block.index].to_numpy(),
                    "pace_s_per_km": pace_s[block.index].to_numpy(),
                }
            ).iloc[::TRACE_STEP_S]
        )
    b["pace_s_per_km"] = b["duration_s"] / b["distance_m"] * 1000
    b["avg_hr"] = [
        mean_over(records, s, e, "heart_rate") for s, e in zip(b["start"], b["end"], strict=True)
    ]
    b["hr_drift_bpm_per_min"] = hr_drift
    b["pace_drift_s_per_km_per_min"] = pace_drift
    b["decoupling_pct"] = decoupling
    cols = ["block_no", "t_min", "heart_rate", "pace_s_per_km"]
    return TempoResult(
        blocks=b,
        traces=pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(columns=cols),
    )


def metrics(r: TempoResult) -> pd.DataFrame:
    b = r.blocks
    # Mean pace over all tempo running, weighting blocks by their time
    mean_pace = b["duration_s"].sum() / b["distance_m"].sum() * 1000 if len(b) else float("nan")
    return metric_frame(
        {
            "tempo_min": b["duration_s"].sum() / 60,
            "mean_block_pace_s_per_km": mean_pace,
            "mean_block_hr": b["avg_hr"].mean(),
            "mean_hr_drift_bpm_per_min": b["hr_drift_bpm_per_min"].mean(),
        },
        {
            "block_pace_s_per_km": b["pace_s_per_km"].tolist(),
            "block_hr_drift_bpm_per_min": b["hr_drift_bpm_per_min"].tolist(),
        },
    )
