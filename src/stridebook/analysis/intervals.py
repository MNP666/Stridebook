"""Interval sessions: pace and HR in each rep, HR recovery in each rest."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stridebook.analysis.common import mean_over, smooth, value_at
from stridebook.analysis.metrics import metric_frame
from stridebook.analysis.segments import blocks
from stridebook.config import Settings
from stridebook.domain import SegmentRole

TRACE_STEP_S = 2  # keep every 2nd second in the overlay traces
PEAK_WINDOW_S = 10  # HR keeps rising briefly after a rep; recovery counts from that peak


@dataclass
class IntervalResult:
    reps: pd.DataFrame  # rep_no, start, end, duration_s, distance_m, pace_s_per_km, avg_hr,
    #                     max_hr, end_hr, hr_drop_60s (of the rest that follows)
    recoveries: pd.DataFrame  # rest_no (= the rep before it), start, duration_s, peak_hr,
    #                           hr_drop_30s, hr_drop_60s, min_hr
    rep_traces: pd.DataFrame  # rep_no, t_s, heart_rate, pace_s_per_km
    rest_traces: pd.DataFrame  # rest_no, t_s, heart_rate


def compute(records: pd.DataFrame, segments: pd.DataFrame, settings: Settings) -> IntervalResult:
    reps = blocks(segments, SegmentRole.ACTIVE).rename(columns={"block_no": "rep_no"})
    rests = blocks(segments, SegmentRole.REST)
    hr = records["heart_rate"]

    reps["pace_s_per_km"] = reps["duration_s"] / reps["distance_m"] * 1000
    reps["avg_hr"] = [mean_over(records, s, e, "heart_rate") for s, e in _spans(reps)]
    reps["max_hr"] = [hr[(hr.index >= s) & (hr.index < e)].max() for s, e in _spans(reps)]
    reps["end_hr"] = [
        mean_over(records, e - pd.Timedelta(seconds=5), e, "heart_rate") for _, e in _spans(reps)
    ]

    rec = []
    for _, r in rests.iterrows():
        start, length = r["start"], r["duration_s"]
        peak = hr[(hr.index >= start) & (hr.index < start + pd.Timedelta(seconds=PEAK_WINDOW_S))]
        peak_hr = float(peak.max())
        drops = {
            f"hr_drop_{s}s": peak_hr
            - value_at(records, start + pd.Timedelta(seconds=s), "heart_rate")
            if length >= s + 5
            else np.nan
            for s in (30, 60)
        }
        rec.append(
            {
                "rest_no": int((reps["end"] <= start).sum()),
                "start": start,
                "duration_s": length,
                "peak_hr": peak_hr,
                **drops,
                "min_hr": float(hr[(hr.index >= start) & (hr.index < r["end"])].min()),
            }
        )
    recoveries = pd.DataFrame(
        rec,
        columns=[
            "rest_no",
            "start",
            "duration_s",
            "peak_hr",
            "hr_drop_30s",
            "hr_drop_60s",
            "min_hr",
        ],
    )
    reps = reps.merge(
        recoveries[["rest_no", "hr_drop_60s"]], left_on="rep_no", right_on="rest_no", how="left"
    ).drop(columns="rest_no")

    return IntervalResult(
        reps=reps.drop(columns="segment_nos"),
        recoveries=recoveries,
        rep_traces=_traces(records, reps, "rep_no", with_pace=True),
        rest_traces=_traces(records, recoveries.assign(end=rests["end"].to_numpy()), "rest_no"),
    )


def _spans(df: pd.DataFrame):
    return zip(df["start"], df["end"], strict=True)


def _traces(records, spans: pd.DataFrame, key: str, with_pace: bool = False) -> pd.DataFrame:
    """Time-aligned HR (and pace) per span, for overlaying reps or rests."""
    speed = smooth(records["speed"], "10s") if with_pace else None
    parts = []
    for no, s, e in zip(spans[key], spans["start"], spans["end"], strict=True):
        mask = (records.index >= s) & (records.index < e)
        part = pd.DataFrame(
            {
                key: no,
                "t_s": (records.index[mask] - s).total_seconds(),
                "heart_rate": records["heart_rate"][mask].to_numpy(),
            }
        )
        if with_pace:
            part["pace_s_per_km"] = (1000 / speed[mask]).where(speed[mask] >= 0.5).to_numpy()
        parts.append(part.iloc[::TRACE_STEP_S])
    cols = [key, "t_s", "heart_rate"] + (["pace_s_per_km"] if with_pace else [])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def metrics(r: IntervalResult) -> pd.DataFrame:
    reps, rec = r.reps, r.recoveries
    pace = reps["pace_s_per_km"]
    return metric_frame(
        {
            "rep_count": len(reps),
            "mean_rep_pace_s_per_km": pace.mean(),
            "rep_pace_cv_pct": pace.std() / pace.mean() * 100 if len(reps) > 1 else np.nan,
            "mean_rep_hr": reps["avg_hr"].mean(),
            "mean_hr_drop_60s": rec["hr_drop_60s"].mean(),
        },
        {
            "rep_pace_s_per_km": pace.tolist(),
            "rep_avg_hr": reps["avg_hr"].tolist(),
            "rep_end_hr": reps["end_hr"].tolist(),
            "rest_hr_drop_60s": rec["hr_drop_60s"].tolist(),
        },
    )
