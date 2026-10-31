"""Segments: the parts of a run with a role (warmup / active / rest / cooldown).

Coros files carry no lap labels, so roles are proposed from lap speeds and then confirmed or
corrected by the user. A segment is one lap for now; consecutive segments with the same role
form a block (one rep, one tempo block), so a rep split by an autolap still counts once.
"""

import pandas as pd

from stridebook.analysis.common import mean_over
from stridebook.domain import SegmentRole

SEGMENT_COLUMNS = ["segment_no", "lap_index", "start", "end", "duration_s", "distance_m", "role"]

STEADY_CV = 0.08  # lap-speed variation below this is one steady effort
FAST_FRACTION = 0.88  # a lap at least this fraction of the fastest laps' speed is active


def propose_segments(laps: pd.DataFrame) -> pd.DataFrame:
    """One segment per lap, with a suggested role.

    Steady runs (lap speeds within STEADY_CV) are all active. Otherwise laps at least
    FAST_FRACTION of the upper-quartile lap speed are active; slower laps before the first
    active lap are warmup, after the last one cooldown, and in between rest.
    """
    if laps.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    seg = pd.DataFrame(
        {
            "segment_no": range(1, len(laps) + 1),
            "lap_index": laps["message_index"].to_numpy(),
            "start": laps["start_time"].to_numpy(),
            "end": laps["timestamp"].to_numpy(),
            "duration_s": laps["total_timer_time"].to_numpy(),
            "distance_m": laps["total_distance"].to_numpy(),
        }
    )
    speed = laps["avg_speed"].reset_index(drop=True)
    # Ignore short laps (the partial last autolap) when judging steadiness
    full = speed[seg["distance_m"] >= 200]
    if len(full) < 3 or full.std() / full.mean() < STEADY_CV:
        seg["role"] = SegmentRole.ACTIVE.value
        return seg
    fast = speed >= FAST_FRACTION * speed.quantile(0.75)
    first, last = fast.idxmax(), fast[::-1].idxmax()
    roles = []
    for i, is_fast in fast.items():
        if is_fast:
            roles.append(SegmentRole.ACTIVE)
        elif i < first:
            roles.append(SegmentRole.WARMUP)
        elif i > last:
            roles.append(SegmentRole.COOLDOWN)
        else:
            roles.append(SegmentRole.REST)
    seg["role"] = [r.value for r in roles]
    return seg


def blocks(segments: pd.DataFrame, role: SegmentRole | str) -> pd.DataFrame:
    """Runs of consecutive segments with `role`, merged: block_no, start, end, duration_s,
    distance_m, segment_nos."""
    role = SegmentRole(role)
    run_id = (segments["role"] != segments["role"].shift()).cumsum()
    picked = segments[segments["role"] == role.value]
    out = picked.groupby(run_id[picked.index]).agg(
        start=("start", "min"),
        end=("end", "max"),
        duration_s=("duration_s", "sum"),
        distance_m=("distance_m", "sum"),
        segment_nos=("segment_no", list),
    )
    out.insert(0, "block_no", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def segment_of(records: pd.DataFrame, segments: pd.DataFrame) -> pd.Series:
    """segment_no of every record (start <= t < end; the last segment includes its end)."""
    out = pd.Series(pd.NA, index=records.index, dtype="Int64")
    if segments.empty:
        return out
    starts = pd.DatetimeIndex(segments["start"])
    ends = pd.DatetimeIndex(segments["end"])
    i = starts.searchsorted(records.index, side="right") - 1
    inside = (i >= 0) & (records.index <= ends[i.clip(0)])
    out[inside] = segments["segment_no"].to_numpy()[i[inside]]
    return out


def segment_summary(records: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Segments plus avg pace and HR, for showing them to the user."""
    out = segments.copy()
    out["pace_s_per_km"] = out["duration_s"] / out["distance_m"].where(out["distance_m"] > 0) * 1000
    out["avg_hr"] = [
        mean_over(records, s, e, "heart_rate")
        for s, e in zip(out["start"], out["end"], strict=True)
    ]
    return out
