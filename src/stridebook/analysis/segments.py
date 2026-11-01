"""Segments: the parts of a run with a role (warmup / active / rest / cooldown / steady).

Coros files carry no lap labels, so roles are proposed from lap speeds and then confirmed or
corrected by the user. A segment is one lap for now; consecutive segments with the same role
form a block (one rep, one tempo block), so a rep split by an autolap still counts once.

Long runs have roles of their own: a burn-in (warmup), pace blocks (active) and steady running
around them. The runner laps at each block's start and end, so a block is whole laps.
"""

import numpy as np
import pandas as pd

from stridebook.analysis.common import mean_over
from stridebook.domain import SegmentRole

SEGMENT_COLUMNS = ["segment_no", "lap_index", "start", "end", "duration_s", "distance_m", "role"]

STEADY_CV = 0.08  # lap-speed variation below this is one steady effort
FAST_FRACTION = 0.88  # a lap at least this fraction of the fastest laps' speed is active

MAX_BLOCKS = 3  # pace blocks a long run is checked for
BLOCK_CONTRAST = 0.05  # a suggested pace block is at least 5 % faster than the steady either side
MIN_BLOCK_S = 480  # and at least 8 min long, so one downhill km isn't taken for a block


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


def long_run_segments(segments: pd.DataFrame, n_blocks: int, burn_in: int) -> pd.DataFrame:
    """Long-run roles: the first `burn_in` laps warmup, `n_blocks` pace blocks active, and the
    rest steady.

    Laps are grouped by pace. The blocks are the runs of consecutive laps that, together with
    the steady running between them, are best described by one pace per part (least squares
    on lap pace, weighted by lap time). Steady parts may differ in pace, since running after a
    block is often slower than before it.
    """
    out = segments.copy()
    burn_in = int(np.clip(burn_in, 0, len(out)))
    roles = np.array(
        [SegmentRole.WARMUP.value] * burn_in + [SegmentRole.STEADY.value] * (len(out) - burn_in),
        dtype=object,
    )
    pace, weight = _lap_pace(out.iloc[burn_in:])
    cuts = _partition(pace, weight, n_blocks)
    for j in range(1, len(cuts) - 1, 2):
        roles[burn_in + cuts[j] : burn_in + cuts[j + 1]] = SegmentRole.ACTIVE.value
    out["role"] = roles
    return out


def suggest_blocks(segments: pd.DataFrame, burn_in: int) -> int:
    """How many pace blocks a long run seems to have: the most, up to MAX_BLOCKS, for which
    every block is at least MIN_BLOCK_S long and BLOCK_CONTRAST faster than the steady running
    either side of it."""
    pace, weight = _lap_pace(segments.iloc[burn_in:])
    for k in range(MAX_BLOCKS, 0, -1):
        cuts = _partition(pace, weight, k)
        if len(cuts) == 2 * k + 2 and all(
            _is_block(pace, weight, cuts, j) for j in range(1, 2 * k, 2)
        ):
            return k
    return 0


def burn_in_laps(segments: pd.DataFrame, km: float) -> int:
    """The number of leading laps whose end is nearest `km` (at least one lap is left)."""
    ends = np.concatenate([[0.0], segments["distance_m"].cumsum().to_numpy() / 1000])
    return int(min(np.abs(ends - km).argmin(), max(len(segments) - 1, 0)))


def burn_in_count(segments: pd.DataFrame) -> int:
    """The number of leading warmup laps."""
    return int(np.cumprod(segments["role"].to_numpy() == SegmentRole.WARMUP.value).sum())


def _lap_pace(segments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pace (s/m) and weight (lap time) of each lap. A lap without distance weighs nothing."""
    pace = segments["duration_s"] / segments["distance_m"].where(segments["distance_m"] > 0)
    pace = pace.to_numpy(dtype=float, copy=True)
    weight = segments["duration_s"].to_numpy(dtype=float, copy=True)
    bad = ~np.isfinite(pace)
    pace[bad], weight[bad] = 0.0, 0.0
    return pace, weight


def _partition(pace: np.ndarray, weight: np.ndarray, k: int) -> list[int]:
    """Cut the laps into steady, block, steady, ..., block, steady (2k + 1 parts) so the
    weighted squared error around each part's mean pace is smallest. Returns the cuts: part j
    is laps cuts[j]:cuts[j + 1]. The first and last steady parts may be empty; blocks and the
    steady parts between them are not. Dynamic programming over the cut positions."""
    n = len(pace)
    k = min(k, (n + 1) // 2)  # k blocks need k laps, plus one between each pair
    if k <= 0:
        return [0, n]
    cw, cs, cq = (
        np.concatenate([[0.0], np.cumsum(x)]) for x in (weight, weight * pace, weight * pace**2)
    )

    def sse(a: int, b: int) -> float:
        w = cw[b] - cw[a]
        return 0.0 if w <= 0 else (cq[b] - cq[a]) - (cs[b] - cs[a]) ** 2 / w

    parts = 2 * k + 1
    best = np.full((parts, n + 1), np.inf)
    back = np.zeros((parts, n + 1), dtype=int)
    best[0] = [sse(0, i) for i in range(n + 1)]
    for j in range(1, parts):
        may_be_empty = j == parts - 1
        for i in range(n + 1):
            for a in range(i + 1 if may_be_empty else i):
                cost = best[j - 1, a] + sse(a, i)
                if cost < best[j, i]:
                    best[j, i], back[j, i] = cost, a
    cuts = [n]
    for j in range(parts - 1, 0, -1):
        cuts.append(int(back[j, cuts[-1]]))
    return [0, *cuts[::-1]]


def _is_block(pace: np.ndarray, weight: np.ndarray, cuts: list[int], j: int) -> bool:
    """Whether part j is long enough and clearly faster than the steady parts beside it."""

    def mean(a: int, b: int) -> float:
        w = weight[a:b].sum()
        return float((weight[a:b] * pace[a:b]).sum() / w) if w > 0 else np.nan

    block = mean(cuts[j], cuts[j + 1])
    beside = [mean(cuts[j - 1], cuts[j]), mean(cuts[j + 1], cuts[j + 2])]
    beside = [p for p in beside if np.isfinite(p)]
    return (
        weight[cuts[j] : cuts[j + 1]].sum() >= MIN_BLOCK_S
        and bool(beside)
        and all(block <= (1 - BLOCK_CONTRAST) * p for p in beside)
    )


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
