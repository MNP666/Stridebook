"""A short label for a workout's shape, derived from its confirmed segments.

"5x6:00 r1:30", "6x1000m r2:30", "2x15:00 r3:00", "1x25:00", a race distance ("10k"), or a
long run's pace blocks ("2x5000m MP"). It lets the trends page compare like with like. Reps
are labelled by time or distance, whichever is rounder: a 6:00 rep is time-based, a 1000 m
rep distance-based.
"""

import pandas as pd

from stridebook.analysis.segments import blocks
from stridebook.domain import BlockPace, RunCategory, SegmentRole

RACE_DISTANCES = {"5k": 5000, "10k": 10_000, "half": 21_097.5, "marathon": 42_195}
RACE_TOLERANCE = 0.03


def derive_structure(
    segments: pd.DataFrame,
    category: RunCategory | None = None,
    distance_m: float | None = None,
    block_pace: BlockPace | None = None,
) -> str | None:
    if category == RunCategory.RACE:
        return race_distance_label(distance_m) if distance_m else None
    if segments.empty or (segments["role"] == SegmentRole.ACTIVE.value).all():
        return None  # one steady effort has no structure
    reps = blocks(segments, SegmentRole.ACTIVE)
    if reps.empty:
        return None  # a long run without pace blocks
    labels = _labels(reps)
    text = f"{len(labels)}x{labels[0]}" if len(set(labels)) == 1 else "-".join(labels)
    rests = blocks(segments, SegmentRole.REST)
    if len(rests):
        text += f" r{pd.Series(_labels(rests)).mode().iloc[0]}"
    if category == RunCategory.LONG_RUN and block_pace:
        text += f" {BlockPace(block_pace).short}"
    return text


def race_distance_label(distance_m: float) -> str:
    for label, d in RACE_DISTANCES.items():
        if abs(distance_m - d) / d <= RACE_TOLERANCE:
            return label
    return f"{distance_m / 1000:.1f} km"


def _labels(b: pd.DataFrame) -> list[str]:
    t, d = b["duration_s"], b["distance_m"]
    t_err = ((t - t.map(_nice_time)).abs() / t).mean()
    d_err = ((d - d.map(_nice_distance)).abs() / d.where(d > 0)).mean()
    if pd.notna(d_err) and d_err < t_err:
        return [f"{_nice_distance(x):.0f}m" for x in d]
    return [format_duration(_nice_time(x)) for x in t]


def _nice_time(s: float) -> float:
    step = 5 if s < 120 else 15
    return round(s / step) * step


def _nice_distance(m: float) -> float:
    step = 50 if m < 1000 else 100
    return round(m / step) * step


def format_duration(seconds: float) -> str:
    """90 -> '1:30', 3725 -> '1:02:05'."""
    seconds = round(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
