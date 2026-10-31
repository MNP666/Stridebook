"""Text formatting for values shown to the user."""

import math
from datetime import datetime, timedelta

from stridebook.analysis.metrics import METRICS
from stridebook.analysis.structure import format_duration


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pace(seconds_per_km: float) -> str:
    """248.4 -> '4:08'."""
    return "–" if is_missing(seconds_per_km) else format_duration(seconds_per_km)


def duration(seconds: float) -> str:
    return "–" if is_missing(seconds) else format_duration(seconds)


def number(x: float, digits: int = 0) -> str:
    return "–" if is_missing(x) else f"{x:,.{digits}f}"


def signed(x: float, digits: int = 1) -> str:
    return "–" if is_missing(x) else f"{x:+.{digits}f}"


def metric_value(metric: str, value: float) -> str:
    """A metrics-table value in its natural form: paces as m:ss, durations as h:mm:ss."""
    unit = METRICS[metric].unit
    if unit == "s/km":
        return f"{pace(value)} /km"
    if unit == "s":
        return duration(value)
    digits = 0 if unit in ("bpm", "") else 1 if unit in ("%", "km", "min") else 2
    return f"{number(value, digits)} {unit}".strip()


def local_time(start: datetime, utc_offset: timedelta | None) -> datetime:
    return start + (utc_offset or timedelta(0))


def when(start: datetime, utc_offset: timedelta | None) -> str:
    """'Tue 11 Aug 2026, 07:39' in the run's local time."""
    return local_time(start, utc_offset).strftime("%a %-d %b %Y, %H:%M")
