"""Shared building blocks: derived columns, smoothing, zones, binning, efficiency.

Everything under analysis/ follows Stridelab's rules so it can move there unchanged:
pandas/numpy only, DataFrame/Series in and out, NaN-preserving, time-based windows,
derived columns carry a unit suffix.
"""

import numpy as np
import pandas as pd

from stridebook.config import HrZone

RUNNING_SPEED = 1.8  # m/s; slower counts as walking or standing for mechanics


def add_derived(records: pd.DataFrame) -> pd.DataFrame:
    """A copy of records with cadence_spm, pace_s_per_km, distance_km and a running flag."""
    out = records.copy()
    if "cadence" in out:
        spm = 2 * out["cadence"]
        if "fractional_cadence" in out:
            spm = spm + 2 * out["fractional_cadence"].fillna(0)
        out["cadence_spm"] = spm
    if "speed" in out:
        out["pace_s_per_km"] = (1000 / out["speed"]).where(out["speed"] >= 0.5)
        out["running"] = out["speed"] >= RUNNING_SPEED
    if "distance" in out:
        out["distance_km"] = out["distance"] / 1000
    return out


def running_mask(records: pd.DataFrame) -> pd.Series:
    """True where the runner is running (all True when there is no speed channel)."""
    if "running" in records:
        return records["running"].astype(bool)
    return pd.Series(True, index=records.index)


def smooth(s: pd.Series, window: str = "15s") -> pd.Series:
    """Centred time-based moving average that keeps NaN where the input is NaN."""
    return s.rolling(window, center=True, min_periods=1).mean().where(s.notna())


def dt_seconds(index: pd.DatetimeIndex, max_gap_s: float = 5.0) -> pd.Series:
    """Seconds each record stands for; gaps (pauses) are capped so they don't count."""
    dt = pd.Series(index, index=index).diff().dt.total_seconds().shift(-1)
    return dt.fillna(1.0).clip(upper=max_gap_s)


def time_in_zones(records: pd.DataFrame, zones: tuple[HrZone, ...]) -> pd.Series:
    """Seconds spent in each zone, as a sum of Δt (not a row count)."""
    dt = dt_seconds(records.index)
    bins = [z.min for z in zones] + [zones[-1].max]
    zone = pd.cut(records["heart_rate"], bins=bins, labels=[z.name for z in zones], right=False)
    return dt.groupby(zone, observed=False).sum()


def efficiency(speed: pd.Series, heart_rate: pd.Series) -> float:
    """Efficiency factor: metres per minute per heartbeat. Higher is fitter."""
    hr = heart_rate.mean()
    return float(speed.mean() * 60 / hr) if hr else float("nan")


def decoupling_pct(records: pd.DataFrame, skip_s: float = 600) -> float:
    """Pace:HR decoupling. Efficiency of the first half vs the second half of running time.

    Positive means HR rose relative to pace. The first `skip_s` seconds are left out while
    HR settles. Below ~5% is usually read as aerobically well-coupled.
    """
    run = records[running_mask(records) & (records["elapsed_s"] >= skip_s)]
    if len(run) < 120:
        return float("nan")
    half = len(run) // 2
    first, second = run.iloc[:half], run.iloc[half:]
    ef1 = efficiency(first["speed"], first["heart_rate"])
    ef2 = efficiency(second["speed"], second["heart_rate"])
    return float((ef1 - ef2) / ef1 * 100)


def slope_per_min(s: pd.Series) -> float:
    """Least-squares slope of a series against time, per minute."""
    s = s.dropna()
    if len(s) < 30:
        return float("nan")
    minutes = (s.index - s.index[0]).total_seconds() / 60
    return float(np.polyfit(minutes, s.to_numpy(), 1)[0])


def bin_by_distance(
    records: pd.DataFrame, columns: list[str], bin_m: float = 1000.0
) -> pd.DataFrame:
    """Per-distance-bin means of `columns` (running records only), plus pace from bin time.

    One row per bin: bin_no (1-based), start_km, distance_m, duration_s, pace_s_per_km, and the
    mean of each column. The last bin may be partial.
    """
    df = records.assign(bin_no=(records["distance"] // bin_m).astype(int) + 1)
    dt = dt_seconds(df.index)
    grouped = df.groupby("bin_no")
    first = grouped["distance"].min()
    # A bin ends where the next one starts, so no distance falls between bins
    end = first.shift(-1).fillna(grouped["distance"].max())
    out = pd.DataFrame(
        {
            "start_km": (first // bin_m) * bin_m / 1000,
            "distance_m": end - first,
            "duration_s": dt.groupby(df["bin_no"]).sum(),
        }
    )
    out["pace_s_per_km"] = out["duration_s"] / out["distance_m"] * 1000
    running = df[running_mask(df)]
    present = [c for c in columns if c in running]
    out = out.join(running.groupby("bin_no")[present].mean())
    return out.reset_index()


def mean_over(records: pd.DataFrame, start, end, column: str) -> float:
    """Mean of a column over [start, end)."""
    s = records.loc[(records.index >= start) & (records.index < end), column]
    return float(s.mean()) if len(s) else float("nan")


def value_at(records: pd.DataFrame, when, column: str, window_s: float = 5) -> float:
    """Mean of a column over the `window_s` seconds starting at `when` (NaN past the end)."""
    return mean_over(records, when, when + pd.Timedelta(seconds=window_s), column)
