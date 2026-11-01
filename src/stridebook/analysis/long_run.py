"""Long runs: steady-state drift and running mechanics, and the pace blocks if there are any.

The laps carry long-run roles (see analysis.segments): the burn-in is warmup, pace blocks are
active, and the rest is steady. Drift and mechanics are judged on steady running only, since
the burn-in hasn't settled yet and pace blocks run at another pace.
"""

from dataclasses import dataclass

import pandas as pd

from stridebook.analysis import tempo
from stridebook.analysis.common import (
    bin_by_distance,
    decoupling_pct,
    efficiency,
    mean_over,
    running_mask,
)
from stridebook.analysis.hr_fit import part_metrics
from stridebook.analysis.metrics import metric_frame
from stridebook.analysis.segments import blocks
from stridebook.config import Settings
from stridebook.domain import SegmentRole

# Columns other vendors may lack; missing ones are simply left out
MECHANICS = ["cadence_spm", "step_length", "vertical_oscillation", "vertical_ratio", "stance_time"]
CHANGE_METRICS = {
    "cadence_spm": "cadence_change_pct",
    "step_length": "step_length_change_pct",
    "vertical_oscillation": "vertical_oscillation_change_pct",
    "stance_time": "stance_time_change_pct",
}
PART_ORDER = ["warmup", "steady", "active", "rest", "cooldown"]
MIN_BIN_M = 100  # a shorter last km is too noisy to show


@dataclass
class LongRunResult:
    bins: pd.DataFrame  # per km: bin_no, start_km, pace_s_per_km, heart_rate, ef, mechanics
    parts: pd.DataFrame  # runs of laps with one role, in order: role, block_no (1.. per role),
    #                      start, end, start_km, end_km, duration_s, distance_m, segment_nos,
    #                      pace_s_per_km, avg_hr, ef, mechanics
    steady: dict[str, float]  # pace_s_per_km, avg_hr, ef over steady running
    decoupling_pct: float  # steady running, first half vs second half
    changes_pct: dict[str, float]  # column -> % change from the first to the last third (steady)
    blocks: tempo.TempoResult | None  # the pace blocks, analysed like tempo blocks


def compute(
    records: pd.DataFrame, segments: pd.DataFrame, settings: Settings, bin_m: float = 1000
) -> LongRunResult:
    bins = bin_by_distance(records, ["heart_rate", "speed", *MECHANICS], bin_m=bin_m)
    bins = bins[bins["distance_m"] >= MIN_BIN_M].copy()
    bins["ef"] = bins["speed"] * 60 / bins["heart_rate"]
    # Laps without long-run roles (all active) are one steady run with no blocks
    if not (segments["role"] == SegmentRole.STEADY.value).any():
        segments = segments.replace({"role": {SegmentRole.ACTIVE.value: SegmentRole.STEADY.value}})
    role = records["segment_no"].map(segments.set_index("segment_no")["role"])
    steady = records[role.eq(SegmentRole.STEADY.value).fillna(False).astype(bool)]
    steady = steady[running_mask(steady)]
    steady_laps = segments[segments["role"] == SegmentRole.STEADY.value]

    third = max(1, len(steady) // 3)
    first, last = steady.iloc[:third], steady.iloc[-third:]
    changes = {
        c: float((last[c].mean() - first[c].mean()) / first[c].mean() * 100)
        for c in CHANGE_METRICS
        if c in steady and first[c].mean()
    }
    has_blocks = (segments["role"] == SegmentRole.ACTIVE.value).any()
    return LongRunResult(
        bins=bins,
        parts=_parts(records, segments),
        steady={
            "pace_s_per_km": _pace(
                steady_laps["duration_s"].sum(), steady_laps["distance_m"].sum()
            ),
            "avg_hr": float(steady["heart_rate"].mean()),
            "ef": efficiency(steady["speed"], steady["heart_rate"]),
        },
        decoupling_pct=decoupling_pct(steady.assign(elapsed_s=0.0), skip_s=0),
        changes_pct=changes,
        blocks=tempo.compute(records, segments, settings) if has_blocks else None,
    )


def _parts(records: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive laps with one role, with their pace, HR, efficiency and mechanics."""
    present = [r for r in PART_ORDER if (segments["role"] == r).any()]
    parts = pd.concat([blocks(segments, r).assign(role=r) for r in present])
    parts = parts.sort_values("start", ignore_index=True)
    parts["start_km"] = [records["distance"].asof(t) / 1000 for t in parts["start"]]
    parts["end_km"] = parts["start_km"] + parts["distance_m"] / 1000
    parts["pace_s_per_km"] = [
        _pace(t, d) for t, d in zip(parts["duration_s"], parts["distance_m"], strict=True)
    ]
    spans = list(zip(parts["start"], parts["end"], strict=True))
    run = records[running_mask(records)]
    within = [run[(run.index >= s) & (run.index < e)] for s, e in spans]
    parts["avg_hr"] = [mean_over(records, s, e, "heart_rate") for s, e in spans]
    parts["ef"] = [efficiency(g["speed"], g["heart_rate"]) for g in within]
    for c in MECHANICS:
        if c in run:
            parts[c] = [float(g[c].mean()) for g in within]
    return parts


def _pace(duration_s: float, distance_m: float) -> float:
    return duration_s / distance_m * 1000 if distance_m > 0 else float("nan")


def metrics(r: LongRunResult) -> pd.DataFrame:
    run_values = {
        "steady_pace_s_per_km": r.steady["pace_s_per_km"],
        "steady_hr": r.steady["avg_hr"],
        "steady_ef": r.steady["ef"],
        "decoupling_pct": r.decoupling_pct,
    } | {CHANGE_METRICS[c]: v for c, v in r.changes_pct.items()}
    part_values = {}
    if r.blocks is not None:
        b = r.blocks.blocks
        run_values |= {
            "mean_block_pace_s_per_km": b["duration_s"].sum() / b["distance_m"].sum() * 1000,
            "mean_block_hr": b["avg_hr"].mean(),
            "mean_hr_drift_bpm_per_min": b["hr_drift_bpm_per_min"].mean(),
        }
        part_values = {
            "block_pace_s_per_km": b["pace_s_per_km"].tolist(),
            "block_avg_hr": b["avg_hr"].tolist(),
            "block_hr_drift_bpm_per_min": b["hr_drift_bpm_per_min"].tolist(),
            **part_metrics(r.blocks.hr_fits, "block"),
        }
    return metric_frame(run_values, part_values)
