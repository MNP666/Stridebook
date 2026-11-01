"""The long-format metrics table that the trends page reads.

Every analysis turns its result into rows of (part_no, metric, value, unit). part_no is the
1-based rep or block a value belongs to, or <NA> for a whole-run value. A new trend is a new
entry in METRICS; the table's shape never changes.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from stridebook.analysis.hr_fit import MODELS, metric_name

METRIC_COLUMNS = ["part_no", "metric", "value", "unit"]


@dataclass(frozen=True)
class MetricInfo:
    label: str
    unit: str  # "s/km" is shown as min:sec pace, "s" as a duration
    per_part: bool = False  # one value per rep/block rather than per run
    lower_is_better: bool = False
    digits: int | None = None  # decimals shown; None picks them from the unit
    # The hr_fit model a fitted value comes from. Metrics that differ only by model share a
    # label, and the trends page lists them once with a model choice.
    model: str | None = None


_PACE = "s/km"

METRICS: dict[str, MetricInfo] = {
    # overview: every run
    "distance_km": MetricInfo("Distance", "km"),
    "duration_s": MetricInfo("Time", "s"),
    "avg_pace_s_per_km": MetricInfo("Average pace", _PACE, lower_is_better=True),
    "avg_hr": MetricInfo("Average HR", "bpm"),
    "max_hr": MetricInfo("Max HR", "bpm"),
    "ef": MetricInfo("Efficiency factor", "m/min/bpm"),
    # easy
    "hr_at_ref_pace": MetricInfo("HR at reference pace", "bpm", lower_is_better=True),
    "decoupling_pct": MetricInfo("Pace:HR decoupling", "%", lower_is_better=True),
    # intervals
    "mean_rep_pace_s_per_km": MetricInfo("Mean rep pace", _PACE, lower_is_better=True),
    "rep_pace_cv_pct": MetricInfo("Rep pace variation", "%", lower_is_better=True),
    "mean_rep_hr": MetricInfo("Mean rep HR", "bpm"),
    "mean_hr_drop_60s": MetricInfo("Mean HR drop, 60 s into rest", "bpm"),
    "rep_count": MetricInfo("Reps", ""),
    "rep_pace_s_per_km": MetricInfo("Rep pace", _PACE, per_part=True, lower_is_better=True),
    "rep_avg_hr": MetricInfo("Rep average HR", "bpm", per_part=True),
    "rep_end_hr": MetricInfo("Rep end HR", "bpm", per_part=True),
    "rest_hr_drop_60s": MetricInfo("HR drop, 60 s into rest", "bpm", per_part=True),
    # tempo
    "tempo_min": MetricInfo("Time at tempo", "min"),
    "mean_block_pace_s_per_km": MetricInfo("Mean block pace", _PACE, lower_is_better=True),
    "mean_block_hr": MetricInfo("Mean block HR", "bpm"),
    "mean_hr_drift_bpm_per_min": MetricInfo(
        "Mean HR drift in blocks", "bpm/min", lower_is_better=True
    ),
    "block_pace_s_per_km": MetricInfo("Block pace", _PACE, per_part=True, lower_is_better=True),
    "block_avg_hr": MetricInfo("Block average HR", "bpm", per_part=True),
    "block_hr_drift_bpm_per_min": MetricInfo(
        "HR drift in block", "bpm/min", per_part=True, lower_is_better=True
    ),
    # long run: steady running (after the burn-in, outside pace blocks), whose pace blocks
    # share the tempo block metrics
    "steady_pace_s_per_km": MetricInfo("Steady pace", _PACE, lower_is_better=True),
    "steady_hr": MetricInfo("Steady HR", "bpm"),
    "steady_ef": MetricInfo("Steady efficiency factor", "m/min/bpm"),
    # long run: change from the first to the last third of steady running
    "cadence_change_pct": MetricInfo("Cadence change, last vs first third", "%"),
    "step_length_change_pct": MetricInfo("Step length change, last vs first third", "%"),
    "vertical_oscillation_change_pct": MetricInfo(
        "Vertical oscillation change, last vs first third", "%", lower_is_better=True
    ),
    "stance_time_change_pct": MetricInfo(
        "Stance time change, last vs first third", "%", lower_is_better=True
    ),
    # race
    "finish_time_s": MetricInfo("Finish time", "s", lower_is_better=True),
    "split_ratio": MetricInfo("Second half / first half", "", digits=3),
    "pace_cv_pct": MetricInfo("Km pace variation", "%", lower_is_better=True),
}

# HR response fits (intervals and tempo): each model's parameters for every rep or block
_HR_FIT = {  # fit column -> label, unit, lower is better, digits
    "tau_s": ("HR rise τ", "s", True, None),
    "plateau_bpm": ("HR plateau", "bpm", False, None),
    "drift_bpm_per_min": ("HR drift, fitted", "bpm/min", True, None),
    "beta": ("HR stretch β", "", False, 2),
}
for _part in ("rep", "block"):
    for _model, _spec in MODELS.items():
        for _column in _spec.trended:
            _label, _unit, _lower, _digits = _HR_FIT[_column]
            METRICS[metric_name(_part, _model, _column)] = MetricInfo(
                f"{_part.capitalize()} {_label}",
                _unit,
                per_part=True,
                lower_is_better=_lower,
                digits=_digits,
                model=_model,
            )


def metric_frame(
    run_values: Mapping[str, float], part_values: Mapping[str, Sequence[float]] | None = None
) -> pd.DataFrame:
    """Rows for whole-run values and per-part values. Unknown metric names raise KeyError."""
    rows = [
        {"part_no": pd.NA, "metric": k, "value": float(v), "unit": METRICS[k].unit}
        for k, v in run_values.items()
    ]
    for k, values in (part_values or {}).items():
        unit = METRICS[k].unit
        rows += [
            {"part_no": i, "metric": k, "value": float(v), "unit": unit}
            for i, v in enumerate(values, start=1)
        ]
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    df["part_no"] = df["part_no"].astype("Int64")
    df["value"] = df["value"].astype(float)
    return df
