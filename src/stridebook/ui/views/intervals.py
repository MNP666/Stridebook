import pandas as pd
import panel as pn

from stridebook.analysis.intervals import IntervalResult
from stridebook.analysis.metrics import METRICS
from stridebook.config import Settings
from stridebook.domain import ProcessedRun
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import html, hv_pane, tiles
from stridebook.ui.theme import palette


def view(p: ProcessedRun, r: IntervalResult, settings: Settings):
    if r.reps.empty:
        return html('<div class="hint">No active segments. Mark the reps on the Edit tab.</div>')
    m = p.metrics[p.metrics["part_no"].isna()].set_index("metric")["value"]
    stats = tiles(
        [
            ("Reps", fmt.number(len(r.reps)), p.run.meta.structure or ""),
            ("Mean rep pace", f"{fmt.pace(m['mean_rep_pace_s_per_km'])} /km"),
            (
                METRICS["rep_pace_cv_pct"].label,
                f"{fmt.number(m['rep_pace_cv_pct'], 1)} %",
                "lower is more even",
            ),
            ("Mean rep HR", f"{fmt.number(m['mean_rep_hr'])} bpm"),
            (
                "HR drop, 60 s into rest",
                f"{fmt.number(m['mean_hr_drop_60s'])} bpm",
                "mean over rests, from the peak",
            ),
        ]
    )
    c = palette()
    reps = plots.with_pace_label(r.reps)
    pace = plots.dots(
        reps,
        "rep_no",
        "pace_s_per_km",
        color=c.pace,
        extra=["pace", "avg_hr"],
        tooltips=[("Rep", "@rep_no"), ("Pace", "@pace /km"), ("Avg HR", "@avg_hr{0} bpm")],
    )
    line = plots.curve(reps, "rep_no", "pace_s_per_km", color=c.pace).opts(tools=[])
    pace_chart = plots.style(
        plots.pace_axis(line * pace, "Rep"),
        240,
        title="Rep pace",
        xticks=reps["rep_no"].tolist(),
        ylim=plots.padded_ylim(reps["pace_s_per_km"]),
    )

    traces = r.rep_traces.assign(time=r.rep_traces["t_s"].map(fmt.duration))
    hr_reps = plots.ordered_overlay(
        traces,
        "rep_no",
        "t_s",
        "heart_rate",
        prefix="Rep",
        extra=["time"],
        tooltips=[("Into rep", "@time"), ("HR", "@heart_rate{0} bpm")],
        title="HR in each rep",
        xlabel="Time into rep (min:sec)",
        ylabel="HR (bpm)",
        xformatter=plots.mmss(),
    )
    rests = r.rest_traces.assign(time=r.rest_traces["t_s"].map(fmt.duration))
    hr_rests = plots.ordered_overlay(
        rests,
        "rest_no",
        "t_s",
        "heart_rate",
        prefix="After rep",
        extra=["time"],
        tooltips=[("Into rest", "@time"), ("HR", "@heart_rate{0} bpm")],
        title="HR recovery in each rest",
        xlabel="Time into rest (min:sec)",
        ylabel="HR (bpm)",
        xformatter=plots.mmss(),
    )

    table = pd.DataFrame(
        {
            "Rep": r.reps["rep_no"],
            "Duration": r.reps["duration_s"].map(fmt.duration),
            "Distance (m)": r.reps["distance_m"].round().astype(int),
            "Pace /km": r.reps["pace_s_per_km"].map(fmt.pace),
            "Avg HR": r.reps["avg_hr"].round().astype("Int64"),
            "Max HR": r.reps["max_hr"].round().astype("Int64"),
            "End HR": r.reps["end_hr"].round().astype("Int64"),
            "HR drop 60 s": r.reps["hr_drop_60s"].round().astype("Int64"),
        }
    )
    return pn.Column(
        stats,
        pn.Row(
            hv_pane(pace_chart, sizing_mode="stretch_width"),
            hv_pane(hr_reps, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        pn.Row(
            hv_pane(hr_rests, sizing_mode="stretch_width"),
            pn.widgets.Tabulator(
                table,
                show_index=False,
                disabled=True,
                layout="fit_data_stretch",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
