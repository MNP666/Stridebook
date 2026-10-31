import pandas as pd
import panel as pn

from stridebook.analysis.tempo import SETTLE_S, TempoResult
from stridebook.config import Settings
from stridebook.domain import ProcessedRun
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import html, linked, tiles


def view(p: ProcessedRun, r: TempoResult, settings: Settings):
    b = r.blocks
    if b.empty:
        return html(
            '<div class="hint">No active segments. Mark the tempo blocks on the Edit tab.</div>'
        )
    mean_pace = b["duration_s"].sum() / b["distance_m"].sum() * 1000
    stats = tiles(
        [
            ("Time at tempo", fmt.duration(b["duration_s"].sum()), f"{len(b)} block(s)"),
            ("Mean block pace", f"{fmt.pace(mean_pace)} /km"),
            ("Mean block HR", f"{fmt.number(b['avg_hr'].mean())} bpm"),
            (
                "HR drift",
                f"{fmt.signed(b['hr_drift_bpm_per_min'].mean(), 2)} bpm/min",
                f"slope after the first {SETTLE_S // 60} min of each block",
            ),
            (
                "Decoupling",
                f"{fmt.number(b['decoupling_pct'].mean(), 1)} %",
                "within blocks, first vs second half",
            ),
        ]
    )
    traces = plots.with_pace_label(r.traces)
    traces["time"] = (traces["t_min"] * 60).map(fmt.duration)
    traces["pace_s_per_km"] = traces["pace_s_per_km"].where(traces["pace_s_per_km"] < 600)
    hr = plots.ordered_overlay(
        traces,
        "block_no",
        "t_min",
        "heart_rate",
        prefix="Block",
        extra=["time"],
        tooltips=[("Into block", "@time"), ("HR", "@heart_rate{0} bpm")],
        height=220,
        title="HR through each block (30 s smoothing)",
        xlabel="",
        ylabel="HR (bpm)",
    )
    pace = plots.ordered_overlay(
        traces,
        "block_no",
        "t_min",
        "pace_s_per_km",
        prefix="Block",
        extra=["time", "pace"],
        tooltips=[("Into block", "@time"), ("Pace", "@pace /km")],
        height=220,
        title="Pace through each block",
        xlabel="Time into block (min)",
    )
    pace = plots.pace_axis(pace, "Time into block (min)")
    charts = linked(hr, pace)
    table = pd.DataFrame(
        {
            "Block": b["block_no"],
            "Duration": b["duration_s"].map(fmt.duration),
            "Distance (km)": (b["distance_m"] / 1000).round(2),
            "Pace /km": b["pace_s_per_km"].map(fmt.pace),
            "Avg HR": b["avg_hr"].round().astype("Int64"),
            "HR drift (bpm/min)": b["hr_drift_bpm_per_min"].round(2),
            "Pace drift (s/km per min)": b["pace_drift_s_per_km_per_min"].round(2),
            "Decoupling (%)": b["decoupling_pct"].round(1),
        }
    )
    return pn.Column(
        stats,
        charts,
        pn.widgets.Tabulator(
            table,
            show_index=False,
            disabled=True,
            layout="fit_data_stretch",
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
