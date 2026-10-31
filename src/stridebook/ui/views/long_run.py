import pandas as pd
import panel as pn

from stridebook.analysis.long_run import LongRunResult
from stridebook.config import Settings
from stridebook.domain import ProcessedRun
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import linked, tiles
from stridebook.ui.theme import palette

# column -> (title, unit, colour role)
PANELS = {
    "pace_s_per_km": ("Pace", "min/km", "pace"),
    "heart_rate": ("Heart rate", "bpm", "hr"),
    "ef": ("Efficiency factor", "m/min per beat", "other"),
    "cadence_spm": ("Cadence", "steps/min", "other"),
    "step_length": ("Step length", "mm", "other"),
    "vertical_oscillation": ("Vertical oscillation", "mm", "other"),
    "vertical_ratio": ("Vertical ratio", "%", "other"),
    "stance_time": ("Stance time", "ms", "other"),
}


def view(p: ProcessedRun, r: LongRunResult, settings: Settings):
    ch = r.changes_pct
    stats = tiles(
        [
            (
                "Pace:HR decoupling",
                f"{fmt.number(r.decoupling_pct, 1)} %",
                "first vs second half; < 5 % is well coupled",
            ),
            *[
                (label, f"{fmt.signed(ch[col])} %", "last third vs first third")
                for col, label in [
                    ("cadence_spm", "Cadence"),
                    ("step_length", "Step length"),
                    ("vertical_oscillation", "Vertical oscillation"),
                    ("stance_time", "Stance time"),
                ]
                if col in ch
            ],
        ]
    )
    c = palette()
    bins = plots.with_pace_label(r.bins)
    bins["km"] = bins["bin_no"]
    charts = []
    for col, (title, unit, role) in PANELS.items():
        if col not in bins or bins[col].isna().all():
            continue
        color = getattr(c, role)
        value = "@pace /km" if col == "pace_s_per_km" else f"@{{{col}}}{{0.0}} {unit}"
        tips = [("Km", "@km"), (title, value)]
        el = plots.curve(bins, "km", col, color=color, extra=["pace"]).opts(tools=[]) * plots.dots(
            bins, "km", col, color=color, extra=["pace"], tooltips=tips
        )
        el = plots.style(el, 200, title=title, xlabel="Km", ylabel=unit)
        charts.append(plots.pace_axis(el, "Km") if col == "pace_s_per_km" else el)
    grid = linked(*charts, ncols=2)
    table = pd.DataFrame(
        {
            "Km": r.bins["bin_no"],
            "Pace /km": r.bins["pace_s_per_km"].map(fmt.pace),
            "HR": r.bins["heart_rate"].round().astype("Int64"),
        }
        | {
            PANELS[c][0]: r.bins[c].round(1)
            for c in PANELS
            if c in r.bins and c not in ("pace_s_per_km", "heart_rate")
        }
    )
    return pn.Column(
        stats,
        grid,
        pn.widgets.Tabulator(
            table,
            show_index=False,
            disabled=True,
            layout="fit_data_stretch",
            sizing_mode="stretch_width",
            height=320,
        ),
        sizing_mode="stretch_width",
    )
