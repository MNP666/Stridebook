import holoviews as hv
import pandas as pd
import panel as pn

from stridebook.analysis.race import RaceResult
from stridebook.config import Settings
from stridebook.domain import ProcessedRun
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import linked, tiles
from stridebook.ui.theme import palette


def view(p: ProcessedRun, r: RaceResult, settings: Settings):
    ratio = r.split_ratio
    split_note = (
        "even" if abs(ratio - 1) < 0.01 else "positive split" if ratio > 1 else "negative split"
    )
    stats = tiles(
        [
            ("Finish time", fmt.duration(r.finish_time_s), p.run.meta.structure or ""),
            ("Average pace", f"{fmt.pace(r.avg_pace_s_per_km)} /km"),
            ("Second half / first half", fmt.number(ratio, 3), split_note),
            ("Km pace variation", f"{fmt.number(r.pace_cv_pct, 1)} %", "lower is more even"),
            ("Average HR", f"{fmt.number(r.avg_hr)} bpm"),
        ]
    )
    c = palette()
    s = plots.with_pace_label(r.splits)
    s["km"] = s["bin_no"]
    s["at"] = s["cumulative_s"].map(fmt.duration)
    tips = [("Km", "@km"), ("Pace", "@pace /km"), ("Elapsed", "@at")]
    pace = plots.curve(s, "km", "pace_s_per_km", color=c.pace, extra=["pace", "at"]).opts(tools=[])
    pace = pace * plots.dots(
        s, "km", "pace_s_per_km", color=c.pace, extra=["pace", "at"], tooltips=tips
    )
    mean = hv.HLine(r.avg_pace_s_per_km).opts(color=c.muted, line_width=1)
    pace = plots.style(
        plots.pace_axis(pace * mean, ""),
        230,
        title="Pace per km (line: average)",
        ylim=plots.padded_ylim(s["pace_s_per_km"]),
    )
    hr = plots.curve(s, "km", "heart_rate", color=c.hr).opts(tools=[]) * plots.dots(
        s, "km", "heart_rate", color=c.hr, tooltips=[("Km", "@km"), ("HR", "@heart_rate{0} bpm")]
    )
    hr = plots.style(hr, 200, title="Heart rate per km", xlabel="Km", ylabel="HR (bpm)")
    table = pd.DataFrame(
        {
            "Km": r.splits["bin_no"],
            "Split": r.splits["duration_s"].map(fmt.duration),
            "Pace /km": r.splits["pace_s_per_km"].map(fmt.pace),
            "Elapsed": r.splits["cumulative_s"].map(fmt.duration),
            "HR": r.splits["heart_rate"].round().astype("Int64"),
        }
    )
    return pn.Column(
        stats,
        pn.Row(
            linked(pace, hr),
            pn.widgets.Tabulator(
                table,
                show_index=False,
                disabled=True,
                width=380,
                height=470,
                layout="fit_data_stretch",
            ),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
