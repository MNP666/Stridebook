import panel as pn

from stridebook.analysis.easy import SETTLE_S, EasyResult
from stridebook.config import Settings
from stridebook.domain import ProcessedRun
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import html, hv_pane, tiles
from stridebook.ui.theme import palette


def view(p: ProcessedRun, r: EasyResult, settings: Settings):
    c = palette()
    ref = fmt.pace(settings.reference_pace_s_per_km)
    stats = tiles(
        [
            ("Efficiency factor", fmt.number(r.ef, 2), f"after the first {SETTLE_S // 60} min"),
            (f"HR at {ref} /km", f"{fmt.number(r.hr_at_ref_pace)} bpm", "implied by the EF"),
            (
                "Pace:HR decoupling",
                f"{fmt.number(r.decoupling_pct, 1)} %",
                "first vs second half; < 5 % is well coupled",
            ),
        ]
    )
    ef = r.ef_trace.assign(time=(r.ef_trace["elapsed_min"] * 60).map(fmt.duration))
    ef_chart = plots.style(
        plots.curve(
            ef,
            "elapsed_min",
            "ef",
            color=c.other,
            extra=["time"],
            tooltips=[("Time", "@time"), ("EF", "@ef{0.00}")],
        ),
        240,
        title="Efficiency factor, 5-min rolling",
        xlabel="Time (min)",
        ylabel="m/min per beat",
    )
    hp = plots.with_pace_label(r.hr_vs_pace)
    scatter = plots.dots(
        hp,
        "pace_s_per_km",
        "heart_rate",
        color=c.hr,
        extra=["pace"],
        tooltips=[("Pace", "@pace /km"), ("HR", "@heart_rate{0} bpm")],
    )
    scatter = plots.style(
        scatter.opts(xformatter=plots.mmss(), invert_xaxis=True),
        240,
        title="HR against pace, 30 s averages",
        xlabel="Pace (min/km), faster to the right",
        ylabel="HR (bpm)",
    )
    return pn.Column(
        stats,
        pn.Row(
            hv_pane(ef_chart, sizing_mode="stretch_width"),
            hv_pane(scatter, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        html(
            '<div class="hint">A rising efficiency factor at the same easy effort is the '
            "clearest sign of aerobic fitness; compare it across runs on the Trends page.</div>"
        ),
        sizing_mode="stretch_width",
    )
