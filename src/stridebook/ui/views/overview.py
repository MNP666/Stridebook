import pandas as pd
import panel as pn

from stridebook.analysis.overview import OverviewResult
from stridebook.config import Settings
from stridebook.domain import ProcessedRun
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import hv_pane, linked, tiles
from stridebook.ui.theme import palette


def view(p: ProcessedRun, r: OverviewResult, settings: Settings):
    stats = tiles(
        [
            ("Distance", f"{fmt.number(r.distance_m / 1000, 2)} km"),
            ("Time", fmt.duration(r.duration_s)),
            ("Average pace", f"{fmt.pace(r.avg_pace_s_per_km)} /km"),
            ("Average HR", f"{fmt.number(r.avg_hr)} bpm"),
            ("Max HR", f"{fmt.number(r.max_hr)} bpm"),
            ("Efficiency factor", fmt.number(r.ef, 2), "m/min per beat, running only"),
        ]
    )
    zones = pd.DataFrame(
        {"zone": r.zones.index.astype(str), "minutes": r.zones.to_numpy() / 60}
    ).assign(time=lambda d: (d["minutes"] * 60).map(fmt.duration))
    zone_chart = plots.bars_h(
        zones,
        "zone",
        "minutes",
        color=palette().hr,
        tooltips=[("Zone", "@zone"), ("Time", "@time")],
        height=220,
        title="Time in HR zones",
        xlabel="",
        ylabel="Minutes",
    )
    below = [hv_pane(zone_chart)]
    if (route := plots.route_map(p.records)) is not None:
        below.append(hv_pane(route))
    return pn.Column(
        stats,
        linked(*plots.time_traces(p.records, p.run.segments)),
        pn.Row(*below, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
