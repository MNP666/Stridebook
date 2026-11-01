"""Long runs: steady-state drift and mechanics per km, the parts of the run (burn-in, steady,
pace blocks), and the pace blocks analysed like tempo blocks."""

import holoviews as hv
import pandas as pd
import panel as pn

from stridebook.analysis.long_run import LongRunResult
from stridebook.analysis.tempo import SETTLE_S
from stridebook.config import Settings
from stridebook.domain import ProcessedRun
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import html, linked, tiles
from stridebook.ui.theme import palette
from stridebook.ui.views.tempo import block_details

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
PART_LABELS = {
    "warmup": "Burn-in",
    "steady": "Steady",
    "active": "Block",
    "rest": "Rest",
    "cooldown": "Cooldown",
}
SPAN_ALPHA = {"warmup": 0.16, "active": 0.10}
# Mechanics in the parts table, which is wide already: column -> (header, decimals). Vertical
# ratio is left out, since it follows from vertical oscillation and step length.
SHORT = {
    "cadence_spm": ("Cadence", 1),
    "step_length": ("Step (mm)", 0),
    "vertical_oscillation": ("Vert. osc. (mm)", 1),
    "stance_time": ("Stance (ms)", 0),
}


def view(p: ProcessedRun, r: LongRunResult, settings: Settings):
    block_pace = p.run.meta.block_pace
    items = [_stats(r), *_charts(r), _parts_table(r, block_pace)]
    if r.blocks is not None:
        items += _blocks(p, r)
    items.append(_km_table(r))
    return pn.Column(*items, sizing_mode="stretch_width")


def _stats(r: LongRunResult):
    ch = r.changes_pct
    burn_in = r.parts[r.parts["role"] == "warmup"]
    burn_in_km = burn_in["distance_m"].sum() / 1000
    burn_in_pace = burn_in["duration_s"].sum() / burn_in_km if burn_in_km else float("nan")
    steady_note = "after the burn-in" + (", outside pace blocks" if r.blocks is not None else "")
    return tiles(
        [
            (
                "Burn-in",
                f"{fmt.number(burn_in_km, 1)} km" if len(burn_in) else "None",
                f"at {fmt.pace(burn_in_pace)} /km, left out below"
                if len(burn_in)
                else "set it on the Edit tab",
            ),
            ("Steady pace", f"{fmt.pace(r.steady['pace_s_per_km'])} /km", steady_note),
            ("Steady HR", f"{fmt.number(r.steady['avg_hr'])} bpm", steady_note),
            (
                "Pace:HR decoupling",
                f"{fmt.number(r.decoupling_pct, 1)} %",
                "steady running, first vs second half; < 5 % is well coupled",
            ),
            *[
                (label, f"{fmt.signed(ch[col])} %", "steady running, last vs first third")
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


def _charts(r: LongRunResult) -> list:
    """Per-km charts with the burn-in and pace blocks shaded, and a key for the shading."""
    c = palette()
    span_color = {"warmup": c.muted, "active": c.pace}
    shaded = r.parts[r.parts["role"].isin(list(SPAN_ALPHA))]
    spans = [
        hv.VSpans(
            shaded.loc[shaded["role"] == role, ["start_km", "end_km"]], kdims=["start_km", "end_km"]
        ).opts(color=span_color[role], alpha=SPAN_ALPHA[role], line_alpha=0)
        for role in SPAN_ALPHA
        if (shaded["role"] == role).any()
    ]
    bins = plots.with_pace_label(r.bins)
    bins["km"] = bins["bin_no"]
    # Each km's point sits in its middle, so the shaded parts line up with it
    bins["mid_km"] = bins["start_km"] + bins["distance_m"] / 2000
    charts = []
    for col, (title, unit, role) in PANELS.items():
        if col not in bins or bins[col].isna().all():
            continue
        color = getattr(c, role)
        value = "@pace /km" if col == "pace_s_per_km" else f"@{{{col}}}{{0.0}} {unit}"
        tips = [("Km", "@km"), (title, value)]
        line = plots.curve(bins, "mid_km", col, color=color, extra=["km", "pace"]).opts(tools=[])
        pts = plots.dots(bins, "mid_km", col, color=color, extra=["km", "pace"], tooltips=tips)
        el = hv.Overlay([*spans, line, pts])
        el = plots.style(el, 200, title=title, xlabel="Distance (km)", ylabel=unit)
        charts.append(plots.pace_axis(el, "Distance (km)") if col == "pace_s_per_km" else el)
    out = []
    if spans:
        keys = "".join(
            f'<span class="key" style="background:{span_color[role]};height:10px;'
            f'opacity:{min(1.0, SPAN_ALPHA[role] * 3):.2f}"></span>{PART_LABELS[role]}'
            f"{'s' if role == 'active' else ''}"
            for role in SPAN_ALPHA
            if (shaded["role"] == role).any()
        )
        out.append(html(f'<div class="meta">Per km.{keys}</div>'))
    out.append(linked(*charts, ncols=2))
    return out


def _part_names(parts: pd.DataFrame, block_pace) -> list[str]:
    """'Burn-in', 'Steady 1', 'MP block 1', ...: numbered when a role comes more than once."""
    counts = parts["role"].value_counts()
    names = []
    for role, no in zip(parts["role"], parts["block_no"], strict=True):
        name = PART_LABELS[role]
        if role == "active" and block_pace:
            name = f"{block_pace.short} block"
        names.append(f"{name} {no}" if counts[role] > 1 else name)
    return names


def _parts_table(r: LongRunResult, block_pace) -> pn.Column:
    parts = r.parts
    table = pd.DataFrame(
        {
            "Part": _part_names(parts, block_pace),
            "Laps": [f"{s[0]}–{s[-1]}" if len(s) > 1 else str(s[0]) for s in parts["segment_nos"]],
            "Km": [
                f"{a:.1f}–{b:.1f}" for a, b in zip(parts["start_km"], parts["end_km"], strict=True)
            ],
            "Duration": parts["duration_s"].map(fmt.duration),
            "Pace /km": parts["pace_s_per_km"].map(fmt.pace),
            "Avg HR": parts["avg_hr"].round().astype("Int64"),
            "EF": parts["ef"].round(2),
        }
        | {
            name: parts[c].round(digits).astype("Int64" if digits == 0 else float)
            for c, (name, digits) in SHORT.items()
            if c in parts
        }
    )
    return pn.Column(
        pn.pane.Markdown("#### Parts of the run", margin=(10, 10, 0, 10)),
        pn.widgets.Tabulator(
            table,
            show_index=False,
            disabled=True,
            frozen_columns=["Part"],
            layout="fit_data_stretch",
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )


def _blocks(p: ProcessedRun, r: LongRunResult) -> list:
    b = r.blocks.blocks
    mean_pace = b["duration_s"].sum() / b["distance_m"].sum() * 1000
    faster = r.steady["pace_s_per_km"] - mean_pace
    block_pace = p.run.meta.block_pace
    return [
        pn.pane.Markdown("#### Pace blocks", margin=(10, 10, 0, 10)),
        tiles(
            [
                (
                    "Pace blocks",
                    p.run.meta.structure or f"{len(b)} block(s)",
                    block_pace.label if block_pace else "",
                ),
                ("Mean block pace", f"{fmt.pace(mean_pace)} /km"),
                ("Faster than steady", f"{fmt.number(faster)} s/km"),
                ("Mean block HR", f"{fmt.number(b['avg_hr'].mean())} bpm"),
                (
                    "HR drift",
                    f"{fmt.signed(b['hr_drift_bpm_per_min'].mean(), 2)} bpm/min",
                    f"within blocks, after their first {SETTLE_S // 60} min",
                ),
            ]
        ),
        *block_details(r.blocks),
    ]


def _km_table(r: LongRunResult):
    table = pd.DataFrame(
        {
            "Km": r.bins["bin_no"],
            "Pace /km": r.bins["pace_s_per_km"].map(fmt.pace),
            "HR": r.bins["heart_rate"].round().astype("Int64"),
            "Efficiency factor": r.bins["ef"].round(2),
        }
        | {
            PANELS[c][0]: r.bins[c].round(1)
            for c in PANELS
            if c in r.bins and c not in ("pace_s_per_km", "heart_rate", "ef")
        }
    )
    return pn.widgets.Tabulator(
        table,
        show_index=False,
        disabled=True,
        layout="fit_data_stretch",
        sizing_mode="stretch_width",
        height=320,
    )
