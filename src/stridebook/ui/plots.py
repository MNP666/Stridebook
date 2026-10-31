"""HoloViews building blocks with the house style: 2px lines, hairline grid, one y-axis per
chart (HR and pace are stacked charts sharing x, never a dual axis), pace on an inverted
min:sec axis so faster is up, and hover tooltips on every mark.
"""

import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models import CustomJSTickFormatter
from holoviews.element import tiles as hv_tiles

from stridebook.domain import SegmentRole
from stridebook.ui import format as fmt
from stridebook.ui.theme import ordinal, palette

hv.extension("bokeh", logo=False)

MMSS = CustomJSTickFormatter(
    code="const t = Math.round(tick); return Math.floor(t / 60) + ':' + "
    "String(t % 60).padStart(2, '0');"
)


HMMSS = CustomJSTickFormatter(
    code="const t = Math.round(tick); const h = Math.floor(t / 3600); "
    "const m = Math.floor((t % 3600) / 60); const s = String(t % 60).padStart(2, '0'); "
    "return h ? h + ':' + String(m).padStart(2, '0') + ':' + s : m + ':' + s;"
)


def style(element, height: int = 200, **opts):
    p = palette()
    return element.opts(
        responsive=True,
        height=height,
        show_grid=True,
        gridstyle={"grid_line_color": p.grid, "grid_line_width": 1},
        toolbar="right",
        default_tools=["pan", "box_zoom", "wheel_zoom", "reset", "save"],
        **opts,
    )


def with_pace_label(df: pd.DataFrame, column: str = "pace_s_per_km") -> pd.DataFrame:
    return df.assign(pace=df[column].map(fmt.pace))


def curve(df, x, y, *, color, label=None, extra=(), tooltips=None, **opts):
    """A 2px line with a vertical-line hover."""
    el = hv.Curve(df, x, [y, *extra], label=label) if label else hv.Curve(df, x, [y, *extra])
    return el.opts(
        color=color,
        line_width=2,
        tools=["hover"],
        hover_mode="vline",
        hover_tooltips=tooltips,
        **opts,
    )


def dots(df, x, y, *, color, extra=(), tooltips=None, label=None, **opts):
    el = hv.Scatter(df, x, [y, *extra], label=label) if label else hv.Scatter(df, x, [y, *extra])
    return el.opts(color=color, size=9, tools=["hover"], hover_tooltips=tooltips, **opts)


def pace_axis(element, xlabel: str = ""):
    return element.opts(yformatter=MMSS, invert_yaxis=True, ylabel="Pace (min/km)", xlabel=xlabel)


def padded_ylim(values: pd.Series, min_pad: float = 8.0) -> tuple[float, float]:
    """A y-range with at least `min_pad` either side, so a 1 s/km difference between reps
    doesn't fill the whole chart."""
    lo, hi = float(values.min()), float(values.max())
    pad = max(min_pad, (hi - lo) * 0.2)
    return lo - pad, hi + pad


def active_spans(records: pd.DataFrame, segments: pd.DataFrame):
    """Shade active segments (reps, blocks) behind a time chart; x in minutes from start."""
    if segments.empty or (segments["role"] == SegmentRole.ACTIVE.value).all():
        return None
    t0 = records.index[0]
    active = segments[segments["role"] == SegmentRole.ACTIVE.value]
    spans = pd.DataFrame(
        {
            "x0": (pd.DatetimeIndex(active["start"]) - t0).total_seconds() / 60,
            "x1": (pd.DatetimeIndex(active["end"]) - t0).total_seconds() / 60,
        }
    )
    return hv.VSpans(spans, kdims=["x0", "x1"]).opts(color=palette().pace, alpha=0.08, line_alpha=0)


def time_traces(records: pd.DataFrame, segments: pd.DataFrame, height: int = 190):
    """Stacked HR and pace against minutes from start, active segments shaded."""
    p = palette()
    df = records[["elapsed_s", "heart_rate", "pace_s_per_km"]].iloc[::3].copy()
    df["elapsed_min"] = df["elapsed_s"] / 60
    df["time"] = df["elapsed_s"].map(fmt.duration)
    # Gaps while standing would otherwise draw near-vertical pace lines to the axis limit
    df["pace_s_per_km"] = df["pace_s_per_km"].where(df["pace_s_per_km"] < 600)
    df = with_pace_label(df)
    hr = curve(
        df,
        "elapsed_min",
        "heart_rate",
        color=p.hr,
        extra=["time"],
        tooltips=[("Time", "@time"), ("HR", "@heart_rate{0} bpm")],
    )
    pace = curve(
        df,
        "elapsed_min",
        "pace_s_per_km",
        color=p.pace,
        extra=["time", "pace"],
        tooltips=[("Time", "@time"), ("Pace", "@pace /km")],
    )
    spans = active_spans(records, segments)
    if spans is not None:
        hr, pace = spans * hr, spans * pace
    hr = style(hr, height, ylabel="HR (bpm)", xlabel="", title="Heart rate")
    pace = style(pace_axis(pace, "Time (min)"), height, title="Pace")
    return hr, pace


def ordered_overlay(df, key, x, y, *, prefix, height=240, extra=(), tooltips=None, **opts):
    """One line per rep/block, coloured along the ordinal ramp (rep 1 lightest)."""
    keys = sorted(df[key].unique())
    colors = ordinal(len(keys))
    curves = [
        curve(
            df[df[key] == k], x, y, color=c, label=f"{prefix} {k}", extra=extra, tooltips=tooltips
        )
        for k, c in zip(keys, colors, strict=True)
    ]
    overlay = hv.Overlay(curves) if curves else hv.Overlay([hv.Curve([])])
    return style(overlay, height, legend_position="right", **opts)


def route_map(records: pd.DataFrame, height: int = 300):
    """The GPS track on map tiles, or None without positions."""
    if not {"position_lat", "position_long"} <= set(records.columns):
        return None
    pos = records[["position_lat", "position_long"]].dropna().iloc[::5]
    if pos.empty:
        return None
    x, y = hv.util.transform.lon_lat_to_easting_northing(
        pos["position_long"].to_numpy(), pos["position_lat"].to_numpy()
    )
    # OpenStreetMap needs no API key (Carto and Stadia tiles now do)
    path = hv.Path([np.column_stack([x, y])]).opts(color=palette().pace, line_width=2)
    tiles = hv_tiles.OSM()
    return (tiles * path).opts(
        responsive=True,
        height=height,
        xaxis=None,
        yaxis=None,
        title="Route",
        default_tools=["pan", "wheel_zoom", "reset"],
    )


def bars_h(df, category, value, *, color, tooltips, height=200, **opts):
    el = hv.Bars(df, category, [value, *[c for c in df.columns if c not in (category, value)]])
    return style(
        el.opts(
            color=color,
            invert_axes=True,
            bar_width=0.6,
            tools=["hover"],
            hover_tooltips=tooltips,
            line_alpha=0,
        ),
        height,
        **opts,
    )
