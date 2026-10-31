"""Trends: one metric over time, narrowed by category, structure and tags.

Filters narrow left to right. Per-rep metrics (rep 1..n as separate lines) are offered once a
structure is chosen, since rep 3 of a 5x6:00 is only comparable with rep 3 of another 5x6:00.
"""

import holoviews as hv
import pandas as pd
import panel as pn

from stridebook.analysis.metrics import METRICS
from stridebook.domain import RunCategory
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import html, hv_pane
from stridebook.ui.state import AppState
from stridebook.ui.theme import ordinal, palette

ALL, ANY = "all", "Any"
PART_PREFIX = {RunCategory.INTERVAL.value: "Rep", RunCategory.TEMPO.value: "Block"}


class TrendsPage(pn.viewable.Viewer):
    def __init__(self, state: AppState, **params):
        super().__init__(**params)
        self.state = state
        runs = state.store.runs()
        self.data = state.store.metrics().merge(runs[["run_id", "title"]], on="run_id")
        present = set(runs["category"])
        default = RunCategory.INTERVAL.value if RunCategory.INTERVAL.value in present else ALL
        self.category = pn.widgets.Select(
            label="Category",
            options={"All runs": ALL}
            | {c.label: c.value for c in RunCategory if c.value in present},
            value=default,
            width=180,
        )
        self.structure = pn.widgets.Select(label="Structure", width=180)
        self.tag_filter = pn.widgets.MultiChoice(
            label="Tags (all of)", options=state.known_tags(), width=240
        )
        self.metric = pn.widgets.Select(label="Metric", width=320)
        self.chart = hv_pane(sizing_mode="stretch_width")
        self.table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            disabled=True,
            selectable=1,
            layout="fit_data_stretch",
            sizing_mode="stretch_width",
            height=320,
        )
        self._ids: list[str] = []
        self.table.param.watch(self._selected, "selection")
        self.category.param.watch(self._category_changed, "value")
        for w in (self.structure, self.tag_filter):
            w.param.watch(self._filters_changed, "value")
        self.metric.param.watch(self._render, "value")
        self._category_changed()

    # filtering

    def _rows(self, category=True, structure=True, tags=True) -> pd.DataFrame:
        d = self.data
        if category and self.category.value != ALL:
            d = d[d["category"] == self.category.value]
        if structure and self.structure.value not in (None, ANY):
            d = d[d["structure"] == self.structure.value]
        if tags and self.tag_filter.value:
            wanted = set(self.tag_filter.value)
            d = d[d["tags"].map(lambda t: wanted <= set(t))]
        return d

    def _category_changed(self, *_):
        structures = sorted(s for s in self._rows(structure=False)["structure"].dropna().unique())
        self.structure.options = [ANY, *structures]
        self.structure.value = ANY
        self._filters_changed()

    def _filters_changed(self, *_):
        rows = self._rows()
        per_part_ok = self.structure.value not in (None, ANY)
        names = [
            m
            for m in METRICS
            if m in set(rows["metric"]) and (per_part_ok or not METRICS[m].per_part)
        ]
        options = {METRICS[m].label: m for m in names}
        current = self.metric.value
        self.metric.options = options
        overview = {"distance_km", "duration_s", "avg_pace_s_per_km", "avg_hr", "max_hr", "ef"}
        specific = [m for m in names if m not in overview]
        self.metric.value = current if current in names else (specific or names or [None])[0]
        self._render()

    # output

    def _render(self, *_):
        metric = self.metric.value
        rows = self._rows()
        rows = rows[rows["metric"] == metric] if metric else rows.iloc[0:0]
        if rows.empty:
            self.chart.object = None
            self.table.value = pd.DataFrame({"": ["No runs match these filters."]})
            self._ids = []
            return
        info = METRICS[metric]
        self.chart.object = (
            self._per_part(rows, info) if info.per_part else self._per_run(rows, info)
        )
        self._fill_table(rows, metric, info)

    def _per_run(self, rows: pd.DataFrame, info):
        c = palette()
        d = rows.sort_values("start").assign(
            date=lambda x: x["start"].dt.strftime("%d %b %Y"),
            shown=lambda x: x["value"].map(lambda v: fmt.metric_value(self.metric.value, v)),
        )
        d["trend"] = d["value"].rolling(5, min_periods=1, center=True).median()
        color = c.pace if info.unit == "s/km" else c.hr if info.unit == "bpm" else c.other
        tips = [("Date", "@date"), ("Run", "@title"), (info.label, "@shown")]
        pts = plots.dots(
            d,
            "start",
            "value",
            color=color,
            extra=["date", "title", "shown"],
            tooltips=tips,
            label="Run",
        )
        trend = hv.Curve(d, "start", "trend", label="Rolling median, 5 runs").opts(
            color=c.ink2, line_width=2
        )
        return self._axes(plots.style(trend * pts, 360, legend_position="top_left"), info)

    def _per_part(self, rows: pd.DataFrame, info):
        d = (
            rows[rows["part_no"].notna()]
            .sort_values("start")
            .assign(
                date=lambda x: x["start"].dt.strftime("%d %b %Y"),
                shown=lambda x: x["value"].map(lambda v: fmt.metric_value(self.metric.value, v)),
            )
        )
        prefix = PART_PREFIX.get(self.category.value, "Part")
        parts = sorted(d["part_no"].unique())
        layers = []
        for part, color in zip(parts, ordinal(len(parts)), strict=True):
            g = d[d["part_no"] == part]
            tips = [("Date", "@date"), (prefix, str(part)), (info.label, "@shown")]
            layers.append(
                plots.curve(
                    g,
                    "start",
                    "value",
                    color=color,
                    label=f"{prefix} {part}",
                    extra=["date", "shown"],
                    tooltips=tips,
                )
            )
            layers.append(
                plots.dots(
                    g,
                    "start",
                    "value",
                    color=color,
                    extra=["date", "shown"],
                    tooltips=tips,
                    label=f"{prefix} {part}",
                )
            )
        return self._axes(plots.style(hv.Overlay(layers), 360, legend_position="right"), info)

    def _axes(self, element, info):
        element = element.opts(title=info.label, xlabel="", ylabel=info.unit)
        if info.unit == "s/km":
            return element.opts(yformatter=plots.MMSS, invert_yaxis=True, ylabel="min/km")
        if info.unit == "s":
            return element.opts(yformatter=plots.HMMSS, ylabel="")
        return element

    def _fill_table(self, rows: pd.DataFrame, metric: str, info):
        if info.per_part:
            wide = rows.pivot_table(
                index=["run_id", "start", "title", "structure"], columns="part_no", values="value"
            ).reset_index()
            prefix = PART_PREFIX.get(self.category.value, "Part")
            parts = [c for c in wide.columns if isinstance(c, int | float)]
            values = {
                f"{prefix} {int(p)}": wide[p].map(lambda v: fmt.metric_value(metric, v))
                for p in parts
            }
        else:
            wide = rows[rows["part_no"].isna()]
            values = {info.label: wide["value"].map(lambda v: fmt.metric_value(metric, v))}
        wide = wide.assign(**{k: v.to_numpy() for k, v in values.items()})
        wide = wide.sort_values("start", ascending=False)
        self._ids = wide["run_id"].tolist()
        self.table.value = pd.DataFrame(
            {
                "Date": wide["start"].dt.strftime("%d %b %Y").to_numpy(),
                "Run": wide["title"].to_numpy(),
                **{k: wide[k].to_numpy() for k in values},
            }
        )
        self.table.selection = []

    def _selected(self, event):
        if event.new and self._ids:
            self.state.open_run(self._ids[event.new[0]])

    def __panel__(self):
        if self.data.empty:
            return pn.Column(
                pn.pane.Markdown("## Trends"),
                html(
                    '<div class="hint">No runs yet. Import one, or start the app with '
                    "<code>--demo</code> for a 12-week demo block.</div>"
                ),
            )
        return pn.Column(
            pn.pane.Markdown("## Trends"),
            pn.Row(self.category, self.structure, self.tag_filter, self.metric),
            self.chart,
            html('<div class="hint">Click a run in the table to open it.</div>'),
            self.table,
            sizing_mode="stretch_width",
        )
