"""Reusable UI pieces: stat tiles, tag chips, and the run editor used on import and on the
run page's Edit tab."""

from collections.abc import Callable
from html import escape

import pandas as pd
import panel as pn

from stridebook.analysis.common import add_derived
from stridebook.analysis.segments import MAX_BLOCKS, blocks, burn_in_count, segment_summary
from stridebook.config import Settings
from stridebook.domain import BlockPace, Run, RunCategory, SegmentRole
from stridebook.pipeline import default_title, group_long_run, restructure
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.theme import stylesheet


def html(body: str, **kwargs) -> pn.pane.HTML:
    return pn.pane.HTML(body, stylesheets=[stylesheet()], sizing_mode="stretch_width", **kwargs)


def hv_pane(obj=None, **kwargs) -> pn.pane.HoloViews:
    """A HoloViews pane whose axes stay its own. By default Panel links the axes of every pane
    on the page that shares a dimension name, which ties unrelated charts together (rep and
    rest HR overlays, say)."""
    kwargs.setdefault("sizing_mode", "stretch_width")
    kwargs.setdefault("linked_axes", False)
    return pn.pane.HoloViews(obj, **kwargs)


def linked(*elements, ncols: int = 1) -> pn.GridBox:
    """Charts stacked (or in a grid) as separate panes whose shared x-axis pans and zooms
    together. Used instead of a HoloViews Layout, whose Bokeh grid is sometimes laid out with
    a zero-size frame inside a tab."""
    panes = [hv_pane(e, linked_axes=True) for e in elements]
    return pn.GridBox(*panes, ncols=ncols, sizing_mode="stretch_width")


def tiles(items: list[tuple[str, str] | tuple[str, str, str]]) -> pn.pane.HTML:
    """Stat tiles: (label, value) or (label, value, note). A numeric value may end in a unit
    after a space ('12.1 km'), which is set smaller."""
    cells = []
    for label, value, *note in items:
        number, _, unit = value.partition(" ")
        if not number or not (number[0].isdigit() or number[0] in "+-–"):
            number, unit = value, ""
        unit_html = f'<span class="unit">{escape(unit)}</span>' if unit else ""
        note_html = f'<div class="note">{escape(note[0])}</div>' if note else ""
        cells.append(
            f'<div class="tile"><div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(number)}{unit_html}</div>{note_html}</div>'
        )
    return html(f'<div class="tiles">{"".join(cells)}</div>')


def chips(tags: list[str]) -> str:
    return "".join(f'<span class="chip">{escape(t)}</span>' for t in tags)


def parse_tags(text: str) -> list[str]:
    seen = []
    for t in text.split(","):
        t = t.strip().lower()
        if t and t not in seen:
            seen.append(t)
    return seen


CATEGORY_OPTIONS = {c.label: c.value for c in RunCategory}
ROLE_VALUES = [r.value for r in SegmentRole]
MAX_BURN_IN_LAPS = 6


class RunEditor(pn.viewable.Viewer):
    """Category, title, structure, tags, notes and segment roles of one run, with a live
    preview. Editing a role re-derives the structure; nothing is saved until `on_save`.

    A long run also gets its burn-in, number of pace blocks and block pace. Changing the first
    two regroups the laps by pace; roles edited in the table update them in turn.
    """

    def __init__(
        self,
        run: Run,
        on_save: Callable[[Run], None],
        save_label: str = "Save run",
        known_tags: list[str] | None = None,
        settings: Settings | None = None,
        **params,
    ):
        super().__init__(**params)
        self._run = run
        self._on_save = on_save
        self._settings = settings or Settings()
        self._records = add_derived(run.parsed.records)
        self._distance_m = run.parsed.summary.get("distance_m")
        meta = run.meta
        self._auto_title = meta.title == default_title(
            meta.category, meta.structure, self._distance_m
        )

        self.category_input = pn.widgets.RadioButtonGroup(
            label="Category",
            options=CATEGORY_OPTIONS,
            value=meta.category.value,
            color="primary",
            variant="outline",
        )
        self.title_input = pn.widgets.TextInput(label="Title", value=meta.title)
        self.structure_input = pn.widgets.TextInput(
            label="Structure",
            value=meta.structure or "",
            placeholder="derived from the segments, e.g. 5x6:00 r1:30",
        )
        hint = f"Known tags: {', '.join(known_tags)}" if known_tags else "e.g. track, heat, hills"
        self.tags_input = pn.widgets.TextInput(
            label="Tags (comma-separated)", value=", ".join(meta.tags), placeholder=hint
        )
        self.notes_input = pn.widgets.TextAreaInput(label="Notes", value=meta.notes, height=90)
        self.table = pn.widgets.Tabulator(
            self._segment_frame(),
            show_index=False,
            layout="fit_data_stretch",
            editors={c: None for c in self._segment_frame().columns}
            | {"Role": {"type": "list", "values": ROLE_VALUES}},
            height=min(420, 34 * (len(run.segments) + 1) + 10),
            width=560,
        )
        self.table.on_edit(self._role_edited)
        self.preview = pn.Column(self._preview(), sizing_mode="stretch_width")
        self.save = pn.widgets.Button(label=save_label, color="primary", width=160)
        self.save.on_click(self._save)
        self.category_input.param.watch(self._category_changed, "value")

        self.burn_in_input = pn.widgets.Select(label="Burn-in", width=190)
        self.blocks_input = pn.widgets.Select(label="Pace blocks", width=130)
        self.block_pace_input = pn.widgets.Select(
            label="Block pace",
            options={p.label: p.value for p in BlockPace},
            value=(meta.block_pace or BlockPace.MARATHON).value,
            width=190,
        )
        self.long_run = pn.Column(
            pn.Row(self.burn_in_input, self.blocks_input, self.block_pace_input),
            html(
                '<div class="hint">Laps are grouped by pace: the burn-in first, then the '
                "fastest runs of laps as pace blocks, and the rest as steady running. Drift and "
                "mechanics use the steady running only.</div>"
            ),
            sizing_mode="stretch_width",
        )
        self._syncing = False
        self._sync_long_run()
        self.burn_in_input.param.watch(self._regroup, "value")
        self.blocks_input.param.watch(self._regroup, "value")
        self.block_pace_input.param.watch(self._block_pace_changed, "value")

    def _segment_frame(self) -> pd.DataFrame:
        s = segment_summary(self._records, self._run.segments)
        t0 = self._records.index[0]
        return pd.DataFrame(
            {
                "#": s["segment_no"],
                "Start": [fmt.duration((pd.Timestamp(x) - t0).total_seconds()) for x in s["start"]],
                "Duration": s["duration_s"].map(fmt.duration),
                "Distance (m)": s["distance_m"].round().astype(int),
                "Pace": s["pace_s_per_km"].map(fmt.pace),
                "Avg HR": s["avg_hr"].round().astype("Int64"),
                "Role": s["role"],
            }
        )

    def _preview(self):
        return linked(*plots.time_traces(self._records, self._run.segments, height=170))

    def _role_edited(self, event) -> None:
        segments = self._run.segments.copy()
        segments.loc[segments.index[event.row], "role"] = event.value
        self._run = restructure(
            self._run, segments=segments, category=RunCategory(self.category_input.value)
        )
        self._sync_derived()
        self._sync_long_run()
        self.preview.objects = [self._preview()]

    def _category_changed(self, event) -> None:
        # Moving into or out of the long-run category proposes the roles afresh. The block pace
        # choice is kept, for a run that comes back to being a long run.
        self._run = restructure(
            self._run,
            settings=self._settings,
            category=RunCategory(event.new),
            block_pace=BlockPace(self.block_pace_input.value),
        )
        self._segments_changed()

    def _regroup(self, _event) -> None:
        if self._syncing:
            return
        self._run = group_long_run(
            self._run,
            n_blocks=self.blocks_input.value,
            burn_in=self.burn_in_input.value,
            block_pace=BlockPace(self.block_pace_input.value),
        )
        self._segments_changed()

    def _block_pace_changed(self, event) -> None:
        if self._run.meta.block_pace and not self._syncing:
            self._run = restructure(self._run, block_pace=BlockPace(event.new))
            self._sync_derived()

    def _segments_changed(self) -> None:
        self.table.value = self._segment_frame()
        self.preview.objects = [self._preview()]
        self._sync_derived()
        self._sync_long_run()

    def _sync_derived(self) -> None:
        meta = self._run.meta
        self.structure_input.value = meta.structure or ""
        if self._auto_title:
            self.title_input.value = default_title(meta.category, meta.structure, self._distance_m)

    def _sync_long_run(self) -> None:
        """Show the long-run choices for a long run, set to what its roles say."""
        segments = self._run.segments
        burn_in = burn_in_count(segments)
        n_blocks = len(blocks(segments, SegmentRole.ACTIVE))
        km = segments["distance_m"].cumsum() / 1000
        laps = range(1, max(min(MAX_BURN_IN_LAPS, len(segments) - 1), burn_in) + 1)
        self._syncing = True
        try:
            self.burn_in_input.param.update(
                options={"None": 0}
                | {f"{n} lap{'s' * (n > 1)} · {km.iloc[n - 1]:.1f} km": n for n in laps},
                value=burn_in,
            )
            self.blocks_input.param.update(
                options={"None": 0} | {str(n): n for n in range(1, max(MAX_BLOCKS, n_blocks) + 1)},
                value=n_blocks,
            )
            if self._run.meta.block_pace:
                self.block_pace_input.value = self._run.meta.block_pace.value
        finally:
            self._syncing = False
        self.block_pace_input.visible = n_blocks > 0
        self.long_run.visible = self._run.meta.category == RunCategory.LONG_RUN

    def _save(self, _event) -> None:
        run = restructure(
            self._run,
            category=RunCategory(self.category_input.value),
            title=self.title_input.value.strip(),
            structure=self.structure_input.value.strip() or None,
            tags=parse_tags(self.tags_input.value),
            notes=self.notes_input.value,
        )
        self._on_save(run)

    def __panel__(self):
        return pn.Column(
            self.category_input,
            pn.Row(
                self.title_input, self.structure_input, self.tags_input, sizing_mode="stretch_width"
            ),
            self.notes_input,
            pn.pane.Markdown("#### Segments", margin=(10, 10, 0, 10)),
            html(
                '<div class="hint">Roles are suggested from lap speeds. Correct any that are '
                "wrong: the analyses use them, and so does the structure.</div>"
            ),
            self.long_run,
            pn.Row(self.table, self.preview, sizing_mode="stretch_width"),
            self.save,
            sizing_mode="stretch_width",
        )
