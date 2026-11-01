"""One run: header, the overview, the category's analyses, and an Edit tab."""

from html import escape

import panel as pn

from stridebook.analysis.registry import ANALYSES
from stridebook.domain import Run
from stridebook.ui import format as fmt
from stridebook.ui.components import RunEditor, chips, html
from stridebook.ui.state import AppState
from stridebook.ui.views import VIEWS


class RunPage(pn.viewable.Viewer):
    def __init__(self, state: AppState, run_id: str, **params):
        super().__init__(**params)
        self.state = state
        self.processed = state.store.get(run_id)

    def _header(self):
        meta = self.processed.run.meta
        summary = self.processed.run.parsed.summary
        parts = [
            f"<b>{escape(meta.category.label)}</b>",
            escape(fmt.when(meta.start, summary.get("utc_offset"))),
        ]
        if meta.structure:
            parts.append(f"structure <b>{escape(meta.structure)}</b>")
        notes = (
            f'<p class="meta" style="white-space:pre-wrap">{escape(meta.notes)}</p>'
            if meta.notes
            else ""
        )
        return pn.Column(
            pn.pane.Markdown(f"## {escape(meta.title or meta.category.label)}", margin=(0, 10)),
            html(f'<div class="meta">{" · ".join(parts)} {chips(meta.tags)}</div>{notes}'),
            sizing_mode="stretch_width",
        )

    def _edit_tab(self):
        def save(run: Run):
            self.state.save(run)
            pn.state.notifications.success("Saved and reprocessed", duration=2000)

        delete = pn.widgets.Button(label="Delete run", color="danger", variant="outline", width=140)
        confirm = pn.widgets.Button(
            label="Yes, delete it", color="danger", width=140, visible=False
        )
        delete.on_click(lambda _: setattr(confirm, "visible", True))
        confirm.on_click(lambda _: self.state.delete(self.processed.run_id))
        editor = RunEditor(
            self.processed.run,
            on_save=save,
            save_label="Save changes",
            known_tags=self.state.known_tags(),
            settings=self.state.settings,
        )
        return pn.Column(
            editor, pn.layout.Divider(), pn.Row(delete, confirm), sizing_mode="stretch_width"
        )

    def __panel__(self):
        p, settings = self.processed, self.state.settings
        tabs = [
            (ANALYSES[name].label, VIEWS[name](p, result, settings))
            for name, result in p.results.items()
        ]
        tabs.append(("Edit", self._edit_tab()))
        return pn.Column(
            self._header(),
            pn.Tabs(*tabs, dynamic=True, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
