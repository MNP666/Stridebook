"""The sidebar: navigation and the list of saved runs."""

import pandas as pd
import panel as pn

from stridebook.domain import RunCategory
from stridebook.ui.state import AppState

ALL = "All"


class Library(pn.viewable.Viewer):
    def __init__(self, state: AppState, **params):
        super().__init__(**params)
        self.state = state
        self.new_run = pn.widgets.Button(
            label="New run", color="primary", sizing_mode="stretch_width"
        )
        self.trends = pn.widgets.Button(
            label="Trends", color="default", sizing_mode="stretch_width"
        )
        self.new_run.on_click(lambda _: setattr(state, "page", "import"))
        self.trends.on_click(lambda _: setattr(state, "page", "trends"))
        self.category = pn.widgets.Select(
            label="Show",
            options=[ALL] + [c.label for c in RunCategory],
            value=ALL,
            sizing_mode="stretch_width",
        )
        self.table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["Date", "Run"]),
            show_index=False,
            disabled=True,
            selectable=1,
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=560,
            widths={"Date": 90},
        )
        self._ids: list[str] = []
        self.table.param.watch(self._selected, "selection")
        self.category.param.watch(self._refresh, "value")
        state.param.watch(self._refresh, "revision")
        self._refresh()

    def _refresh(self, *_):
        runs = self.state.store.runs()
        if self.category.value != ALL and len(runs):
            label_to_value = {c.label: c.value for c in RunCategory}
            runs = runs[runs["category"] == label_to_value[self.category.value]]
        self._ids = runs["run_id"].tolist()
        self.table.value = pd.DataFrame(
            {
                "Date": [s.strftime("%d %b %y") for s in runs["start"]],
                "Run": runs["title"].tolist(),
            }
        )
        self.table.selection = []

    def _selected(self, event):
        if event.new:
            self.state.open_run(self._ids[event.new[0]])

    def __panel__(self):
        return pn.Column(
            pn.Row(self.new_run, self.trends, sizing_mode="stretch_width"),
            pn.pane.Markdown("#### Runs", margin=(10, 10, 0, 10)),
            self.category,
            self.table,
            sizing_mode="stretch_width",
        )
