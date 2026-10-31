"""Smoke tests: every page and view builds its Bokeh models without errors."""

import panel as pn
import pytest

from stridebook.app import create_app
from stridebook.sources.toy import ToySource
from stridebook.store.memory import MemoryStore
from stridebook.ui.pages.importer import ImportPage
from stridebook.ui.pages.run import RunPage
from stridebook.ui.pages.trends import TrendsPage
from stridebook.ui.state import AppState


@pytest.fixture(scope="module")
def state(demo, settings):
    store = MemoryStore()
    for p in demo:
        store.save(p)
    return AppState(store=store, source=ToySource(), settings=settings)


def test_app_builds(state):
    template = create_app(state.store, state.source, state.settings)
    pn.panel(template.main[0]).get_root()
    pn.panel(template.sidebar[0]).get_root()


@pytest.mark.parametrize("category", ["easy", "interval", "tempo", "long_run", "race"])
def test_run_page_builds(state, category):
    runs = state.store.runs()
    run_id = runs.loc[runs["category"] == category, "run_id"].iloc[0]
    for tab in RunPage(state, run_id).__panel__()[1]:
        pn.panel(tab).get_root()


def test_trends_per_part(state):
    page = TrendsPage(state)
    page.structure.value = "5x6:00 r1:30"
    page.metric.value = "rep_pace_s_per_km"
    pn.panel(page).get_root()
    assert len(page.table.value) == 4


def test_import_and_save(state):
    page = ImportPage(state)
    page.file.filename = "x.fit"
    page.file.value = b"pretend fit bytes"
    pn.panel(page).get_root()
    page.editor.tags_input.value = "Hills, test"
    page.editor.save.clicks += 1
    assert state.page == "run"
    assert state.store.get(state.run_id).run.meta.tags == ["hills", "test"]
