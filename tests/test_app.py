"""Smoke tests: every page and view builds its Bokeh models without errors."""

import panel as pn
import pytest
from bokeh.document import Document

from stridebook.analysis.hr_fit import MODELS
from stridebook.analysis.metrics import METRICS
from stridebook.app import create_app
from stridebook.sources.toy import ToySource
from stridebook.store.memory import MemoryStore
from stridebook.ui.pages.importer import ImportPage
from stridebook.ui.pages.run import RunPage
from stridebook.ui.pages.trends import TrendsPage
from stridebook.ui.state import AppState
from stridebook.ui.views.hr_response import HrResponse


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


def test_pages_render_in_two_sessions(state):
    """Each browser session has its own Bokeh document, and a model may belong to only one.
    A model shared between sessions (a module-level tick formatter, say) fails in the second."""
    runs = state.store.runs()
    run_id = runs.loc[runs["category"] == "interval", "run_id"].iloc[0]
    for _ in range(2):
        doc = Document()
        trends = TrendsPage(state)
        trends.structure.value = "5x6:00 r1:30"
        trends.metric.value = "rep_pace_s_per_km"
        for view in [*RunPage(state, run_id).__panel__()[1], trends]:
            doc.add_root(pn.panel(view).get_root(doc))


@pytest.mark.parametrize(("analysis", "part"), [("intervals", "Rep"), ("tempo", "Block")])
def test_hr_response_switches_model(state, analysis, part):
    runs = state.store.runs()
    category = "interval" if analysis == "intervals" else analysis
    run_id = runs.loc[runs["category"] == category, "run_id"].iloc[0]
    section = HrResponse(state.store.get(run_id).results[analysis].hr_fits, part)
    pn.panel(section).get_root()
    for name in MODELS:
        section.model_input.value = name
        pn.panel(section.body).get_root()


def test_trends_per_part(state):
    page = TrendsPage(state)
    page.structure.value = "5x6:00 r1:30"
    page.metric.value = "rep_pace_s_per_km"
    pn.panel(page).get_root()
    assert len(page.table.value) == 4


def test_trends_hr_model_choice(state):
    page = TrendsPage(state)
    page.structure.value = "5x6:00 r1:30"
    page.metric.value = "rep_hr_exp_drift_tau_s"
    assert page.hr_model.visible and page.hr_model.value == "exp_drift"
    # Each fitted value is listed once
    assert list(page.metric.options).count(METRICS["rep_hr_exp_drift_tau_s"].label) == 1
    page.hr_model.value = "stretched"
    assert page.metric.value == "rep_hr_stretched_tau_s"
    assert len(page.table.value) == 4
    # Drift only comes from the drift model, so that is the only model offered
    page.metric.value = "rep_hr_exp_drift_drift_bpm_per_min"
    assert list(page.hr_model.options.values()) == ["exp_drift"]
    page.metric.value = "rep_pace_s_per_km"
    assert not page.hr_model.visible


def test_import_and_save(state):
    page = ImportPage(state)
    page.file.filename = "x.fit"
    page.file.value = b"pretend fit bytes"
    pn.panel(page).get_root()
    page.editor.tags_input.value = "Hills, test"
    page.editor.save.clicks += 1
    assert state.page == "run"
    assert state.store.get(state.run_id).run.meta.tags == ["hills", "test"]
