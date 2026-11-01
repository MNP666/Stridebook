"""Smoke tests: every page and view builds its Bokeh models without errors."""

from types import SimpleNamespace

import panel as pn
import pytest
from bokeh.document import Document

from stridebook import pipeline
from stridebook.analysis.hr_fit import MODELS
from stridebook.analysis.metrics import METRICS
from stridebook.analysis.segments import burn_in_count
from stridebook.app import create_app
from stridebook.sources import toy
from stridebook.sources.toy import ToySource
from stridebook.store.memory import MemoryStore
from stridebook.ui.components import RunEditor
from stridebook.ui.pages.importer import ImportPage
from stridebook.ui.pages.run import RunPage
from stridebook.ui.pages.trends import TrendsPage
from stridebook.ui.state import AppState
from stridebook.ui.views.hr_response import HrResponse
from tests.conftest import START


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


@pytest.mark.parametrize("pace", ["MP", "HMP"])
def test_long_run_with_pace_blocks_builds(state, pace):
    runs = state.store.runs()
    run_id = runs.loc[runs["structure"].str.endswith(f" {pace}", na=False), "run_id"].iloc[0]
    assert state.store.get(run_id).results["long_run"].blocks is not None
    for tab in RunPage(state, run_id).__panel__()[1]:
        pn.panel(tab).get_root()


def test_editor_groups_long_run_laps(settings):
    parsed = toy.generate("long_run_mp", start=START, seed=2)
    saved = []
    editor = RunEditor(
        pipeline.new_run(parsed, run_id="x", settings=settings), saved.append, settings=settings
    )
    pn.panel(editor).get_root()
    assert editor.long_run.visible
    assert (editor.burn_in_input.value, editor.blocks_input.value) == (3, 2)
    assert editor.structure_input.value == "2x5000m MP"

    def roles():
        return "".join(r[0] for r in editor.table.value["Role"])

    editor.block_pace_input.value = "hmp"
    assert editor.structure_input.value == "2x5000m HMP"
    assert editor.title_input.value == "Long run 24 km with 2x5000m HMP"
    editor.blocks_input.value = 0
    assert "a" not in roles() and editor.structure_input.value == ""
    assert not editor.block_pace_input.visible
    editor.burn_in_input.value = 4
    assert roles().startswith("wwwws")
    editor.blocks_input.value = 2
    assert editor.structure_input.value == "2x5000m HMP"  # the block pace is remembered
    # Moving out of the long-run category hides the choices and proposes tempo roles
    editor.category_input.value = "tempo"
    assert not editor.long_run.visible and "s" not in roles()
    editor.category_input.value = "long_run"
    assert editor.blocks_input.value == 2
    assert editor.structure_input.value == "2x5000m HMP"
    # A role edited in the table moves the choices with it
    editor._role_edited(SimpleNamespace(row=3, value="warmup"))
    assert editor.burn_in_input.value == 4
    editor.save.clicks += 1
    assert saved[0].meta.block_pace == "hmp" and burn_in_count(saved[0].segments) == 4


def test_trends_long_run_blocks(state):
    page = TrendsPage(state)
    page.category.value = "long_run"
    page.structure.value = "3x3000m HMP"
    page.metric.value = "block_pace_s_per_km"
    pn.panel(page).get_root()
    assert list(page.table.value.columns) == ["Date", "Run", "Block 1", "Block 2", "Block 3"]


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
