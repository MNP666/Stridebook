from stridebook import pipeline
from stridebook.domain import RunCategory
from stridebook.store.base import RUN_COLUMNS
from stridebook.store.memory import MemoryStore


def test_runs_and_metrics_tables(demo):
    store = MemoryStore()
    for p in demo:
        store.save(p)
    runs = store.runs()
    assert list(runs.columns) == RUN_COLUMNS
    assert runs["start"].is_monotonic_decreasing
    m = store.metrics()
    reps = m[(m["structure"] == "5x6:00 r1:30") & (m["metric"] == "rep_pace_s_per_km")]
    assert set(reps["part_no"]) == {1, 2, 3, 4, 5}
    assert reps["run_id"].nunique() == 4  # every other Tuesday of 8 weeks


def test_reprocessing_keeps_user_inputs(demo, settings):
    store = MemoryStore()
    p = demo[0]
    store.save(p)
    edited = pipeline.restructure(p.run, tags=["hills"], notes="windy", category=RunCategory.TEMPO)
    store.save(pipeline.process(edited, settings))
    again = store.get(p.run_id)
    assert again.run.meta.tags == ["hills"]
    assert again.run.meta.notes == "windy"
    assert "tempo" in again.results
    assert len(store) == 1
