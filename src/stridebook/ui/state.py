import param

from stridebook import pipeline
from stridebook.config import Settings
from stridebook.domain import ProcessedRun, Run
from stridebook.sources.base import RunSource
from stridebook.store.base import RunStore


class AppState(param.Parameterized):
    """What the pages share: the store, the source, settings, and where the user is."""

    page = param.Selector(objects=["import", "run", "trends"], default="import")
    run_id = param.String(default=None, allow_None=True)
    revision = param.Integer(default=0, doc="Bumped whenever the store changes")

    def __init__(self, store: RunStore, source: RunSource, settings: Settings, **params):
        super().__init__(**params)
        self.store = store
        self.source = source
        self.settings = settings

    def open_run(self, run_id: str) -> None:
        self.param.update(run_id=run_id, page="run")

    def save(self, run: Run) -> ProcessedRun:
        """Process and store a run, then show it."""
        processed = pipeline.process(run, self.settings)
        self.store.save(processed)
        self.revision += 1
        self.open_run(run.meta.run_id)
        return processed

    def delete(self, run_id: str) -> None:
        self.store.delete(run_id)
        self.revision += 1
        self.param.update(run_id=None, page="import")

    def known_tags(self) -> list[str]:
        runs = self.store.runs()
        return sorted({t for tags in runs["tags"] for t in tags}) if len(runs) else []
