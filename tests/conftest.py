from datetime import UTC, datetime

import pytest

from stridebook import pipeline
from stridebook.config import Settings
from stridebook.domain import ProcessedRun
from stridebook.sources import toy

START = datetime(2026, 10, 6, 7, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


def make(name: str, settings: Settings, seed: int = 1, **kwargs) -> ProcessedRun:
    parsed = toy.generate(name, start=START, seed=seed, **kwargs)
    run = pipeline.new_run(parsed, run_id=f"test-{name}", settings=settings)
    return pipeline.process(run, settings)


@pytest.fixture(scope="session")
def demo(settings) -> list[ProcessedRun]:
    return pipeline.demo_runs(settings, weeks=8)
