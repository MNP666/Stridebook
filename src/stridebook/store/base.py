from typing import Protocol

import pandas as pd

from stridebook.domain import ProcessedRun

RUN_COLUMNS = [
    "run_id",
    "start",
    "category",
    "title",
    "structure",
    "tags",
    "distance_km",
    "duration_s",
    "avg_hr",
]


class RunStore(Protocol):
    """Where runs live. In memory for now; Parquet/DuckDB later behind the same methods.

    A durable store keeps three kinds of data apart: the raw file (source of truth), the user's
    inputs (RunMeta, segments) and everything derived (records, results, metrics), which can be
    rebuilt from the first two with pipeline.process.
    """

    def save(self, processed: ProcessedRun) -> None:
        """Insert or replace a run."""
        ...

    def get(self, run_id: str) -> ProcessedRun: ...

    def delete(self, run_id: str) -> None: ...

    def __contains__(self, run_id: object) -> bool: ...

    def runs(self) -> pd.DataFrame:
        """One row per run (RUN_COLUMNS), newest first."""
        ...

    def metrics(self) -> pd.DataFrame:
        """Every run's metrics, long format: run_id, start, category, structure, tags, part_no,
        metric, value, unit."""
        ...
