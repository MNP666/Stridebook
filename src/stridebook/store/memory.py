import pandas as pd

from stridebook.analysis.metrics import METRIC_COLUMNS
from stridebook.domain import ProcessedRun
from stridebook.store.base import RUN_COLUMNS


class MemoryStore:
    """A RunStore that keeps everything in a dict. Gone when the app stops."""

    def __init__(self) -> None:
        self._runs: dict[str, ProcessedRun] = {}
        self._runs_df: pd.DataFrame | None = None
        self._metrics_df: pd.DataFrame | None = None

    def save(self, processed: ProcessedRun) -> None:
        self._runs[processed.run_id] = processed
        self._invalidate()

    def get(self, run_id: str) -> ProcessedRun:
        return self._runs[run_id]

    def delete(self, run_id: str) -> None:
        del self._runs[run_id]
        self._invalidate()

    def __contains__(self, run_id: object) -> bool:
        return run_id in self._runs

    def __len__(self) -> int:
        return len(self._runs)

    def runs(self) -> pd.DataFrame:
        if self._runs_df is None:
            rows = [_run_row(p) for p in self._runs.values()]
            df = pd.DataFrame(rows, columns=RUN_COLUMNS)
            self._runs_df = df.sort_values("start", ascending=False, ignore_index=True)
        return self._runs_df

    def metrics(self) -> pd.DataFrame:
        if self._metrics_df is None:
            keys = ["run_id", "start", "category", "structure", "tags"]
            parts = [p.metrics.assign(run_id=p.run_id) for p in self._runs.values()]
            if parts:
                long = pd.concat(parts, ignore_index=True)
                self._metrics_df = long.merge(self.runs()[keys], on="run_id")[keys + METRIC_COLUMNS]
            else:
                self._metrics_df = pd.DataFrame(columns=keys + METRIC_COLUMNS)
        return self._metrics_df

    def _invalidate(self) -> None:
        self._runs_df = self._metrics_df = None


def _run_row(p: ProcessedRun) -> dict:
    m = p.run.meta
    overview = p.results.get("overview")
    return {
        "run_id": m.run_id,
        "start": m.start,
        "category": m.category.value,
        "title": m.title,
        "structure": m.structure,
        "tags": list(m.tags),
        "distance_km": overview.distance_m / 1000 if overview else None,
        "duration_s": overview.duration_s if overview else None,
        "avg_hr": overview.avg_hr if overview else None,
    }
