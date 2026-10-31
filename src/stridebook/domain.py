"""Core types shared by every layer.

ParsedRun mirrors what Stridelab returns (records_df(), laps_df(), summary()), so a source can
swap the toy generator for `stridelab.read` without anything downstream changing.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import pandas as pd


class RunCategory(StrEnum):
    EASY = "easy"
    INTERVAL = "interval"
    TEMPO = "tempo"
    LONG_RUN = "long_run"
    RACE = "race"

    @property
    def label(self) -> str:
        return _CATEGORY_LABELS[self]


_CATEGORY_LABELS = {
    RunCategory.EASY: "Easy run",
    RunCategory.INTERVAL: "Interval session",
    RunCategory.TEMPO: "Tempo run",
    RunCategory.LONG_RUN: "Long run",
    RunCategory.RACE: "Race",
}


class SegmentRole(StrEnum):
    """FIT's lap `intensity` vocabulary. Coros leaves it out of its files, so we derive it."""

    WARMUP = "warmup"
    ACTIVE = "active"
    REST = "rest"
    COOLDOWN = "cooldown"


@dataclass
class ParsedRun:
    """One decoded activity, in Stridelab's shapes.

    records: UTC DatetimeIndex, FIT profile column names (heart_rate, speed, cadence in rpm,
             step_length in mm, ...), plus elapsed_s and lap_index.
    laps:    one row per lap, with message_index, start_time, timestamp (lap end), total_*, avg_*.
    summary: Stridelab's Activity.summary() keys (start, distance_m, timer_time_s, ...).
    """

    records: pd.DataFrame
    laps: pd.DataFrame
    summary: dict[str, Any]


@dataclass
class RunMeta:
    """What the user says about a run. Never overwritten by reprocessing."""

    run_id: str
    start: datetime
    category: RunCategory
    title: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    structure: str | None = None  # e.g. "5x6:00 r1:30", "6x1000m r2:30", "10k"
    source_name: str = ""


@dataclass
class Run:
    """A parsed file plus the user's inputs: metadata and confirmed segments."""

    meta: RunMeta
    parsed: ParsedRun
    segments: pd.DataFrame  # analysis.segments.SEGMENT_COLUMNS


@dataclass
class ProcessedRun:
    """A run with everything derived from it. Rebuildable from `run` at any time."""

    run: Run
    records: pd.DataFrame  # parsed records + derived columns (cadence_spm, pace_s_per_km, ...)
    results: dict[str, Any]  # analysis name -> that analysis's result dataclass
    metrics: pd.DataFrame  # analysis.metrics.METRIC_COLUMNS

    @property
    def run_id(self) -> str:
        return self.run.meta.run_id


def run_id_for(data: bytes) -> str:
    """Stable id from the file's bytes, so importing the same file twice is detected."""
    return hashlib.sha256(data).hexdigest()[:16]
