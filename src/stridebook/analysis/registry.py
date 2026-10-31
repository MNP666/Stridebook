"""Which analyses run for which category.

Each analysis is a pure `compute(records, segments, settings) -> result` plus a
`metrics(result) -> DataFrame` for the trends table. Views for them live in ui/views and are
looked up by the same name.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from stridebook.analysis import easy, intervals, long_run, overview, race, tempo
from stridebook.domain import RunCategory


@dataclass(frozen=True)
class Analysis:
    name: str
    label: str
    compute: Callable[..., Any]
    metrics: Callable[[Any], pd.DataFrame]


ANALYSES = {
    a.name: a
    for a in (
        Analysis("overview", "Overview", overview.compute, overview.metrics),
        Analysis("easy", "Aerobic efficiency", easy.compute, easy.metrics),
        Analysis("intervals", "Reps & recovery", intervals.compute, intervals.metrics),
        Analysis("tempo", "Threshold blocks", tempo.compute, tempo.metrics),
        Analysis("long_run", "Mechanics & drift", long_run.compute, long_run.metrics),
        Analysis("race", "Splits & pacing", race.compute, race.metrics),
    )
}

BY_CATEGORY = {
    RunCategory.EASY: ["overview", "easy"],
    RunCategory.INTERVAL: ["overview", "intervals"],
    RunCategory.TEMPO: ["overview", "tempo"],
    RunCategory.LONG_RUN: ["overview", "long_run"],
    RunCategory.RACE: ["overview", "race"],
}


def for_category(category: RunCategory) -> list[Analysis]:
    return [ANALYSES[name] for name in BY_CATEGORY[category]]
