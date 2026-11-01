"""From a parsed file to a processed run. The only module the UI calls to (re)compute anything.

    parsed --new_run--> Run (proposed segments, suggested category and structure)
           --user edits meta and segments-->
    Run --process--> ProcessedRun (derived records, results per analysis, metrics)

`process` is pure and cheap, so it reruns after every edit; the user's inputs live in Run and
are never touched by it.
"""

from dataclasses import replace

import pandas as pd

from stridebook.analysis.common import add_derived
from stridebook.analysis.metrics import METRIC_COLUMNS
from stridebook.analysis.registry import for_category
from stridebook.analysis.segments import (
    blocks,
    burn_in_laps,
    long_run_segments,
    propose_segments,
    segment_of,
    suggest_blocks,
)
from stridebook.analysis.structure import derive_structure
from stridebook.config import Settings
from stridebook.domain import (
    BlockPace,
    ParsedRun,
    ProcessedRun,
    Run,
    RunCategory,
    RunMeta,
    SegmentRole,
)
from stridebook.sources.toy import demo_history

RACE_HR_FRACTION = 0.88  # a steady run averaging this share of max HR looks like a race


def new_run(
    parsed: ParsedRun, *, run_id: str, source_name: str = "", settings: Settings | None = None
) -> Run:
    """Wrap a parsed file with proposed segments and a suggested category, title and structure."""
    settings = settings or Settings()
    category = suggest_category(parsed, propose_segments(parsed.laps), settings)
    segments = suggest_segments(parsed.laps, category, settings)
    block_pace = _block_pace(segments, category, None)
    distance_m = parsed.summary.get("distance_m")
    structure = derive_structure(segments, category, distance_m, block_pace)
    meta = RunMeta(
        run_id=run_id,
        start=parsed.summary["start"],
        category=category,
        title=default_title(category, structure, distance_m),
        structure=structure,
        block_pace=block_pace,
        source_name=source_name,
    )
    return Run(meta=meta, parsed=parsed, segments=segments)


def suggest_category(parsed: ParsedRun, segments: pd.DataFrame, settings: Settings) -> RunCategory:
    s = parsed.summary
    race_like = (s.get("avg_heart_rate_bpm") or 0) >= RACE_HR_FRACTION * settings.max_hr
    long = (s.get("distance_m") or 0) >= 18_000 or (s.get("timer_time_s") or 0) >= 90 * 60
    if long and not race_like:
        return RunCategory.LONG_RUN  # pace blocks in a long run don't make it a tempo run
    reps = blocks(segments, SegmentRole.ACTIVE)
    if (segments["role"] != SegmentRole.ACTIVE.value).any() and len(reps):
        return RunCategory.INTERVAL if reps["duration_s"].median() < 600 else RunCategory.TEMPO
    if race_like:
        return RunCategory.RACE
    return RunCategory.EASY


def suggest_segments(laps: pd.DataFrame, category: RunCategory, settings: Settings) -> pd.DataFrame:
    """Proposed segments for a category. A long run's laps are grouped into the burn-in, the
    pace blocks it seems to have, and steady running."""
    segments = propose_segments(laps)
    if category != RunCategory.LONG_RUN:
        return segments
    burn_in = burn_in_laps(segments, settings.burn_in_km)
    return long_run_segments(segments, suggest_blocks(segments, burn_in), burn_in)


def default_title(
    category: RunCategory, structure: str | None, distance_m: float | None = None
) -> str:
    """'Interval session 5x6:00 r1:30', 'Long run 24 km', 'Long run 26 km with 2x5000m MP'."""
    if category == RunCategory.LONG_RUN and distance_m:
        title = f"{category.label} {distance_m / 1000:.0f} km"
        return f"{title} with {structure}" if structure else title
    if structure:
        return f"{category.label} {structure}"
    if distance_m:
        return f"{category.label} {distance_m / 1000:.0f} km"
    return category.label


def restructure(
    run: Run,
    segments: pd.DataFrame | None = None,
    settings: Settings | None = None,
    **meta_changes,
) -> Run:
    """A copy of the run with new segments and/or metadata, and its structure re-derived
    unless the caller sets one.

    Long runs have roles of their own, so moving a run into or out of the long-run category
    proposes its segments afresh (unless the caller gives them).
    """
    meta = replace(run.meta, **meta_changes)
    if segments is None:
        segments = run.segments
        if (meta.category == RunCategory.LONG_RUN) != (run.meta.category == RunCategory.LONG_RUN):
            segments = suggest_segments(run.parsed.laps, meta.category, settings or Settings())
    meta.block_pace = _block_pace(segments, meta.category, meta.block_pace)
    if "structure" not in meta_changes:
        meta.structure = derive_structure(
            segments, meta.category, run.parsed.summary.get("distance_m"), meta.block_pace
        )
    return Run(meta=meta, parsed=run.parsed, segments=segments)


def group_long_run(
    run: Run, n_blocks: int, burn_in: int, block_pace: BlockPace | None = None
) -> Run:
    """A long run with its laps grouped afresh: `burn_in` laps of burn-in, then `n_blocks`
    pace blocks found from lap paces, and steady running."""
    return restructure(
        run,
        segments=long_run_segments(run.segments, n_blocks, burn_in),
        block_pace=block_pace or run.meta.block_pace,
    )


def _block_pace(
    segments: pd.DataFrame, category: RunCategory, current: BlockPace | str | None
) -> BlockPace | None:
    """A long run with pace blocks keeps its block pace (marathon pace unless set); any other
    run has none."""
    if category != RunCategory.LONG_RUN or not (segments["role"] == SegmentRole.ACTIVE.value).any():
        return None
    return BlockPace(current) if current else BlockPace.MARATHON


def process(run: Run, settings: Settings) -> ProcessedRun:
    records = add_derived(run.parsed.records)
    records["segment_no"] = segment_of(records, run.segments)
    results, metrics = {}, []
    for analysis in for_category(run.meta.category):
        result = analysis.compute(records, run.segments, settings)
        results[analysis.name] = result
        metrics.append(analysis.metrics(result))
    table = (
        pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(columns=METRIC_COLUMNS)
    )
    return ProcessedRun(run=run, records=records, results=results, metrics=table)


def demo_runs(settings: Settings, weeks: int = 12, seed: int = 0) -> list[ProcessedRun]:
    """A processed demo training block, as if each run had been imported and labelled."""
    out = []
    for demo in demo_history(weeks, seed=seed):
        run = new_run(demo.parsed, run_id=f"demo-{demo.seed:08x}", settings=settings)
        category = demo.workout.category
        run = restructure(
            run,
            settings=settings,
            category=category,
            block_pace=demo.workout.block_pace,
            tags=demo.tags,
            source_name="demo",
        )
        run.meta.title = default_title(
            category, run.meta.structure, demo.parsed.summary["distance_m"]
        )
        out.append(process(run, settings))
    return out
