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
from stridebook.analysis.segments import blocks, propose_segments, segment_of
from stridebook.analysis.structure import derive_structure
from stridebook.config import Settings
from stridebook.domain import ParsedRun, ProcessedRun, Run, RunCategory, RunMeta, SegmentRole
from stridebook.sources.toy import demo_history

RACE_HR_FRACTION = 0.88  # a steady run averaging this share of max HR looks like a race


def new_run(
    parsed: ParsedRun, *, run_id: str, source_name: str = "", settings: Settings | None = None
) -> Run:
    """Wrap a parsed file with proposed segments and a suggested category, title and structure."""
    settings = settings or Settings()
    segments = propose_segments(parsed.laps)
    category = suggest_category(parsed, segments, settings)
    structure = derive_structure(segments, category, parsed.summary.get("distance_m"))
    meta = RunMeta(
        run_id=run_id,
        start=parsed.summary["start"],
        category=category,
        title=default_title(category, structure, parsed.summary.get("distance_m")),
        structure=structure,
        source_name=source_name,
    )
    return Run(meta=meta, parsed=parsed, segments=segments)


def suggest_category(parsed: ParsedRun, segments: pd.DataFrame, settings: Settings) -> RunCategory:
    reps = blocks(segments, SegmentRole.ACTIVE)
    if (segments["role"] != SegmentRole.ACTIVE.value).any() and len(reps):
        return RunCategory.INTERVAL if reps["duration_s"].median() < 600 else RunCategory.TEMPO
    s = parsed.summary
    if (s.get("avg_heart_rate_bpm") or 0) >= RACE_HR_FRACTION * settings.max_hr:
        return RunCategory.RACE
    if (s.get("distance_m") or 0) >= 18_000 or (s.get("timer_time_s") or 0) >= 90 * 60:
        return RunCategory.LONG_RUN
    return RunCategory.EASY


def default_title(
    category: RunCategory, structure: str | None, distance_m: float | None = None
) -> str:
    """'Interval session 5x6:00 r1:30', or 'Long run 24 km' when there is no structure."""
    if structure:
        return f"{category.label} {structure}"
    if distance_m:
        return f"{category.label} {distance_m / 1000:.0f} km"
    return category.label


def restructure(run: Run, segments: pd.DataFrame | None = None, **meta_changes) -> Run:
    """A copy of the run with new segments and/or metadata, and its structure re-derived
    unless the caller sets one."""
    segments = run.segments if segments is None else segments
    meta = replace(run.meta, **meta_changes)
    if "structure" not in meta_changes:
        meta.structure = derive_structure(
            segments, meta.category, run.parsed.summary.get("distance_m")
        )
    return Run(meta=meta, parsed=run.parsed, segments=segments)


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
        run = restructure(run, category=category, tags=demo.tags, source_name="demo")
        run.meta.title = default_title(
            category, run.meta.structure, demo.parsed.summary["distance_m"]
        )
        out.append(process(run, settings))
    return out
