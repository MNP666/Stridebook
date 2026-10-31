"""One view per analysis, keyed by the analysis name in analysis.registry."""

from stridebook.ui.views import easy, intervals, long_run, overview, race, tempo

VIEWS = {
    "overview": overview.view,
    "easy": easy.view,
    "intervals": intervals.view,
    "tempo": tempo.view,
    "long_run": long_run.view,
    "race": race.view,
}
