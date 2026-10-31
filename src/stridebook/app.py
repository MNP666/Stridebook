"""The Panel app and the `stridebook` command."""

import argparse
from collections.abc import Sequence

import panel as pn

from stridebook import pipeline
from stridebook.config import Settings, load_settings, resolve_data_dir
from stridebook.sources.base import RunSource
from stridebook.sources.toy import ToySource
from stridebook.store.base import RunStore
from stridebook.store.memory import MemoryStore
from stridebook.ui.pages.importer import ImportPage
from stridebook.ui.pages.library import Library
from stridebook.ui.pages.run import RunPage
from stridebook.ui.pages.trends import TrendsPage
from stridebook.ui.state import AppState
from stridebook.ui.theme import palette

pn.extension("tabulator", notifications=True)


def create_app(store: RunStore, source: RunSource, settings: Settings):
    """One browser session's app. The store is shared by all sessions."""
    state = AppState(store=store, source=source, settings=settings)
    main = pn.Column(sizing_mode="stretch_width")

    def render(*_):
        if state.page == "run" and state.run_id in store:
            main.objects = [RunPage(state, state.run_id)]
        elif state.page == "trends":
            main.objects = [TrendsPage(state)]
        else:
            main.objects = [ImportPage(state)]

    state.param.watch(render, ["page", "run_id", "revision"])
    render()
    return pn.template.FastListTemplate(
        title="Stridebook",
        sidebar=[Library(state)],
        main=[main],
        accent=palette().pace,
        sidebar_width=340,
        theme_toggle=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stridebook", description=__doc__)
    parser.add_argument("--demo", action="store_true", help="start with a demo training block")
    parser.add_argument("--weeks", type=int, default=12, help="weeks of demo data (default 12)")
    parser.add_argument("--data-dir", help="data folder (default $STRIDEBOOK_DATA or ./data)")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--no-show", action="store_true", help="don't open a browser")
    args = parser.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    settings = load_settings(data_dir)
    store = MemoryStore()
    if args.demo:
        for processed in pipeline.demo_runs(settings, weeks=args.weeks):
            store.save(processed)
        print(f"Loaded {len(store)} demo runs")
    pn.serve(
        lambda: create_app(store, ToySource(), settings),
        port=args.port,
        show=not args.no_show,
        title="Stridebook",
    )
