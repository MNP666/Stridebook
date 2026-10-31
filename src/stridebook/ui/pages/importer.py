"""The landing page: upload a .fit file (or generate a demo run), label it, save it."""

from datetime import UTC, datetime, timedelta

import numpy as np
import panel as pn

from stridebook import pipeline
from stridebook.domain import ParsedRun, run_id_for
from stridebook.sources import toy
from stridebook.ui import format as fmt
from stridebook.ui.components import RunEditor, html, tiles
from stridebook.ui.state import AppState


class ImportPage(pn.viewable.Viewer):
    def __init__(self, state: AppState, **params):
        super().__init__(**params)
        self.state = state
        self.file = pn.widgets.FileInput(accept=".fit", multiple=False)
        self.file.param.watch(self._uploaded, "value")
        self.demo = pn.widgets.Select(
            label="Demo workout", options={v: k for k, v in toy.WORKOUTS.items()}, width=240
        )
        self.generate = pn.widgets.Button(label="Generate", color="default", align="end")
        self.generate.on_click(self._generate)
        self.draft = pn.Column(sizing_mode="stretch_width")
        self.editor: RunEditor | None = None

    def _uploaded(self, event):
        if not event.new:
            return
        data, name = event.new, self.file.filename
        run_id = run_id_for(data)
        if run_id in self.state.store:
            self._already_imported(run_id, name)
            return
        self._open(self.state.source.parse(data, name), run_id, name)

    def _generate(self, _):
        seed = int(np.random.default_rng().integers(2**31))
        start = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(hours=2)
        parsed = toy.generate(self.demo.value, start=start, seed=seed)
        self._open(parsed, f"toy-{seed:08x}", f"{toy.WORKOUTS[self.demo.value]} (demo)")

    def _already_imported(self, run_id: str, name: str):
        button = pn.widgets.Button(label="Open it", color="primary", width=120)
        button.on_click(lambda _: self.state.open_run(run_id))
        self.draft.objects = [
            html(f'<div class="meta"><b>{name}</b> is already imported.</div>'),
            button,
        ]

    def _open(self, parsed: ParsedRun, run_id: str, name: str):
        run = pipeline.new_run(
            parsed, run_id=run_id, source_name=name, settings=self.state.settings
        )
        s = parsed.summary
        day, clock = fmt.when(s["start"], s.get("utc_offset")).split(", ")
        summary = tiles(
            [
                ("Date", day, clock),
                ("Distance", f"{fmt.number((s['distance_m'] or 0) / 1000, 2)} km"),
                ("Time", fmt.duration(s["timer_time_s"])),
                ("Average HR", f"{fmt.number(s['avg_heart_rate_bpm'])} bpm"),
                ("Laps", fmt.number(s["laps"])),
                ("Device", s.get("device") or "–"),
            ]
        )
        self.editor = RunEditor(run, on_save=self.state.save, known_tags=self.state.known_tags())
        self.draft.objects = [
            pn.pane.Markdown(f"### {name}"),
            summary,
            html(
                f'<div class="hint">Suggested category: <b>{run.meta.category.label}</b>. '
                "Change it below if it's wrong.</div>"
            ),
            self.editor,
        ]

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("## Import a run"),
            html(
                '<div class="hint">Phase 1: files are not decoded yet. The toy source turns any '
                "uploaded file into a synthetic run (the same file always gives the same run). "
                "Stridelab will replace it.</div>"
            ),
            pn.Row(
                pn.Column(pn.pane.Markdown("**Upload a .fit file**"), self.file),
                pn.Spacer(width=40),
                pn.Column(
                    pn.pane.Markdown("**…or generate a demo run**"),
                    pn.Row(self.demo, self.generate),
                ),
            ),
            self.draft,
            sizing_mode="stretch_width",
        )
