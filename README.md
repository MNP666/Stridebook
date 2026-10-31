# Stridebook

A local [Panel](https://panel.holoviz.org) dashboard for analysing running workouts from `.fit` files and trending them over time. Processing will come from [Stridelab](../Stridelab); for now a toy data source stands in for it.

- **Import:** upload a `.fit` file (or generate a demo workout), confirm the suggested category, title, tags, notes and the role of each lap (warmup / active / rest / cooldown).
- **Run pages:** an overview for every run, plus analysis by category:
  - **Interval:** rep paces, HR in each rep, HR recovery in each rest.
  - **Tempo:** HR and pace drift through each threshold block.
  - **Long run:** per-km mechanics (cadence, step length, vertical oscillation and ratio, stance time) and cardiac drift.
  - **Race:** splits and pacing evenness.
  - **Easy:** aerobic efficiency.
- **Trends:** any metric over time, narrowed by category, workout structure (`5x6:00 r1:30`, `6x1000m r2:30`, `10k`) and tags. With a structure chosen, per-rep metrics show rep 1, rep 2, … as separate lines.

## Running it

```bash
uv sync
uv run stridebook --demo        # 12 weeks of demo training; opens http://localhost:5006
uv run stridebook               # empty, ready to import
uv run stridebook --help        # --weeks, --port, --data-dir, --no-show
```

**Phase 1 status:**
- Uploaded files are not decoded yet: the toy source turns any file into a synthetic run (the same file always gives the same run).
- Nothing is written to disk: runs live in memory until the app stops.

## Personal settings

Copy `settings.example.toml` to `data/settings.toml` and set your max HR, HR zones and the reference pace for easy runs. The format is a superset of Stridelab's `hr_zones.toml`, so that file can be copied in as is.

The data folder is `./data` by default. Change it with `--data-dir` or `$STRIDEBOOK_DATA`. `data/` is gitignored, so routes and personal data never end up in the repository.

## Layout

```
src/stridebook/
  domain.py        RunCategory, SegmentRole, ParsedRun, RunMeta, Run, ProcessedRun
  config.py        data folder and settings.toml
  sources/         .fit bytes → ParsedRun. toy.py now; a Stridelab adapter later
  analysis/        pure pandas: segments, structure, metrics, one module per category, registry
  pipeline.py      new_run (suggest segments/category/structure) and process (run every analysis)
  store/           RunStore interface; memory.py now, Parquet/DuckDB later
  ui/              Panel only: theme, plots, components, pages/, views/ (one view per analysis)
  app.py           create_app() and the `stridebook` command
tests/             toy-data shape, analyses, store, and smoke tests that build every page
```

The processing side (everything but `ui/` and `app.py`) never imports Panel, HoloViews or Bokeh, and ruff enforces this (`TID251`). Analysis code follows Stridelab's rules, so it can move into Stridelab unchanged once it has proven useful.

## Development

```bash
uv run pytest -q
uv run ruff check --fix . && uv run ruff format .
```
