# Stridebook

A local, single-user Panel dashboard for running workouts from `.fit` files. Other people clone it from GitHub and run it themselves; nothing is hosted. The sibling project Stridelab (`../Stridelab`) will decode the files and hold the processing that proves useful. Phase 1 is toy data only: no Stridelab import, and no Parquet/DuckDB files are written.

## Commands

```bash
uv sync
uv run stridebook --demo                       # app with 12 weeks of demo runs on :5006
uv run pytest -q
uv run ruff check --fix . && uv run ruff format .
```

Use uv for everything (`uv add`, `uv add --dev`, `uv run`), never pip.

## Rules

1. **Two sides, one direction.** `domain`, `config`, `sources`, `analysis`, `pipeline` and `store` never import panel, param, holoviews, hvplot or bokeh (ruff `TID251` enforces it). `ui/` and `app.py` may import anything. The UI changes data only through `pipeline` and `AppState`.
2. **Analysis code follows Stridelab's rules**, so it can be moved there unchanged:
   - pandas, numpy and scipy only; DataFrame or Series in and out
   - time-based windows; NaN-preserving
   - FIT profile column names; derived columns carry a unit suffix (`cadence_spm`, `pace_s_per_km`)
3. **The toy source mirrors Stridelab's output** (`records_df()`, `laps_df()`, `summary()`: UTC DatetimeIndex, `cadence` as one-foot rpm, `step_length` in mm, `lap_index`). `tests/test_toy.py` pins this format. Swapping in Stridelab must change nothing downstream.
4. **User inputs vs derived data.**
   - `RunMeta` and the confirmed segments are the user's inputs.
   - `ProcessedRun` (derived records, results, metrics) is rebuilt by `pipeline.process`, which must never change the inputs.
   - A durable store keeps the raw file, the inputs and the derived data apart.
5. **Metrics are long format** (`part_no, metric, value, unit`). `part_no` is the rep or block number, or `<NA>` for the whole run. Every metric name must be registered in `analysis/metrics.METRICS`.
6. **A new analysis** is three things:
   - a module in `analysis/` with `compute(records, segments, settings)` and `metrics(result)`
   - an entry in `analysis/registry.py`
   - a view in `ui/views/` registered in `VIEWS` under the same name

## UI conventions

- **Colours:** the tokens in `ui/theme.py`. Pace is blue, HR orange, everything else aqua. Reps and blocks use `ordinal(n)`. Never put two y-scales on one chart: stack HR and pace instead.
- **HoloViews panes:** build them with `components.hv_pane` (axes not linked) or `components.linked(...)` (separate panes with a shared x-axis). Don't use HoloViews `Layout`: its Bokeh grid sometimes gets a zero-size frame inside a run-page tab.
- **Panel 1.10 renames:**
  - widgets take `label=`, not `name=`
  - buttons take `color=` / `variant=`, not `button_type` / `button_style`
  - a `Viewer` already has a `name` param, and a Column a `tags` param, so don't reuse those as attribute names
- **Map tiles** are OpenStreetMap. Carto and Stadia tiles now need an API key.
