"""The HR response section under the interval and tempo analyses, and a long run's pace blocks:
the chosen model fitted to each rep or block, its parameters part by part, and how well each
model fits."""

from html import escape

import pandas as pd
import panel as pn

from stridebook.analysis.hr_fit import MODELS, HrFits
from stridebook.ui import format as fmt
from stridebook.ui import plots
from stridebook.ui.components import html, hv_pane, linked
from stridebook.ui.theme import palette

DEFAULT_MODEL = "exp_drift"
NCOLS = 3  # small multiples per row


class HrResponse(pn.viewable.Viewer):
    """`part` names one rep or block in labels: 'Rep' or 'Block'."""

    def __init__(self, fits: HrFits, part: str, **params):
        super().__init__(**params)
        self._fits = fits
        self._part = part
        self.model_input = pn.widgets.RadioButtonGroup(
            options={m.label: name for name, m in MODELS.items()},
            value=DEFAULT_MODEL,
            color="primary",
            variant="outline",
        )
        self.body = pn.Column(self._body(), sizing_mode="stretch_width")
        self.model_input.param.watch(self._model_changed, "value")

    def _model_changed(self, _event) -> None:
        self.body.objects = [self._body()]

    def _body(self):
        name = self.model_input.value
        model = MODELS[name]
        params = self._fits.params[self._fits.params["model"] == name]
        c = palette()
        key = (
            f'<div class="meta">HR(t) = <b>{escape(model.formula)}</b>: '
            f"{escape(model.description)}. t is the time into the {self._part.lower()}."
            f'<span class="key" style="background:{c.hr}"></span>HR'
            f'<span class="key" style="background:{c.ink2}"></span>Model</div>'
        )
        items = [html(key), self._small_multiples(name)]
        if params["part_no"].nunique() > 1:
            items.append(self._by_part(params, name))
        items.append(_table(self._table(params, name)))
        return pn.Column(*items, sizing_mode="stretch_width")

    def _small_multiples(self, name: str):
        """Each part's HR with the model over it, on shared axes."""
        c = palette()
        curves = self._fits.curves.assign(
            model_hr=self._fits.curves[f"fit_{name}"],
            time=self._fits.curves["t_s"].map(fmt.duration),
        )
        values = pd.concat([curves["heart_rate"], curves["model_hr"]]).dropna()
        ylim = (float(values.min()) - 3, float(values.max()) + 3)
        xlim = (0, float(curves["t_s"].max()))
        into = f"Into {self._part.lower()}"
        charts = []
        for no, d in curves.groupby("part_no"):
            hr = plots.curve(
                d,
                "t_s",
                "heart_rate",
                color=c.hr,
                extra=["time", "model_hr"],
                tooltips=[
                    (into, "@time"),
                    ("HR", "@heart_rate{0} bpm"),
                    ("Model", "@model_hr{0} bpm"),
                ],
            ).opts(line_width=1.5, alpha=0.6)
            fitted = plots.curve(d, "t_s", "model_hr", color=c.ink2).opts(tools=[])
            charts.append(
                plots.style(
                    hr * fitted,
                    180,
                    title=f"{self._part} {no}",
                    xlim=xlim,
                    ylim=ylim,
                    xticks=plots.time_ticks(xlim[1]),
                    xformatter=plots.mmss(),
                    xlabel=f"{into} (min:sec)",
                    ylabel="HR (bpm)",
                )
            )
        return linked(*charts, ncols=min(NCOLS, len(charts)))

    def _by_part(self, params: pd.DataFrame, name: str):
        """The model's parameters rep by rep (or block by block)."""
        c = palette()
        # column, title, unit, colour, smallest y padding, tooltip format
        specs = [
            ("plateau_bpm", "Plateau", "bpm", c.hr, 2, "{0}"),
            ("tau_s", "Time constant τ", "s", c.other, 3, "{0}"),
        ]
        if name == "exp_drift":
            specs.append(("drift_bpm_per_min", "Drift", "bpm/min", c.other, 0.3, "{0.00}"))
        elif name == "stretched":
            specs.append(("beta", "Stretch β", "", c.other, 0.05, "{0.00}"))
        charts = []
        for column, title, unit, color, pad, number in specs:
            d = params[["part_no", column]].dropna()
            if d.empty:
                continue
            line = plots.curve(d, "part_no", column, color=color).opts(tools=[])
            dots = plots.dots(
                d,
                "part_no",
                column,
                color=color,
                tooltips=[(self._part, "@part_no"), (title, f"@{column}{number} {unit}")],
            )
            charts.append(
                hv_pane(
                    plots.style(
                        line * dots,
                        200,
                        title=title,
                        xlabel=self._part,
                        ylabel=unit,
                        xticks=d["part_no"].tolist(),
                        xlim=(d["part_no"].min() - 0.5, d["part_no"].max() + 0.5),
                        ylim=plots.padded_ylim(d[column], pad),
                    )
                )
            )
        return pn.Row(*charts, sizing_mode="stretch_width")

    def _table(self, params: pd.DataFrame, name: str) -> pd.DataFrame:
        table = pd.DataFrame(
            {
                self._part: params["part_no"],
                "HR at start": params["hr_start_bpm"].round().astype("Int64"),
                "Plateau": params["plateau_bpm"].round().astype("Int64"),
                "τ (s)": params["tau_s"].round().astype("Int64"),
            }
        )
        if name == "exp_drift":
            table["Drift (bpm/min)"] = params["drift_bpm_per_min"].round(2)
        elif name == "stretched":
            table["β"] = params["beta"].round(2)
        table["RMSE (bpm)"] = params["rmse_bpm"].round(1)
        return table

    def _comparison(self):
        """How well each model fits: BIC summed over the parts, and which model wins each."""
        p = self._fits.params.dropna(subset=["bic"])
        if p.empty:
            return pn.Column()
        parts = p["part_no"].nunique()
        total = p.groupby("model")["bic"].sum().reindex(list(MODELS))
        rmse = p.groupby("model")["rmse_bpm"].mean().reindex(list(MODELS))
        wins = p.loc[p.groupby("part_no")["bic"].idxmin(), "model"].value_counts()
        table = pd.DataFrame(
            {
                "Model": [m.label for m in MODELS.values()],
                "Mean RMSE (bpm)": rmse.round(2).to_numpy(),
                "ΔBIC": (total - total.min()).round().astype("Int64").to_numpy(),
                f"Lowest BIC, {self._part.lower()}s": [
                    f"{wins.get(name, 0)} of {parts}" for name in MODELS
                ],
            }
        )
        return pn.Column(
            pn.pane.Markdown("#### Which model fits best", margin=(10, 10, 0, 10)),
            html(
                '<div class="hint">BIC weighs the fit against the number of parameters. Lower '
                "is better, and a difference above about 10 is strong evidence for the lower "
                "one. ΔBIC is summed over all parts, relative to the best model.</div>"
            ),
            _table(table, width=640),
            sizing_mode="stretch_width",
        )

    def __panel__(self):
        part = self._part.lower()
        return pn.Column(
            pn.pane.Markdown("#### HR response", margin=(10, 10, 0, 10)),
            html(
                f'<div class="hint">A model fitted to the HR of each {part}, from its start. '
                "τ is how fast HR rises (the time to 63% of the rise), the plateau is where it "
                "levels off, and the drift is how much it keeps climbing. A stand-in fit until "
                "Stridelab does the fitting.</div>"
            ),
            self.model_input,
            self.body,
            self._comparison(),
            sizing_mode="stretch_width",
        )


def _table(df: pd.DataFrame, width: int | None = None) -> pn.widgets.Tabulator:
    sizing = {"width": width} if width else {"sizing_mode": "stretch_width"}
    return pn.widgets.Tabulator(
        df, show_index=False, disabled=True, layout="fit_data_stretch", **sizing
    )
