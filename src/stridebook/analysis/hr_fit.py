"""HR response within a rep or block: how fast HR rises, where it levels off and how much it
drifts, from a model fitted to each part. With t the seconds since the part started:

- exp:        HR₀ + A·(1 − exp(−t/τ))            rises to a plateau and stays there
- exp_drift:  HR₀ + A·(1 − exp(−t/τ)) + s·t      the same, plus a linear drift
- stretched:  HR₀ + A·(1 − exp(−(t/τ)^β))        β < 1 rises fast, then keeps creeping up

τ is the time to 63% of the rise in all three, and the plateau is HR₀ + A.

The fit here is a stand-in until Stridelab fits these models (with lmfit). τ and β are found
by a grid search; HR₀, A and s are linear, so least squares solves them at each grid point.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_SAMPLES = 30
FIT_STEP_S = 2  # fit every 2nd second: plenty for time constants of tens of seconds
TRACE_STEP_S = 2
TAU_MIN_S = 2.0
TAU_GRID_N = 60
BETA_GRID = np.linspace(0.3, 1.0, 15)


@dataclass(frozen=True)
class HrModel:
    label: str
    formula: str
    description: str
    n_params: int
    trended: tuple[str, ...]  # fit columns that become per-part metrics


MODELS = {
    "exp": HrModel(
        "Exponential",
        "HR₀ + A·(1 − exp(−t/τ))",
        "rises to a plateau and stays there",
        3,
        ("tau_s", "plateau_bpm"),
    ),
    "exp_drift": HrModel(
        "Exponential + drift",
        "HR₀ + A·(1 − exp(−t/τ)) + s·t",
        "rises to a plateau, then drifts up at s bpm/min",
        4,
        ("tau_s", "plateau_bpm", "drift_bpm_per_min"),
    ),
    "stretched": HrModel(
        "Stretched exponential",
        "HR₀ + A·(1 − exp(−(t/τ)^β))",
        "with β < 1, rises fast and then keeps creeping up ever more slowly",
        4,
        ("tau_s", "plateau_bpm", "beta"),
    ),
}

FIT_COLUMNS = [
    "hr_start_bpm",
    "rise_bpm",
    "plateau_bpm",
    "tau_s",
    "drift_bpm_per_min",
    "beta",
    "rmse_bpm",
    "bic",
]


@dataclass
class HrFits:
    params: pd.DataFrame  # part_no, model, *FIT_COLUMNS: one row per part and model
    curves: pd.DataFrame  # part_no, t_s, heart_rate, fit_exp, fit_exp_drift, fit_stretched


def fit_parts(records: pd.DataFrame, parts: pd.DataFrame, key: str) -> HrFits:
    """Every model fitted to the HR of each part (rep or block). `parts` has `key`, start, end."""
    rows, curves = [], []
    for no, s, e in zip(parts[key], parts["start"], parts["end"], strict=True):
        hr = records.loc[(records.index >= s) & (records.index < e), "heart_rate"]
        t = (hr.index - s).total_seconds().to_numpy()
        y = hr.to_numpy(dtype=float, na_value=np.nan)
        curve = pd.DataFrame({"part_no": no, "t_s": t, "heart_rate": y})
        for name in MODELS:
            p = fit(t[::FIT_STEP_S], y[::FIT_STEP_S], name)
            rows.append({"part_no": no, "model": name, **p})
            curve[f"fit_{name}"] = predict(t, p, name)
        curves.append(curve.iloc[::TRACE_STEP_S])
    curve_cols = ["part_no", "t_s", "heart_rate"] + [f"fit_{name}" for name in MODELS]
    return HrFits(
        params=pd.DataFrame(rows, columns=["part_no", "model", *FIT_COLUMNS]),
        curves=(
            pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=curve_cols)
        ),
    )


def metric_name(part: str, model: str, column: str) -> str:
    """The metric a fitted value is trended as: ('rep', 'exp_drift', 'tau_s') ->
    'rep_hr_exp_drift_tau_s'."""
    return f"{part}_hr_{model}_{column}"


def part_metrics(fits: HrFits, part: str) -> dict[str, list[float]]:
    """Each model's trended parameters part by part, for metrics.metric_frame."""
    out = {}
    for name, model in MODELS.items():
        p = fits.params[fits.params["model"] == name].sort_values("part_no")
        for column in model.trended:
            out[metric_name(part, name, column)] = p[column].tolist()
    return out


def fit(t_s: np.ndarray, hr: np.ndarray, model: str) -> dict[str, float]:
    """One model fitted to HR against seconds since the part started (all NaN if too short)."""
    t, y = np.asarray(t_s, dtype=float), np.asarray(hr, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < MIN_SAMPLES:
        return dict.fromkeys(FIT_COLUMNS, np.nan)
    drift = model == "exp_drift"
    # τ beyond half the part would put the plateau outside the data (and trade off with drift)
    taus = np.geomspace(TAU_MIN_S, max(2 * TAU_MIN_S, (t[-1] - t[0]) / 2), TAU_GRID_N)
    betas = BETA_GRID if model == "stretched" else np.array([1.0])
    tau, beta = _search(t, y, taus, betas, drift)
    tau, beta = _search(t, y, _around(taus, tau, log=True), _around(betas, beta), drift)

    x = np.column_stack([np.ones_like(t), _rise(t, tau, beta)] + ([t] if drift else []))
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    sse = float(((x @ coef - y) ** 2).sum())
    n = len(t)
    return {
        "hr_start_bpm": float(coef[0]),
        "rise_bpm": float(coef[1]),
        "plateau_bpm": float(coef[0] + coef[1]),
        "tau_s": tau,
        "drift_bpm_per_min": float(coef[2] * 60) if drift else np.nan,
        "beta": beta if model == "stretched" else np.nan,
        "rmse_bpm": float(np.sqrt(sse / n)),
        "bic": float(n * np.log(max(sse / n, 1e-12)) + MODELS[model].n_params * np.log(n)),
    }


def predict(t_s: np.ndarray, params: dict[str, float], model: str) -> np.ndarray:
    """The fitted model's HR at `t_s` (NaN where the fit failed)."""
    beta = params["beta"] if model == "stretched" else 1.0
    slope = params["drift_bpm_per_min"] / 60 if model == "exp_drift" else 0.0
    rise = params["rise_bpm"] * _rise(t_s, params["tau_s"], beta)
    return params["hr_start_bpm"] + rise + slope * t_s


def _rise(t, tau, beta=1.0):
    return 1 - np.exp(-((t / tau) ** beta))


def _search(t, y, taus, betas, drift: bool) -> tuple[float, float]:
    """The grid point (τ, β) whose least-squares fit leaves the smallest residual, solved for
    all grid points at once through the normal equations."""
    tau, beta = (a.ravel() for a in np.meshgrid(taus, betas))
    cols = [np.ones((len(t), 1)), _rise(t[:, None], tau, beta)]
    if drift:
        cols.append(t[:, None] / t.max())  # scaled for conditioning; the residual is the same
    p = len(cols)
    xtx, xty = np.empty((len(tau), p, p)), np.empty((len(tau), p))
    for i in range(p):
        xty[:, i] = (cols[i] * y[:, None]).sum(axis=0)
        for j in range(i, p):
            xtx[:, i, j] = xtx[:, j, i] = (cols[i] * cols[j]).sum(axis=0)
    coef = (np.linalg.pinv(xtx) @ xty[..., None])[..., 0]
    sse = y @ y - (coef * xty).sum(axis=1)
    best = int(np.argmin(sse))
    return float(tau[best]), float(beta[best])


def _around(grid: np.ndarray, value: float, n: int = 15, log: bool = False) -> np.ndarray:
    """A finer grid between the neighbours of `value` in `grid`."""
    if len(grid) == 1:
        return grid
    i = int(np.argmin(np.abs(grid - value)))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    return np.geomspace(lo, hi, n) if log else np.linspace(lo, hi, n)
