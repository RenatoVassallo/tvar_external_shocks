"""Estimation of one Threshold VAR, its generalized impulse responses and the pass-through.

All the econometrics is MacroPy's ``ThresholdVAR`` (exact posterior over the
regime split, Normal-inverse-Wishart draws per regime, generalized impulse
responses with regime switching along the horizon). This module only maps a
:class:`ModelSpec` to its arguments and summarises the draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from MacroPy import ThresholdVAR

from .config import ModelSpec

__all__ = ["fit", "girfs", "girfs_by_band", "regime_irfs", "pass_through", "asymmetry", "regime_table",
           "pandemic_residuals"]


def fit(spec: ModelSpec, data: pd.DataFrame, horizon: int, progress: bool = False) -> ThresholdVAR:
    """Estimate the Threshold VAR of ``spec`` on ``data``."""
    Y = data.loc[spec.start:spec.end, list(spec.variables)].dropna()
    covid = {}
    c = spec.covid
    if c.get("mode"):
        covid = dict(covid_mode=c["mode"], covid_window=tuple(c["window"]))
        if c["mode"] == "dummies":
            covid["covid_drop_lags"] = c.get("drop_lags", True)
        else:
            covid["covid_n_free"] = c.get("n_free", 3)
    b_exo = None
    if spec.external_block:                  # the first variables form a linear block that ignores the rest
        n = len(spec.variables)
        b_exo = np.zeros((n, n), dtype=int)
        b_exo[:spec.external_block, spec.external_block:] = 1
    model = ThresholdVAR(Y, threshold_var=spec.threshold_var, lags=spec.lags, max_delay=spec.max_delay,
                         delay=spec.delay, threshold=spec.threshold, post_draws=spec.post_draws,
                         burnin=spec.burnin, hor=horizon, irf_1std=0, seed=spec.seed,
                         prior_params=dict(spec.prior), b_exo=b_exo, **covid)
    model.sample_posterior(show_progress=progress)
    return model


def girfs(model: ThresholdVAR, spec: ModelSpec, n_draws: int, reps: int,
          histories: dict | None = None) -> dict[str, np.ndarray]:
    """GIRFs to the external shock (``spec.external``, ``spec.shock`` units on impact).

    Returns ``{label: array (draws, horizon, variables)}``; labels are
    ``"pesimismo"`` and ``"optimismo"`` for the regime split (regime 1 is the
    low-confidence regime), or the keys of ``histories``.
    """
    out = model.compute_irfs(shock="level", n_draws=n_draws, reps=reps, scale=spec.shock,
                             show_progress=False, histories=histories)
    if histories is None:
        out = {"pesimismo": out["regime1"], "optimismo": out["regime2"]}
    return {k: v["level"][:, :, :, spec.shock_index] for k, v in out.items() if v}


def regime_irfs(model: ThresholdVAR, spec: ModelSpec, horizon: int, n_draws: int | None = None):
    """Conventional impulse responses of each regime's linear VAR (the economy
    never leaves the regime), to the external Cholesky shock of ``spec.shock``
    units. Reported only to compare with the GIRFs: in a regime-switching model
    they ignore that the shock itself can move the economy across regimes."""
    if spec.external_block:
        raise NotImplementedError("conventional regime IRFs are computed for the unrestricted models only")
    d = model.draws
    sel = slice(None) if n_draws is None else np.linspace(0, d["tar"].shape[0] - 1, n_draws).astype(int)
    out = {}
    L, N, k = model.lags, model.N, spec.shock_index
    for label, B, S in (("pesimismo", d["B1"][sel], d["S1"][sel]), ("optimismo", d["B2"][sel], d["S2"][sel])):
        C = np.linalg.cholesky(S)
        impact = C[:, :, k] / C[:, k:k + 1, k] * spec.shock
        lags = np.zeros((B.shape[0], L * N))
        resp = np.empty((B.shape[0], horizon, N))
        for h in range(horizon):
            y = np.einsum("pk,pkn->pn", lags, B[:, :L * N, :]) + (impact if h == 0 else 0.0)
            resp[:, h] = y
            lags = np.concatenate([y, lags[:, :-N]], axis=1)
        out[label] = resp
    return out


def girfs_by_band(model: ThresholdVAR, spec: ModelSpec, bands, n_draws: int, reps: int):
    """GIRFs averaged over the months whose threshold variable, ``d`` periods
    back, lies in each band ``[lo, hi)``: how far from the threshold the
    economy is when the shock hits."""
    d = spec.delay or int(np.median(model.draws["delay"]))
    lagged = model.y_df[spec.threshold_var].shift(d).reindex(model.sample_dates)
    histories = {f"[{lo}, {hi})": ((lagged >= lo) & (lagged < hi)).to_numpy() for lo, hi in bands}
    histories = {k: v for k, v in histories.items() if v.any()}
    counts = {k: int(v.sum()) for k, v in histories.items()}
    return girfs(model, spec, n_draws, reps, histories=histories), counts


def pass_through(resp: np.ndarray, spec: ModelSpec, horizons, response: str = "g_gdp",
                 quantiles=(50, 16, 84, 5, 95)) -> pd.DataFrame:
    """Cumulative response of ``response`` over cumulative response of the
    external variable, per draw, scaled to ``spec.shock`` units:
    ``shock * sum_{h<H} r_y(h) / sum_{h<H} r_x(h)``."""
    g, x = spec.variables.index(response), spec.shock_index
    rows = {}
    for H in horizons:
        ratio = resp[:, :H, g].sum(1) / resp[:, :H, x].sum(1) * spec.shock
        ratio = ratio[np.isfinite(ratio)]
        rows[H] = dict(zip([f"p{q}" for q in quantiles], np.percentile(ratio, quantiles)))
        rows[H]["prob_positive"] = float((ratio > 0).mean())
    out = pd.DataFrame(rows).T
    out.index.name = "horizonte"
    return out


def asymmetry(resp: dict, spec: ModelSpec, horizons, response: str = "g_gdp") -> pd.DataFrame:
    """Posterior of the difference in pass-through, optimism minus pessimism.

    Both GIRFs come from the same posterior draws (row ``i`` of each array is
    draw ``i``), so the difference is computed draw by draw: its posterior
    probability of being positive is the evidence for the asymmetry itself.
    """
    g, x, rows = spec.variables.index(response), spec.shock_index, {}
    for H in horizons:
        pts = {k: resp[k][:, :H, g].sum(1) / resp[k][:, :H, x].sum(1) * spec.shock
               for k in ("pesimismo", "optimismo")}
        diff = pts["optimismo"] - pts["pesimismo"]
        diff = diff[np.isfinite(diff)]
        rows[H] = {"p50": np.median(diff), "p16": np.percentile(diff, 16), "p84": np.percentile(diff, 84),
                   "prob_optimism_larger": float((diff > 0).mean())}
    out = pd.DataFrame(rows).T
    out.index.name = "horizonte"
    return out


def regime_table(model: ThresholdVAR) -> pd.DataFrame:
    """Posterior probability of the low-confidence regime by date, with the
    threshold variable (``NaN`` on the pandemic dates dropped by the dummies)."""
    p2 = model.regime_probability()
    return pd.DataFrame({"p_pesimismo": 1.0 - p2, model.threshold_name: model.y_df[model.threshold_name]
                         .reindex(p2.index)})


def pandemic_residuals(model: ThresholdVAR, spec: ModelSpec) -> pd.DataFrame:
    """Posterior of the pandemic dummies, recovered from a model that drops the
    window rows: with a diffuse prior on the dummy of month ``t`` its coefficient
    equals the one-step residual ``y_t - x_t' B_{regime(t)}`` (Cascaldi-Garcia, 2025).

    Returns draws in long format: ``date, variable, draw, value``.
    """
    d = model.draws
    full = model.data
    start = model.start
    win = model.covid_dates
    rows = []
    L, N = model.lags, model.N
    f = full[:, model.tarvar]
    for date in win:
        t = model.dates.get_loc(date)
        x = np.concatenate([full[t - j] for j in range(1, L + 1)] + [[1.0]])
        low = f[t - d["delay"]] <= d["tar"]                      # regime of month t in each draw
        fitted = np.where(low[:, None], np.einsum("k,nkv->nv", x, d["B1"]), np.einsum("k,nkv->nv", x, d["B2"]))
        resid = full[t] - fitted
        for v, name in enumerate(model.names):
            rows.append(pd.DataFrame({"date": date, "variable": name, "draw": np.arange(len(resid)),
                                      "value": resid[:, v]}))
    return pd.concat(rows, ignore_index=True)
