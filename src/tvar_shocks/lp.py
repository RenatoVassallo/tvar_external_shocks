"""State-dependent local projections: a check on the TVAR that imposes no VAR dynamics.

Following Ramey and Zubairy (2018), the pass-through at horizon ``H`` is the
cumulative multiplier ``m_H`` of the regression

    sum_{j<H} y_{t+j} = a_s + m_{H,s} sum_{j<H} x_{t+j} + G_s w_{t-1} + u_t,   s = S_t,

estimated by two-stage least squares in each state with ``x_t`` (the external
variable at ``t``) as the instrument, ``w_{t-1}`` the lags of every variable of
the model (so ``x_t`` is the external innovation of the recursive VAR, as in
Plagborg-Møller and Wolf, 2021) and the state ``S_t`` set by the threshold
variable ``d`` months before the shock, as in the TVAR. The state is fixed at
the time of the shock: the projection averages over whatever regimes the
economy goes through afterwards, like the generalized impulse responses.
Standard errors are Newey-West with ``H`` lags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelSpec

__all__ = ["state_dependent_pass_through"]


def _newey_west(Z: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    g = Z * u[:, None]
    S = g.T @ g
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = g[l:].T @ g[:-l]
        S += w * (G + G.T)
    return S


def _iv(Y, X, Z, lags):
    """Exactly identified 2SLS with HAC covariance; returns coefficients and covariance."""
    ZX_inv = np.linalg.inv(Z.T @ X)
    b = ZX_inv @ (Z.T @ Y)
    u = Y - X @ b
    V = ZX_inv @ _newey_west(Z, u, lags) @ ZX_inv.T
    return b, V


def state_dependent_pass_through(db: pd.DataFrame, spec: ModelSpec, horizons, threshold: float,
                                 delay: int, end: str, exclude: tuple | None = None,
                                 response: str = "g_gdp", min_obs_per_regressor: int = 3,
                                 state: pd.Series | None = None) -> pd.DataFrame:
    """Cumulative multiplier of ``response`` with respect to the external variable,
    by state, scaled to ``spec.shock``. ``exclude`` drops the observations whose
    controls or outcome window touch the given ``(start, end)`` dates. A state with fewer than
    ``min_obs_per_regressor`` observations per regressor is reported as missing. ``state``, a
    boolean series (True in the first state), replaces the threshold rule, e.g. for a date split."""
    d = db.loc[spec.start:end, list(spec.variables)].dropna()
    x_name = spec.external
    state_low = (d[spec.threshold_var].shift(delay) < threshold) if state is None else \
        state.reindex(d.index).fillna(False).astype(bool)
    lagged = pd.concat({f"{c}_l{l}": d[c].shift(l) for c in spec.variables for l in range(1, spec.lags + 1)},
                       axis=1)
    rows = []
    for H in horizons:
        ycum = sum(d[response].shift(-j) for j in range(H))
        xcum = sum(d[x_name].shift(-j) for j in range(H))
        frame = pd.concat({"ycum": ycum, "xcum": xcum, "z": d[x_name], "low": state_low}, axis=1).join(lagged)
        frame = frame.dropna()
        if exclude is not None:
            a, b = pd.Timestamp(exclude[0]), pd.Timestamp(exclude[1])
            first = frame.index - pd.DateOffset(months=spec.lags)
            last = frame.index + pd.DateOffset(months=H - 1)
            frame = frame[(last < a) | (first > b)]
        for state, sub in (("pesimismo", frame[frame["low"]]), ("optimismo", frame[~frame["low"]])):
            W = np.column_stack([np.ones(len(sub)), sub[lagged.columns].to_numpy()])
            X = np.column_stack([sub["xcum"].to_numpy(), W])
            Z = np.column_stack([sub["z"].to_numpy(), W])
            if len(sub) < min_obs_per_regressor * X.shape[1]:          # too few months in the state
                rows.append({"model": spec.name, "group": state, "horizonte": H, "n": len(sub),
                             **{k: np.nan for k in ("p50", "p16", "p84", "p5", "p95", "se", "first_stage")}})
                continue
            b, V = _iv(sub["ycum"].to_numpy(), X, Z, H)
            m, se = b[0] * spec.shock, np.sqrt(V[0, 0]) * spec.shock
            first_stage = np.linalg.lstsq(Z, sub["xcum"].to_numpy(), rcond=None)[0][0]
            rows.append({"model": spec.name, "group": state, "horizonte": H, "n": len(sub),
                         "p50": m, "p16": m - se, "p84": m + se, "p5": m - 1.645 * se, "p95": m + 1.645 * se,
                         "se": se, "first_stage": first_stage})
    return pd.DataFrame(rows)
