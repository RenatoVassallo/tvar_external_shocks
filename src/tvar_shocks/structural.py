"""A small structural model of the transmission mechanism (``soe.model``, solved with irispie).

The model separates the two ways an external shock reaches real GDP measured at
constant prices. Foreign demand raises export volumes directly. A higher export
price raises income, not volumes, so it reaches GDP only if that income is
spent, mostly through investment, whose response to the expected return is
scaled by the confidence parameter ``phi`` (real options under uncertainty).

The external variables follow the median paths of the TVAR's generalized
impulse responses in each regime, so the model and the TVAR face the same
shocks. Three parameters are calibrated to three TVAR moments, the 12-month GDP
pass-through of the China shock in optimism and of the export price shock in
both regimes: the export elasticity ``eta``, the investment sensitivity to the
expected price ``a_p`` and the relative responsiveness in pessimism
``phi_pes`` (``phi = 1`` in optimism). Everything else is fixed in
``config.toml``; the China pass-through in pessimism, the 24-month pass-through
and the investment and revenue responses are predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

__all__ = ["level_from_yoy", "yoy", "Structural", "calibrate", "estimate"]

MODEL_FILE = Path(__file__).with_name("soe.model")
SHOCKS = {"chn": ("yf", 1.0), "ipx": ("p", 10.0)}
REGIMES = ("pesimismo", "optimismo")


def level_from_yoy(g: np.ndarray) -> np.ndarray:
    """Level path implied by a path of 12-month changes: ``x(h) = g(h) + x(h - 12)``."""
    x = np.zeros(len(g))
    for h in range(len(g)):
        x[h] = g[h] + (x[h - 12] if h >= 12 else 0.0)
    return x


def yoy(x: np.ndarray) -> np.ndarray:
    """12-month change of a level path that starts from the steady state."""
    return x - np.concatenate([np.zeros(12), x])[:len(x)]


class Structural:
    """``soe.model`` parsed once; ``simulate`` reassigns the parameters and solves."""

    def __init__(self, length: int = 400):
        import irispie as ir
        self._ir = ir
        self.model = ir.Simultaneous.from_file(str(MODEL_FILE), linear=True)
        self.length = length
        start = ir.mm(2000, 1)
        self.span = ir.Span(start, start + length - 1)

    def simulate(self, params: dict, p_path: np.ndarray, yf_path: np.ndarray) -> dict[str, np.ndarray]:
        ir, T = self._ir, self.length
        self.model.assign(**params)
        self.model.steady()
        self.model.solve()
        db = ir.Databox.steady(self.model, self.span)
        for name, path in (("e_p", p_path), ("e_yf", yf_path)):
            full = np.full(T, path[-1], dtype=float)            # paths stay at their last value
            full[:len(path)] = path
            db[name] = ir.Series(periods=self.span, values=full)
        out = self.model.simulate(db, self.span)
        out = out[0] if isinstance(out, tuple) else out
        return {k: out[k].get_data(self.span).ravel() for k in ("y", "c", "i", "x", "rev", "p", "yf")}

    def response(self, params: dict, paths: dict, shock: str, regime: str) -> dict[str, np.ndarray]:
        path = paths[(shock, regime)]
        zero = np.zeros(len(path))
        r = self.simulate(params, path, zero) if shock == "ipx" else self.simulate(params, zero, path)
        return {k: v[:len(path)] for k, v in r.items()}


def external_paths(results: Path) -> dict:
    """Median TVAR response of each external variable (12-month change per unit of shock) as a level path."""
    paths = {}
    for shock, (_, scale) in SHOCKS.items():
        z = np.load(results / f"girf_{shock}.npz")
        for regime in REGIMES:
            g = np.median(z[f"regime_{regime}"], axis=0)[:, 0] / scale
            paths[(shock, regime)] = level_from_yoy(g)
    return paths


def pass_through(resp: np.ndarray, ext: np.ndarray, H: int, scale: float) -> float:
    """Same statistic as the TVAR: cumulative 12-month change of the response over that of the shock."""
    return float(scale * yoy(resp)[:H].sum() / yoy(ext)[:H].sum())


def _params(fixed: dict, eta: float, a_p: float, phi: float) -> dict:
    return {**fixed, "eta": eta, "a_p": a_p, "phi": phi}


def calibrate(model: Structural, paths: dict, fixed: dict, targets: dict, H: int = 12):
    """``(eta, a_p, phi_pes)`` matching China in optimism and the export price shock in both regimes."""
    def moments(z):
        eta, a_p, phi_pes = z
        out = []
        for shock, regime, phi in (("chn", "optimismo", 1.0), ("ipx", "optimismo", 1.0), ("ipx", "pesimismo", phi_pes)):
            r = model.response(_params(fixed, eta, a_p, phi), paths, shock, regime)
            ext, scale = SHOCKS[shock]
            out.append(pass_through(r["y"], r[ext], H, scale) - targets[(shock, regime)])
        return out
    sol = least_squares(moments, x0=[0.5, 1.0, 0.5], bounds=([0.0, 0.0, 0.0], [5.0, 50.0, 1.5]), xtol=1e-10)
    return dict(zip(("eta", "a_p", "phi_pes"), sol.x)), float(np.abs(sol.fun).max())


def _summary(model, paths, fixed, cal, horizons) -> tuple[pd.DataFrame, dict]:
    rows, irfs = [], {}
    for shock, (ext, scale) in SHOCKS.items():
        for regime in REGIMES:
            for label, phi in (("modelo", cal["phi_pes"] if regime == "pesimismo" else 1.0),
                               ("misma_respuesta", 1.0)):
                if label == "misma_respuesta" and regime == "optimismo":
                    continue
                r = model.response(_params(fixed, cal["eta"], cal["a_p"], phi), paths, shock, regime)
                irfs.update({f"{label}__{shock}__{regime}__{k}": yoy(v) * scale for k, v in r.items()})
                for H in horizons:
                    for var in ("y", "i", "c", "rev"):
                        rows.append({"version": label, "shock": shock, "regime": regime, "H": H, "variable": var,
                                     "pass_through": pass_through(r[var], r[ext], H, scale)})
    return pd.DataFrame(rows), irfs


def estimate(cfg: dict, results: Path) -> None:
    """Calibrate the model, save its pass-through, a sensitivity table and the response paths."""
    s = cfg["structural"]
    pt = pd.read_csv(results / "pass_through.csv")
    pt = pt[(pt["variant"] == "base") & (pt["response"] == "g_gdp") & (pt["horizonte"] == 12)]
    targets = {(r.model, r.group): r.p50 for r in pt.itertuples()}
    paths = external_paths(results)
    model = Structural()
    cal, err = calibrate(model, paths, s["fixed"], targets)
    table, irfs = _summary(model, paths, s["fixed"], cal, cfg["girf"]["pass_through"][:2])
    table.to_csv(results / "structural_pass_through.csv", index=False)
    pd.DataFrame([{**s["fixed"], **cal, "max_moment_error": err}]).to_csv(results / "structural_calibration.csv",
                                                                         index=False)
    np.savez_compressed(results / "structural_irfs.npz", **irfs)
    rows = []                                                  # how much the predictions depend on fixed values
    for a_y in s["sensitivity"]["a_y"]:
        for mpc in s["sensitivity"]["mpc"]:
            fixed = {**s["fixed"], "a_y": a_y, "mpc": mpc}
            c, e = calibrate(model, paths, fixed, targets)
            r = {k: model.response(_params(fixed, c["eta"], c["a_p"], c["phi_pes"] if k[1] == "pesimismo" else 1.0),
                                   paths, *k) for k in (("chn", "pesimismo"), ("ipx", "pesimismo"))}
            same = model.response(_params(fixed, c["eta"], c["a_p"], 1.0), paths, "ipx", "pesimismo")
            rows.append({"a_y": a_y, "mpc": mpc, **c, "max_moment_error": e,
                         "chn_pes_12": pass_through(r[("chn", "pesimismo")]["y"], r[("chn", "pesimismo")]["yf"], 12, 1.0),
                         "ipx_pes_12_same_phi": pass_through(same["y"], same["p"], 12, 10.0)})
    pd.DataFrame(rows).to_csv(results / "structural_sensitivity.csv", index=False)
