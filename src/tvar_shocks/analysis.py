"""Run every estimation of the paper and store the results in ``output/results``.

Figures and tables are drawn from these files only (``figures.py``,
``tables.py``), so they can be regenerated without re-estimating.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import data as dbase
from .config import model_specs, results_dir, variant_config
from .lp import state_dependent_pass_through
from .model import (asymmetry, fit, girfs, girfs_by_band, pandemic_residuals, pass_through, regime_irfs,
                    regime_table)

__all__ = ["estimate_prior_grid", "estimate_baseline", "estimate_robustness", "estimate_threshold_grid",
           "estimate_threshold_posterior", "estimate_local_projections"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _info(model, spec, variant: str) -> dict:
    p_low = 1.0 - model.regime_probability()
    lml = model.log_marginal_likelihood()
    ts = model.threshold_summary()
    return {"model": spec.name, "variant": variant, "T": model.T,
            "sample": f"{model.sample_dates[0]:%Y-%m} a {model.sample_dates[-1]:%Y-%m}",
            "threshold": float(ts.loc["threshold (Z*)", "p50"]), "delay": int(np.median(model.draws["delay"])),
            "months_pessimism": int((p_low > 0.5).sum()), "months_optimism": int((p_low < 0.5).sum()),
            "log_ml_tvar": float(lml["TVAR"]), "log_ml_linear": float(lml["linear VAR"]),
            "log_bayes_factor": float(lml["log Bayes factor"]), "stable_share": float(model.stable_share)}


def _pt_frame(resp: dict, spec, horizons, variant: str) -> pd.DataFrame:
    """Pass-through of GDP (and of fiscal revenues or capital goods imports when the model has them)."""
    frames = []
    for response in [v for v in ("g_gdp", "g_rev", "g_rev3", "g_kimp") if v in spec.variables]:
        for group, arr in resp.items():
            pt = pass_through(arr, spec, horizons, response=response).reset_index()
            pt.insert(0, "group", group)
            pt.insert(0, "response", response)
            frames.append(pt)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "variant", variant)
    out.insert(0, "model", spec.name)
    return out


def estimate_prior_grid(cfg: dict) -> None:
    """Marginal likelihood of the TVAR and of the linear VAR over the prior grid
    (``tau = 10 lambda``), to choose the hyperparameters (Giannone, Lenza and Primiceri, 2015)."""
    data = dbase.load()
    grid = cfg["robustness"]["prior_grid"]
    rows = []
    for lam in grid["lamdaP"]:
        for eps in grid["epsilonP"]:
            prior = {**cfg["prior"], "lamdaP": lam, "tauP": 10 * lam, "epsilonP": eps}
            for name, spec in model_specs(cfg, prior=prior, post_draws=20).items():
                model = fit(spec, data, 2)
                lml = model.log_marginal_likelihood()
                rows.append({"model": name, "lamdaP": lam, "tauP": 10 * lam, "epsilonP": eps,
                             "log_ml_tvar": lml["TVAR"], "log_ml_linear": lml["linear VAR"],
                             "log_bayes_factor": lml["log Bayes factor"],
                             "delay_mode": int(model.split_posterior.groupby("delay")["probability"].sum().idxmax())})
    pd.DataFrame(rows).to_csv(results_dir() / "prior_grid.csv", index=False)
    _log("prior grid done")


def estimate_baseline(cfg: dict, quick: bool = False) -> None:
    """Baseline models: regimes, GIRFs by regime and by confidence band,
    pass-through, model comparison and the pandemic dummies."""
    results_dir().mkdir(parents=True, exist_ok=True)
    g = cfg["girf"]
    n_draws, reps = (100, 5) if quick else (g["n_draws"], g["reps"])
    data = dbase.load()
    infos, pts, asym = [], [], []
    for name, spec in model_specs(cfg).items():
        _log(f"baseline {name}: estimating")
        model = fit(spec, data, g["horizon"])
        resp = girfs(model, spec, n_draws, reps)
        bands, counts = girfs_by_band(model, spec, g["confidence_bands"], n_draws, reps)
        linear = regime_irfs(model, spec, g["horizon"])
        _log(f"baseline {name}: GIRFs done")
        np.savez_compressed(results_dir() / f"girf_{name}.npz", variables=np.array(spec.variables),
                            **{f"regime_{k}": v for k, v in resp.items()},
                            **{f"band_{k}": v for k, v in bands.items()},
                            **{f"linear_{k}": v for k, v in linear.items()})
        regime_table(model).to_csv(results_dir() / f"regimes_{name}.csv")
        res = pandemic_residuals(model, spec)
        res.to_csv(results_dir() / f"pandemic_dummies_{name}.csv.gz", index=False)
        info = _info(model, spec, "base")
        info["band_months"] = json.dumps(counts)
        info["band_delay"] = spec.delay or int(np.median(model.draws["delay"]))
        infos.append(info)
        pts += [_pt_frame(resp, spec, g["pass_through"], "base"),
                _pt_frame(bands, spec, g["pass_through"], "base_bands"),
                _pt_frame(linear, spec, g["pass_through"], "base_linear_irf")]
        for method, r in (("girf", resp), ("linear_irf", linear)):
            asym.append(asymmetry(r, spec, g["pass_through"]).reset_index().assign(model=name, method=method))
    pd.DataFrame(infos).to_csv(results_dir() / "model_info.csv", index=False)
    pd.concat(pts).to_csv(results_dir() / "pass_through.csv", index=False)
    pd.concat(asym).to_csv(results_dir() / "asymmetry.csv", index=False)


def estimate_robustness(cfg: dict, quick: bool = False) -> None:
    """Each ``[robustness.variants]`` entry: same pipeline, GIRFs by regime only."""
    g = cfg["girf"]
    n_draws, reps = (100, 5) if quick else (cfg["robustness"]["grid_n_draws"], cfg["robustness"]["grid_reps"])
    data = dbase.load()
    infos, pts, resps = [], [], {}
    for variant in cfg["robustness"]["variants"]:
        vcfg = variant_config(cfg, variant)
        for name, spec in model_specs(vcfg).items():
            if not dbase.available(data, spec.variables):       # political noise index not distributed
                _log(f"robustness {variant}/{name}: skipped, series not available")
                continue
            _log(f"robustness {variant}/{name}")
            model = fit(spec, data, g["horizon"])
            resp = girfs(model, spec, n_draws, reps)
            infos.append(_info(model, spec, variant))
            pts.append(_pt_frame(resp, spec, g["pass_through"], variant))
            resps.update({f"{variant}__{name}__{k}": v for k, v in resp.items()})
    np.savez_compressed(results_dir() / "girf_robustness.npz", **resps)
    pd.DataFrame(infos).to_csv(results_dir() / "model_info_robustness.csv", index=False)
    pd.concat(pts).to_csv(results_dir() / "pass_through_robustness.csv", index=False)


def estimate_threshold_grid(cfg: dict, quick: bool = False) -> None:
    """Baseline models with the threshold fixed at each value of the grid:
    regime sizes, marginal likelihood (a profile over thresholds) and pass-through."""
    g, rob = cfg["girf"], cfg["robustness"]
    n_draws, reps = (60, 4) if quick else (rob["grid_n_draws"], rob["grid_reps"])
    data = dbase.load()
    infos, pts = [], []
    for z in rob["threshold_grid"]:
        for name, spec in model_specs(cfg, threshold=float(z)).items():
            model = fit(spec, data, g["horizon"])
            info = _info(model, spec, f"threshold_{z}")
            info["grid_threshold"] = z
            infos.append(info)
            pts.append(_pt_frame(girfs(model, spec, n_draws, reps), spec, g["pass_through"], f"threshold_{z}")
                       .assign(grid_threshold=z))
        _log(f"threshold grid: {z} done")
    pd.DataFrame(infos).to_csv(results_dir() / "threshold_grid_info.csv", index=False)
    pd.concat(pts).to_csv(results_dir() / "threshold_grid_pass_through.csv", index=False)


def estimate_threshold_posterior(cfg: dict) -> None:
    """Threshold and delay estimated jointly (exact posterior, flat prior on the
    threshold over the admissible splits, uniform prior on the delay)."""
    g, rob = cfg["girf"], cfg["robustness"]
    data = dbase.load()
    frames, infos = [], []
    for name, spec in model_specs(cfg, threshold=None, delay=None, max_delay=rob["estimated_max_delay"],
                                  prior={**cfg["prior"], "tarvariance": 1e6}).items():
        _log(f"estimated threshold {name}")
        model = fit(spec, data, g["horizon"])
        sp = model.split_posterior.copy()
        sp.insert(0, "model", name)
        frames.append(sp)
        infos.append(_info(model, spec, "estimated_threshold"))
    pd.concat(frames).to_csv(results_dir() / "threshold_posterior.csv", index=False)
    pd.DataFrame(infos).to_csv(results_dir() / "model_info_estimated_threshold.csv", index=False)



def estimate_local_projections(cfg: dict) -> None:
    """State-dependent local projections (Ramey and Zubairy, 2018) for the baseline models."""
    est, lp = cfg["estimation"], cfg["lp"]
    data = dbase.load()
    frames = []
    for sample, opts in lp["samples"].items():
        for threshold in lp["thresholds"]:
            for name, spec in model_specs(cfg).items():
                out = state_dependent_pass_through(data, spec, lp["horizons"], threshold, lp["delay"],
                                                   end=opts["end"],
                                                   exclude=tuple(opts["exclude"]) if "exclude" in opts else None)
                frames.append(out.assign(sample=sample, threshold=threshold))
    covid = cfg["covid"]["window"]
    for variant, delay in lp.get("state_variants", {}).items():   # another state variable
        for name, spec in model_specs(variant_config(cfg, variant)).items():
            if not dbase.available(data, spec.variables):
                _log(f"local projections {variant}/{name}: skipped, series not available")
                continue
            out = state_dependent_pass_through(data, spec, lp["horizons"], spec.threshold, delay, end=spec.end,
                                               exclude=(f"{covid[0]}-01", f"{covid[1]}-01"))
            frames.append(out.assign(sample=variant, threshold=spec.threshold))
    for split in lp.get("date_splits", []):                        # a break instead of a state
        after = pd.Series(data.index >= pd.Timestamp(split), index=data.index)
        for name, spec in model_specs(cfg).items():
            out = state_dependent_pass_through(data, spec, lp["horizons"], 0.0, 0, end=cfg["data"]["end"],
                                               exclude=(f"{covid[0]}-01", f"{covid[1]}-01"), state=after)
            frames.append(out.assign(sample=f"corte_{split[:7]}", threshold=np.nan))
    pd.concat(frames).to_csv(results_dir() / "lp_pass_through.csv", index=False)
    _log("local projections done")


def estimate_structural(cfg: dict) -> None:
    """Small structural model calibrated to the baseline GIRFs (needs the baseline stage)."""
    from . import structural
    structural.estimate(cfg, results_dir())
    _log("structural model done")
