"""Command line: ``python -m tvar_shocks <stage> [--quick]``.

Stages
------
data        rebuild the database from the sources and compare it with the frozen copy
prior       marginal likelihood over the prior grid (choice of hyperparameters)
baseline    baseline models: regimes, GIRFs, pass-through, pandemic dummies
robustness  robustness variants of config.toml
grid        threshold grid (regimes, marginal likelihood, pass-through)
threshold   threshold and delay estimated jointly
lp          state-dependent local projections (Ramey and Zubairy, 2018)
structural  small structural model calibrated to the baseline TVAR (soe.model, irispie)
figures     all figures (from output/results)
tables      all tables (from output/results)
all         every stage above except data
"""
from __future__ import annotations

import argparse
import os
import time

from .config import OUTPUT, load_config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tvar_shocks", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["data", "prior", "baseline", "robustness", "grid", "threshold", "lp",
                                          "structural",
                                          "figures", "tables", "all"])
    parser.add_argument("--quick", action="store_true",
                        help="few GIRF draws, to check that the pipeline runs (numbers are not the paper's)")
    parser.add_argument("--config", default=None, help="path to config.toml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.quick:                                     # never overwrite the paper's results
        os.environ["TVAR_SHOCKS_OUTPUT"] = str(OUTPUT / "quick")
    t0 = time.time()

    if args.stage == "data":
        from .data import build, compare
        new = build()
        print(compare(new).round(4).to_string())
        print("The frozen copy (data/database.csv) is not modified.")
        return

    from . import analysis, figures, tables
    steps = {"prior": lambda: analysis.estimate_prior_grid(cfg),
             "baseline": lambda: analysis.estimate_baseline(cfg, args.quick),
             "robustness": lambda: analysis.estimate_robustness(cfg, args.quick),
             "grid": lambda: analysis.estimate_threshold_grid(cfg, args.quick),
             "threshold": lambda: analysis.estimate_threshold_posterior(cfg),
             "lp": lambda: analysis.estimate_local_projections(cfg),
             "structural": lambda: analysis.estimate_structural(cfg),
             "figures": lambda: figures.make_all(cfg),
             "tables": lambda: tables.make_all(cfg)}
    order = list(steps) if args.stage == "all" else [args.stage]
    for stage in order:
        steps[stage]()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
