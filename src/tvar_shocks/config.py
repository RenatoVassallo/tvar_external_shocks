"""Read ``config.toml`` and turn it into model specifications.

Every number that shapes a result lives in ``config.toml``; the code only
reads it. A robustness variant is the base configuration with a few keys
overridden (``[robustness.variants.<name>]``).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ROOT", "OUTPUT", "output_dir", "results_dir", "figures_dir", "tables_dir", "load_config",
           "ModelSpec", "model_specs", "variant_config"]

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"


def output_dir() -> Path:
    """Where results, figures and tables go: ``output/`` unless ``TVAR_SHOCKS_OUTPUT`` says otherwise
    (``--quick`` runs write to ``output/quick`` so they never overwrite the paper's results)."""
    return Path(os.environ.get("TVAR_SHOCKS_OUTPUT", OUTPUT))


def results_dir() -> Path:
    path = output_dir() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = output_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tables_dir() -> Path:
    path = output_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Path | str | None = None) -> dict:
    """The parsed ``config.toml`` (repository root by default)."""
    with open(path or ROOT / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def variant_config(cfg: dict, name: str | None) -> dict:
    """Base configuration with the overrides of robustness variant ``name``."""
    if name is None:
        return cfg
    over = dict(cfg["robustness"]["variants"][name])
    over.pop("description", None)
    over.pop("short", None)
    if "models" in over:                     # a variant with its own models replaces the base ones
        cfg = {**cfg, "models": {}}
    return _merge(cfg, over)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to estimate one Threshold VAR."""

    name: str
    variables: tuple
    shock: float
    label: str
    short: str
    start: str
    end: str
    lags: int
    threshold_var: str
    threshold: float | None
    delay: int | None
    max_delay: int
    post_draws: int
    burnin: float
    seed: int
    prior: dict = field(hash=False)
    covid: dict = field(hash=False)
    shock_var: str | None = None             # shocked variable (default: the first one)
    external_block: int = 0                  # first `n` variables form a linear, exogenous block (MacroPy `b_exo`)

    @property
    def external(self) -> str:
        return self.shock_var or self.variables[0]

    @property
    def shock_index(self) -> int:
        return self.variables.index(self.external)

    def key(self) -> str:
        """Short hash of the specification, used to name cached results."""
        payload = json.dumps(self.__dict__, sort_keys=True, default=str).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


def model_specs(cfg: dict, **overrides) -> dict[str, ModelSpec]:
    """One :class:`ModelSpec` per entry of ``[models]``; ``overrides`` replace
    estimation fields (e.g. ``threshold=48.0``)."""
    est, data = cfg["estimation"], cfg["data"]
    threshold = est["threshold"]
    specs = {}
    for name, m in cfg["models"].items():
        kw = dict(
            name=name, variables=tuple(m["variables"]), shock=float(m["shock"]), label=m["label"],
            short=m["short"], start=data["start"], end=data["end"], lags=est["lags"],
            threshold_var=est["threshold_var"],
            threshold=None if math.isnan(threshold) else float(threshold),
            delay=est["delay"] or None, max_delay=est["max_delay"], post_draws=est["post_draws"],
            burnin=est["burnin"], seed=est["seed"], prior=dict(cfg["prior"]), covid=dict(cfg["covid"]),
            shock_var=m.get("shock_var"), external_block=int(est.get("external_block", 0)),
        )
        kw.update(overrides)
        specs[name] = ModelSpec(**kw)
    return specs
