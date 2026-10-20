"""Replication package: asymmetric effects of external shocks on the Peruvian
economy and the role of business confidence (Sánchez y Vassallo).

Run everything with ``python -m tvar_shocks all`` (see README.md).
"""
from .config import ROOT, OUTPUT, load_config, model_specs, variant_config
from . import data

__all__ = ["ROOT", "OUTPUT", "load_config", "model_specs", "variant_config", "data"]
__version__ = "1.0.0"
