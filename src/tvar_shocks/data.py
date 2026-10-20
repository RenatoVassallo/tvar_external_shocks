"""The monthly database: frozen copy, sources and the script that rebuilds it.

``load()`` reads the frozen copy that every number in the paper uses.
``build()`` rebuilds the database from the original sources (BCRP API, FRED
and ``data/raw/external.csv``) with the transformations of the paper, and
``compare()`` reports how far a rebuild is from the frozen copy: the BCRP
revises the national accounts, the terms of trade and the export price index,
so a rebuild made today differs slightly from the frozen copy.
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .config import ROOT

__all__ = ["DATA", "SOURCES", "RAW_EXTERNAL", "RAW_POLITICAL_NOISE", "DERIVED", "OPTIONAL", "load", "available",
           "sources", "build", "compare", "yoy"]

DATA = ROOT / "data" / "database.csv"
SOURCES = ROOT / "data" / "sources.csv"
RAW_EXTERNAL = ROOT / "data" / "raw" / "external.csv"
# The political noise index is not distributed (see data/README.md). When the file is present, load()
# adds `pni` (3-month average) and `pni_neg = -pni`, and the pipeline also runs the variants that use them.
RAW_POLITICAL_NOISE = ROOT / "data" / "raw" / "political_noise.csv"
OPTIONAL = ("pni", "pni_neg")   # series that exist only when the political noise file is present
DERIVED = OPTIONAL              # series added by load() that are not stored in the frozen copy

# BCRP series codes (https://estadisticas.bcrp.gob.pe/estadisticas/series/)
BCRP_MONTHLY = {
    "exp": "PD38045AM",   # expectativas de la economía a 3 meses (índice de difusión)
    "gdp": "PN01773AM",   # PBI, índice 2007 = 100
    "cpi": "PN38705PM",   # IPC Lima Metropolitana
    "int": "PN07819NM",   # tasa interbancaria promedio
    "xpi": "PN38915BM",   # índice de precios de exportación
    "tot": "PN38923BM",   # términos de intercambio
    "ner": "PN01210PM",   # tipo de cambio bancario promedio
    "rev": "PN38689FM",   # ingresos corrientes del gobierno central, términos reales
    "kimp": "PD37978BM",  # volumen de importación de bienes de capital sin materiales de construcción
}
BCRP_QUARTERLY = {"gdpq": "PN02538AQ"}               # PBI real trimestral (completa 2002-2003)
FRED = {"ip_usa": "INDPRO", "fed": "FEDFUNDS"}


def load() -> pd.DataFrame:
    """Frozen monthly database (first column ``date``), plus the political noise index when its file exists.

    ``pni_neg`` is the index with the sign flipped: MacroPy's regime 1 is the one with the threshold
    variable at or below the threshold, so thresholding ``pni_neg`` makes regime 1 the high-noise
    regime, the counterpart of pessimism.
    """
    db = pd.read_csv(DATA, index_col="date", parse_dates=True)
    if RAW_POLITICAL_NOISE.exists():
        raw = pd.read_csv(RAW_POLITICAL_NOISE, index_col="date", parse_dates=True)["pni_raw"]
        db["pni"] = raw.rolling(3).mean().reindex(db.index)
        db["pni_neg"] = -db["pni"]
    return db


def available(db: pd.DataFrame, variables) -> bool:
    """True when every series in ``variables`` is in ``db``; False when the only missing ones are
    optional (the political noise index), so that the caller can skip the exercise."""
    missing = set(variables) - set(db.columns)
    if missing - set(OPTIONAL):
        raise KeyError(f"series not in the database: {sorted(missing)}")
    return not missing


def sources() -> pd.DataFrame:
    """Dictionary of the database: description, transformation, source and code of each series."""
    return pd.read_csv(SOURCES)


def yoy(s: pd.Series | pd.DataFrame, periods: int = 12):
    """Interannual growth rate, ``100 (x_t / x_{t-12} - 1)``."""
    return 100 * (s / s.shift(periods) - 1)


# ------------------------------------------------------------------ rebuild
def _bcrp(codes: dict, freq: str) -> pd.DataFrame:
    from MacroPy import get_bcrp_data
    out = get_bcrp_data({code: name for name, code in codes.items()}, frequency=freq,
                        start_period="2000-1", date_index=True)
    if out is None or len(out) == 0:
        raise ConnectionError(f"the BCRP API returned nothing for {list(codes.values())}")
    out.index = pd.DatetimeIndex(out.index)
    return out.astype(float)


def _fred(code: str) -> pd.Series:
    raw = urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}",
                                 timeout=60).read()
    d = pd.read_csv(io.BytesIO(raw))
    s = pd.to_numeric(d[code], errors="coerce")
    s.index = pd.DatetimeIndex(d.iloc[:, 0])
    return s.dropna()


def _quarterly_to_monthly(q: pd.Series) -> pd.Series:
    """Not-a-knot cubic spline through the quarterly values placed at the last
    month of each quarter (MATLAB ``retime(..., 'spline')``)."""
    q = q.dropna()
    t = lambda d: d.year * 12 + d.month
    spline = CubicSpline([t(d) for d in q.index], q.to_numpy(), bc_type="not-a-knot")
    months = pd.date_range(q.index[0], q.index[-1], freq="MS")
    return pd.Series(spline([t(d) for d in months]), index=months)


def build(end: str | None = None) -> pd.DataFrame:
    """Rebuild the database from the sources (needs internet)."""
    m = _bcrp(BCRP_MONTHLY, "M")
    gdpq = _quarterly_to_monthly(_bcrp(BCRP_QUARTERLY, "Q")["gdpq"])
    ext = pd.read_csv(RAW_EXTERNAL, index_col="date", parse_dates=True)
    usa = yoy(_fred(FRED["ip_usa"])).rename("g_ip_usa")
    fed = _fred(FRED["fed"]).rename("fed")

    db = pd.DataFrame(index=m.index)
    db["g_ip_chn"] = ext["g_ip_chn"]
    db["g_ip_usa"] = usa
    db["fed"] = fed
    db["g_tot"] = yoy(m["tot"])
    db["g_xpi"] = yoy(m["xpi"])
    db["exp"] = m["exp"]
    db["g_gdp"] = yoy(m["gdp"]).fillna(yoy(gdpq).reindex(db.index))   # monthly GDP index from 2003
    db["g_cpi"] = yoy(m["cpi"])
    db["int"] = m["int"]
    db["g_ner"] = yoy(m["ner"])
    db["g_kimp"] = yoy(m["kimp"])
    db["g_rev"] = yoy(m["rev"].rolling(12).sum())                       # 12-month sum: no tax calendar
    db.index.name = "date"
    db = db.loc[db["exp"].first_valid_index():]
    return db.loc[:end] if end else db


def compare(new: pd.DataFrame, old: pd.DataFrame | None = None) -> pd.DataFrame:
    """Largest and mean absolute difference per column between a rebuild and the frozen copy."""
    old = load() if old is None else old
    rows = {}
    for col in old.columns.intersection(new.columns):
        d = (new[col] - old[col]).reindex(old.index).abs().dropna()
        rows[col] = {"n": len(d), "max_abs_diff": d.max() if len(d) else np.nan,
                     "mean_abs_diff": d.mean() if len(d) else np.nan}
    return pd.DataFrame(rows).T
