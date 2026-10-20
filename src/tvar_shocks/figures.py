"""Figures of the paper, drawn from ``output/results`` and the frozen database.

Grayscale, serif type, no titles inside the figures (captions go in the
paper). Each ``fig_*`` function writes one PDF (for LaTeX) and one PNG
(preview) to ``output/figures`` and returns the path of the PDF.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as dbase
from .config import figures_dir, model_specs, results_dir, variant_config

__all__ = ["make_all"]

# regime: (line colour, band colour, line style, label)
REGIME = {"pesimismo": ("0.45", "0.55", "--", "Régimen 1 (pesimismo)"),
          "optimismo": ("0.0", "0.25", "-", "Régimen 2 (optimismo)")}
LABELS = {"g_ip_chn": "Producción industrial de China", "g_xpi": "Precios de exportación",
          "g_gdp": "Crecimiento del PBI", "g_cpi": "Inflación", "int": "Tasa interbancaria",
          "exp": "Confianza empresarial", "g_ip_usa": "Producción industrial de EE.UU.",
          "g_tot": "Términos de intercambio", "g_rev": "Ingresos fiscales"}
STYLE = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
         "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42}


def _gdp(pt: pd.DataFrame) -> pd.DataFrame:
    """Rows of a pass-through table that refer to GDP."""
    return pt[pt["response"] == "g_gdp"] if "response" in pt else pt


def _save(fig, name: str):
    # no creation date in the PDF, so that a rerun reproduces the committed files byte for byte
    fig.savefig(figures_dir() / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(figures_dir() / f"{name}.png", dpi=150)
    plt.close(fig)
    return figures_dir() / f"{name}.pdf"


def _band(ax, arr, reg, label=None, cred=0.68):
    """Median and central ``cred`` band of ``arr`` (draws, horizon); horizon 1 = impact."""
    line, fill, ls, default = REGIME[reg]
    h = np.arange(1, arr.shape[1] + 1)
    lo, med, hi = np.percentile(arr, [50 * (1 - cred), 50, 100 - 50 * (1 - cred)], axis=0)
    ax.fill_between(h, lo, hi, color=fill, alpha=0.22, lw=0)
    ax.plot(h, med, color=line, ls=ls, lw=1.6, label=label or default)


def _zero(ax):
    ax.axhline(0, color="0.5", lw=0.6)


def _shade_below(ax, s: pd.Series, level: float, color="0.88"):
    below = (s < level).to_numpy()
    idx = s.index
    starts = np.flatnonzero(np.diff(np.r_[False, below].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[below, False].astype(int)) == -1)
    for a, b in zip(starts, ends):
        ax.axvspan(idx[a] - pd.Timedelta(days=15), idx[b] + pd.Timedelta(days=15), color=color, lw=0, zorder=0)


def _years(ax, step):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _band_delay() -> int:
    info = pd.read_csv(results_dir() / "model_info.csv")
    return int(info["band_delay"].iloc[0]) if "band_delay" in info else 2


# ------------------------------------------------------------------ descriptive
def fig_confidence(cfg):
    db = dbase.load().loc[cfg["data"]["start"]:cfg["data"]["end"]]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        _shade_below(ax, db["exp"], cfg["estimation"]["threshold"])
        ax.plot(db.index, db["exp"], color="0", lw=1.2)
        ax.axhline(cfg["estimation"]["threshold"], color="0", lw=0.8, ls=":")
        ax.set_ylabel("Puntos")
        _years(ax, 2)
        ax.margins(x=0.01)
        return _save(fig, "fig_confidence")


def fig_stylized_facts(cfg):
    """(a) 10-year rolling correlations of GDP growth with the external variables
    six months earlier, without 2020-03 to 2021-12; (b) subperiod averages."""
    db = dbase.load()
    win, lag = 120, 6
    pand = (db.index >= "2020-03-01") & (db.index <= "2021-12-01")
    y = db["g_gdp"].where(~pand)
    rho = {}
    for col in ("g_ip_chn", "g_xpi"):
        x = db[col].shift(lag).where(~pand)
        rho[col] = x.rolling(win, min_periods=int(0.6 * win)).corr(y)
    periods = {"2003-2012": ("2003-01-01", "2012-12-01"), "2013-2019": ("2013-01-01", "2019-12-01"),
               "2022-2025": ("2022-01-01", "2025-12-01")}
    avg = pd.DataFrame({k: db.loc[a:b, ["g_gdp", "g_tot", "exp"]].mean() for k, (a, b) in periods.items()}).T
    avg.index.name = "periodo"
    avg.round(2).to_csv(results_dir() / "stylized_facts_periods.csv")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        ax.plot(rho["g_ip_chn"].index, rho["g_ip_chn"], color="0", lw=1.2, label="China ($t-6$)")
        ax.plot(rho["g_xpi"].index, rho["g_xpi"], color="0.45", lw=1.2, ls="--", label="Precios de exportación ($t-6$)")
        _zero(ax)
        ax.set_ylabel("Correlación con el crecimiento del PBI")
        _years(ax, 4)
        ax.legend(loc="lower left")
        pa = _save(fig, "fig_facts_rolling")
        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        x = np.arange(len(avg))
        ax.bar(x - 0.2, avg["g_gdp"], 0.38, color="0.75", label="PBI")
        ax.bar(x + 0.2, avg["g_tot"], 0.38, color="0.3", label="Términos de intercambio")
        _zero(ax)
        ax.set_xticks(x); ax.set_xticklabels(avg.index); ax.set_ylabel("Var. % interanual"); ax.set_ylim(-2, 9)
        ax2 = ax.twinx()
        ax2.plot(x, avg["exp"], "D", color="0", ms=5, label="Confianza (eje der.)")
        ax2.axhline(cfg["estimation"]["threshold"], color="0", ls=":", lw=0.8)
        ax2.set_ylim(40, 70); ax2.set_ylabel("Confianza (puntos)")
        ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8,
                  columnspacing=1.0, handlelength=1.5)
        pb = _save(fig, "fig_facts_periods")
    return pa, pb


def fig_variables(cfg):
    db = dbase.load().loc[cfg["data"]["start"]:cfg["data"]["end"]]
    cols = ["g_ip_chn", "g_xpi", "g_gdp", "g_cpi", "int", "exp"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.2), sharex=True)
        for ax, c in zip(axes.ravel(), cols):
            ax.plot(db.index, db[c], color="0", lw=0.9)
            ax.set_title(LABELS[c])
            _years(ax, 4)
        fig.tight_layout()
        return _save(fig, "fig_variables")


# ------------------------------------------------------------------ baseline results
def _girf(name):
    return np.load(results_dir() / f"girf_{name}.npz", allow_pickle=False)


def fig_girf_gdp(cfg):
    """GDP response to each external shock, by regime (one panel per shock)."""
    specs = model_specs(cfg)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(specs), figsize=(6.5, 2.8), squeeze=False)
        for ax, (name, spec) in zip(axes[0], specs.items()):
            z = _girf(name)
            g = list(z["variables"]).index("g_gdp")
            for reg in ("pesimismo", "optimismo"):
                _band(ax, z[f"regime_{reg}"][:, :, g], reg)
            _zero(ax)
            ax.set_xlim(1, z["regime_optimismo"].shape[1]); ax.set_xlabel("Meses")
            ax.set_title(f"({'ab'[list(specs).index(name)]}) {spec.label}, +{spec.shock:g} p.p.")
        axes[0, 0].set_ylabel("Puntos porcentuales")
        axes[0, 0].legend(loc="upper right")
        fig.tight_layout()
        return _save(fig, "fig_girf_gdp")


def fig_girf_all(cfg):
    paths = []
    for name, spec in model_specs(cfg).items():
        z = _girf(name)
        variables = list(z["variables"])
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.6))
            for ax, v in zip(axes.ravel(), variables):
                for reg in ("pesimismo", "optimismo"):
                    _band(ax, z[f"regime_{reg}"][:, :, variables.index(v)], reg)
                _zero(ax); ax.set_title(LABELS[v]); ax.set_xlim(1, z["regime_optimismo"].shape[1])
            axes.ravel()[-1].axis("off")
            h, l = axes[0, 0].get_legend_handles_labels()
            axes.ravel()[-1].legend(h, l, loc="center")
            fig.tight_layout()
            paths.append(_save(fig, f"fig_girf_all_{name}"))
    return paths


def _band_label(group: str) -> str:
    """``"[45, 50)"`` -> ``"45-50"``; open-ended bands as ``"< 45"`` and ``"60 o más"``."""
    lo, hi = (float(v) for v in group.strip("[)").split(","))
    if lo <= 0:
        return f"< {hi:g}"
    if hi >= 100:
        return f"{lo:g} o más"
    return f"{lo:g}-{hi:g}"


def _dot_whisker(ax, x, d, color="0", marker="o", label=None):
    ax.vlines(x, d["p5"], d["p95"], colors=color, lw=0.8)
    ax.vlines(x, d["p16"], d["p84"], colors=color, lw=2.6)
    ax.plot(x, d["p50"], marker, color=color, mfc="white", ms=5, zorder=3, label=label)


def fig_pass_through_bands(cfg):
    """Pass-through by the level of confidence when the shock hits."""
    pt = _gdp(pd.read_csv(results_dir() / "pass_through.csv"))
    pt = pt[pt["variant"] == "base_bands"]
    info = pd.read_csv(results_dir() / "model_info.csv").set_index("model")
    specs = model_specs(cfg)
    horizons = cfg["girf"]["pass_through"][:2]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(specs), len(horizons), figsize=(6.5, 2.6 * len(specs)), squeeze=False)
        for i, (name, spec) in enumerate(specs.items()):
            counts = json.loads(info.loc[name, "band_months"])
            for j, H in enumerate(horizons):
                ax = axes[i, j]
                d = pt[(pt["model"] == name) & (pt["horizonte"] == H)].reset_index(drop=True)
                for k, row in d.iterrows():
                    upper = float(row["group"].split(",")[1].strip(" )"))
                    _dot_whisker(ax, k, row, color="0.5" if upper <= cfg["estimation"]["threshold"] else "0")
                _zero(ax)
                ax.set_xticks(range(len(d)))
                ax.set_xticklabels([f"{_band_label(g)}\n({counts[g]})" for g in d["group"]], fontsize=8)
                ax.set_title(f"({'abcd'[2 * i + j]}) {spec.short}, +{spec.shock:g} p.p.: {H} meses")
            axes[i, 0].set_ylabel("Traspaso al PBI (p.p.)")
        fig.supxlabel(f"Confianza empresarial {_band_delay()} meses antes del choque (puntos; meses en el tramo)",
                      fontsize=9)
        fig.tight_layout()
        return _save(fig, "fig_pass_through_bands")


def fig_threshold_grid(cfg):
    info = pd.read_csv(results_dir() / "threshold_grid_info.csv")
    pt = _gdp(pd.read_csv(results_dir() / "threshold_grid_pass_through.csv"))
    specs = model_specs(cfg)
    H = cfg["girf"]["pass_through"][0]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(specs), figsize=(6.5, 5.0), sharex=True, squeeze=False)
        for j, (name, spec) in enumerate(specs.items()):
            ax = axes[0, j]
            for k, reg in enumerate(("pesimismo", "optimismo")):
                d = pt[(pt["model"] == name) & (pt["group"] == reg) & (pt["horizonte"] == H)].sort_values("grid_threshold")
                x = d["grid_threshold"] + (k - 0.5) * 0.3
                line, _, _, label = REGIME[reg]
                ax.vlines(x, d["p16"], d["p84"], colors=line, lw=2)
                ax.plot(x, d["p50"], "o" if reg == "optimismo" else "s", color=line, mfc="white", ms=4, label=label)
            _zero(ax)
            ax.set_title(f"({'ab'[j]}) {spec.short}: traspaso a {H} meses")
            ax = axes[1, j]
            d = info[info["model"] == name].sort_values("grid_threshold")
            ax.plot(d["grid_threshold"], d["log_ml_tvar"] - d["log_ml_tvar"].max(), "o-", color="0", ms=3, lw=0.9)
            ax.axvline(cfg["estimation"]["threshold"], color="0.5", ls=":", lw=0.8)
            ax.set_xlabel("Umbral de confianza (puntos)")
            if j == 0:
                ax.set_ylabel("log verosimilitud marginal\n(respecto al máximo)")
        axes[0, 0].set_ylabel("Traspaso al PBI (p.p.)")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return _save(fig, "fig_threshold_grid")


def _bin_probability(d: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Posterior mass per bin; the density is flat within each split interval under the flat prior."""
    out = np.zeros(len(edges) - 1)
    for lo, up, prob in d[["lower", "upper", "probability"]].itertuples(index=False):
        if up <= lo:
            k = np.searchsorted(edges, lo, side="right") - 1
            if 0 <= k < len(out):
                out[k] += prob
            continue
        overlap = np.clip(np.minimum(edges[1:], up) - np.maximum(edges[:-1], lo), 0, None)
        out += prob * overlap / (up - lo)
    return out


def fig_threshold_posterior(cfg):
    """Posterior probability of the threshold in one-point bins, delay integrated out."""
    sp = pd.read_csv(results_dir() / "threshold_posterior.csv")
    specs = model_specs(cfg)
    edges = np.arange(35.0, 61.0, 1.0)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(specs), figsize=(6.5, 2.5), squeeze=False, sharey=True)
        for ax, (name, spec) in zip(axes[0], specs.items()):
            d = sp[sp["model"] == name]
            mass = _bin_probability(d, edges)
            ax.bar(edges[:-1], mass, width=1.0, align="edge", color="0.55", edgecolor="white", lw=0.5)
            delay = d.groupby("delay")["probability"].sum()
            ax.text(0.97, 0.95, f"Pr($d$ = {delay.idxmax()}) = {delay.max():.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8)
            ax.axvline(cfg["estimation"]["threshold"], color="0", ls=":", lw=0.8)
            ax.set_title(f"({'ab'[list(specs).index(name)]}) {spec.label}"); ax.set_xlabel("Umbral (puntos)")
            ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, 1.05)
        axes[0, 0].set_ylabel("Probabilidad posterior")
        fig.tight_layout()
        return _save(fig, "fig_threshold_posterior")


def fig_pandemic_dummies(cfg, name="ipx"):
    """Posterior of the pandemic dummies (one-step residuals of the dropped months)."""
    res = pd.read_csv(results_dir() / f"pandemic_dummies_{name}.csv.gz", parse_dates=["date"])
    q = res.groupby(["variable", "date"])["value"].quantile([0.05, 0.16, 0.5, 0.84, 0.95]).unstack()
    q.columns = ["p5", "p16", "p50", "p84", "p95"]
    variables = list(model_specs(cfg)[name].variables)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.2), sharex=True)
        for ax, v in zip(axes.ravel(), variables):
            d = q.loc[v].reset_index()
            for k, row in d.iterrows():
                _dot_whisker(ax, k, row)
            _zero(ax); ax.set_title(LABELS[v])
            ax.set_xticks(range(0, len(d), 3)); ax.set_xticklabels([f"{t:%Y-%m}" for t in d["date"][::3]], rotation=45, fontsize=8)
        axes.ravel()[-1].axis("off")
        axes[-2, -1].xaxis.set_tick_params(labelbottom=True)  # the panel above the empty slot needs dates
        fig.tight_layout()
        return _save(fig, "fig_pandemic_dummies")


def fig_methods(cfg):
    """Pass-through by regime: conventional regime IRF, GIRF and state-dependent local projections."""
    pt = _gdp(pd.read_csv(results_dir() / "pass_through.csv"))
    lp = pd.read_csv(results_dir() / "lp_pass_through.csv")
    lp = lp[(lp["sample"] == "completa_sin_covid") & (lp["threshold"] == cfg["estimation"]["threshold"])]
    sources = [("IRF convencional por régimen", pt[pt["variant"] == "base_linear_irf"], "0.6", "s"),
               ("GIRF (base)", pt[pt["variant"] == "base"], "0", "o"),
               ("Proyecciones locales", lp, "0.35", "^")]
    specs = model_specs(cfg)
    horizons = cfg["lp"]["horizons"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(specs), len(horizons), figsize=(6.5, 2.6 * len(specs)), squeeze=False)
        for i, (name, spec) in enumerate(specs.items()):
            for j, H in enumerate(horizons):
                ax = axes[i, j]
                for r, reg in enumerate(("pesimismo", "optimismo")):
                    for k, (label, d, color, marker) in enumerate(sources):
                        row = d[(d["model"] == name) & (d["group"] == reg) & (d["horizonte"] == H)]
                        if row.empty or row["p50"].isna().all():
                            continue
                        x = r * 4 + k
                        row = row.iloc[0]
                        ax.vlines(x, row["p16"], row["p84"], colors=color, lw=2.4)
                        ax.plot(x, row["p50"], marker, color=color, mfc="white", ms=5, zorder=3,
                                label=label if (r == 0 and i == 0 and j == 0) else None)
                _zero(ax)
                ax.set_xticks([1, 5]); ax.set_xticklabels(["Pesimismo", "Optimismo"])
                ax.set_title(f"({'abcd'[2 * i + j]}) {spec.short}, +{spec.shock:g} p.p.: {H} meses")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return _save(fig, "fig_methods")


def fig_robustness(cfg):
    """GDP response under alternative specifications against the baseline, by regime."""
    base = {name: _girf(name) for name in model_specs(cfg)}
    rob = np.load(results_dir() / "girf_robustness.npz")
    pairs = [("chn", "usa__usa", "Prod. industrial de EE.UU."), ("ipx", "tot__tot", "Términos de intercambio")]
    specs = model_specs(cfg)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(pairs), figsize=(6.5, 4.8), sharex=True)
        for j, (b, alt, title) in enumerate(pairs):
            g = list(base[b]["variables"]).index("g_gdp")
            for i, reg in enumerate(("pesimismo", "optimismo")):
                ax = axes[i, j]
                _band(ax, base[b][f"regime_{reg}"][:, :, g], reg, f"Base: {specs[b].label}")
                arr = rob[f"{alt}__{reg}"][:, :, 1]
                ax.plot(np.arange(1, arr.shape[1] + 1), np.median(arr, axis=0), color="0", lw=1.0, ls=":", label=title)
                _zero(ax)
                ax.set_title(f"({'abcd'[2 * i + j]}) {'Pesimismo' if reg == 'pesimismo' else 'Optimismo'}", fontsize=9)
                if j == 0:
                    ax.set_ylabel("PBI (p.p.)")
                ax.legend(fontsize=7, loc="upper right")
        for ax in axes[1]:
            ax.set_xlabel("Meses")
        fig.tight_layout()
        return _save(fig, "fig_robustness")


def fig_joint_block(cfg):
    """GDP response to both shocks in the joint system with block exogeneity, against the baseline medians."""
    rob = np.load(results_dir() / "girf_robustness.npz")
    specs = model_specs(variant_config(cfg, "joint_block"))
    base = {name: _girf(name) for name in model_specs(cfg)}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(specs), figsize=(6.5, 2.8), squeeze=False)
        for ax, letter, (name, spec) in zip(axes[0], "ab", specs.items()):
            g = spec.variables.index("g_gdp")
            b = base[name.split("_")[0]]
            gb = list(b["variables"]).index("g_gdp")
            for reg in ("pesimismo", "optimismo"):
                _band(ax, rob[f"joint_block__{name}__{reg}"][:, :, g], reg)
                med = np.median(b[f"regime_{reg}"][:, :, gb], axis=0)
                ax.plot(np.arange(1, med.size + 1), med, color=REGIME[reg][0], ls=":", lw=1.0,
                        label=f"Base, {reg}" if reg == "pesimismo" else "Base, optimismo")
            _zero(ax)
            ax.set_xlim(1, med.size); ax.set_xlabel("Meses")
            ax.set_title(f"({letter}) {spec.label}, +{spec.shock:g} p.p.")
        axes[0, 0].set_ylabel("Puntos porcentuales")
        axes[0, 0].legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        return _save(fig, "fig_joint_block")


def fig_model(cfg, horizon: int = 48):
    """Structural model against the TVAR: GDP (top) and investment (bottom) by regime."""
    irf = np.load(results_dir() / "structural_irfs.npz")
    h = np.arange(1, horizon + 1)
    titles = {"chn": "Producción industrial de China, +1 p.p.", "ipx": "Precios de exportación, +10 p.p."}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
        for j, shock in enumerate(("chn", "ipx")):
            tvar = np.load(results_dir() / f"girf_{shock}.npz")
            ax = axes[0, j]
            for reg in ("optimismo", "pesimismo"):
                line, _, ls, label = REGIME[reg]
                ax.plot(h, irf[f"modelo__{shock}__{reg}__y"][:horizon], color=line, ls=ls, lw=1.5,
                        label=f"Modelo, {reg}")
                med = np.median(tvar[f"regime_{reg}"], axis=0)[:horizon, 1]
                ax.plot(h[::3], med[::3], ls="none", marker="o" if reg == "optimismo" else "^", ms=3.2,
                        mfc="white" if reg == "pesimismo" else "0", mec=line, mew=0.8, label=f"TVAR, {reg}")
            if shock == "ipx":
                ax.plot(h, irf[f"misma_respuesta__{shock}__pesimismo__y"][:horizon], color="0.45", ls=":", lw=1.2,
                        label="Modelo, pesimismo con la misma\nrespuesta de la inversión")
            _zero(ax)
            ax.set_title(f"({'ab'[j]}) PBI: {titles[shock]}")
            ax = axes[1, j]
            for reg in ("optimismo", "pesimismo"):
                line, _, ls, _ = REGIME[reg]
                ax.plot(h, irf[f"modelo__{shock}__{reg}__i"][:horizon], color=line, ls=ls, lw=1.5)
            _zero(ax)
            ax.set_title(f"({'cd'[j]}) Inversión: {titles[shock]}")
            ax.set_xlabel("Meses")
        axes[0, 0].set_ylabel("p.p."); axes[1, 0].set_ylabel("p.p.")
        handles, labels = [], []
        for ax in axes[0]:
            for hd, lb in zip(*ax.get_legend_handles_labels()):
                if lb not in labels:
                    handles.append(hd); labels.append(lb)
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8)
        fig.tight_layout()
        return _save(fig, "fig_model")


def make_all(cfg) -> None:
    for f in (fig_confidence, fig_stylized_facts, fig_variables, fig_girf_gdp, fig_girf_all,
              fig_pass_through_bands, fig_threshold_grid, fig_threshold_posterior, fig_pandemic_dummies,
              fig_robustness, fig_joint_block, fig_methods, fig_model):
        try:
            f(cfg)
            print(f"  {f.__name__}: ok", flush=True)
        except FileNotFoundError as err:
            print(f"  {f.__name__}: skipped ({err.filename} missing; run its stage first)", flush=True)
