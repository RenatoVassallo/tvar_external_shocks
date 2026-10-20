"""LaTeX tables of the paper (``output/tables``), drawn from ``output/results``.

Plain ``booktabs`` tables with ``threeparttable`` notes; no colours.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import model_specs, results_dir, tables_dir, variant_config
from .model import asymmetry

__all__ = ["make_all"]

# Table 4: alternative specifications of the baseline (the state and sample checks have their own table)
TABLE4_VARIANTS = ["lenza_primiceri", "prior_variance", "delay_2", "prior_loose", "usa", "tot", "joint_block",
                   "tot_block", "fiscal"]
REGIME = {"pesimismo": "Pesimismo ($c_{t-d} \\leq 50$)", "optimismo": "Optimismo ($c_{t-d} > 50$)"}


def _read_pt(name: str, response: str = "g_gdp") -> pd.DataFrame:
    pt = pd.read_csv(results_dir() / name)
    return pt[pt["response"] == response] if "response" in pt else pt


def _fmt(x: float) -> str:
    return "n.d." if pd.isna(x) else f"{x:.2f}".replace("-0.00", "0.00")


def _cell(r) -> str:
    return f"{_fmt(r['p50'])} & [{_fmt(r['p16'])}, {_fmt(r['p84'])}]"


def _write(name: str, body: str) -> None:
    (tables_dir() / f"{name}.tex").write_text(body, encoding="utf-8")


def _table(caption: str, label: str, colspec: str, header: str, rows: str, note: str,
           placement: str = "t", source: str = "Estimaciones de los autores") -> str:
    """Booktabs table without outer column padding; ``placement="H"`` pins appendix tables.
    Every table carries a source line, as the journal requires."""
    return (f"\\begin{{table}}[{placement}]\n\\centering\n\\begin{{threeparttable}}\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n\\small\n\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}\n\\toprule\n{header}\\midrule\n{rows}"
            f"\\bottomrule\n\\end{{tabular}}\n\\begin{{tablenotes}}[flushleft]\n\\footnotesize\n"
            f"\\item \\textit{{Nota:}} {note} \\textit{{Fuente:}} {source}.\n\\end{{tablenotes}}\n"
            f"\\end{{threeparttable}}\n\\end{{table}}\n")


def _horizon_header(H) -> str:
    top = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h} meses}}" for h in H)
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(H)))
    sub = " & ".join("Mediana & Int. 68\\%" for _ in H)
    return f" & {top} \\\\\n{rules}\n & {sub} \\\\\n"


def table_pass_through(cfg) -> None:
    pt = _read_pt("pass_through.csv")
    pt = pt[pt["variant"] == "base"]
    asym = pd.read_csv(results_dir() / "asymmetry.csv")
    asym = asym[asym["method"] == "girf"]
    H = cfg["girf"]["pass_through"]
    rows = ""
    for name, spec in model_specs(cfg).items():
        rows += f"\\multicolumn{{{1 + 2 * len(H)}}}{{l}}{{\\textit{{{spec.label}, +{spec.shock:g} p.p.}}}} \\\\\n"
        for reg in ("pesimismo", "optimismo"):
            d = pt[(pt["model"] == name) & (pt["group"] == reg)].set_index("horizonte")
            rows += f"\\quad {REGIME[reg]} & " + " & ".join(_cell(d.loc[h]) for h in H) + " \\\\\n"
        a = asym[asym["model"] == name].set_index("horizonte")
        rows += ("\\quad $\\Pr(\\text{optimismo} > \\text{pesimismo})$ & "
                 + " & ".join(f"\\multicolumn{{2}}{{c}}{{{a.loc[h, 'prob_optimism_larger']:.2f}}}" for h in H)
                 + " \\\\\n\\addlinespace\n")
    note = ("Efecto traspaso: respuesta acumulada del crecimiento del PBI dividida por la respuesta acumulada "
            "de la variable externa, escalada al tamaño del choque, calculada extracción por extracción a "
            "partir de funciones impulso-respuesta generalizadas que promedian sobre las historias de cada "
            "régimen y permiten que el régimen cambie a lo largo del horizonte. Mediana y percentiles 16 y 84 "
            "de la distribución posterior. La última fila de cada panel es la proporción de extracciones en "
            "que el traspaso en optimismo supera al de pesimismo.")
    _write("tab_pass_through", _table("Efecto traspaso de los choques externos al PBI", "tab:pass_through",
                                      "l" + "cc" * len(H), _horizon_header(H), rows, note))


def table_pass_through_bands(cfg) -> None:
    pt = _read_pt("pass_through.csv")
    pt = pt[pt["variant"] == "base_bands"]
    info = pd.read_csv(results_dir() / "model_info.csv").set_index("model")
    H = cfg["girf"]["pass_through"]
    rows = ""
    for name, spec in model_specs(cfg).items():
        counts = json.loads(info.loc[name, "band_months"])
        rows += f"\\multicolumn{{{1 + 2 * len(H)}}}{{l}}{{\\textit{{{spec.label}, +{spec.shock:g} p.p.}}}} \\\\\n"
        d = pt[pt["model"] == name]
        for group in d["group"].unique():
            lo, hi = (float(v) for v in group.strip("[)").split(","))
            label = (f"$c < {hi:g}$" if lo <= 0 else
                     (f"$c \\geq {lo:g}$" if hi >= 100 else f"${lo:g} \\leq c < {hi:g}$"))
            dd = d[d["group"] == group].set_index("horizonte")
            rows += (f"\\quad {label} ({counts[group]} meses) & "
                     + " & ".join(_cell(dd.loc[h]) for h in H) + " \\\\\n")
        rows += "\\addlinespace\n"
    delay = int(info["band_delay"].iloc[0])
    note = (f"Funciones impulso-respuesta generalizadas promediadas sobre los meses en que la confianza "
            f"empresarial $c$, {delay} meses antes del choque, se ubicaba en cada intervalo. Mismo modelo y "
            f"misma distribución posterior que la Tabla~\\ref{{tab:pass_through}}; el régimen puede cambiar a "
            f"lo largo del horizonte.")
    _write("tab_pass_through_bands", _table("Efecto traspaso según el nivel de confianza al momento del choque",
                                            "tab:pass_through_bands", "l" + "cc" * len(H), _horizon_header(H),
                                            rows, note, placement="H"))


def _variant_asymmetry(cfg, variant: str, model: str, response: str, H: int) -> float:
    """Pr(regime 2 > regime 1) for a robustness variant, draw by draw from its saved GIRFs."""
    spec = model_specs(variant_config(cfg, variant))[model]
    z = np.load(results_dir() / "girf_robustness.npz")
    resp = {g: z[f"{variant}__{model}__{g}"] for g in ("pesimismo", "optimismo")}
    return float(asymmetry(resp, spec, [H], response).loc[H, "prob_optimism_larger"])


def _state_row(label, d, prob=None) -> str:
    cells = [d[d["group"] == g].iloc[0] for g in ("pesimismo", "optimismo")]
    p = "" if prob is None or pd.isna(prob) else f"{prob:.2f}"
    return f"{label} & " + " & ".join(_cell(c) for c in cells) + f" & {p} \\\\\n"


def table_state(cfg) -> None:
    """Pass-through by sample: the full sample against the sample that ends in 2019."""
    H = cfg["girf"]["pass_through"][0]
    base = _read_pt("pass_through.csv")
    base = base[(base["variant"] == "base") & (base["horizonte"] == H)]
    asym = pd.read_csv(results_dir() / "asymmetry.csv")
    asym = asym[(asym["method"] == "girf") & (asym["horizonte"] == H)].set_index("model")
    rob = _read_pt("pass_through_robustness.csv")
    rob = rob[rob["horizonte"] == H]
    rows = ""
    for shock, label in (("chn", "Producción industrial de China, +1 p.p."),
                         ("ipx", "Índice de precios de exportación, +10 p.p.")):
        rows += f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\\n"
        for text, variant in (("2002-2025 (base)", None), ("2002-2019", "pre2020")):
            if variant is None:
                d, prob = base[base["model"] == shock], asym.loc[shock, "prob_optimism_larger"]
            else:
                d = rob[(rob["variant"] == variant) & (rob["model"] == shock)]
                prob = _variant_asymmetry(cfg, variant, shock, "g_gdp", H)
            rows += _state_row(f"\\quad {text}", d, prob)
        rows += "\\addlinespace\n"
    header = (" & \\multicolumn{2}{c}{Pesimismo} & \\multicolumn{2}{c}{Optimismo} & \\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
              " & Mediana & Int. 68\\% & Mediana & Int. 68\\% & Pr \\\\\n")
    note = (f"Efecto traspaso al PBI a {H} meses, como en la Tabla~\\ref{{tab:pass_through}}. La muestra hasta "
            "2019 se estima sin tratamiento de la pandemia y deja 29 meses de pesimismo. Pr: probabilidad "
            "posterior de que el traspaso sea mayor en optimismo.")
    _write("tab_state", _table(f"Efecto traspaso a {H} meses en la muestra completa y hasta 2019",
                               "tab:state", ">{\\raggedright\\arraybackslash}p{6.2cm}ccccc", header, rows, note))


def table_investment(cfg) -> None:
    """Pass-through to capital goods imports (volume) by state."""
    H = cfg["girf"]["pass_through"][:2]
    rob = _read_pt("pass_through_robustness.csv", response="g_kimp")
    conf = ("pesimismo", "optimismo", "$\\Pr(\\text{optimismo} > \\text{pesimismo})$")
    specs = [("2008-2025", "investment", "inv", conf),
             ("2008-2019", "investment_pre2020", "inv", conf)]
    rows = ""
    for shock, label in (("chn", "Producción industrial de China, +1 p.p."),
                         ("ipx", "Índice de precios de exportación, +10 p.p.")):
        rows += f"\\multicolumn{{{1 + 2 * len(H)}}}{{l}}{{\\textit{{{label}}}}} \\\\\n"
        for text, variant, suffix, names in specs:
            model = f"{shock}_{suffix}"
            d = rob[(rob["variant"] == variant) & (rob["model"] == model)]
            rows += f"\\quad {text} & & & & \\\\\n"
            for g, gl in zip(("pesimismo", "optimismo"), names[:2]):
                dd = d[d["group"] == g].set_index("horizonte")
                rows += f"\\qquad {gl.capitalize()} & " + " & ".join(_cell(dd.loc[h]) for h in H) + " \\\\\n"
            probs = [_variant_asymmetry(cfg, variant, model, "g_kimp", h) for h in H]
            rows += (f"\\qquad {names[2]} & "
                     + " & ".join(f"\\multicolumn{{2}}{{c}}{{{p:.2f}}}" for p in probs) + " \\\\\n")
        rows += "\\addlinespace\n"
    note = ("Respuesta acumulada del crecimiento interanual del volumen de importación de bienes de capital "
            "(sin materiales de construcción, BCRP) dividida por la respuesta acumulada de la variable externa, "
            "escalada al tamaño del choque, en el modelo base ampliado con esa serie ordenada después del PBI. "
            "La serie empieza en 2007, de modo que la muestra empieza en 2008. Mediana y percentiles 16 y 84; la "
            "probabilidad se calcula extracción por extracción.")
    _write("tab_investment", _table("Respuesta de la importación de bienes de capital según el estado",
                                    "tab:investment", "l" + "cc" * len(H), _horizon_header(H), rows, note))

FIXED_DOC = [  # parameter, symbol, rationale
    ("w_c", "$w_c$", "Consumo privado (64\\% del PBI) por su contenido doméstico (85\\%)"),
    ("w_i", "$w_i$", "Inversión privada (20\\% del PBI) por su contenido doméstico (55\\%)"),
    ("w_x", "$w_x$", "Exportaciones (26\\% del PBI)"),
    ("h", "$h$", "Inercia mensual del consumo"),
    ("mpc", "$m$", "Respuesta del consumo al ingreso"),
    ("chi", "$\\chi$", "Parte del ingreso extraordinario que reciben los hogares, en unidades de consumo"),
    ("beta", "$\\beta$", "Descuento mensual, incluida la depreciación"),
    ("rho_i", "$\\rho_i$", "Ajuste gradual de la inversión (vida media de 6.6 meses)"),
    ("a_y", "$a_y$", "Sensibilidad de la inversión a la demanda esperada"),
    ("tau_p", "$\\tau_p$", "Elasticidad de los ingresos fiscales al precio de exportación"),
    ("tau_y", "$\\tau_y$", "Elasticidad de los ingresos fiscales a la actividad"),
]


def table_calibration(cfg) -> None:
    """Parameters of the structural model and sensitivity of its predictions."""
    cal = pd.read_csv(results_dir() / "structural_calibration.csv").iloc[0]
    sens = pd.read_csv(results_dir() / "structural_sensitivity.csv")
    rows = "\\multicolumn{3}{l}{\\textit{Fijados con las cuentas nacionales (BCRP, promedios 2010-2019)}} \\\\\n"
    for key, sym, text in FIXED_DOC[:3]:
        rows += f"\\quad {sym} & {cal[key]:g} & {text} \\\\\n"
    rows += "\\addlinespace\n\\multicolumn{3}{l}{\\textit{Supuestos del ejercicio}} \\\\\n"
    for key, sym, text in FIXED_DOC[3:]:
        rows += f"\\quad {sym} & {cal[key]:g} & {text} \\\\\n"
    rows += "\\addlinespace\n\\multicolumn{3}{l}{\\textit{Calibrados con el traspaso a 12 meses del TVAR}} \\\\\n"
    for key, sym, text in (("eta", "$\\eta$", "Elasticidad de las exportaciones a la demanda externa"),
                           ("a_p", "$a_p$", "Sensibilidad de la inversión al precio esperado"),
                           ("phi_pes", "$\\phi_{\\text{pes}}$",
                            "Sensibilidad relativa de la inversión en pesimismo ($\\phi = 1$ en optimismo)")):
        rows += f"\\quad {sym} & {cal[key]:.2f} & {text} \\\\\n"
    rng = lambda c: f"{sens[c].min():.2f} a {sens[c].max():.2f}"
    note = (f"Modelo de la Sección~\\ref{{sec:mecanismo}}, en frecuencia mensual. Las participaciones son las "
            "de las cuentas nacionales del BCRP multiplicadas por el contenido doméstico de cada componente; "
            "los parámetros de dinámica y de respuesta son supuestos del ejercicio dentro de rangos usuales en "
            "la literatura, sin una fuente específica. Los tres parámetros calibrados reproducen el traspaso a "
            "12 meses de la demanda externa en optimismo y de los precios de exportación en ambos regímenes. "
            "Sensibilidad: con $a_y \\in \\{"
            + ", ".join(f"{v:g}" for v in sorted(sens['a_y'].unique())) + "\\}$ y $m \\in \\{"
            + ", ".join(f"{v:g}" for v in sorted(sens['mpc'].unique())) + "\\}$, "
            f"$\\phi_{{\\text{{pes}}}}$ va de {rng('phi_pes')}, el traspaso predicho de la demanda externa en "
            f"pesimismo va de {rng('chn_pes_12')} (0.20 en el TVAR) y el traspaso de los precios en pesimismo "
            f"con la sensibilidad de optimismo va de {rng('ipx_pes_12_same_phi')}.")
    header = "Parámetro & Valor & Descripción \\\\\n"
    _write("tab_calibration", _table("Parámetros del modelo", "tab:calibration",
                                     "cc>{\\raggedright\\arraybackslash}p{10cm}", header, rows, note,
                                     placement="H", source="Cálculos de los autores"))


PAPER_VARIANTS = ["lenza_primiceri", "pre2020", "usa", "tot", "joint_block"]   # Table 3 of the paper


def _robustness_table(cfg, variants, lp_thresholds, name: str, label: str, caption: str) -> None:
    """Pass-through at 12 months by specification, with Pr(optimism > pessimism) where it is defined."""
    H = cfg["girf"]["pass_through"][0]
    est_thr = cfg["estimation"]["threshold"]
    base = _read_pt("pass_through.csv")
    rob = _read_pt("pass_through_robustness.csv")
    rob = rob[rob["variant"].isin(variants)]
    rob["order"] = rob["variant"].map({v: i for i, v in enumerate(variants)})
    rob = rob.sort_values("order", kind="stable").drop(columns="order")
    frames = [base[base["variant"].isin(["base", "base_linear_irf"])], rob]
    lp = pd.read_csv(results_dir() / "lp_pass_through.csv")
    lp = lp[(lp["sample"] == "completa_sin_covid") & lp["threshold"].isin(lp_thresholds)]
    frames.append(lp.assign(variant=lp["threshold"].map(lambda t: f"lp_{t:g}")))
    pt = pd.concat(frames)
    asym = pd.read_csv(results_dir() / "asymmetry.csv")
    asym = asym[asym["horizonte"] == H].set_index(["model", "method"])
    descr = {"base": "Especificación base (GIRF)",
             "base_linear_irf": "IRF convencional por régimen",
             f"lp_{est_thr:g}": "Proyecciones locales",
             "lp_45": "Proyecciones locales, pesimismo profundo ($c < 45$)",
             **{k: v.get("short", v["description"]) for k, v in cfg["robustness"]["variants"].items()}}
    shock = {"chn": "China", "ipx": "IPX", "usa": "EE.UU.", "tot": "TI"}
    rows = ""
    for (variant, model), d in pt[pt["horizonte"] == H].groupby(["variant", "model"], sort=False):
        cells = [d[d["group"] == reg].iloc[0] for reg in ("pesimismo", "optimismo")]
        if variant == "base":
            prob = asym.loc[(model, "girf"), "prob_optimism_larger"]
        elif variant == "base_linear_irf":
            prob = asym.loc[(model, "linear_irf"), "prob_optimism_larger"]
        elif variant.startswith("lp_"):
            prob = None
        else:
            prob = _variant_asymmetry(cfg, variant, model, "g_gdp", H)
        rows += (f"{descr.get(variant, variant)} & {shock[model.split('_')[0]]} & "
                 + " & ".join(_cell(c) for c in cells)
                 + (" & " if prob is None else f" & {prob:.2f}") + " \\\\\n")
    header = (" & & \\multicolumn{2}{c}{Pesimismo} & \\multicolumn{2}{c}{Optimismo} & \\\\\n"
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
              "Especificación & Choque & Mediana & Int. 68\\% & Mediana & Int. 68\\% & Pr \\\\\n")
    note = ("Efecto traspaso al PBI a 12 meses, muestra 2002-2025 salvo donde se indica. China e IPX: choques de "
            "+1 p.p. en la producción industrial de China y de +10 p.p. en el índice de precios de exportación; "
            "EE.UU.: +1 p.p. en la producción industrial de EE.UU.; TI: +10 p.p. en los términos de intercambio. "
            "Pr: probabilidad posterior de que el traspaso sea mayor en optimismo, calculada extracción por "
            "extracción. Las proyecciones locales son el multiplicador acumulado de Ramey y Zubairy (2018) "
            "estimado por mínimos cuadrados en dos etapas en cada estado, sin los meses de 2020-21, con bandas "
            "de $\\pm 1$ error estándar de Newey-West, no comparables con los intervalos de credibilidad; el "
            f"estado es la confianza {cfg['lp']['delay']} meses antes del choque.")
    _write(name, _table(caption, label, ">{\\raggedright\\arraybackslash}p{5.2cm}lccccc", header, rows, note))


def table_robustness(cfg) -> None:
    est_thr = cfg["estimation"]["threshold"]
    _robustness_table(cfg, PAPER_VARIANTS, [est_thr], "tab_robustness", "tab:robustness",
                      "Efecto traspaso a 12 meses bajo especificaciones alternativas")
    _robustness_table(cfg, TABLE4_VARIANTS + ["pre2020"], [est_thr, 45.0], "tab_robustness_full",
                      "tab:robustness_full", "Efecto traspaso a 12 meses bajo todas las especificaciones alternativas")


def table_fiscal(cfg) -> None:
    pt = pd.read_csv(results_dir() / "pass_through_robustness.csv")
    pt = pt[pt["variant"] == "fiscal"]
    H = cfg["girf"]["pass_through"][:2]                 # 12 and 24 months: the 60-month horizon adds width, not information
    names = {"g_rev": "Ingresos", "g_gdp": "PBI"}
    labels = {"chn_fiscal": "Producción industrial de China, +1 p.p.",
              "ipx_fiscal": "Índice de precios de exportación, +10 p.p."}
    rows = ""
    for name, d in pt.groupby("model", sort=False):
        rows += f"\\multicolumn{{{1 + 2 * len(H)}}}{{l}}{{\\textit{{{labels[name]}}}}} \\\\\n"
        for response in ("g_rev", "g_gdp"):
            for reg in ("pesimismo", "optimismo"):
                dd = d[(d["response"] == response) & (d["group"] == reg)].set_index("horizonte")
                rows += f"\\quad {names[response]}, {reg} & " + " & ".join(_cell(dd.loc[h]) for h in H) + " \\\\\n"
            probs = [_variant_asymmetry(cfg, "fiscal", name, response, h) for h in H]
            rows += (f"\\quad {names[response]}, $\\Pr(\\text{{opt.}} > \\text{{pes.}})$ & "
                     + " & ".join(f"\\multicolumn{{2}}{{c}}{{{p:.2f}}}" for p in probs) + " \\\\\n")
        rows += "\\addlinespace\n"
    note = ("Modelo base ampliado con los ingresos corrientes reales del gobierno central (variación porcentual "
            "interanual de la suma móvil de 12 meses), ordenados después del PBI; muestra 2002-2025. Efecto "
            "traspaso calculado con funciones impulso-respuesta generalizadas, como en la "
            "Tabla~\\ref{tab:pass_through}: mediana y percentiles 16 y 84 de la distribución posterior, y "
            "proporción de extracciones en que el traspaso en optimismo supera al de pesimismo.")
    _write("tab_fiscal", _table("Efecto traspaso a los ingresos fiscales y al PBI", "tab:fiscal",
                                "l" + "cc" * len(H), _horizon_header(H), rows, note))


def table_prior_grid(cfg) -> None:
    g = pd.read_csv(results_dir() / "prior_grid.csv")
    g["rel"] = g.groupby("model")["log_ml_tvar"].transform(lambda s: s - s.max())
    wide = g.pivot_table(index=["lamdaP", "epsilonP"], columns="model", values=["rel", "log_bayes_factor"])
    rows = ""
    for (lam, eps), r in wide.iterrows():
        rows += (f"{lam:g} & {eps:g} & {r[('rel', 'chn')]:.1f} & {r[('log_bayes_factor', 'chn')]:.1f} & "
                 f"{r[('rel', 'ipx')]:.1f} & {r[('log_bayes_factor', 'ipx')]:.1f} \\\\\n")
    header = (" & & \\multicolumn{2}{c}{China} & \\multicolumn{2}{c}{IPX} \\\\\n"
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n$\\lambda$ & $\\varepsilon$ & $\\Delta \\log p(Y)$ & "
              "$\\log BF$ & $\\Delta \\log p(Y)$ & $\\log BF$ \\\\\n")
    note = ("$\\Delta \\log p(Y)$: logaritmo de la verosimilitud marginal del TVAR respecto al máximo de la "
            "grilla. $\\log BF$: logaritmo del factor de Bayes del TVAR frente al VAR lineal con el mismo prior. "
            "$\\tau = 10\\lambda$ en todos los casos; umbral en 50 y rezago integrado sobre 1 a 6 meses.")
    _write("tab_prior_grid", _table("Verosimilitud marginal según los hiperparámetros del prior",
                                    "tab:prior_grid", "cccccc", header, rows, note, placement="H"))


def table_models(cfg) -> None:
    """Sample, regime sizes and marginal likelihoods (CSV, for the text)."""
    frames = [pd.read_csv(results_dir() / f) for f in ("model_info.csv", "model_info_robustness.csv",
                                                      "model_info_estimated_threshold.csv")
              if (results_dir() / f).exists()]
    pd.concat(frames, ignore_index=True).to_csv(tables_dir() / "models_summary.csv", index=False)
    for f in ("asymmetry.csv", "lp_pass_through.csv"):
        if (results_dir() / f).exists():
            pd.read_csv(results_dir() / f).round(3).to_csv(tables_dir() / f, index=False)


def make_all(cfg) -> None:
    for f in (table_pass_through, table_pass_through_bands, table_robustness, table_state, table_investment,
              table_fiscal, table_prior_grid, table_calibration, table_models):
        try:
            f(cfg)
            print(f"  {f.__name__}: ok", flush=True)
        except FileNotFoundError as err:
            print(f"  {f.__name__}: skipped ({err.filename} missing; run its stage first)", flush=True)
