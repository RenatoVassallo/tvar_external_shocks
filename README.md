# Does the transmission of external shocks depend on the state of business confidence?

Replication package for Sánchez and Vassallo, *¿Depende la transmisión de los choques externos del
estado de la confianza empresarial? Evidencia para el Perú, 2002-2025* (revised version). All the
code, figures and tables are produced in Python with [`MacroPy`](https://github.com/RenatoVassallo/MacroPy)
from a frozen copy of the data. The paper is written in Spanish; the labels printed in the figures and
tables are therefore in Spanish, while the code, the configuration and this documentation are in English.

## Contents

```
config.toml             every specification decision (sample, prior, threshold, COVID, GIRF, robustness, model)
data/                   frozen database, series dictionary and sources (see data/README.md)
src/tvar_shocks/        the package
  config.py             reads config.toml and builds the model specifications
  data.py               loads the frozen database; rebuilds it from the sources
  model.py              threshold VAR, GIRFs, regime IRFs, pass-through, pandemic dummies
  lp.py                 state-dependent local projections (Ramey and Zubairy, 2018)
  structural.py, soe.model  small structural model (irispie), calibrated to the TVAR
  analysis.py           runs the estimations and stores the results in output/results
  figures.py, tables.py figures (PDF and PNG) and tables (LaTeX) from output/results
tests/                  data, specification and estimation tests
notebooks/              replication.ipynb: the main results (Figure 3, Table 2, the joint system) step by step
output/                 results, figures and tables, written by the pipeline (not tracked; `make all` regenerates them)
```

## Installation

Python 3.11. With `uv`:

```bash
uv sync
```

or with `pip` in a virtual environment:

```bash
pip install -e . && pip install pytest
```

The package needs `MacroPy` 0.3.3: `ThresholdVAR` with a fixed threshold and delay (`threshold=`,
`delay=`), the pandemic treatment (`covid_mode=`), responses averaged over chosen histories
(`compute_irfs(histories=...)`) and block exogeneity (`b_exo=`). `pyproject.toml` points to the
released wheel.

## Replication

```bash
make all        # or: python -m tvar_shocks all   (about 10 minutes on a 10-core laptop)
make test       # tests
make quick      # checks that everything runs in 2 minutes; the numbers are not the paper's
make data       # rebuilds the database from the sources and compares it with the frozen copy
```

Single stages: `python -m tvar_shocks baseline | robustness | grid | threshold | lp | structural |
figures | tables`. Figures and tables are drawn from `output/results` only, so changing a figure does
not require re-estimating. `notebooks/replication.ipynb` walks through the main results (the two
baseline models, Table 2, Figure 3 and the joint system with block exogeneity) in about a minute.

The seeds are fixed in `config.toml`: a full run from scratch reproduces `output/results`, the figures
and the tables exactly (checked on 2026-10-20 after a `make clean`: maximum difference zero in every
results file, PDFs and PNGs byte-identical, in about 8 minutes).

## From the paper to the code

| Paper | File | Function |
|:-|:-|:-|
| Figure 1: business confidence and the 50-point threshold | `fig_confidence` | `figures.fig_confidence` |
| Figure 2: (a) rolling correlations, (b) averages by subperiod | `fig_facts_rolling`, `fig_facts_periods` | `figures.fig_stylized_facts` |
| Figure 3: GDP response by regime (GIRF) | `fig_girf_gdp` | `figures.fig_girf_gdp` |
| Table 2: pass-through and posterior probability of the asymmetry | `tab_pass_through.tex` | `tables.table_pass_through` |
| Figure 4 and Table A2: pass-through by the level of confidence when the shock hits | `fig_pass_through_bands`, `tab_pass_through_bands.tex` | `figures.fig_pass_through_bands`, `tables.table_pass_through_bands` |
| Figure 5: conventional IRF, GIRF and local projections | `fig_methods` | `figures.fig_methods` |
| Figure 6: alternative thresholds and marginal likelihood | `fig_threshold_grid` | `figures.fig_threshold_grid` |
| Table 3: alternative specifications | `tab_robustness.tex` | `tables.table_robustness` |
| Table 4: full sample and sample up to 2019 | `tab_state.tex` | `tables.table_state` |
| Figure 7: joint system with block exogeneity | `fig_joint_block` | `figures.fig_joint_block` |
| Figure 8: structural model against the TVAR | `fig_model` | `figures.fig_model` |
| Table 5: capital goods imports by state | `tab_investment.tex` | `tables.table_investment` |
| Table 6: fiscal revenue | `tab_fiscal.tex` | `tables.table_fiscal` |
| Figure A1: model variables | `fig_variables` | `figures.fig_variables` |
| Table A1: marginal likelihood by prior | `tab_prior_grid.tex` | `tables.table_prior_grid` |
| Figure A2: estimated threshold and delay | `fig_threshold_posterior` | `figures.fig_threshold_posterior` |
| Figures A3 and A4: responses of every variable | `fig_girf_all_chn`, `fig_girf_all_ipx` | `figures.fig_girf_all` |
| Figure A5: other external variables | `fig_robustness` | `figures.fig_robustness` |
| Figure A6: pandemic dummies | `fig_pandemic_dummies` | `figures.fig_pandemic_dummies` |
| Table A3: model parameters and sensitivity | `tab_calibration.tex` | `tables.table_calibration` |

Not in the package: the MS-VAR of the previous version, and the political noise index of Section 6.4
of the paper, which is not distributed (see `data/README.md`); the results that use it are reported in
the paper's text only, and the `pni*` variants of `config.toml` are skipped unless the index file is
present.

## The method, briefly

- **Threshold VAR** with two regimes (Alessandri and Mumtaz, 2017), $\mathbf{y}_t$ = (external
  variable, GDP growth, inflation, interbank rate, business confidence), two lags, pessimism regime
  when confidence $d$ months earlier is at or below 50 points. The delay $d$ is estimated with a
  uniform prior on 1 to 6; the posterior concentrates on 4.
- **Prior** of Bańbura, Giannone and Reichlin (2010) by dummy observations, one per regime, with
  hyperparameters chosen by marginal likelihood (Giannone, Lenza and Primiceri, 2015):
  $\lambda = 0.1$, $\tau = 10\lambda$, $\varepsilon = 1$ (`analysis.estimate_prior_grid`).
  `MacroPy` computes the **exact** posterior: with the threshold fixed, the marginal likelihood of
  each delay is closed form, the prior is conjugate in each regime and the draws of
  $(d, B_i, \Sigma_i)$ are independent, with no Gibbs chain to converge. Explosive draws are rejected
  (41% of the proposals are accepted with China and 52% with the export price index).
- **Pandemic**: *pandemic priors* (Cascaldi-Garcia, 2022) with diffuse dummies for January 2020 to
  June 2021, equivalent to taking those months out of the likelihood, together with the two following
  months, whose lags contain the outliers.
- **Generalized impulse responses** (Koop, Pesaran and Potter, 1996): for each draw, the difference
  between simulated paths with and without the shock, with the same future shocks, averaged over the
  histories of each regime; the regime can change along the horizon when simulated confidence crosses
  the threshold. The shock is 1 pp (China) or 10 pp (export prices) on impact, with recursive
  identification and the external variable first.
- **Pass-through**: cumulative response of GDP over the cumulative response of the external variable,
  scaled to the size of the shock, draw by draw.
- **Joint system with block exogeneity** (variants `joint_block`, `tot_block`;
  `estimation.external_block = 2`): China's industrial production and the price (export price index or
  terms of trade) form a linear external block that does not respond to Peru (`ThresholdVAR(b_exo=...)`);
  the domestic block keeps the threshold and responds to the current and lagged external variables.
  One model per shock (`shock_var`), with the same estimation. Without the restriction, the joint
  system depends on the ordering of the external variables and was discarded.
- **Episode or state** (`config.toml`, variants `pre2020`, `investment*`, `fiscal_pre2020`, and the
  optional `pni*`): the sample up to 2019, capital goods imports as an additional variable, the
  political noise index as the state variable (only when its file is present), and local projections
  with the index or with a date split as the state (`[lp]`).
- **Structural model** (`src/tvar_shocks/soe.model`, `structural.py`, solved with `irispie`): linear,
  monthly, with forward-looking investment; the external variables follow the median GIRFs of the TVAR
  as anticipated shocks. The state changes only the sensitivity of investment; three parameters are
  calibrated to the 12-month pass-through of the baseline TVAR and the rest are fixed in `[structural]`.

## Differences with the previous version (MATLAB)

1. **GIRFs instead of conventional IRFs by regime.** Conventional IRFs assume the economy never leaves
   the regime; in a threshold model the shock itself, and the response of confidence, can change it.
   `fig_methods` shows both.
2. **Prior chosen by marginal likelihood.** The previous code used $\lambda = 1$, $\tau = 10$,
   $\varepsilon = 10^{-5}$, with a much lower marginal likelihood; the `prior_loose` variant reproduces it.
3. **Estimated delay and exact sampler.** With the threshold fixed at 50 (prior variance 0.0001) the
   Metropolis step had nothing to sample, and in `c_tvar_covid.m` the delay is fixed at 2 (variant
   `delay_2`). Here the threshold is fixed explicitly (`threshold`), the delay is integrated with its
   exact posterior, and the threshold grid and the joint posterior of threshold and delay are reported
   separately.
4. **Scale of the prior.** The text of the previous version said standard deviations in one paragraph
   and variances in another; the code used variances. Here: standard deviations, as in Bańbura et al.
   (2010) (`sigma_scale = "sd"`); the `prior_variance` variant reproduces the previous convention.
5. **Structural model.** The DSGE of the previous version (`soe_spirits.model`, IRIS) passed both shocks
   through the same sentiment state and fixed two parameters per regime by hand, one with the opposite
   sign under pessimism: it predicted an asymmetry also for external demand, which the GIRFs do not
   find. The new model changes a single parameter between states, is calibrated to the TVAR and its
   uncalibrated predictions are checked against the data.
6. **Pandemic.** The previous version included 18 dummies as regressors with regime-specific
   coefficients; here they have a diffuse prior, which is equivalent to excluding those months (and
   the two following ones) from the likelihood. The moments that scale the prior are also computed
   without the pandemic.
7. **One system per shock, stated explicitly.** The previous text described a single system with China
   and the export price index; its code estimated one system per shock. This version does the same and
   says so, and adds the joint system with block exogeneity as a robustness exercise.

Validation: with the conventions of the previous code (variance scaling, moments with the pandemic,
conventional IRFs) `MacroPy` reproduces Table 1 of the previous version under optimism to two decimals
(for example 0.24 [0.15, 0.32] against 0.24 [0.15, 0.33] for China at 12 months and 0.62 against 0.65
for the export price index). Under pessimism the medians differ more (0.18 against 0.10 for China),
within bands almost one point wide.
