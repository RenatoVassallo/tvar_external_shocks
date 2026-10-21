# Data

## `database.csv`: the frozen copy

Monthly database used by every estimation in the paper, from 2002-04 to 2026-04 (the estimation
sample ends in 2025-12; see `config.toml`). First column `date`, first day of each month. The
dictionary of each series (description, transformation, source and code) is in `sources.csv`.

| Column | Series | Transformation | Source and code |
|:-|:-|:-|:-|
| `g_ip_chn` | China, industrial production | year-on-year % change (as published) | NBS China |
| `g_ip_usa` | United States, industrial production | year-on-year % change | FRED `INDPRO` |
| `fed` | Federal funds rate | level | FRED `FEDFUNDS` |
| `g_tot` | Terms of trade | year-on-year % change | BCRP `PN38923BM` |
| `g_xpi` | Export price index | year-on-year % change | BCRP `PN38915BM` |
| `exp` | Expectations of the economy three months ahead (business confidence) | level (diffusion index) | BCRP `PD38045AM` |
| `g_gdp` | Real GDP | year-on-year % change | BCRP `PN01773AM`; `PN02538AQ` for 2002-2003 |
| `g_cpi` | Lima CPI | year-on-year % change | BCRP `PN38705PM` |
| `int` | Interbank interest rate | level | BCRP `PN07819NM` |
| `g_ner` | Nominal exchange rate | year-on-year % change | BCRP `PN01210PM` |
| `g_rev` | Central government current revenue, real, 12-month sum | year-on-year % change | BCRP `PN38689FM` |
| `g_rev3` | Central government current revenue, real, 3-month sum | year-on-year % change | BCRP `PN38689FM` |
| `g_kimp` | Volume of capital goods imports excluding construction materials | year-on-year % change | BCRP `PD37978BM` |

Provenance:

- The first ten columns are the `database_tvar.xlsx` database of the MATLAB code of the previous
  version of the paper (`a_makedatabase.m`), unchanged.
- `g_rev` was added on 2026-10-16, `g_kimp` on 2026-10-17 and `g_rev3` on 2026-10-21 from the BCRP API.
- The political noise index used in the robustness exercise of Section 6.4 of the paper is not
  distributed (it was built from press headlines and snippets collected for the previous version, and
  neither the texts nor the collection code were kept). The results that use it are reported in the
  paper's text only. The pipeline looks for `raw/political_noise.csv` (columns `date`, `pni_raw`, the
  monthly normalised count; the file is listed in `.gitignore`): when it is present, `load()` adds
  `pni` (its 3-month average) and `pni_neg = -pni`, and the `pni*` variants of `config.toml` run;
  otherwise they are skipped.

## How it is built

`python -m tvar_shocks data` rebuilds the database from the sources (BCRP API through `MacroPy`, FRED
and `raw/external.csv`) with the transformations in `src/tvar_shocks/data.py`, and reports the
difference with the frozen copy without modifying it:

- GDP: the monthly index from 2003; the months of 2002 and 2003 without a year-on-year change are
  filled with quarterly GDP interpolated to monthly by a cubic spline, each quarter at its last month.
- Year-on-year changes are `100 (x_t / x_{t-12} - 1)`.
- Fiscal revenue is summed over 3 months (`g_rev3`, the paper's measure) or 12 months (`g_rev`) before taking the change, to remove the income tax calendar.

The BCRP revises the national accounts, the terms of trade and the export price index, so a rebuild
made today differs from the frozen copy in those series (for example, up to 2 points in some months of
`g_tot` and `g_xpi`). That is why the paper uses the frozen copy.

## `raw/external.csv`

Series without a stable public API: China's industrial production (manual download from the NBS,
year-on-year growth as published).
