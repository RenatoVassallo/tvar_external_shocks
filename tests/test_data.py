"""The frozen database is complete and documented."""
import pandas as pd
import pytest

from tvar_shocks import data, load_config, model_specs


def test_monthly_contiguous_index():
    db = data.load()
    expected = pd.date_range(db.index[0], db.index[-1], freq="MS")
    assert db.index.equals(expected)


def test_model_variables_complete_over_the_sample():
    cfg = load_config()
    db = data.load().loc[cfg["data"]["start"]:cfg["data"]["end"]]
    for spec in model_specs(cfg).values():
        assert db[list(spec.variables)].notna().all().all(), spec.name


def test_every_series_is_documented():
    db, src = data.load(), data.sources()
    assert set(db.columns) - set(data.DERIVED) == set(src["variable"])
    assert src["fuente"].notna().all()


def test_raw_external_file_matches_the_database():
    raw = pd.read_csv(data.RAW_EXTERNAL, index_col="date", parse_dates=True)
    db = data.load()
    common = db.index.intersection(raw.index)
    assert (db.loc[common, "g_ip_chn"] - raw.loc[common, "g_ip_chn"]).abs().max() < 1e-6


def test_robustness_variants_have_complete_data():
    cfg = load_config()
    db = data.load()
    from tvar_shocks.config import variant_config
    for variant in cfg["robustness"]["variants"]:
        vcfg = variant_config(cfg, variant)
        for spec in model_specs(vcfg).values():
            if not data.available(db, spec.variables):      # political noise index not distributed
                continue
            sample = db.loc[spec.start:spec.end, list(spec.variables)]
            assert sample.notna().all().all(), (variant, spec.name)


def test_political_noise_index_is_optional(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "RAW_POLITICAL_NOISE", tmp_path / "absent.csv")
    db = data.load()
    assert not set(data.OPTIONAL) & set(db.columns)
    assert data.available(db, ["g_xpi", "g_gdp"]) and not data.available(db, ["g_xpi", "pni_neg"])
    with pytest.raises(KeyError):
        data.available(db, ["g_xpi", "not_a_series"])
