"""Specification handling, the pass-through and a fast end-to-end estimation."""
import numpy as np
import pandas as pd
import pytest

from tvar_shocks import data, load_config, model_specs, variant_config
from tvar_shocks.model import fit, girfs, pass_through


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_pass_through_is_the_ratio_of_cumulated_responses(cfg):
    spec = model_specs(cfg)["ipx"]
    resp = np.zeros((3, 24, 5))
    resp[:, :, 0] = 2.0                      # external variable
    resp[:, :, 1] = 0.1                      # GDP
    pt = pass_through(resp, spec, [12, 24])
    assert np.allclose(pt["p50"], 0.1 / 2.0 * spec.shock)
    assert (pt["prob_positive"] == 1.0).all()


def test_variants_override_only_what_they_name(cfg):
    lp = variant_config(cfg, "lenza_primiceri")
    assert lp["covid"]["mode"] == "lenza-primiceri" and lp["prior"] == cfg["prior"]
    usa = variant_config(cfg, "usa")
    assert list(usa["models"]) == ["usa"] and usa["estimation"] == cfg["estimation"]
    a, b = model_specs(cfg)["chn"], model_specs(cfg)["chn"]
    assert a.key() == b.key() and a.key() != model_specs(cfg, threshold=48.0)["chn"].key()


def test_fixed_threshold_and_delay_define_the_regimes(cfg):
    spec = model_specs(cfg, post_draws=40, delay=3)["chn"]
    model = fit(spec, data.load(), horizon=6)
    assert np.all(model.draws["tar"] == cfg["estimation"]["threshold"])
    assert np.all(model.draws["delay"] == 3)
    low = (model.y_df["exp"].shift(3) <= cfg["estimation"]["threshold"])
    assert np.array_equal(model.draws["e1"][0], low.reindex(model.sample_dates).to_numpy())
    resp = girfs(model, spec, n_draws=3, reps=2)
    assert set(resp) == {"pesimismo", "optimismo"}
    assert np.allclose(resp["optimismo"][:, 0, 0], spec.shock)    # impact of the external shock


def test_joint_block_shocks_the_named_variable(cfg):
    spec = model_specs(variant_config(cfg, "joint_block"), post_draws=40)["ipx_joint"]
    assert spec.external_block == 2 and spec.external == "g_xpi" and spec.shock_index == 1
    model = fit(spec, data.load(), horizon=6)
    assert model.n_ext == 2                                        # MacroPy received the block mask
    resp = girfs(model, spec, n_draws=3, reps=2)
    assert np.allclose(resp["optimismo"][:, 0, 1], spec.shock)      # impact on the IPX, not on China
    fake = np.zeros((3, 24, 6))
    fake[:, :, 1], fake[:, :, 2] = 2.0, 0.1                        # IPX and GDP
    assert np.allclose(pass_through(fake, spec, [12])["p50"], 0.1 / 2.0 * spec.shock)


def test_estimated_delay_stays_in_its_support(cfg):
    spec = model_specs(cfg, post_draws=40)["chn"]
    model = fit(spec, data.load(), horizon=6)
    assert spec.delay is None
    assert set(np.unique(model.draws["delay"])) <= set(range(1, cfg["estimation"]["max_delay"] + 1))
    assert np.isclose(model.split_posterior["probability"].sum(), 1.0)
