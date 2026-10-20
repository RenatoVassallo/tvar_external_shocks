"""The structural model: rational-expectations solution and the property the paper uses."""
import numpy as np
import pytest

pytest.importorskip("irispie")

from tvar_shocks.structural import Structural, level_from_yoy, pass_through, yoy  # noqa: E402

FIXED = dict(w_c=0.54, w_i=0.11, w_x=0.26, h=0.8, mpc=0.5, chi=0.05, beta=0.99, rho_i=0.9, a_y=1.5,
             tau_p=0.2, tau_y=1.0, eta=0.6, a_p=0.7)


@pytest.fixture(scope="module")
def model():
    return Structural(length=300)


def test_level_from_yoy_inverts_yoy():
    g = np.r_[np.linspace(1.0, 0.0, 30), np.zeros(30)]
    assert np.allclose(yoy(level_from_yoy(g)), g)


def test_anticipated_paths_are_perfect_foresight(model):
    p = np.r_[np.linspace(10, 4, 60), np.full(240, 4.0)]
    out = model.simulate({**FIXED, "phi": 1.0}, p, np.zeros(300))
    # the expected discounted price is the discounted sum of the known path
    beta = FIXED["beta"]
    qp0 = (1 - beta) * np.sum(beta ** np.arange(300) * p)
    i0 = (1 - FIXED["rho_i"]) * (FIXED["a_p"] * qp0 + FIXED["a_y"] * 0.0)
    assert out["p"][0] == pytest.approx(10.0)
    assert out["i"][0] > 0 and out["i"][0] < i0 * 2      # investment moves on impact through qp


def test_lower_investment_response_hits_prices_more_than_demand(model):
    path = level_from_yoy(np.r_[np.linspace(1.0, 0.0, 36), np.zeros(24)])
    zero = np.zeros(len(path))
    pts = {}
    for phi in (1.0, 0.3):
        prm = {**FIXED, "phi": phi}
        rp = model.simulate(prm, 10 * path, zero)
        rd = model.simulate(prm, zero, path)
        n = len(path)
        pts[phi] = (pass_through(rp["y"][:n], rp["p"][:n], 12, 10.0), pass_through(rd["y"][:n], rd["yf"][:n], 12, 1.0))
    price_ratio = pts[0.3][0] / pts[1.0][0]
    demand_ratio = pts[0.3][1] / pts[1.0][1]
    assert price_ratio < 0.7 < demand_ratio
