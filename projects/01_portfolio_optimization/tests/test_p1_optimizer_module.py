"""solve_portfolio invariants on small problems."""
import numpy as np

from portopt import solve_portfolio

CFG = {'gamma': 2.0, 'kappa': 0.005, 'c_prop': 0.001, 'c_impact': 0.0005,
       'w_max': 0.30, 'tau_max': 0.40}


def _toy(n=8, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.01, (60, n))
    return rng.normal(0.0004, 0.0002, n), np.cov(A, rowvar=False)


def test_first_solve_feasible_and_optimal():
    mu, cov = _toy()
    w, info = solve_portfolio(mu, cov, None, CFG)
    assert info['status'] == 'optimal'
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all() and (w <= CFG['w_max'] + 1e-6).all()


def test_turnover_cap_respected():
    mu, cov = _toy()
    w_prev = np.ones(8) / 8
    w, info = solve_portfolio(mu, cov, w_prev, CFG)
    assert info['status'] in ('optimal', 'optimal_inaccurate')
    assert np.abs(w - w_prev).sum() <= CFG['tau_max'] + 1e-6


def test_infinite_kappa_freezes_portfolio():
    """The 1.5 finding as a unit test: a large enough linear turnover penalty
    makes holding w_prev optimal — the mechanism behind 0.0000 turnover."""
    mu, cov = _toy()
    w_prev = np.ones(8) / 8
    cfg = dict(CFG, kappa=10.0)
    w, _ = solve_portfolio(mu, cov, w_prev, cfg)
    assert np.abs(w - w_prev).sum() < 1e-4
