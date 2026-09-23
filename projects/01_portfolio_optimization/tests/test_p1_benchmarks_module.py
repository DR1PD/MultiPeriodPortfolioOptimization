"""Benchmark allocators: closed-form and definitional checks."""
import numpy as np

from portopt import equal_weight, min_variance, risk_parity

CFG = {'w_max': 1.0, 'gamma': 2.0}


def test_equal_weight():
    assert np.allclose(equal_weight(5), 0.2)


def test_min_variance_matches_closed_form_interior():
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    vols = np.array([0.015, 0.030, 0.020])
    sigma = np.diag(vols) @ corr @ np.diag(vols)
    w_closed = np.linalg.solve(sigma, np.ones(3))
    w_closed /= w_closed.sum()
    assert (w_closed > 0).all()
    w, info = min_variance(sigma, CFG)  # (w, status) since the 1.5 P1-3 fix
    assert info['status'] == 'optimal'
    assert np.allclose(w, w_closed, atol=1e-4)


def test_risk_parity_equalizes_risk_contributions():
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    vols = np.array([0.015, 0.030, 0.020])
    sigma = np.diag(vols) @ corr @ np.diag(vols)
    w = risk_parity(sigma)
    contrib = w * (sigma @ w)
    assert np.allclose(contrib, contrib.mean(), rtol=1e-4)
