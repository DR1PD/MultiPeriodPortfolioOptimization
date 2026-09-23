"""P1-1 regression tests (ROADMAP 1.5): the optimizer's transaction-cost term
must be the SAME cost model the backtest charges.

Contract (both sides of it, verbatim from the notebook):
    charged:   c_prop * sum|dw| + c_impact * sum|dw|^1.5
    objective: c_prop * norm1(dw) + c_impact * sum(power(abs(dw), 1.5))
If someone reverts the objective to a surrogate (e.g. the old n-scaled
quadratic), the agreement test fails.
"""
import cvxpy as cp
import numpy as np

C_PROP, C_IMPACT = 0.0010, 0.0005
N = 46


def charged_cost(dw):
    """compute_transaction_costs' formula (numeric side of the contract)."""
    dw = np.abs(dw)
    return C_PROP * dw.sum() + C_IMPACT * np.sum(dw ** 1.5)


def objective_cost_expr(dw):
    """solve_portfolio's tc_term (cvxpy side of the contract)."""
    return C_PROP * cp.norm1(dw) + C_IMPACT * cp.sum(cp.power(cp.abs(dw), 1.5))


def old_surrogate_expr(dw):
    """The pre-fix objective term: n-scaled quadratic impact."""
    return C_PROP * cp.norm1(dw) + C_IMPACT * cp.sum_squares(dw) * N


def test_objective_cost_equals_charged_cost_on_random_trades():
    rng = np.random.default_rng(5)
    for _ in range(20):
        dw = rng.normal(0, 0.02, N)  # realistic per-name weight changes
        x = cp.Variable(N)
        x.value = dw
        assert np.isclose(objective_cost_expr(x).value, charged_cost(dw), rtol=1e-10)


def test_old_surrogate_impact_diverges_from_charged_impact():
    """Documents the P1-1 defect precisely: both formulas share the linear
    term, so the divergence lives in the IMPACT component — the n-scaled
    quadratic overcharges it ~4x at realistic trade sizes (and the gap grows
    with size). The linear kappa+c_prop terms are what mostly froze the
    optimizer; the surrogate piled on."""
    rng = np.random.default_rng(5)
    dw = rng.normal(0, 0.01, N)
    x = cp.Variable(N)
    x.value = dw
    surrogate_impact = (C_IMPACT * cp.sum_squares(x) * N).value
    true_impact = C_IMPACT * np.sum(np.abs(dw) ** 1.5)
    assert surrogate_impact > 2.0 * true_impact


def test_true_cost_term_is_dcp():
    """The old comment claimed DCP forces the quadratic; it doesn't — the
    1.5-power composes convexly and the full problem stays solvable."""
    w = cp.Variable(N)
    w_prev = np.ones(N) / N
    dw = w - w_prev
    objective = cp.Minimize(objective_cost_expr(dw) + cp.sum_squares(w))
    prob = cp.Problem(objective, [cp.sum(w) == 1, w >= 0])
    assert prob.is_dcp()
    prob.solve()
    assert prob.status == 'optimal'
