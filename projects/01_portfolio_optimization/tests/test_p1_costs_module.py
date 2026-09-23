"""The 1.5 contract, machine-checked against the REAL module: the optimizer's
objective cost expression equals the backtest's charged cost, always."""
import cvxpy as cp
import numpy as np

from portopt import objective_cost_expr, compute_transaction_costs

CFG = {'c_prop': 0.0010, 'c_impact': 0.0005}


def test_objective_expression_equals_charged_cost_on_random_trades():
    rng = np.random.default_rng(5)
    for _ in range(20):
        w_old = rng.dirichlet(np.ones(46))
        w_new = rng.dirichlet(np.ones(46))
        dw = cp.Variable(46)
        dw.value = w_new - w_old
        expr_val = objective_cost_expr(dw, CFG['c_prop'], CFG['c_impact']).value
        charged, _ = compute_transaction_costs(w_new, w_old, CFG)
        assert np.isclose(expr_val, charged, rtol=1e-10)
