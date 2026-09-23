"""Transaction-cost model — ONE source for both sides of the P1-1 contract:
the optimizer's objective expression and the backtest's realized charge use
the same formulas (machine-checked in tests/test_p1_costs_module.py).
"""
import cvxpy as cp
import numpy as np


def objective_cost_expr(dw, c_prop, c_impact):
    """cvxpy expression for c_prop*||dw||_1 + c_impact*sum|dw|^1.5 — the same
    model compute_transaction_costs charges (ROADMAP 1.5; AUDIT P1-1)."""
    return c_prop * cp.norm1(dw) + c_impact * cp.sum(cp.power(cp.abs(dw), 1.5))


def compute_transaction_costs(w_new, w_old, config):
    """
    Compute realized transaction costs for a rebalancing trade.
    
    Returns:
        total_cost: scalar, total cost as fraction of portfolio value
        cost_breakdown: dict with proportional and impact components
    """
    dw = np.abs(w_new - w_old)
    
    # Proportional: bid-ask spread + commissions
    prop_cost = config['c_prop'] * dw.sum()
    
    # Market impact: |dw|^(3/2) approximation
    impact_cost = config['c_impact'] * np.sum(dw ** 1.5)
    
    total = prop_cost + impact_cost
    
    return total, {
        'proportional': prop_cost,
        'impact': impact_cost,
        'total': total,
        'turnover': dw.sum(),
        'n_trades': (dw > 0.001).sum(),
    }
