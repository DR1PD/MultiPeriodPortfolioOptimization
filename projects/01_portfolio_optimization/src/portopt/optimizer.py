"""Multi-period CVXPY optimizer (extracted verbatim, ROADMAP 2.2)."""
import time

import cvxpy as cp
import numpy as np

from .transaction_costs import objective_cost_expr


def solve_portfolio(mu_hat, cov_matrix, w_prev, config):
    """
    Solve the multi-period convex portfolio optimization problem.
    
    minimize   -mu^T w  +  gamma * w^T Sigma w  +  kappa * ||w - w_prev||_1  +  lambda * TC(dw)
    subject to  1^T w = 1,  w >= 0,  w <= w_max,  ||w - w_prev||_1 <= tau_max
    
    Args:
        mu_hat: (n,) expected returns
        cov_matrix: (n,n) covariance matrix
        w_prev: (n,) previous weights (None for first period)
        config: parameter dictionary
    
    Returns:
        w_opt: (n,) optimal weights
        solve_info: dict with solver status, objective, timing
    """
    n = len(mu_hat)
    w = cp.Variable(n)
    
    gamma = config['gamma']
    kappa = config['kappa']
    c_prop = config['c_prop']
    c_impact = config['c_impact']
    w_max = config['w_max']
    tau_max = config['tau_max']
    
    # Objective components
    # 1. Expected return (maximize = minimize negative)
    ret_term = -mu_hat @ w
    
    # 2. Risk (quadratic form)
    risk_term = gamma * cp.quad_form(w, cov_matrix, assume_PSD=True)
    
    # 3. Turnover penalty + transaction costs
    if w_prev is not None:
        dw = w - w_prev
        # kappa*||dw||_1 is a REGULARIZER (deliberate extra caution for model
        # uncertainty); c_prop*||dw||_1 is the real linear cost. Two distinct
        # knobs on the same term — intentional, documented, not merged (P1-2).
        turnover_penalty = kappa * cp.norm1(dw)
        # P1-1 fix: charge the SAME impact model the backtest charges.
        # |dw|^1.5 is convex and DCP-representable (power cone) — the old
        # 'DCP requires quadratic' comment was wrong, and the n-scaled
        # quadratic surrogate overcharged impact ~4x at realistic sizes.
        tc_term = objective_cost_expr(dw, c_prop, c_impact)  # single-source P1-1 contract
    else:
        dw = None
        turnover_penalty = 0
        tc_term = 0
    
    objective = cp.Minimize(ret_term + risk_term + turnover_penalty + tc_term)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,       # fully invested
        w >= 0,               # long-only
        w <= w_max,           # diversification
    ]
    
    if dw is not None:
        constraints.append(cp.norm1(dw) <= tau_max)  # turnover cap
    
    # Solve
    problem = cp.Problem(objective, constraints)
    t0 = time.time()
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False, max_iters=500)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        except cp.SolverError:
            # Last resort
            problem.solve(solver=cp.OSQP, verbose=False)
    
    solve_time = time.time() - t0
    
    solve_info = {
        'status': problem.status,
        'optimal': problem.status == 'optimal',
        'objective': problem.value if problem.value is not None else np.inf,
        'solve_time': solve_time,
        'solver': problem.solver_stats.solver_name if problem.solver_stats else 'unknown',
    }
    
    if problem.status in ['optimal', 'optimal_inaccurate']:
        w_opt = np.array(w.value).flatten()
        # Clean tiny weights (numerical noise)
        w_opt = np.maximum(w_opt, 0)
        w_opt /= w_opt.sum()  # re-normalize
    else:
        # Fallback: equal weight
        w_opt = np.ones(n) / n
        solve_info['fallback'] = True
    
    return w_opt, solve_info
