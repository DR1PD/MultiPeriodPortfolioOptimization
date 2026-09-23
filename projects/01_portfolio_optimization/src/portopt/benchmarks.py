"""Benchmark allocators (extracted verbatim, ROADMAP 2.2). All hand-rolled
CVXPY/NumPy — no PyPortfolioOpt (see DESIGN library-table correction).
"""
import cvxpy as cp
import numpy as np

from common.data import download_prices


def equal_weight(n):
    """1/N allocation."""
    return np.ones(n) / n


def single_period_mvo(mu_hat, cov_matrix, config):
    """
    Classical mean-variance optimization.
    Same as our optimizer but WITHOUT turnover penalty or TC in objective.
    """
    n = len(mu_hat)
    w = cp.Variable(n)
    
    objective = cp.Minimize(-mu_hat @ w + config['gamma'] * cp.quad_form(w, cov_matrix, assume_PSD=True))
    constraints = [cp.sum(w) == 1, w >= 0, w <= config['w_max']]
    
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except:
        problem.solve(solver=cp.SCS, verbose=False)
    
    if problem.status in ['optimal', 'optimal_inaccurate']:
        w_opt = np.maximum(np.array(w.value).flatten(), 0)
        return w_opt / w_opt.sum(), {'status': problem.status}
    return np.ones(n) / n, {'status': problem.status, 'fallback': True}  # P1-3: no silent fallback


def min_variance(cov_matrix, config):
    """Minimum variance portfolio — ignores expected returns entirely."""
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(w, cov_matrix, assume_PSD=True))
    constraints = [cp.sum(w) == 1, w >= 0, w <= config['w_max']]
    
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except:
        problem.solve(solver=cp.SCS, verbose=False)
    
    if problem.status in ['optimal', 'optimal_inaccurate']:
        w_opt = np.maximum(np.array(w.value).flatten(), 0)
        return w_opt / w_opt.sum(), {'status': problem.status}
    return np.ones(n) / n, {'status': problem.status, 'fallback': True}  # P1-3: no silent fallback


def risk_parity(cov_matrix, n_iter=500, tol=1e-8):
    """
    Risk parity: each asset contributes equally to portfolio variance.
    Solved via iterative reweighting (Maillard, Roncalli, Teiletche, 2010).
    """
    n = cov_matrix.shape[0]
    w = np.ones(n) / n
    
    for _ in range(n_iter):
        sigma_w = cov_matrix @ w
        mctr = sigma_w / np.sqrt(w @ sigma_w)
        
        # Target: equal risk contribution
        w_new = 1 / mctr
        w_new /= w_new.sum()
        
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new
    
    return w


def single_period_mvo(mu_hat, cov_matrix, config):
    """
    Classical mean-variance optimization.
    Same as our optimizer but WITHOUT turnover penalty or TC in objective.
    """
    n = len(mu_hat)
    w = cp.Variable(n)
    
    objective = cp.Minimize(-mu_hat @ w + config['gamma'] * cp.quad_form(w, cov_matrix, assume_PSD=True))
    constraints = [cp.sum(w) == 1, w >= 0, w <= config['w_max']]
    
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except:
        problem.solve(solver=cp.SCS, verbose=False)
    
    if problem.status in ['optimal', 'optimal_inaccurate']:
        w_opt = np.maximum(np.array(w.value).flatten(), 0)
        return w_opt / w_opt.sum(), {'status': problem.status}
    return np.ones(n) / n, {'status': problem.status, 'fallback': True}  # P1-3: no silent fallback


def min_variance(cov_matrix, config):
    """Minimum variance portfolio — ignores expected returns entirely."""
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(w, cov_matrix, assume_PSD=True))
    constraints = [cp.sum(w) == 1, w >= 0, w <= config['w_max']]
    
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except:
        problem.solve(solver=cp.SCS, verbose=False)
    
    if problem.status in ['optimal', 'optimal_inaccurate']:
        w_opt = np.maximum(np.array(w.value).flatten(), 0)
        return w_opt / w_opt.sum(), {'status': problem.status}
    return np.ones(n) / n, {'status': problem.status, 'fallback': True}  # P1-3: no silent fallback


def risk_parity(cov_matrix, n_iter=500, tol=1e-8):
    """
    Risk parity: each asset contributes equally to portfolio variance.
    Solved via iterative reweighting (Maillard, Roncalli, Teiletche, 2010).
    """
    n = cov_matrix.shape[0]
    w = np.ones(n) / n
    
    for _ in range(n_iter):
        sigma_w = cov_matrix @ w
        mctr = sigma_w / np.sqrt(w @ sigma_w)
        
        # Target: equal risk contribution
        w_new = 1 / mctr
        w_new /= w_new.sum()
        
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new
    
    return w


def min_variance(cov_matrix, config):
    """Minimum variance portfolio — ignores expected returns entirely."""
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(w, cov_matrix, assume_PSD=True))
    constraints = [cp.sum(w) == 1, w >= 0, w <= config['w_max']]
    
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except:
        problem.solve(solver=cp.SCS, verbose=False)
    
    if problem.status in ['optimal', 'optimal_inaccurate']:
        w_opt = np.maximum(np.array(w.value).flatten(), 0)
        return w_opt / w_opt.sum(), {'status': problem.status}
    return np.ones(n) / n, {'status': problem.status, 'fallback': True}  # P1-3: no silent fallback


def risk_parity(cov_matrix, n_iter=500, tol=1e-8):
    """
    Risk parity: each asset contributes equally to portfolio variance.
    Solved via iterative reweighting (Maillard, Roncalli, Teiletche, 2010).
    """
    n = cov_matrix.shape[0]
    w = np.ones(n) / n
    
    for _ in range(n_iter):
        sigma_w = cov_matrix @ w
        mctr = sigma_w / np.sqrt(w @ sigma_w)
        
        # Target: equal risk contribution
        w_new = 1 / mctr
        w_new /= w_new.sum()
        
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new
    
    return w


def risk_parity(cov_matrix, n_iter=500, tol=1e-8):
    """
    Risk parity: each asset contributes equally to portfolio variance.
    Solved via iterative reweighting (Maillard, Roncalli, Teiletche, 2010).
    """
    n = cov_matrix.shape[0]
    w = np.ones(n) / n
    
    for _ in range(n_iter):
        sigma_w = cov_matrix @ w
        mctr = sigma_w / np.sqrt(w @ sigma_w)
        
        # Target: equal risk contribution
        w_new = 1 / mctr
        w_new /= w_new.sum()
        
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new
    
    return w


def get_sp500_benchmark(returns, config):
    """S&P 500 buy-and-hold proxy using SPY (common.data, cache-first)."""
    spy = download_prices('SPY', config['start_date'], config['end_date'],
                          cache_name='p1_spy_2005_2024.parquet').squeeze('columns')
    ret_log = np.log(spy / spy.shift(1)).dropna()
    return np.expm1(ret_log)  # G-2: benchmark series in SIMPLE returns
