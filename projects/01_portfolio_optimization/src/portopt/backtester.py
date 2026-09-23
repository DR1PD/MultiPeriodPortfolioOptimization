"""Rolling backtest engine + per-strategy metric aggregation (2.2).

compute_strategy_metrics uses common.metrics.summary keys NATIVELY — the
2.1 key adapter is retired (drill-34 payoff): ann_ret, sharpe_ci_lo/hi,
var95/cvar95 everywhere.
"""
import numpy as np
import pandas as pd

from common.metrics import summary

from .factor_model import estimate_factor_model
from .covariance import estimate_covariance
from .optimizer import solve_portfolio
from .transaction_costs import compute_transaction_costs
from .benchmarks import (equal_weight, single_period_mvo, min_variance,
                         risk_parity)


def run_backtest(returns, ff_factors, config, strategy='optimizer', verbose=True):
    """
    Run a rolling backtest for a given strategy.
    
    Args:
        returns: (T, n) DataFrame of log returns
        ff_factors: (T, 4) DataFrame of factor data
        config: parameter dictionary
        strategy: 'optimizer', 'mvo', 'equal_weight', 'min_variance', 'risk_parity'
        verbose: print progress
    
    Returns:
        results: dict with return series, weight history, cost history, etc.
    """
    lookback = config['lookback_days']
    rebal_freq = config['rebal_frequency']
    n = returns.shape[1]
    
    # Build rebalancing dates
    all_dates = returns.index[lookback:]
    rebal_dates = all_dates[::rebal_freq]
    
    # Storage
    port_returns_gross = []
    port_returns_net = []
    weight_history = []
    cost_history = []
    turnover_history = []
    solve_info_history = []
    dates_out = []
    
    w_prev = None
    
    for i, rebal_date in enumerate(rebal_dates):
        if i == len(rebal_dates) - 1:
            break  # need at least one forward period
        
        # ── ESTIMATION (uses only trailing data) ──
        end_idx = returns.index.get_loc(rebal_date)
        start_idx = end_idx - lookback
        ret_window = returns.iloc[start_idx:end_idx]
        
        # Factor model
        alphas, betas, mu_hat, _ = estimate_factor_model(
            returns, ff_factors, rebal_date, lookback
        )
        
        # Covariance
        cov_matrix, _ = estimate_covariance(
            ret_window, method=config['shrinkage_method'],
            ewma_halflife=config['ewma_halflife']
        )
        
        # ── PORTFOLIO CONSTRUCTION ──
        solve_info = {'status': 'n/a', 'solve_time': 0}
        
        if strategy == 'optimizer':
            w_new, solve_info = solve_portfolio(mu_hat, cov_matrix, w_prev, config)
        elif strategy == 'mvo':
            w_new, solve_info = single_period_mvo(mu_hat, cov_matrix, config)
        elif strategy == 'equal_weight':
            w_new = equal_weight(n)
        elif strategy == 'min_variance':
            w_new, solve_info = min_variance(cov_matrix, config)
        elif strategy == 'risk_parity':
            w_new = risk_parity(cov_matrix)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # ── TRANSACTION COSTS ──
        if w_prev is not None:
            tc, tc_breakdown = compute_transaction_costs(w_new, w_prev, config)
        else:
            tc = 0
            tc_breakdown = {'proportional': 0, 'impact': 0, 'total': 0, 'turnover': 0, 'n_trades': 0}
        
        # ── FORWARD RETURNS ──
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
        fwd_idx = returns.index[(returns.index > rebal_date) & (returns.index <= next_rebal)]
        
        if len(fwd_idx) == 0:
            continue
        
        # G-2 fix: `returns` holds LOG returns (kept for estimation); portfolio
        # aggregation needs SIMPLE returns — w·r_log is neither the portfolio's
        # simple nor log return and understates growth (Jensen).
        fwd_returns_simple = np.expm1(returns.loc[fwd_idx])
        
        # Daily portfolio returns (using fixed weights — no intra-period rebalancing)
        for day in fwd_idx:
            day_ret_simple = fwd_returns_simple.loc[day].values
            gross_ret = w_new @ day_ret_simple
            port_returns_gross.append(gross_ret)
            dates_out.append(day)
        
        # Net return: subtract TC spread over the holding period
        daily_tc = tc / len(fwd_idx) if len(fwd_idx) > 0 else 0
        for day_gross in port_returns_gross[-len(fwd_idx):]:
            port_returns_net.append(day_gross - daily_tc)
        
        # Store
        weight_history.append({'date': rebal_date, 'weights': w_new.copy()})
        cost_history.append(tc_breakdown)
        turnover_history.append(tc_breakdown['turnover'])
        solve_info_history.append(solve_info)
        
        w_prev = w_new.copy()
        
        if verbose and i % 20 == 0:
            print(f"  {strategy}: rebalance {i+1}/{len(rebal_dates)-1} at {rebal_date.date()}")
    
    # Assemble output
    gross_series = pd.Series(port_returns_gross[:len(dates_out)], index=dates_out[:len(port_returns_gross)])
    net_series = pd.Series(port_returns_net[:len(dates_out)], index=dates_out[:len(port_returns_net)])
    
    return {
        'gross_returns': gross_series,
        'net_returns': net_series,
        'weight_history': weight_history,
        'cost_history': cost_history,
        'turnover_history': turnover_history,
        'solve_info': solve_info_history,
        'strategy': strategy,
    }


def compute_strategy_metrics(bt_result, config):
    """Compute metrics for all periods: full, IS, OOS."""
    net = bt_result['net_returns']
    gross = bt_result['gross_returns']
    
    is_mask = net.index <= config['is_end']
    oos_mask = net.index >= config['oos_start']
    
    metrics = {}
    for label, mask in [('full', slice(None)), ('IS', is_mask), ('OOS', oos_mask)]:
        r = net[mask] if not isinstance(mask, slice) else net
        metrics[label] = summary(r)  # common.metrics keys, natively (2.2: adapter retired)
    
    # Add turnover and cost info if available
    if 'turnover_history' in bt_result and bt_result['turnover_history']:
        metrics['avg_turnover'] = np.mean(bt_result['turnover_history'])
        metrics['total_cost'] = sum(c['total'] for c in bt_result['cost_history'])
    else:
        metrics['avg_turnover'] = 0
        metrics['total_cost'] = 0
    
    return metrics


def print_comparison_table(all_metrics, strategy_labels, period='OOS'):
    """Print a formatted comparison table for a given period."""
    print(f"\n{'='*100}")
    print(f"PERFORMANCE COMPARISON — {period} Period")
    print(f"{'='*100}")
    
    header = f"{'Strategy':<24} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'SR 95% CI':>16} {'MaxDD':>8} {'Calmar':>8} {'Sortino':>8} {'CVaR95':>8}"
    print(header)
    print("-" * 100)
    
    for strat in ['optimizer', 'mvo', 'equal_weight', 'min_variance', 'risk_parity', 'sp500']:
        m = all_metrics.get(strat, {}).get(period, {})
        if not m or np.isnan(m.get('ann_ret', np.nan)):
            continue
        
        ci = f"[{m.get('sharpe_ci_lo', 0):.2f}, {m.get('sharpe_ci_hi', 0):.2f}]"
        
        print(f"{strategy_labels.get(strat, strat):<24} "
              f"{m['ann_ret']:>7.2%} "
              f"{m['ann_vol']:>7.2%} "
              f"{m['sharpe']:>8.3f} "
              f"{ci:>16} "
              f"{m['max_dd']:>7.2%} "
              f"{m['calmar']:>8.3f} "
              f"{m['sortino']:>8.3f} "
              f"{m['cvar95']:>7.4f}")
    
    print("-" * 100)
    
    # Turnover and cost summary
    print(f"\n{'Strategy':<24} {'Avg Turnover':>14} {'Total Cost':>14}")
    print("-" * 54)
    for strat in ['optimizer', 'mvo', 'equal_weight', 'min_variance', 'risk_parity']:
        m = all_metrics.get(strat, {})
        print(f"{strategy_labels.get(strat, strat):<24} "
              f"{m.get('avg_turnover', 0):>13.4f} "
              f"{m.get('total_cost', 0)*10000:>12.1f} bps")
