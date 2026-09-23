"""run_backtest smoke + the G-2 aggregation invariant on synthetic data."""
import numpy as np
import pandas as pd

from portopt import run_backtest

CFG = {'lookback_days': 60, 'rebal_frequency': 21, 'gamma': 2.0,
       'kappa': 0.005, 'w_max': 0.5, 'tau_max': 0.4,
       'c_prop': 0.001, 'c_impact': 0.0005,
       'ewma_halflife': 30, 'shrinkage_method': 'ledoit_wolf',
       'is_end': '2015-12-31', 'oos_start': '2016-01-01'}


def _toy(T=300, n=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2015-01-01', periods=T, freq='B')
    ret_log = pd.DataFrame(rng.normal(0.0002, 0.01, (T, n)), index=idx)
    ff = pd.DataFrame({'Mkt-RF': rng.normal(0.0002, 0.008, T),
                       'SMB': rng.normal(0, 0.003, T),
                       'HML': rng.normal(0, 0.003, T),
                       'RF': np.full(T, 0.0001)}, index=idx)
    return ret_log, ff


def test_equal_weight_strategy_is_mean_of_simple_returns():
    ret_log, ff = _toy()
    bt = run_backtest(ret_log, ff, CFG, strategy='equal_weight', verbose=False)
    gross = bt['gross_returns']
    expected = np.expm1(ret_log).mean(axis=1).reindex(gross.index)
    assert np.allclose(gross.values, expected.values, atol=1e-12)


def test_optimizer_backtest_records_statuses():
    ret_log, ff = _toy()
    bt = run_backtest(ret_log, ff, CFG, strategy='optimizer', verbose=False)
    statuses = [s.get('status') for s in bt['solve_info'] if isinstance(s, dict)]
    assert len(statuses) > 0
    assert all(s in ('optimal', 'optimal_inaccurate') for s in statuses)
    assert np.isfinite(bt['net_returns']).all()
