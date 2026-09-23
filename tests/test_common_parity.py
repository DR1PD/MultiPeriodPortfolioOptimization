"""Parity tests (ROADMAP 2.1): common/ functions must reproduce the executed
notebooks' printed values on the cached data, to numerical tolerance.

Golden values below are copied from P2's committed notebook outputs
(post-1.1–1.6). Tests skip when the data/ caches are absent (fresh clone
without downloads) — parity is a statement about the pinned snapshots.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.data import DATA_DIR, clean_prices, download_ff_factors
from common.metrics import summary

P2_PRICES = DATA_DIR / 'p2_prices_2005_2024.parquet'
P2_SPY = DATA_DIR / 'p2_spy_2005_2024.parquet'
FF_CACHE = DATA_DIR / 'p1_ff_factors_2005_2024.parquet'

needs_cache = pytest.mark.skipif(
    not (P2_PRICES.exists() and P2_SPY.exists() and FF_CACHE.exists()),
    reason='data/ caches absent — parity is defined against the pinned snapshots')


@needs_cache
def test_clean_prices_matches_notebook_universe():
    """Notebook printed: 'Final: 151 assets, 4810 days' after cleaning."""
    raw = pd.read_parquet(P2_PRICES)
    prices, ret_log, ret_simple, log = clean_prices(raw)
    assert ret_log.shape == (4810, 151)
    assert ret_simple.shape == (4810, 151)
    assert prices.shape[1] == 151
    # G-2 contract: the two return types are exact transforms of each other
    assert np.allclose(ret_simple.values, np.expm1(ret_log.values))


@needs_cache
def test_metrics_summary_matches_notebook_spy_oos_row():
    """Notebook OOS table row (S&P 500 Buy & Hold):
    Return 14.84%, Vol 19.46%, Sharpe 0.634, CI [-0.18, 1.45],
    MaxDD -33.72%, Sortino 0.768, CVaR95 -0.0300."""
    spy = pd.read_parquet(P2_SPY).squeeze('columns')
    ret_simple = np.expm1(np.log(spy / spy.shift(1)).dropna())
    m = summary(ret_simple[ret_simple.index >= '2018-01-01'])
    assert np.isclose(m['ann_ret'], 0.1484, atol=5e-5)
    assert np.isclose(m['ann_vol'], 0.1946, atol=5e-5)
    assert np.isclose(m['sharpe'], 0.634, atol=5e-4)
    assert np.isclose(m['sharpe_ci_lo'], -0.18, atol=5e-3)
    assert np.isclose(m['sharpe_ci_hi'], 1.45, atol=5e-3)
    assert np.isclose(m['max_dd'], -0.3372, atol=5e-5)
    assert np.isclose(m['sortino'], 0.768, atol=5e-4)
    assert np.isclose(m['cvar95'], -0.0300, atol=5e-5)


@needs_cache
def test_metrics_summary_matches_notebook_full_and_rsp_rows():
    """P2 Full-period table: S&P 500 11.65%/yr (full cached span);
    decay-table OOS row: RSP 11.69%/yr. (The decay table's 11.79% Full SPY
    figure is windowed by the backtest warm-up — not comparable here.)"""
    spy = pd.read_parquet(P2_SPY).squeeze('columns')
    spy_simple = np.expm1(np.log(spy / spy.shift(1)).dropna())
    assert np.isclose(summary(spy_simple)['ann_ret'], 0.1165, atol=1e-4)

    rsp = pd.read_parquet(DATA_DIR / 'p2_rsp_2005_2024.parquet').squeeze('columns')
    rsp_simple = np.expm1(np.log(rsp / rsp.shift(1)).dropna())
    m = summary(rsp_simple[rsp_simple.index >= '2018-01-01'])
    assert np.isclose(m['ann_ret'], 0.1169, atol=1e-4)


@needs_cache
def test_ff_factors_load_real_cached_data():
    """FF3 cache: 5033 days × 4 columns of real Ken French data (decimal)."""
    ff = download_ff_factors('2005-01-01', '2024-12-31',
                             cache_name='p1_ff_factors_2005_2024.parquet')
    assert ff.shape == (5033, 4)
    assert list(ff.columns) == ['Mkt-RF', 'SMB', 'HML', 'RF']
    assert ff['Mkt-RF'].abs().max() < 0.20  # decimal units, not percent
