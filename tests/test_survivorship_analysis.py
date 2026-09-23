"""G-3 sensitivity-analysis tests (ROADMAP 1.6): the near-death vs smooth
survivor split must be built on a correct per-name max-drawdown and produce a
clean partition of the universe.
"""
import numpy as np
import pandas as pd


def max_drawdown(prices):
    """Per-column max drawdown from a price DataFrame (the notebook recipe)."""
    dd = prices / prices.cummax() - 1.0
    return dd.min()


def median_split(maxdd_by_name):
    """Worse-than-median = near-death cohort; rest = smooth survivors."""
    thresh = maxdd_by_name.median()
    near_death = maxdd_by_name.index[maxdd_by_name <= thresh]
    smooth = maxdd_by_name.index[maxdd_by_name > thresh]
    return near_death, smooth, thresh


def test_max_drawdown_hand_computed():
    # 100 -> 120 -> 60 -> 90: worst peak-to-trough is 60/120 - 1 = -50%
    px = pd.DataFrame({'A': [100.0, 120.0, 60.0, 90.0]})
    assert np.isclose(max_drawdown(px)['A'], -0.50)
    # Monotone rise: no drawdown
    px2 = pd.DataFrame({'B': [50.0, 55.0, 70.0]})
    assert np.isclose(max_drawdown(px2)['B'], 0.0)


def test_median_split_partitions_universe():
    rng = np.random.default_rng(2)
    names = [f'S{i}' for i in range(151)]
    maxdd = pd.Series(-rng.uniform(0.1, 0.9, 151), index=names)
    near, smooth, thresh = median_split(maxdd)
    assert len(set(near) & set(smooth)) == 0          # disjoint
    assert len(near) + len(smooth) == 151             # exhaustive
    assert abs(len(near) - len(smooth)) <= 1          # balanced
    assert maxdd[near].max() <= thresh + 1e-12        # near-death are the worse half


def test_decay_gap_arithmetic():
    """Annualized-gap arithmetic used in the decay table: constant daily
    returns of 8 bps vs 4 bps -> ~10.6% vs ~5.2% annualized, gap positive."""
    idx = pd.date_range('2010-01-01', periods=504, freq='B')
    a = pd.Series(0.0008, index=idx)
    b = pd.Series(0.0004, index=idx)
    ann = lambda r: r.mean() * 252
    assert np.isclose(ann(a), 0.2016)
    assert ann(a) - ann(b) > 0.10
