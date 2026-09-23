# Multi-Period Portfolio Optimization with Transaction Costs

**Hypothesis.** Multi-period convex optimization — expected returns from a
rolling Fama-French-3 model, Ledoit-Wolf covariance, an explicit 3/2-power
impact cost in the objective — should beat naive allocation once real
trading frictions are charged.

**Method.** CVXPY optimizer (long-only, 5% weight cap, turnover cap; the
objective charges *exactly* the cost model the backtest charges — machine-
checked contract, `tests/test_p1_cost_alignment.py`), walk-forward monthly
over 2005–2024 on 46 surviving large caps, against five hand-rolled
benchmarks under identical costs. IS 2005–2017 / OOS 2018–2024. Code:
[`src/portopt/`](src/portopt/).

**The honest headline: the optimizer's correct policy was restraint.**
At the documented baseline (κ=0.005) the optimizer never trades — and the
21-cell κ×γ grid shows why that's the *answer*, not a bug: the freeze
boundary sits exactly at the baseline; below it the optimizer trades, costs
barely bite (gross−net ≤ 0.003 Sharpe), **and trading still doesn't pay**
(2/12 trading cells beat equal-weight OOS, within noise; even in-sample
selection prefers a frozen cell). Costs are irrelevant; **forecast quality
is the binding constraint.** Independent rediscovery of DeMiguel et al.
(2009) by the machinery built to refute it.

![Cumulative net returns](../../results/figures/p1_fig05.png)
![The κ×γ trade-off surface](../../results/figures/p1_fig06.png)

Numbers: [`results/tables/p1_metrics.md`](../../results/tables/p1_metrics.md),
[grid](../../results/tables/p1_kappa_gamma_grid.md) ·
Limitations: [`docs/LIMITATIONS.md`](../../docs/LIMITATIONS.md) (survivor
universe, ≈ +350–390 bp/yr selection premium) ·
Discipline: [`results/OOS_DISCIPLINE.md`](../../results/OOS_DISCIPLINE.md)

**Regenerate** (from repo root, data snapshots in `data/`):
```
jupyter nbconvert --to notebook --execute --inplace projects/01_portfolio_optimization/notebooks/exploration.ipynb
```
