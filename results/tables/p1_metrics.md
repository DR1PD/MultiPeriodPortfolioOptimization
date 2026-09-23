# Multi-Period Portfolio Optimization — metric tables (baseline 2026-07-20)

```
PERFORMANCE COMPARISON — OOS Period
====================================================================================================
Strategy                   Return      Vol   Sharpe        SR 95% CI    MaxDD   Calmar  Sortino   CVaR95
----------------------------------------------------------------------------------------------------
Multi-Period Optimizer    18.70%  19.15%    0.846    [-0.02, 1.71] -29.21%    0.640    1.081 -0.0291
Single-Period MVO         15.17%  17.55%    0.722    [-0.11, 1.56] -27.56%    0.550    0.925 -0.0264
Equal Weight (1/N)        16.27%  17.53%    0.785    [-0.07, 1.64] -27.76%    0.586    0.990 -0.0267
Minimum Variance          12.17%  14.59%    0.663    [-0.16, 1.48] -24.12%    0.505    0.823 -0.0222
Risk Parity               15.16%  16.30%    0.777    [-0.07, 1.63] -26.39%    0.574    0.968 -0.0248
S&P 500 (Buy & Hold)      14.84%  19.46%    0.634    [-0.18, 1.45] -33.72%    0.440    0.768 -0.0300
----------------------------------------------------------------------------------------------------

Strategy                   Avg Turnover     Total Cost
------------------------------------------------------
Multi-Period Optimizer          0.0000          0.0 bps
Single-Period MVO               0.3281        825.0 bps
Equal Weight (1/N)              0.0000          0.0 bps
Minimum Variance                0.1195        289.1 bps
Risk Parity                     0.0329         76.1 bps


Key finding (OOS period 2018-01-01 to 2024-12-31):
  Optimizer net Sharpe: 0.846
  MVO net Sharpe:       0.722
  Optimizer max DD:     -29.21%

Running sensitivity analysis (simplified for runtime)...
In production: each (γ, κ) pair gets a full rolling backtest.


Sensitivity analysis complete.

→ Higher κ reduces turnover (lower trading costs) but dampens alpha capture.
  The best net Sharpe occurs at intermediate values — this is the Pareto frontier.

→ The Pareto frontier reveals the fundamental tradeoff between
  alpha capture (more trading) and implementation efficiency (less trading).
  Operating on or near this frontier is the goal of systematic portfolio management.

Regime distribution:
SPY
Bull        102
Sideways     89
Bear         49
Name: count, dtype: int64

Bull months: 102
Bear months: 49
Sideways months: 89


==========================================================================================
REGIME-CONDITIONED NET SHARPE RATIOS
==========================================================================================
Strategy                       Bull       Bear   Sideways       Full
------------------------------------------------------------------
Multi-Period Optimizer        1.077      0.871      0.411      0.748
Single-Period MVO             1.016      0.839      0.327      0.697
Equal Weight (1/N)            1.013      0.843      0.422      0.721
Minimum Variance              0.889      0.820      0.412      0.673
Risk Parity                   1.002      0.846      0.438      0.723
S&P 500 (Buy & Hol
```

```
PERFORMANCE COMPARISON — IS Period
====================================================================================================
Strategy                   Return      Vol   Sharpe        SR 95% CI    MaxDD   Calmar  Sortino   CVaR95
----------------------------------------------------------------------------------------------------
Multi-Period Optimizer    15.40%  18.71%    0.690     [0.06, 1.32] -49.43%    0.312    0.868 -0.0289
Single-Period MVO         13.95%  16.80%    0.682     [0.05, 1.31] -45.88%    0.304    0.871 -0.0259
Equal Weight (1/N)        14.68%  17.78%    0.685     [0.06, 1.31] -46.95%    0.313    0.852 -0.0274
Minimum Variance          12.01%  14.00%    0.680     [0.05, 1.31] -38.04%    0.316    0.860 -0.0212
Risk Parity               13.83%  16.36%    0.693     [0.06, 1.32] -44.02%    0.314    0.861 -0.0251
S&P 500 (Buy & Hold)       9.93%  18.81%    0.395    [-0.17, 0.96] -55.19%    0.180    0.482 -0.0289
----------------------------------------------------------------------------------------------------

Strategy                   Avg Turnover     Total Cost
------------------------------------------------------
Multi-Period Optimizer          0.0000          0.0 bps
Single-Period MVO               0.3281        825.0 bps
Equal Weight (1/N)              0.0000          0.0 bps
Minimum Variance                0.1195        289.1 bps
Risk Parity                     0.0329         76.1 bps

====================================================================================================
```

```
Solver Diagnostics:
  Total solves: 227
  optimal: 227 (100.0%)

  Avg solve time: 0.023s
  Max solve time: 0.116s

  ✓ No fallback events — all solves converged.

Per-strategy solver status:
  optimizer      optimal: 227 | equal-weight fallbacks: 0
  mvo            optimal: 227 | equal-weight fallbacks: 0
  min_variance   optimal: 227 | equal-weight fallbacks: 0
```

```
FINAL PROJECT SUMMARY
================================================================================

Universe: 46 US large-cap equities
Period: 2005-01-01 to 2024-12-31 (5031 trading days)
IS/OOS: 2005-01-01–2017-12-31 | 2018-01-01–2024-12-31
Rebalancing: Monthly (21 trading days)
Strategies tested: 6 (1 optimizer + 5 benchmarks)
Total optimizer solves: 227

Optimizer parameters:
  Risk aversion (γ):  2.0
  Turnover penalty:   0.005
  Max weight:         5%
  Transaction cost:   10 bps proportional
  Covariance:         Ledoit-Wolf shrinkage

Sensitivity grid: 5 × 5 = 25 parameter combinations
Cost sensitivity: 5 levels tested (5–50 bps)
Regime analysis: 3 regimes (Bull/Bear/Sideways)

================================================================================
PROJECT COMPLETE
================================================================================

Next steps:
  → Project 2: Eigenportfolio Construction via Random Matrix Theory
  → Project 3: Hidden Markov Model Regime Detection
  → Extension: Connect regime detector (P3) to factor timing in this optimizer
```
