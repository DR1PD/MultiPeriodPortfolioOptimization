# Multi-Period Portfolio Optimization with Transaction Costs, Turnover Penalties, and Factor-Based Return Forecasts via Disciplined Convex Programming

---

## A. Project Title

**Multi-Period Portfolio Optimization with Transaction Costs, Turnover Penalties, and Factor-Based Return Forecasts via Disciplined Convex Programming**

---

## B. Project Thesis

Real portfolios are not rebalanced in a vacuum. Every trade incurs costs—commissions, spread crossing, market impact—and every rebalancing decision today constrains what you can do tomorrow. Most textbook portfolio optimization (single-period Markowitz) ignores these realities and produces allocations that are theoretically elegant but practically unimplementable. This project builds a multi-period convex optimization framework that jointly optimizes expected return, risk, transaction costs, and turnover over a rolling horizon, using factor-based return forecasts as inputs. The central question is whether embedding realistic frictions into the optimization itself—rather than bolting them on after the fact—produces portfolios with meaningfully better net-of-cost, risk-adjusted performance than naive single-period approaches.

---

## C. Why This Project Is Admissions-Relevant

This project directly signals the competencies that Berkeley MFE, CMU MSCF, Columbia FE, and Princeton MFin/ORFE admissions committees screen for:

**Mathematical maturity.** The core of the project is formulating and solving a disciplined convex program. This requires comfort with convex analysis, quadratic forms, conic constraints, duality, and KKT conditions. These are the same mathematical foundations tested in courses like CMU's Optimization (46-975) and Berkeley's Numerical Methods.

**Programming competence.** The implementation requires writing a modular Python pipeline from scratch—data ingestion, factor construction, covariance estimation, optimizer formulation in CVXPY, rolling-window backtesting, and performance attribution. This is not calling `.fit()` on a library.

**Financial understanding.** Transaction cost modeling, turnover control, risk decomposition, and the distinction between gross and net alpha are things that separate someone who understands how money actually gets managed from someone who just read a textbook. Admissions readers who have worked on desks will recognize this immediately.

**Research discipline.** The project includes benchmark comparisons, out-of-sample testing, sensitivity analysis, regime-conditioned evaluation, and explicit failure-mode documentation. It reads like a research artifact, not a homework assignment.

**ISE alignment.** For an Industrial & Systems Engineering student, this project is native territory. Operations research, constrained optimization, stochastic modeling, and systems-level thinking are the backbone of ISE curricula. This project applies those tools to a financially meaningful problem, making the applicant's background a strength rather than an oddity.

---

## D. Research Question

**Does multi-period convex portfolio optimization with embedded transaction costs and turnover penalties produce statistically significant improvement in net Sharpe ratio, maximum drawdown, and turnover-adjusted information ratio compared to single-period mean-variance optimization, equal-weight portfolios, and risk-parity benchmarks, across both full-sample and regime-conditioned evaluation windows?**

---

## E. Quant Finance Concepts Involved

- Mean-variance optimization (Markowitz, 1952)
- Multi-period portfolio optimization (Boyd, Busseti, Diamond, Kahn, 2017)
- Transaction cost modeling: linear (proportional) and square-root (market impact) components
- Turnover penalties and holding constraints
- Factor-based return forecasting (Fama-French 3/5 factor models)
- Covariance estimation: sample covariance, Ledoit-Wolf shrinkage, exponentially weighted covariance
- Convex optimization: quadratic programming, second-order cone programming (SOCP)
- Risk budgeting and risk parity as benchmark allocation methods
- Rolling-window backtesting with walk-forward validation
- Performance attribution: gross vs. net returns, alpha decomposition
- Maximum drawdown, conditional Value-at-Risk (CVaR), Sharpe ratio under estimation error
- Regime detection for conditional evaluation (bull/bear/sideways classification)

---

## F. Math/Statistics Requirements

The following mathematics must appear explicitly in the project, not as background reading, but as implemented computation:

1. **Quadratic form for portfolio variance:**
   w^T Σ w, where Σ is the covariance matrix and w is the weight vector.

2. **Convex objective formulation:**
   minimize  -μ^T w + γ · w^T Σ w + κ · ||w_{t} - w_{t-1}||₁ + λ · TC(w_{t} - w_{t-1})
   subject to: 1^T w = 1, w ≥ 0 (long-only), ||w||∞ ≤ w_max, turnover ≤ τ_max

   Where:
   - μ = expected return vector (from factor model)
   - γ = risk aversion parameter
   - κ = turnover penalty
   - λ = transaction cost scaling
   - TC(·) = transaction cost function (linear + square-root impact)

3. **Ledoit-Wolf shrinkage estimator:**
   Σ_shrunk = δ · F + (1 - δ) · S
   where F is the structured target (e.g., constant-correlation or single-factor), S is the sample covariance, and δ is the optimal shrinkage intensity computed analytically.

4. **Factor model for expected returns:**
   r_i = α_i + β_i1 · MKT + β_i2 · SMB + β_i3 · HML + ε_i
   Estimated via rolling OLS. Expected returns μ_i = α̂_i + β̂_i · E[factors].

5. **Transaction cost model:**
   TC(Δw) = c_prop · |Δw| + c_impact · |Δw|^(3/2)
   The proportional term captures commissions/spreads; the 3/2-power term approximates square-root market impact (Almgren & Chriss, 2001).

> **[STATUS 2026-07-19 — ROADMAP 1.5]** The optimizer's objective now charges exactly this model (the earlier quadratic surrogate is removed; `tests/test_p1_cost_alignment.py`). κ·‖Δw‖₁ is a separate, deliberate regularizer on top of c_prop (AUDIT P1-2 — documented, not merged). Honest finding: at the base-case κ the optimizer's realized turnover is 0.0000 — it degenerates to buy-and-hold of its first solve; κ calibration is owned by the Phase 3.2 sensitivity grid.

6. **Risk decomposition:**
   Portfolio risk = Σ_i w_i · (Σw)_i / (w^T Σ w)
   Marginal contribution to risk (MCTR) and component contribution to risk for each asset.

7. **Sharpe ratio with estimation error:**
   SE(SR̂) ≈ sqrt((1 + 0.5 · SR²) / T)
   Used to construct confidence intervals around reported Sharpe ratios (Lo, 2002).

8. **Conditional Value-at-Risk (CVaR):**
   CVaR_α = E[L | L ≥ VaR_α]
   Estimated empirically from the return distribution tail.

---

## G. Data

### Primary Data Sources

| Data | Source | Variables | Frequency |
|------|--------|-----------|-----------|
| US equity prices | yfinance (Yahoo Finance API) | Adj. close, volume, market cap | Daily |
| Fama-French factors | Kenneth French Data Library | MKT-RF, SMB, HML, RF — FF3 as implemented [REVISED 2026-07-19: RMW/CMA never used] | Daily |
| Risk-free rate | Kenneth French / FRED | 1-month T-bill rate | Daily |

### Universe Construction

- **Investment universe [REVISED 2026-07-19]:** as implemented, 50 hand-picked 2024-era large caps (46 realized after cleaning) — a **survivor universe**, not a fixed-historical-date list; measured selection premium ≈ +350–390 bp/yr (`docs/LIMITATIONS.md` §1). Point-in-time construction is the intended fix, gated on the WRDS/CRSP access check.
- **Sample period:** January 2005 – December 2024 (20 years). This spans the GFC, post-GFC recovery, COVID crash, and 2022 rate tightening—providing regime diversity.
- **In-sample / out-of-sample split:** 2005–2017 for development and calibration; 2018–2024 for pure out-of-sample evaluation.

### Data Cleaning

- Adjust for splits and dividends (use adjusted close)
- Remove assets with >5% missing data in any rolling window
- Forward-fill gaps of ≤3 days; drop assets with longer gaps
- Winsorize daily returns at ±5 standard deviations to mitigate data errors
- Log all cleaning steps in a data dictionary

---

## H. Modeling Pipeline

### Step 1: Data Ingestion and Cleaning
- Pull adjusted close prices for the equity universe via yfinance
- Download Fama-French daily factor returns
- Compute log returns: r_t = ln(P_t / P_{t-1})
- Apply cleaning rules above
- Store cleaned data in a standardized Parquet format with metadata

### Step 2: Factor Model Estimation (Rolling)
- At each rebalancing date t, use a trailing 252-day window to estimate:
  - Factor betas (β) via OLS regression of each asset's excess returns on FF3 factors
  - Alpha estimates (α̂) as the intercept
  - Expected returns μ = α̂ + β̂ · E[factors], where E[factors] is the trailing mean of factor returns
- Store estimated parameters with timestamps

### Step 3: Covariance Estimation
- Compute sample covariance from the same trailing 252-day window
- Apply Ledoit-Wolf shrinkage (using sklearn.covariance.LedoitWolf or manual implementation)
- Optionally compute exponentially weighted covariance (halflife = 63 days) as a sensitivity check
- Ensure positive semi-definiteness; apply eigenvalue clipping if necessary

### Step 4: Optimizer Formulation (CVXPY)
- Define the decision variable: w ∈ R^n (portfolio weights at time t)
- Formulate the objective:
  ```
  minimize:  -μ^T w  +  γ · quad_form(w, Σ)  +  κ · norm(w - w_prev, 1)  +  λ · TC(w - w_prev)
  ```
- Constraints:
  - sum(w) == 1  (fully invested)
  - w >= 0  (long-only)
  - w <= w_max  (e.g., 5% per asset to enforce diversification)
  - norm(w - w_prev, 1) <= τ_max  (turnover cap)
- Solve using CVXPY with ECOS or SCS solver
- Record solver status, objective value, and solution time

### Step 5: Transaction Cost Computation
- For each rebalancing, compute:
  - Proportional cost: c_prop · Σ|Δw_i| (default: 10 bps per leg)
  - Impact cost: c_impact · Σ|Δw_i|^1.5 (calibrated to typical large-cap impact)
- Subtract total costs from gross portfolio return to get net return

### Step 6: Rolling Backtest
- Rebalance monthly (21 trading days)
- At each rebalance date:
  1. Re-estimate factor model (Step 2)
  2. Re-estimate covariance (Step 3)
  3. Solve optimizer (Step 4)
  4. Record new weights, trades, costs
  5. Compute portfolio return over next period using realized returns
- Track: gross return series, net return series, weight history, turnover history, cost history

### Step 7: Benchmark Construction
- Run the same backtest infrastructure for each benchmark (see Section I)
- Ensure identical rebalancing frequency, data, and cost assumptions

### Step 8: Performance Evaluation
- Compute all metrics from Section J
- Generate comparison tables and visualizations

### Step 9: Sensitivity and Robustness Analysis
- Vary γ (risk aversion): [0.5, 1, 2, 5, 10]
- Vary κ (turnover penalty): [0, 0.001, 0.005, 0.01, 0.05]
- Vary transaction cost assumptions: [5bps, 10bps, 20bps, 50bps]
- Vary covariance estimator: [sample, Ledoit-Wolf, EWMA]
- Vary rebalancing frequency: [weekly, monthly, quarterly]
- Conduct regime-conditioned evaluation (see Section J)

### Step 10: Documentation and Packaging
- Write README, technical appendix, and executive summary
- Package code into reproducible repository structure (see Section L)

---

## I. Benchmarks

The following benchmarks must be implemented with identical data, rebalancing frequency, and cost treatment:

| # | Benchmark | Description | Purpose |
|---|-----------|-------------|---------|
| 1 | **Equal-Weight (1/N)** | Uniform allocation across all assets, rebalanced monthly | Naive diversification baseline. Surprisingly hard to beat (DeMiguel, Garlappi, Uppal, 2009). |
| 2 | **Single-Period Mean-Variance (Markowitz)** | Standard quadratic optimization without transaction costs or turnover penalties | Tests whether the multi-period frictions-aware framework adds value over the classical approach. |
| 3 | **Risk Parity** | Allocate such that each asset contributes equally to portfolio risk | Popular institutional benchmark. Tests whether optimization outperforms a heuristic risk-budgeting approach. |
| 4 | **Minimum Variance** | Minimize w^T Σ w subject to sum(w) = 1, w ≥ 0 | Pure risk-minimization. Tests whether the return forecast (alpha model) adds value or just adds noise. |
| 5 | **S&P 500 (buy-and-hold)** | Market-cap weighted index, no rebalancing | Passive market benchmark. The "what if I just bought the index?" baseline. |

---

## J. Evaluation Framework

> **[STATUS 2026-07-19]** Implemented so far: the metric subset visible in the notebook, a single-window (γ,κ) snapshot, and the 5–50 bps cost curve. Full rolling-backtest grids (γ×κ, covariance estimator, rebalancing frequency, universe size) are **[PLANNED — Phase 3.2]**.

### Performance Metrics

| Metric | Formula / Description |
|--------|-----------------------|
| Annualized return (gross) | Arithmetic daily mean × 252 [as implemented; geometric reserved for equity curves] |
| Annualized return (net) | Arithmetic daily mean × 252 of net returns [as implemented] |
| Annualized volatility | Std dev of daily returns × √252 |
| Sharpe ratio (net) | (Net return - RF) / Volatility |
| Sharpe ratio confidence interval | SR ± 1.96 × SE(SR), where SE ≈ √((1 + 0.5·SR²)/T) |
| Maximum drawdown | Largest peak-to-trough decline |
| Calmar ratio | Net return / Max drawdown |
| Sortino ratio | (Net return - RF) / Downside deviation |
| Information ratio | (Net return - benchmark return) / Tracking error [PLANNED — Phase 2/3] |
| Average monthly turnover | Mean of monthly L1 turnover: Σ|w_{t} - w_{t-1}| |
| Total transaction costs | Cumulative cost over backtest period |
| Turnover-adjusted IR | IR adjusted for realized turnover and cost drag [PLANNED — Phase 2/3] |

### Risk Metrics

| Metric | Description |
|--------|-------------|
| VaR (95%, 99%) | Historical [parametric PLANNED — Phase 2/3] |
| CVaR (95%) | Expected shortfall beyond VaR |
| Volatility of volatility | Rolling 63-day vol, then take std of that series [PLANNED — Phase 2/3] |
| Risk concentration (HHI) | Herfindahl of risk contributions across assets |
| Max position size | Largest single-asset weight over time |

### Robustness Checks

1. **Parameter sensitivity:** Heatmap of net Sharpe across (γ, κ) grid
2. **Cost sensitivity:** Performance degradation curve as transaction costs increase from 5 bps to 50 bps
3. **Covariance sensitivity:** Compare results using sample, Ledoit-Wolf, and EWMA covariance
4. **Rebalancing frequency:** Weekly vs. monthly vs. quarterly
5. **Universe size:** 30 vs. 50 vs. 100 assets

### Regime-Conditioned Evaluation

- Classify months as Bull (S&P 500 return > +2%), Bear (< -2%), or Sideways (between)
- Alternatively: use VIX quintiles as regime proxy
- Report Sharpe, drawdown, and turnover separately per regime
- Key question: Does the optimizer's advantage concentrate in one regime, or persist broadly?

### Out-of-Sample Discipline

- All calibration (factor model estimation, covariance, parameter tuning) uses 2005–2017 only
- All reported performance numbers include 2018–2024 out-of-sample window
- No retroactive parameter adjustment based on OOS results
- Report both in-sample and OOS results side by side with explicit separation

---

## K. Failure Modes

These are the ways this project can go wrong. Each one must be acknowledged in the final writeup:

1. **Estimation error dominates signal.** Factor alpha estimates are noisy. If the alpha forecast is garbage, the optimizer dutifully maximizes garbage. The Minimum Variance benchmark (which ignores alpha entirely) may win, which would be an honest and informative result.

2. **Overfitting to the (γ, κ) grid.** If you tune risk aversion and turnover penalty on in-sample data and evaluate on the same data, you're kidding yourself. The project enforces strict IS/OOS separation, but the sensitivity heatmap still needs honest interpretation.

3. **Covariance estimation breaks in crises.** Sample covariance estimated from a calm period will dramatically underestimate risk during a crisis. Ledoit-Wolf shrinkage helps but doesn't solve the fundamental problem. The regime-conditioned evaluation exposes this.

4. **Transaction cost model is too simple.** The linear + 3/2-power model is a reasonable approximation for large-cap equities but doesn't capture intraday timing, queue priority, or cross-asset impact. This is a limitation, not a fixable bug at this project's scope.

5. **Solver numerical issues.** CVXPY can return suboptimal solutions or fail to converge when the problem is near-degenerate (e.g., near-singular covariance matrix, very tight constraints). The project must log solver status and flag any non-optimal solutions.

6. **Survivorship bias in universe construction.** Using current S&P 500 constituents introduces look-ahead bias. [REVISED 2026-07-19] The implemented universe does NOT use a fixed historical date; the bias is instead measured and documented (`docs/LIMITATIONS.md` §1), with PIT/CRSP construction gated on the WRDS access check.

7. **Turnover penalty may kill alpha.** If the penalty κ is set too high, the optimizer will barely trade, and the portfolio will drift toward buy-and-hold. There's a Pareto frontier between alpha capture and turnover cost—the project must map this frontier explicitly.

8. **Equal-weight may simply win.** The DeMiguel et al. (2009) result showed 1/N beats most optimized portfolios in many settings. If that happens here, it's a real finding, not a failure. The project must honestly report it and explain why (estimation error, parameter instability, etc.).

---

## L. Final GitHub Repository Structure

> **[STATUS 2026-07-19]** Current reality: one executed notebook (`notebooks/exploration.ipynb`), shared `common/` (plotting, hmm_filter), repo-level `tests/` (7 files, 21 tests), `requirements.lock`, `data/` parquet caches. The `src/` module split, per-project config.yaml, notebook series, and results tree below are **[PLANNED — Phase 2 (code) / Phase 3 (results)]**.

```
systematic-portfolio-optimization/
│
├── README.md                          # Executive summary + technical overview
├── requirements.txt                   # Python dependencies with pinned versions
├── environment.yml                    # Conda environment specification
├── config.yaml                        # All parameters: γ, κ, λ, universe, dates, etc.
├── data_dictionary.md                 # Variable definitions, sources, cleaning rules
├── LICENSE                            # MIT or Apache 2.0
│
├── data/
│   ├── raw/                           # Downloaded price data, factor files (gitignored)
│   ├── processed/                     # Cleaned returns, factor betas (gitignored)
│   └── README.md                      # How to reproduce data pull
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # yfinance pulls, FF factor downloads, cleaning
│   ├── factor_model.py                # Rolling OLS factor estimation, alpha extraction
│   ├── covariance.py                  # Sample, Ledoit-Wolf, EWMA covariance estimators
│   ├── optimizer.py                   # CVXPY formulation: objective, constraints, solver
│   ├── transaction_costs.py           # Proportional + impact cost model
│   ├── backtester.py                  # Rolling backtest engine, rebalancing logic
│   ├── benchmarks.py                  # Equal-weight, MVO, risk parity, min-var, index
│   ├── metrics.py                     # Sharpe, drawdown, CVaR, turnover, IR, etc.
│   ├── regime.py                      # Bull/bear/sideways classification
│   └── utils.py                       # Logging, seed setting, config loading
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Universe stats, return distributions, correlation
│   ├── 02_factor_model.ipynb          # Factor beta stability, alpha distribution
│   ├── 03_covariance_analysis.ipynb   # Estimator comparison, eigenvalue spectrum
│   ├── 04_optimization_demo.ipynb     # Single-period optimizer walkthrough
│   ├── 05_backtest_results.ipynb      # Full backtest, benchmark comparison tables
│   ├── 06_sensitivity_analysis.ipynb  # Parameter grids, cost sensitivity, robustness
│   └── 07_regime_analysis.ipynb       # Regime-conditioned performance
│
├── results/
│   ├── figures/                       # All plots as PNGs (high-res)
│   ├── tables/                        # CSV exports of performance tables
│   └── logs/                          # Solver logs, backtest execution logs
│
├── tests/
│   ├── test_optimizer.py              # Unit tests for CVXPY formulation
│   ├── test_costs.py                  # Cost model correctness
│   └── test_metrics.py                # Metric computation validation
│
└── docs/
    ├── technical_appendix.pdf         # Detailed math derivations
    └── references.bib                 # Academic citations
```

---

## M. README Outline

> **[STATUS 2026-07-19]** README build-out is **[PLANNED — Phase 3.1/5]**. No number below may be cited until Phase 3 reruns complete (`results/CHANGELOG.md` tracks provisional values).

The README.md must contain exactly these sections, in this order:

### 1. Title and One-Line Summary
The project title followed by a single sentence explaining what the project does and why.

### 2. Executive Summary (for Non-Specialists)
3–5 sentences explaining the problem, approach, and key finding in plain language. An admissions reader with a finance background but no time to read code should understand the project from this section alone.

### 3. Research Question
The specific question from Section D, stated cleanly.

### 4. Key Results
A summary table showing: Sharpe (net), Max Drawdown, Avg Turnover, and Total Cost for each strategy (multi-period optimizer + all benchmarks). One or two sentences interpreting the main finding.

### 5. Methodology Overview
Brief description of:
- Factor model for return forecasts
- Covariance estimation approach
- Optimizer formulation (objective + constraints, in math notation)
- Transaction cost model
- Backtest design (rebalancing freq, IS/OOS split)

### 6. Repository Structure
The tree from Section L.

### 7. How to Reproduce
Step-by-step instructions:
```
git clone ...
conda env create -f environment.yml
conda activate portfolio-opt
python src/data_loader.py --config config.yaml
jupyter notebook notebooks/05_backtest_results.ipynb
```

### 8. Key Visualizations
Embed 3–4 figures directly in the README:
- Cumulative return plot (all strategies, net of costs)
- Sensitivity heatmap (Sharpe vs. γ and κ)
- Turnover vs. net Sharpe Pareto frontier
- Regime-conditioned performance bars

### 9. Assumptions and Limitations
Honest list from Section K, written concisely.

### 10. References
Academic citations: Boyd et al. (2017), Markowitz (1952), Ledoit & Wolf (2004), DeMiguel et al. (2009), Almgren & Chriss (2001), Lo (2002), Fama & French (1993/2015).

### 11. Author
Name, university, program, contact.

---

## N. Resume / Interview Framing

> **[STATUS 2026-07-19]** These bullets describe the completed end-state. Current numbers are provisional and several structural claims are Phase 2/3 — do not use verbatim until Phase 3 completes.

Use these four bullet points on a resume or when describing the project in an interview:

1. **Built a multi-period portfolio optimization engine in Python using CVXPY that jointly minimizes risk, transaction costs, and turnover over a rolling 20-year horizon across 50+ equities, producing net-of-cost Sharpe ratios and risk-adjusted returns under realistic market frictions.**

2. **Implemented and benchmarked five allocation strategies—multi-period convex optimizer, single-period mean-variance, risk parity, minimum variance, and equal-weight—with identical data, rebalancing frequency, and transaction cost treatment to ensure fair comparison.**

3. **Conducted sensitivity analysis across risk aversion, turnover penalty, cost assumptions, and covariance estimators, mapping the Pareto frontier between alpha capture and implementation cost, and evaluated all strategies under bull, bear, and sideways market regimes.**

4. **Documented the full pipeline as a reproducible research artifact with modular code, configuration files, unit tests, and a technical appendix, structured for graduate admissions review and interview defensibility.**

---

## O. Stretch Extensions

These three extensions increase rigor and can be pursued if time permits:

### Extension 1: Robust Optimization Under Parameter Uncertainty
Replace the point estimates of μ and Σ with uncertainty sets. Formulate the problem as:

minimize max_{(μ, Σ) ∈ U} [ -μ^T w + γ · w^T Σ w ]

where U is an ellipsoidal uncertainty set around the estimated parameters. This converts the problem from a standard QP to a robust counterpart (Goldfarb & Iyengar, 2003). It directly addresses the estimation error failure mode and is a meaningful step toward what real funds do.

**Why this impresses:** It shows you understand that optimization under certainty is a toy problem—real decisions are made under parameter uncertainty.

### Extension 2: Multi-Period Lookahead with Dynamic Programming
Extend the single-step optimization to a true multi-period formulation where the optimizer considers the impact of today's trade on future rebalancing costs. This requires either approximate dynamic programming or a model-predictive control (MPC) approach where you solve a T-step lookahead at each rebalancing date.

**Why this impresses:** MPC is the industry-standard approach at systematic funds like AQR and Two Sigma. Implementing even a 3-step lookahead demonstrates understanding of inter-temporal optimization.

### Extension 3: Factor Timing via Regime-Switching Expected Returns
Replace the static factor model with a regime-switching model where expected factor returns depend on the detected market regime (from Project 3—Hidden Markov Model). In the bull regime, tilt toward momentum and growth factors; in the bear regime, tilt toward value and low-volatility. This creates a bridge between Projects 1 and 3 and shows portfolio-level integration of regime intelligence.

**Why this impresses:** It demonstrates that you think in systems, not isolated models. Connecting regime detection to portfolio construction is exactly how systematic funds operate.

---

## Self-Evaluation Rubric

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Finance relevance | 5/5 | Solves the actual problem systematic funds face: how to trade optimally under realistic frictions. Not a generic ML exercise. |
| Mathematical rigor | 5/5 | Convex optimization, quadratic forms, shrinkage estimation, factor models, CVaR, Sharpe inference—all implemented, not just cited. |
| Programming rigor | 5/5 | Modular pipeline, CVXPY formulation from scratch, rolling backtest, config-driven, unit-tested. Reviewer would infer real coding ability. |
| Statistical validity | 5/5 | Five benchmarks, strict IS/OOS split, sensitivity grids, regime conditioning, confidence intervals on Sharpe. |
| Professional presentation | 5/5 | Clean repo structure, README with embedded results, technical appendix, data dictionary, reproducibility instructions. |
| Interview defensibility | 5/5 | Every design choice (why Ledoit-Wolf? why 3/2-power impact? why monthly rebalancing?) has a defensible answer grounded in the literature. |
| Originality | 4/5 | The framework is known (Boyd et al., 2017), but the specific combination of factor-based alpha, shrinkage covariance, regime evaluation, and Pareto frontier mapping is more thoughtful than a template project. |
| Graduate readiness | 5/5 | Directly prepares for courses in optimization, stochastic processes, computational finance, and econometrics. |

---

## Key Python Libraries and Their Roles

| Library | Role in Project |
|---------|----------------|
| **CVXPY** | Core optimizer. Formulates and solves the convex program (QP/SOCP). The star of the project. |
| **PyPortfolioOpt** | [NOT USED — 2026-07-19] All benchmarks are implemented directly in CVXPY/NumPy in this repo. |
| **yfinance** | Data ingestion. Pulls adjusted close prices and volume for the equity universe. |
| **Qlib** | [NOT USED — 2026-07-19] Factor pipeline is rolling statsmodels OLS implemented in this repo. |
| **NumPy / SciPy** | Matrix operations, eigenvalue decomposition, statistical functions, Ledoit-Wolf implementation. |
| **pandas** | Data manipulation, time-series alignment, rolling windows. |
| **statsmodels** | OLS regression for factor model estimation, statistical tests. |
| **scikit-learn** | Ledoit-Wolf covariance estimator (sklearn.covariance.LedoitWolf). |
| **matplotlib / plotly** | Visualization: cumulative returns, heatmaps, Pareto frontiers, regime overlays. |
| **PyYAML** | Configuration file parsing. |
| **pytest** | Unit testing framework. |

---

## References

1. Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., & Nystrup, P. (2017). Multi-period trading via convex optimization. *Foundations and Trends in Optimization*, 3(1), 1–76.
2. Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77–91.
3. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
4. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *The Review of Financial Studies*, 22(5), 1915–1953.
5. Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.
6. Lo, A. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52.
7. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
8. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.
9. Goldfarb, D., & Iyengar, G. (2003). Robust portfolio selection problems. *Mathematics of Operations Research*, 28(1), 1–38.
10. Kolm, P. N., Tütüncü, R., & Fabozzi, F. J. (2014). 60 years of portfolio optimization. *European Journal of Operational Research*, 234(2), 356–371.
