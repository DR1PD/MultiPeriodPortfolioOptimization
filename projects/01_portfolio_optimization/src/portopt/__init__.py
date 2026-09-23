"""P1 strategy package (DESIGN Section L via src-layout; ROADMAP 2.2)."""
from .factor_model import estimate_factor_model
from .covariance import estimate_covariance
from .transaction_costs import objective_cost_expr, compute_transaction_costs
from .optimizer import solve_portfolio
from .benchmarks import (equal_weight, single_period_mvo, min_variance,
                         risk_parity, get_sp500_benchmark)
from .backtester import (run_backtest, compute_strategy_metrics,
                         print_comparison_table)
from .regime import classify_regime
