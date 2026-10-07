"""quant-gbm: GBM and regime-switching Monte Carlo risk simulation."""

from .data import load_prices, log_returns
from .gbm import GBMParams, estimate_gbm, gbm_return_quantile, simulate_gbm
from .regime import (RegimeModel, filtered_probabilities, fit_hmm, select_n_states,
                     simulate_regime_gbm)
from .risk import comparison_table, max_drawdowns, summarize, terminal_returns, var_cvar
from .validation import backtest_var, christoffersen_independence, evaluate, kupiec_pof

__version__ = "0.1.0"

__all__ = [
    "load_prices", "log_returns",
    "GBMParams", "estimate_gbm", "simulate_gbm", "gbm_return_quantile",
    "RegimeModel", "fit_hmm", "select_n_states", "filtered_probabilities", "simulate_regime_gbm",
    "terminal_returns", "var_cvar", "max_drawdowns", "summarize", "comparison_table",
    "backtest_var", "kupiec_pof", "christoffersen_independence", "evaluate",
]
