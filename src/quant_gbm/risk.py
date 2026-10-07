"""Risk metrics on simulated price paths. Losses are reported as positive fractions."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def terminal_returns(paths: np.ndarray) -> np.ndarray:
    """Simple return from the first to the last column of each path."""
    return paths[:, -1] / paths[:, 0] - 1.0


def var_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Historical-simulation VaR and CVaR (expected shortfall) at tail probability `alpha`.

    VaR  = -q_alpha(returns)
    CVaR = -E[returns | returns <= q_alpha]
    """
    r = np.asarray(returns, dtype=float)
    q = np.quantile(r, alpha)
    return float(-q), float(-r[r <= q].mean())


def max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Maximum peak-to-trough drawdown along each path (positive fraction)."""
    running_max = np.maximum.accumulate(paths, axis=1)
    return -(paths / running_max - 1.0).min(axis=1)


def summarize(paths: np.ndarray, alphas=(0.01, 0.05)) -> dict:
    """Headline statistics of a simulated path set."""
    s0 = paths[0, 0]
    final = paths[:, -1]
    rets = terminal_returns(paths)
    dd = max_drawdowns(paths)

    out = {
        "start_price": float(s0),
        "mean_price": float(final.mean()),
        "median_price": float(np.median(final)),
        "std_price": float(final.std(ddof=1)),
        "min_price": float(final.min()),
        "max_price": float(final.max()),
        "mean_return": float(rets.mean()),
        "median_return": float(np.median(rets)),
        "prob_loss": float((rets < 0).mean()),
        "median_max_drawdown": float(np.median(dd)),
        "p95_max_drawdown": float(np.quantile(dd, 0.95)),
    }
    for a in alphas:
        v, c = var_cvar(rets, a)
        tag = f"{int(round((1 - a) * 100))}"
        out[f"VaR_{tag}"] = v
        out[f"CVaR_{tag}"] = c
    for p in PERCENTILES:
        out[f"p{p:02d}_price"] = float(np.percentile(final, p))
    return out


def comparison_table(path_sets: dict[str, np.ndarray], alphas=(0.01, 0.05)) -> pd.DataFrame:
    """One column per model, one row per metric."""
    return pd.DataFrame({name: summarize(p, alphas) for name, p in path_sets.items()})
