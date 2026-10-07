"""Geometric Brownian Motion: estimation, simulation and closed-form quantiles.

Model:  dS_t = mu * S_t dt + sigma * S_t dW_t
Exact discretisation:  ln(S_{t+dt}/S_t) = (mu - sigma^2/2) dt + sigma sqrt(dt) Z
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class GBMParams:
    mu: float                 # annualised arithmetic drift
    sigma: float              # annualised volatility
    periods_per_year: int = 252

    @property
    def dt(self) -> float:
        return 1.0 / self.periods_per_year

    @property
    def log_drift(self) -> float:
        """Annualised drift of log returns, mu - sigma^2 / 2."""
        return self.mu - 0.5 * self.sigma**2


def estimate_gbm(log_returns, periods_per_year: int = 252) -> GBMParams:
    """Estimate GBM parameters from per-bar log returns.

    Because E[log return] = (mu - sigma^2/2) dt, the arithmetic drift is recovered as
    mu = mean * periods_per_year + sigma^2 / 2.
    """
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("Need at least two returns to estimate GBM parameters.")
    sigma = r.std(ddof=1) * np.sqrt(periods_per_year)
    mu = r.mean() * periods_per_year + 0.5 * sigma**2
    return GBMParams(mu=float(mu), sigma=float(sigma), periods_per_year=periods_per_year)


def draw_shocks(rng: np.random.Generator, size, kind: str = "normal", df: float = 4.0) -> np.ndarray:
    """Unit-variance shocks: standard normal, or Student-t rescaled to unit variance."""
    if kind == "normal":
        return rng.standard_normal(size)
    if kind == "t":
        if df <= 2:
            raise ValueError("Student-t degrees of freedom must exceed 2 for finite variance.")
        return rng.standard_t(df, size) * np.sqrt((df - 2.0) / df)
    raise ValueError(f"Unknown shock kind '{kind}'. Use 'normal' or 't'.")


def simulate_gbm(
    s0: float,
    params: GBMParams,
    n_steps: int,
    n_paths: int = 10_000,
    shocks: str = "normal",
    df: float = 4.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate price paths. Returns an array of shape (n_paths, n_steps + 1); column 0 is s0."""
    rng = np.random.default_rng(seed)
    dt = params.dt
    z = draw_shocks(rng, (n_paths, n_steps), shocks, df)
    increments = params.log_drift * dt + params.sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return s0 * np.exp(log_paths)


def gbm_return_quantile(params: GBMParams, horizon: int, q: float) -> float:
    """Closed-form q-quantile of the simple return over `horizon` bars under normal GBM."""
    t = horizon * params.dt
    z = stats.norm.ppf(q)
    return float(np.exp(params.log_drift * t + params.sigma * np.sqrt(t) * z) - 1.0)
