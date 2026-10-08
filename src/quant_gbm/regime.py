"""Regime-switching GBM driven by a Gaussian HMM on log returns.

Conventions follow https://github.com/mehmetoaltekin/hmm-regime-detection:
  * states are re-indexed by emission variance (state 0 = lowest volatility),
  * the current regime is taken from forward-filtered probabilities P(S_t | Y_1:t) only,
  * the number of states is chosen by BIC (or AIC).

Simulation: each path samples its own regime chain from the fitted transition matrix,
starting from today's filtered regime distribution, and draws log returns from the
active state's Gaussian emission. Regime persistence, switching and the resulting
volatility clustering are therefore part of the simulated distribution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

_SCALE = 100.0  # fit on percentage returns for numerical stability


@dataclass
class RegimeModel:
    means: np.ndarray       # per-bar mean log return of each state, shape (K,)
    stds: np.ndarray        # per-bar std of log returns of each state, shape (K,)
    transmat: np.ndarray    # (K, K), rows sum to 1
    startprob: np.ndarray   # (K,)
    loglik: float           # log-likelihood of the (unscaled) returns
    n_obs: int

    @property
    def n_states(self) -> int:
        return len(self.means)

    @property
    def n_params(self) -> int:
        k = self.n_states
        return (k - 1) + k * (k - 1) + 2 * k

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * np.log(self.n_obs)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    def expected_durations(self) -> np.ndarray:
        """Expected number of bars spent in each state per visit, 1 / (1 - p_kk)."""
        stay = np.clip(np.diag(self.transmat), 0, 1 - 1e-12)
        return 1.0 / (1.0 - stay)

    def stationary_distribution(self) -> np.ndarray:
        vals, vecs = np.linalg.eig(self.transmat.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
        return v / v.sum()

    def summary(self, periods_per_year: int = 252) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ann_log_drift": self.means * periods_per_year,
                "ann_vol": self.stds * np.sqrt(periods_per_year),
                "p_stay": np.diag(self.transmat),
                "expected_duration_bars": self.expected_durations(),
                "long_run_share": self.stationary_distribution(),
            },
            index=pd.Index(range(self.n_states), name="state"),
        )


def fit_hmm(log_returns, n_states: int, n_init: int = 10, seed: int = 0) -> RegimeModel:
    """Fit a K-state Gaussian HMM with multiple random starts; keep the best likelihood."""
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size

    if n_states == 1:
        mu, sd = r.mean(), r.std(ddof=0)
        ll = stats.norm.logpdf(r, mu, sd).sum()
        return RegimeModel(np.array([mu]), np.array([sd]), np.ones((1, 1)), np.ones(1), float(ll), n)

    from hmmlearn.hmm import GaussianHMM

    x = (r * _SCALE).reshape(-1, 1)
    best, best_ll = None, -np.inf
    for i in range(n_init):
        model = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=500, tol=1e-6, random_state=seed + i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(x)
                ll = model.score(x)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = model, ll
    if best is None:
        raise RuntimeError(f"HMM with {n_states} states failed to converge.")

    means = best.means_.ravel() / _SCALE
    stds = np.sqrt(np.asarray(best.covars_).reshape(n_states, -1)[:, 0]) / _SCALE
    order = np.argsort(stds)  # state 0 = lowest volatility
    loglik = best_ll + n * np.log(_SCALE)  # change of variables back to raw returns
    return RegimeModel(
        means=means[order],
        stds=stds[order],
        transmat=best.transmat_[np.ix_(order, order)],
        startprob=best.startprob_[order],
        loglik=float(loglik),
        n_obs=n,
    )


def select_n_states(log_returns, candidates=(1, 2, 3, 4), criterion: str = "bic",
                    n_init: int = 10, seed: int = 0) -> tuple[RegimeModel, pd.DataFrame]:
    """Fit each candidate K and return the best model plus the comparison table."""
    if criterion not in ("bic", "aic"):
        raise ValueError("criterion must be 'bic' or 'aic'")
    models, rows = {}, []
    for k in candidates:
        try:
            m = fit_hmm(log_returns, k, n_init=n_init, seed=seed)
        except RuntimeError:
            continue
        models[k] = m
        rows.append({"n_states": k, "loglik": m.loglik, "n_params": m.n_params, "aic": m.aic, "bic": m.bic})
    table = pd.DataFrame(rows).set_index("n_states")
    best_k = int(table[criterion].idxmin())
    return models[best_k], table


def filtered_probabilities(model: RegimeModel, log_returns) -> np.ndarray:
    """Forward-algorithm filtered probabilities P(S_t | Y_1:t), shape (T, K). No smoothing."""
    r = np.asarray(log_returns, dtype=float)
    dens = stats.norm.pdf(r[:, None], model.means[None, :], model.stds[None, :]) + 1e-300
    out = np.empty_like(dens)
    alpha = model.startprob * dens[0]
    out[0] = alpha / alpha.sum()
    for t in range(1, len(r)):
        alpha = (out[t - 1] @ model.transmat) * dens[t]
        out[t] = alpha / alpha.sum()
    return out


def simulate_regime_gbm(
    s0: float,
    model: RegimeModel,
    current_probs: np.ndarray,
    n_steps: int,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Markov-switching simulation.

    Returns (paths, states): paths has shape (n_paths, n_steps + 1), states (n_paths, n_steps).
    """
    rng = np.random.default_rng(seed)
    k = model.n_states
    cum = np.cumsum(model.transmat, axis=1)
    cum[:, -1] = 1.0

    current = rng.choice(k, size=n_paths, p=np.asarray(current_probs) / np.sum(current_probs))
    states = np.empty((n_paths, n_steps), dtype=np.int64)
    for t in range(n_steps):
        u = rng.random(n_paths)
        current = np.minimum((u[:, None] > cum[current]).sum(axis=1), k - 1)
        states[:, t] = current

    increments = model.means[states] + model.stds[states] * rng.standard_normal((n_paths, n_steps))
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return s0 * np.exp(log_paths), states
