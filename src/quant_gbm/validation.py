"""Walk-forward VaR backtesting: does the model's tail actually hold out of sample?

At each forecast origin t the model is estimated on the trailing `window` returns only,
a VaR for the next `horizon` bars is produced, and it is compared with the realised return.
Origins are spaced `step` bars apart (default: `horizon`, i.e. non-overlapping windows,
so breach indicators are approximately independent and the tests below are valid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .gbm import estimate_gbm, gbm_return_quantile
from .regime import filtered_probabilities, fit_hmm, simulate_regime_gbm


def backtest_var(
    prices: pd.Series,
    method: str = "gbm",
    horizon: int = 21,
    window: int = 756,
    alpha: float = 0.05,
    step: int | None = None,
    n_states: int = 2,
    n_paths: int = 4000,
    periods_per_year: int = 252,
    seed: int = 0,
) -> pd.DataFrame:
    """Return one row per forecast origin with columns: var, realized, breach."""
    if method not in ("gbm", "regime"):
        raise ValueError("method must be 'gbm' or 'regime'")
    step = step or horizon
    r = np.diff(np.log(prices.to_numpy(dtype=float)))
    rows = []
    for i, t in enumerate(range(window, len(r) - horizon + 1, step)):
        train = r[t - window: t]
        realized = float(np.exp(r[t: t + horizon].sum()) - 1.0)
        if method == "gbm":
            q = gbm_return_quantile(estimate_gbm(train, periods_per_year), horizon, alpha)
        else:
            model = fit_hmm(train, n_states, n_init=3, seed=seed)
            probs = filtered_probabilities(model, train)[-1]
            paths, _ = simulate_regime_gbm(1.0, model, probs, horizon, n_paths, seed=seed + i)
            q = float(np.quantile(paths[:, -1] - 1.0, alpha))
        rows.append({"date": prices.index[t], "var": -q, "realized": realized, "breach": realized < q})
    return pd.DataFrame(rows).set_index("date")


def kupiec_pof(breaches, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures test. H0: breach rate == alpha."""
    b = np.asarray(breaches, dtype=bool)
    n, x = b.size, int(b.sum())
    p_hat = x / n if n else np.nan
    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat) if 0 < x < n else 0.0
    lr = -2.0 * (ll0 - ll1)
    return {"n": n, "breaches": x, "expected": alpha * n, "breach_rate": p_hat,
            "LR_pof": lr, "p_value": float(stats.chi2.sf(lr, 1))}


def christoffersen_independence(breaches) -> dict:
    """Christoffersen (1998) test that breaches do not cluster. H0: independence."""
    b = np.asarray(breaches, dtype=int)
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum()); n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum()); n11 = int(((prev == 1) & (curr == 1)).sum())

    def _ll(p, stay, move):
        return (stay * np.log(1 - p) if stay else 0.0) + (move * np.log(p) if move and p > 0 else 0.0)

    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(len(curr), 1)
    lr = -2.0 * (_ll(pi, n00 + n10, n01 + n11) - (_ll(pi0, n00, n01) + _ll(pi1, n10, n11)))
    lr = max(lr, 0.0)
    return {"LR_ind": lr, "p_value": float(stats.chi2.sf(lr, 1))}


def evaluate(backtest: pd.DataFrame, alpha: float) -> dict:
    out = kupiec_pof(backtest["breach"], alpha)
    ind = christoffersen_independence(backtest["breach"])
    out["kupiec_p_value"] = out.pop("p_value")
    out["LR_ind"] = ind["LR_ind"]
    out["independence_p_value"] = ind["p_value"]
    out["avg_var"] = float(backtest["var"].mean())
    return out
