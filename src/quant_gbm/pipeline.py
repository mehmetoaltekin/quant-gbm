"""End-to-end analysis: data -> parameters -> three simulation models -> risk -> backtest -> report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting
from .data import load_prices, log_returns
from .gbm import estimate_gbm, simulate_gbm
from .regime import filtered_probabilities, fit_hmm, select_n_states, simulate_regime_gbm
from .risk import comparison_table
from .validation import backtest_var, evaluate


@dataclass
class Config:
    ticker: str | None = "GC=F"
    csv: str | None = None
    label: str | None = None
    start: str | None = None
    end: str | None = None
    years: float = 10.0
    horizon: int = 252
    n_paths: int = 10_000
    n_states: int | None = None       # None = choose by BIC from 1..4
    t_df: float = 4.0                 # Student-t degrees of freedom for the fat-tail variant
    periods_per_year: int = 252
    seed: int = 42
    backtest: bool = True
    backtest_horizon: int = 21
    backtest_window: int = 756
    alpha: float = 0.05
    output_dir: str = "outputs"

    @property
    def name(self) -> str:
        return self.label or self.ticker or Path(self.csv).stem


def run(cfg: Config) -> dict:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ppy = cfg.periods_per_year

    # 1. Data
    prices = load_prices(cfg.ticker if cfg.csv is None else None, cfg.csv, cfg.start, cfg.end, cfg.years)
    r = log_returns(prices)
    s0 = float(prices.iloc[-1])

    # 2. Constant-parameter GBM
    gbm_params = estimate_gbm(r.to_numpy(), ppy)

    # 3. Regime model (full-sample fit; see README on look-ahead)
    if cfg.n_states is None:
        regime_model, ic_table = select_n_states(r.to_numpy(), seed=cfg.seed)
    else:
        regime_model = fit_hmm(r.to_numpy(), cfg.n_states, seed=cfg.seed)
        ic_table = pd.DataFrame([{"n_states": cfg.n_states, "loglik": regime_model.loglik,
                                  "n_params": regime_model.n_params, "aic": regime_model.aic,
                                  "bic": regime_model.bic}]).set_index("n_states")
    probs = filtered_probabilities(regime_model, r.to_numpy())
    current_probs = probs[-1]

    # 4. Simulations
    path_sets = {
        "GBM (normal)": simulate_gbm(s0, gbm_params, cfg.horizon, cfg.n_paths, seed=cfg.seed),
        f"GBM (Student-t, df={cfg.t_df:g})": simulate_gbm(s0, gbm_params, cfg.horizon, cfg.n_paths,
                                                          shocks="t", df=cfg.t_df, seed=cfg.seed),
    }
    if regime_model.n_states > 1:
        regime_paths, _ = simulate_regime_gbm(s0, regime_model, current_probs, cfg.horizon,
                                              cfg.n_paths, seed=cfg.seed)
        path_sets[f"Regime-switching ({regime_model.n_states} states)"] = regime_paths

    # 5. Risk
    risk = comparison_table(path_sets, alphas=(0.01, cfg.alpha))

    # 6. Out-of-sample backtest
    bt_results, bt_eval = {}, pd.DataFrame()
    if cfg.backtest:
        bt_results["GBM"] = backtest_var(prices, "gbm", cfg.backtest_horizon, cfg.backtest_window,
                                         cfg.alpha, periods_per_year=ppy, seed=cfg.seed)
        if regime_model.n_states > 1:
            bt_results["Regime"] = backtest_var(prices, "regime", cfg.backtest_horizon, cfg.backtest_window,
                                                cfg.alpha, n_states=regime_model.n_states,
                                                periods_per_year=ppy, seed=cfg.seed)
        bt_eval = pd.DataFrame({k: evaluate(v, cfg.alpha) for k, v in bt_results.items()})

    # 7. Save tables
    empirical = _empirical_stats(r, ppy)
    risk.to_csv(out / "risk_comparison.csv")
    ic_table.to_csv(out / "model_selection.csv")
    regime_model.summary(ppy).to_csv(out / "regime_parameters.csv")
    pd.DataFrame(probs, index=r.index, columns=[f"state_{k}" for k in range(regime_model.n_states)]) \
        .to_csv(out / "filtered_probabilities.csv")
    for k, v in bt_results.items():
        v.to_csv(out / f"backtest_{k.lower()}.csv")
    if not bt_eval.empty:
        bt_eval.to_csv(out / "backtest_evaluation.csv")

    # 8. Figures
    first_name = next(iter(path_sets))
    plotting.fan_chart(path_sets[first_name], f"{cfg.name}: {cfg.n_paths:,} GBM paths, {cfg.horizon} days",
                       str(out / "fan_gbm.png"))
    if regime_model.n_states > 1:
        reg_name = list(path_sets)[-1]
        plotting.fan_chart(path_sets[reg_name], f"{cfg.name}: {cfg.n_paths:,} regime-switching paths",
                           str(out / "fan_regime.png"))
        plotting.regime_history(prices, probs, f"{cfg.name}: filtered HMM regimes", str(out / "regimes.png"))
    plotting.terminal_distributions(path_sets, f"{cfg.name}: {cfg.horizon}-day return distribution",
                                    str(out / "terminal_distributions.png"), cfg.alpha)
    if bt_results:
        plotting.var_backtest(bt_results, cfg.alpha, cfg.backtest_horizon, str(out / "var_backtest.png"))

    # 9. Report
    results = {
        "config": asdict(cfg),
        "data": {"first_date": str(prices.index[0].date()), "last_date": str(prices.index[-1].date()),
                 "n_prices": int(len(prices)), "last_price": s0},
        "empirical": empirical,
        "gbm_params": {"mu": gbm_params.mu, "sigma": gbm_params.sigma, "log_drift": gbm_params.log_drift},
        "regime": {"n_states": regime_model.n_states, "current_probs": current_probs.tolist(),
                   "summary": regime_model.summary(ppy).to_dict(orient="index")},
        "risk": risk.to_dict(),
        "backtest": bt_eval.to_dict(),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    (out / "REPORT.md").write_text(_report(cfg, results, risk, ic_table, regime_model.summary(ppy), bt_eval))
    return results


def _empirical_stats(r: pd.Series, ppy: int) -> dict:
    from scipy import stats
    x = r.to_numpy()
    jb = stats.jarque_bera(x)
    sq = x**2 - (x**2).mean()
    acf1_sq = float((sq[1:] * sq[:-1]).sum() / (sq**2).sum())
    return {
        "ann_log_return": float(x.mean() * ppy),
        "ann_vol": float(x.std(ddof=1) * np.sqrt(ppy)),
        "skew": float(stats.skew(x)),
        "excess_kurtosis": float(stats.kurtosis(x)),
        "jarque_bera_p": float(jb.pvalue),
        "acf1_squared_returns": acf1_sq,
        "worst_day": float(x.min()),
        "best_day": float(x.max()),
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _report(cfg, res, risk, ic_table, regime_tbl, bt_eval) -> str:
    d, e, g = res["data"], res["empirical"], res["gbm_params"]
    lines = [
        f"# {cfg.name} — GBM & Regime-Switching Risk Simulation",
        "",
        f"Data: {d['first_date']} to {d['last_date']} ({d['n_prices']:,} prices). Last price: {d['last_price']:,.2f}.",
        f"Horizon: {cfg.horizon} bars, {cfg.n_paths:,} paths per model, seed {cfg.seed}.",
        "",
        "## 1. Empirical return properties",
        "",
        "| Statistic | Value |", "|---|---|",
        f"| Annualised log return | {_pct(e['ann_log_return'])} |",
        f"| Annualised volatility | {_pct(e['ann_vol'])} |",
        f"| Skewness | {e['skew']:.3f} |",
        f"| Excess kurtosis | {e['excess_kurtosis']:.3f} |",
        f"| Jarque–Bera p-value | {e['jarque_bera_p']:.2e} |",
        f"| Lag-1 autocorrelation of squared returns | {e['acf1_squared_returns']:.3f} |",
        f"| Worst / best day (log) | {_pct(e['worst_day'])} / {_pct(e['best_day'])} |",
        "",
        "Excess kurtosis > 0 and positive autocorrelation in squared returns are the two GBM assumptions "
        "(normal shocks, constant volatility) that the data most directly contradicts.",
        "",
        "## 2. GBM parameters",
        "",
        f"mu (arithmetic drift) = {_pct(g['mu'])}, sigma = {_pct(g['sigma'])}, "
        f"mu − sigma²/2 = {_pct(g['log_drift'])}.",
        "",
        "## 3. HMM regimes (state 0 = lowest volatility)",
        "",
        ic_table.round(2).to_markdown(),
        "",
        regime_tbl.round(4).to_markdown(),
        "",
        "Current filtered regime probabilities: "
        + ", ".join(f"state {k}: {p:.1%}" for k, p in enumerate(res["regime"]["current_probs"])),
        "",
        "## 4. Simulated risk (horizon returns; VaR/CVaR are losses)",
        "",
        _risk_markdown(risk),
        "",
    ]
    if not bt_eval.empty:
        lines += [
            f"## 5. Out-of-sample VaR backtest ({cfg.backtest_horizon}-bar horizon, "
            f"{int((1 - cfg.alpha) * 100)}% VaR, {cfg.backtest_window}-bar rolling window)",
            "",
            bt_eval.round(4).to_markdown(),
            "",
            "Kupiec p < 0.05 rejects correct coverage; independence p < 0.05 indicates clustered breaches.",
            "",
        ]
    lines += [
        "## Caveats",
        "",
        "- Sections 3–4 use parameters fitted on the full sample. Only Section 5 is strictly out of sample.",
        "- Simulations are model-based scenarios under stated assumptions, not forecasts.",
        "- Research use only. Not investment advice.",
    ]
    return "\n".join(lines) + "\n"


def _risk_markdown(risk: pd.DataFrame) -> str:
    pct_rows = [r for r in risk.index if r.startswith(("VaR", "CVaR")) or r.endswith(("return", "drawdown")) or r == "prob_loss"]
    price_rows = [r for r in risk.index if r not in pct_rows]
    fmt = risk.copy().astype(object)
    for r in pct_rows:
        fmt.loc[r] = [_pct(v) for v in risk.loc[r]]
    for r in price_rows:
        fmt.loc[r] = [f"{v:,.2f}" for v in risk.loc[r]]
    return fmt.to_markdown()
