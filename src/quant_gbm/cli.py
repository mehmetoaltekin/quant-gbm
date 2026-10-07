"""Command-line interface:  quant-gbm --ticker GC=F --years 10"""

from __future__ import annotations

import argparse

from .pipeline import Config, run


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="quant-gbm",
                                description="GBM and regime-switching Monte Carlo risk simulation.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--ticker", default="GC=F", help="Yahoo Finance symbol (default: GC=F, gold futures)")
    src.add_argument("--csv", help="Local CSV with a date column and a close column (MT5 exports supported)")
    p.add_argument("--label", help="Display name used in charts and report")
    p.add_argument("--start", help="Start date YYYY-MM-DD (overrides --years)")
    p.add_argument("--end", help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--years", type=float, default=10.0, help="Look-back in years (default: 10)")
    p.add_argument("--horizon", type=int, default=252, help="Simulation horizon in bars (default: 252)")
    p.add_argument("--paths", type=int, default=10_000, help="Paths per model (default: 10000)")
    p.add_argument("--states", type=int, help="Number of HMM states (default: chosen by BIC from 1-4)")
    p.add_argument("--t-df", type=float, default=4.0, help="Student-t df for fat-tail variant (default: 4)")
    p.add_argument("--periods-per-year", type=int, default=252, help="Bars per year (252 daily equities/gold, 365 crypto)")
    p.add_argument("--alpha", type=float, default=0.05, help="VaR tail probability (default: 0.05)")
    p.add_argument("--no-backtest", action="store_true", help="Skip the out-of-sample VaR backtest")
    p.add_argument("--backtest-horizon", type=int, default=21)
    p.add_argument("--backtest-window", type=int, default=756)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    a = p.parse_args(argv)

    cfg = Config(
        ticker=None if a.csv else a.ticker, csv=a.csv, label=a.label, start=a.start, end=a.end,
        years=a.years, horizon=a.horizon, n_paths=a.paths, n_states=a.states, t_df=a.t_df,
        periods_per_year=a.periods_per_year, seed=a.seed, backtest=not a.no_backtest,
        backtest_horizon=a.backtest_horizon, backtest_window=a.backtest_window,
        alpha=a.alpha, output_dir=a.out,
    )
    res = run(cfg)
    g, d = res["gbm_params"], res["data"]
    print(f"{cfg.name}: {d['first_date']} -> {d['last_date']}  ({d['n_prices']} prices)")
    print(f"GBM: mu={g['mu']:.2%}  sigma={g['sigma']:.2%}   HMM states: {res['regime']['n_states']}")
    print(f"Report written to {cfg.output_dir}/REPORT.md")


if __name__ == "__main__":
    main()
