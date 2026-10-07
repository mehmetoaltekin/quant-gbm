# quant-gbm

[![CI](https://github.com/mehmetoaltekin/quant-gbm/actions/workflows/ci.yml/badge.svg)](https://github.com/mehmetoaltekin/quant-gbm/actions)

Monte Carlo risk simulation for any asset under three models:

1. **GBM** with normal shocks, the textbook baseline.
2. **GBM with Student-t shocks**, the same volatility but with fat tails.
3. **Regime-switching GBM**, driven by a Gaussian HMM.

Each model produces **VaR, CVaR, a drawdown distribution and percentile bands**. The project then asks the question most GBM demos skip: *does the predicted tail hold up out of sample?* It answers with a walk-forward VaR backtest scored by the Kupiec and Christoffersen tests.

The regime layer follows the methodology of [hmm-regime-detection](https://github.com/mehmetoaltekin/hmm-regime-detection):

- States are sorted by variance, so state 0 is always the lowest-volatility regime.
- The current regime comes from forward-filtered probabilities only, never from smoothed ones.
- The number of states is chosen by BIC.

> Research use only. Nothing here is financial or investment advice.

---

## 1. Model

**GBM.** Prices follow

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,$$

simulated with the exact log-space discretisation

$$\ln\frac{S_{t+\Delta t}}{S_t} = \left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z.$$

Since $\mathbb{E}[\ln(S_{t+\Delta t}/S_t)] = (\mu - \sigma^2/2)\Delta t$, the arithmetic drift is estimated as $\hat\mu = \bar r \cdot N + \hat\sigma^2/2$, where $N$ is the number of periods per year.

**Fat-tail variant.** $Z$ is replaced by a Student-t variable rescaled to unit variance. Volatility stays the same and only the tails get heavier.

**Regime-switching GBM.** A $K$-state Gaussian HMM is fitted to log returns:

$$P(S_t = j \mid S_{t-1} = i) = A_{ij}, \qquad r_t \mid S_t = k \sim \mathcal{N}(m_k, s_k^2).$$

Each simulated path then works as follows:

- It draws its starting regime from today's filtered distribution $P(S_T \mid r_{1:T})$.
- It evolves its own Markov chain through $A$.
- It draws each return from the active regime.

Regime persistence, switching and the resulting volatility clustering are therefore part of the simulated distribution.

**Validation.** At every forecast origin, each model is re-estimated on the trailing window only. It forecasts the $h$-bar VaR, which is compared with the realised return. Origins do not overlap, so breaches are approximately independent. Two tests score the result:

- **Kupiec POF** tests whether the breach rate equals $\alpha$.
- **Christoffersen** tests whether breaches cluster.

## 2. Installation

```bash
git clone https://github.com/mehmetoaltekin/quant-gbm.git
cd quant-gbm
pip install -e ".[dev]"
pytest -q
```

## 3. Usage

### Command line: any ticker

```bash
quant-gbm --ticker GC=F --label XAUUSD --years 10          # gold
quant-gbm --ticker SPY --years 15 --horizon 126             # S&P 500, 6-month horizon
quant-gbm --ticker BTC-USD --periods-per-year 365           # crypto trades every day
quant-gbm --ticker NVDA --states 3 --paths 20000            # force 3 regimes
quant-gbm --csv XAUUSD_D1.csv --label "XAUUSD spot"         # your own data / MT5 export
```

Run `quant-gbm --help` to see every option. Results go to `outputs/` by default:

| File | Content |
|---|---|
| `REPORT.md` | Full write-up with every table filled from your run |
| `risk_comparison.csv` | VaR / CVaR / drawdown / percentiles per model |
| `model_selection.csv` | Log-likelihood, AIC, BIC for K = 1..4 |
| `regime_parameters.csv` | Per-regime volatility, drift, persistence, expected duration |
| `filtered_probabilities.csv` | Daily filtered regime probabilities |
| `backtest_*.csv`, `backtest_evaluation.csv` | Out-of-sample VaR forecasts and test statistics |
| `fan_gbm.png`, `fan_regime.png` | Simulated paths with percentile bands |
| `terminal_distributions.png` | Horizon-return distribution, all models overlaid |
| `regimes.png` | Price coloured by filtered regime |
| `var_backtest.png` | VaR forecasts against realised returns, with breaches marked |

### Python API

```python
from quant_gbm import (load_prices, log_returns, estimate_gbm, simulate_gbm,
                       select_n_states, filtered_probabilities, simulate_regime_gbm,
                       comparison_table, backtest_var, evaluate)

prices = load_prices(ticker="GC=F", years=10)
r = log_returns(prices).to_numpy()

gbm = estimate_gbm(r)
paths_gbm = simulate_gbm(prices.iloc[-1], gbm, n_steps=252, n_paths=10_000, seed=42)

model, ic = select_n_states(r)                        # BIC over K = 1..4
probs_now = filtered_probabilities(model, r)[-1]
paths_reg, states = simulate_regime_gbm(prices.iloc[-1], model, probs_now, 252, 10_000, seed=42)

print(comparison_table({"GBM": paths_gbm, "Regime": paths_reg}))

bt = backtest_var(prices, method="regime", horizon=21, n_states=model.n_states)
print(evaluate(bt, alpha=0.05))
```

### CSV input

Any CSV with a date column and a `close` column works. The delimiter is detected automatically, and MetaTrader 5 exports (`<DATE> <TIME> ... <CLOSE>`, tab-separated) are read directly. To use a different price column, pass `column=` in Python.

## 4. Worked example: XAUUSD

```bash
python examples/xauusd.py                    # GC=F futures as a daily gold proxy
python examples/xauusd.py --csv XAUUSD_D1.csv  # spot gold from your broker
```

Yahoo's spot symbol `XAUUSD=X` has frequent gaps and stale quotes. COMEX gold futures (`GC=F`) are therefore the default proxy. For true spot data, export daily bars from MT5.

## 5. Reading the results

- **Horizon matters.** At a 1-year horizon, the three models often give similar terminal distributions, because summing many daily shocks pulls them toward normality. The fat-tail and regime effects show most clearly at short horizons (`--horizon 21`), in the drawdown statistics, and when the series currently sits in a high-volatility regime.
- **Starting state matters.** The regime model starts from today's filtered probabilities. Its forecast therefore depends on current market conditions, while constant-parameter GBM gives the same forecast in any market.
- **Trust the backtest over the fan chart.** Section 4 of the report shows what a model *implies*. Section 5 shows whether those implications held on unseen data.

## 6. Limitations

- **In-sample fit.** The headline simulation uses parameters fitted on the full sample. Only the VaR backtest is strictly out of sample.
- **Gaussian emissions.** Within a regime, returns are still normal. Heavy tails come only from regime mixing, unless you use the Student-t variant.
- **Constant transition matrix.** Regime switching is time-homogeneous and does not respond to macro variables.
- **No jumps or event risk.** Overnight gaps, central-bank surprises and geopolitical shocks are not modelled explicitly.
- **Drift uncertainty.** The standard error of an annual drift estimate is roughly $\sigma/\sqrt{T}$, which is about 4–5 percentage points over 10 years for gold. The simulated median is far less certain than the volatility.
- **Local optima.** EM can converge to a local optimum. The code uses multiple random starts, but results can still depend on the seed.

## 7. Project structure

```
quant-gbm/
├── src/quant_gbm/
│   ├── data.py         # Yahoo Finance / CSV / MT5 loading
│   ├── gbm.py          # estimation, simulation, closed-form quantiles
│   ├── regime.py       # HMM fit, BIC selection, forward filter, Markov-switching simulation
│   ├── risk.py         # VaR, CVaR, drawdowns, summary tables
│   ├── validation.py   # walk-forward VaR backtest, Kupiec & Christoffersen tests
│   ├── plotting.py     # figures
│   ├── pipeline.py     # end-to-end run + REPORT.md generation
│   └── cli.py          # `quant-gbm` command
├── examples/xauusd.py
├── tests/
└── .github/workflows/ci.yml
```

## 8. References

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3).
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2).
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, 3(2).
- Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review*, 39(4).
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

## Related

- [hmm-regime-detection](https://github.com/mehmetoaltekin/hmm-regime-detection): walk-forward Gaussian HMM regime identification.

## License

MIT
