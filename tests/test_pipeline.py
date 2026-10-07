import numpy as np
import pandas as pd

from quant_gbm.pipeline import Config, run


def test_pipeline_from_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 1500
    sd = np.where((np.arange(n) // 250) % 2 == 0, 0.006, 0.018)
    prices = 1500 * np.exp(np.cumsum(rng.normal(0.0003, sd)))
    dates = pd.bdate_range("2019-01-01", periods=n)
    csv = tmp_path / "synthetic.csv"
    pd.DataFrame({"Date": dates, "Close": prices}).to_csv(csv, index=False)

    res = run(Config(ticker=None, csv=str(csv), start="2000-01-01", horizon=63, n_paths=2000,
                     n_states=2, backtest_window=500, output_dir=str(tmp_path / "out")))
    out = tmp_path / "out"
    for f in ["REPORT.md", "results.json", "risk_comparison.csv", "fan_gbm.png",
              "fan_regime.png", "regimes.png", "var_backtest.png", "backtest_evaluation.csv"]:
        assert (out / f).exists(), f
    assert res["regime"]["n_states"] == 2
