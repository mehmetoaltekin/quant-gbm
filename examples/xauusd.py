"""Worked example: 10 years of gold (XAUUSD proxy) through the full pipeline.

Yahoo's spot symbol XAUUSD=X has frequent gaps and stale quotes, so COMEX gold futures (GC=F)
are used as a daily proxy. To use true spot XAUUSD from your broker, export daily bars from
MetaTrader 5 and pass the file instead:

    python examples/xauusd.py --csv XAUUSD_D1.csv
"""

import argparse
from pathlib import Path

from quant_gbm.pipeline import Config, run

parser = argparse.ArgumentParser()
parser.add_argument("--csv", help="Optional MT5 / broker export of daily XAUUSD bars")
args = parser.parse_args()

cfg = Config(
    ticker=None if args.csv else "GC=F",
    csv=args.csv,
    label="XAUUSD",
    years=10,
    horizon=252,
    n_paths=10_000,
    n_states=None,          # BIC chooses between 1 and 4 regimes
    output_dir="outputs/xauusd",
)
run(cfg)
print(Path(cfg.output_dir, "REPORT.md").read_text())
