"""Price loading from Yahoo Finance or a local CSV (including MetaTrader 5 exports)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(
    ticker: str | None = None,
    csv: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    years: float = 10.0,
    column: str | None = None,
) -> pd.Series:
    """Return a clean, sorted, strictly positive close-price series.

    Exactly one of ``ticker`` (Yahoo Finance symbol) or ``csv`` must be given.
    If ``start`` is omitted, the last ``years`` years up to ``end`` (or today) are used.
    """
    if (ticker is None) == (csv is None):
        raise ValueError("Provide exactly one of `ticker` or `csv`.")

    end_d = pd.Timestamp(end).date() if end else date.today()
    start_d = pd.Timestamp(start).date() if start else end_d - timedelta(days=int(365.25 * years))

    if csv is not None:
        prices = _from_csv(Path(csv), column)
        prices = prices.loc[str(start_d): str(end_d)]
    else:
        prices = _from_yfinance(ticker, str(start_d), str(end_d))

    return _clean(prices)


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily (per-bar) log returns."""
    return np.log(prices).diff().dropna()


def _from_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for ticker downloads: pip install yfinance") from exc

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for '{ticker}' ({start} to {end}).")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker) columns
        df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
    return df["Close"].rename(ticker)


def _from_csv(path: Path, column: str | None) -> pd.Series:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:  # MT5 export: separate date & time
        idx = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        date_col = next((c for c in ("datetime", "date", "time", "timestamp") if c in df.columns), df.columns[0])
        idx = pd.to_datetime(df[date_col])

    col = (column or "close").lower()
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path.name}. Available: {list(df.columns)}")
    return pd.Series(df[col].astype(float).values, index=idx, name=path.stem)


def _clean(prices: pd.Series) -> pd.Series:
    prices = prices.astype(float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if len(prices) < 100:
        raise ValueError(f"Only {len(prices)} valid observations; need at least 100.")
    return prices
