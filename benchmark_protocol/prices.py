"""Price-history loaders for the two frozen universes used in this repo.

Both come from yfinance auto-adjusted close prices over 2021-01-04 .. 2025-12-31
(1255 trading days, no missing values), stored as plain CSVs.

    QUBO universe (25 stocks)  ->  data/prices/prices_daily.csv
    MIQP universe (50 stocks)  ->  data/prices/prices_miqp_daily.csv

Use `load_prices(universe="qubo")` or `load_prices(universe="miqp")`.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "prices"

PRICES_CSV = _DATA_DIR / "prices_daily.csv"            # legacy alias for the 25-stock file
MIQP_PRICES_CSV = _DATA_DIR / "prices_miqp_daily.csv"

UNIVERSE = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META",
    "JPM", "GS", "BAC",
    "JNJ", "UNH", "PFE",
    "KO", "PG", "NKE", "WMT",
    "XOM", "CVX",
    "CAT", "HON", "UPS",
    "DIS", "NFLX", "TSLA", "COST",
]

MIQP_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "NVDA", "JPM", "V",
    "JNJ", "WMT", "PG", "MA", "XOM", "UNH", "HD", "CVX", "LLY", "BAC",
    "ABBV", "KO", "AVGO", "PEP", "COST", "ORCL", "TMO", "CSCO", "PFE", "ABT",
    "ADBE", "NKE", "DIS", "MCD", "WFC", "DHR", "PM", "AMD", "TXN", "VZ",
    "NEE", "HON", "RTX", "AMGN", "INTC", "COP", "LOW", "IBM", "GE", "CAT",
]

WINDOWS = {
    "in_sample":     ("2022-01-01", "2024-12-31"),
    "out_of_sample": ("2025-01-01", "2025-12-31"),
    "full":          ("2021-01-04", "2025-12-31"),
}

_CSV_BY_UNIVERSE = {
    "qubo": PRICES_CSV,
    "miqp": MIQP_PRICES_CSV,
}

_TICKERS_BY_UNIVERSE = {
    "qubo": UNIVERSE,
    "miqp": MIQP_UNIVERSE,
}


def load_prices(
    start: str | None = None,
    end: str | None = None,
    tickers: list[str] | None = None,
    universe: str = "qubo",
) -> "pd.DataFrame":
    """Daily close prices indexed by Date.

    universe : "qubo" (25 stocks, default) or "miqp" (50 stocks).
    start, end : ISO date strings, inclusive.
    tickers : subset of the chosen universe.
    """
    import pandas as pd

    if universe not in _CSV_BY_UNIVERSE:
        raise ValueError(f"unknown universe: {universe!r} (use 'qubo' or 'miqp')")
    csv_path = _CSV_BY_UNIVERSE[universe]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"prices file not found at {csv_path}. "
            "Run scripts/fetch_prices.py to regenerate."
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    if tickers is not None:
        missing = set(tickers) - set(df.columns)
        if missing:
            raise ValueError(f"unknown tickers in {universe} universe: {sorted(missing)}")
        df = df[tickers]
    if start is not None or end is not None:
        df = df.loc[start:end]
    return df


def load_returns(
    start: str | None = None,
    end: str | None = None,
    tickers: list[str] | None = None,
    universe: str = "qubo",
    log_returns: bool = False,
) -> "pd.DataFrame":
    """Daily simple (or log) returns from the chosen universe."""
    import numpy as np

    px = load_prices(start=start, end=end, tickers=tickers, universe=universe)
    if log_returns:
        return np.log(px / px.shift(1)).dropna()
    return px.pct_change().dropna()
