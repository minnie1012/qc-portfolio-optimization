"""Regenerate the frozen price CSVs from yfinance.

    pip install yfinance pandas
    python scripts/fetch_prices.py --universe qubo    # 25 stocks -> prices_daily.csv
    python scripts/fetch_prices.py --universe miqp    # 50 stocks -> prices_miqp_daily.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from benchmark_protocol.prices import (
    PRICES_CSV,
    MIQP_PRICES_CSV,
    UNIVERSE,
    MIQP_UNIVERSE,
)

CONFIG = {
    "qubo": (UNIVERSE, PRICES_CSV),
    "miqp": (MIQP_UNIVERSE, MIQP_PRICES_CSV),
}


def fetch(start: str, end: str, tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers].copy()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    return prices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=list(CONFIG), default="qubo")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--out", default=None, help="override output path")
    args = parser.parse_args()

    tickers, default_out = CONFIG[args.universe]
    out_path = Path(args.out) if args.out else default_out

    print(f"fetching {len(tickers)} tickers ({args.universe} universe) "
          f"from {args.start} to {args.end}")
    df = fetch(args.start, args.end, tickers)
    nan_count = int(df.isna().sum().sum())
    print(f"got shape={df.shape}, NaN count={nan_count}")
    if nan_count > 0:
        before = len(df)
        df = df.dropna()
        print(f"dropping {before - len(df)} rows with NaNs -> {df.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, float_format="%.6f")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
