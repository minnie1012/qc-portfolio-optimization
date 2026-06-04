"""Treasury yield-curve data loader and proxy-return utilities.

The U.S. Treasury publishes daily yield-curve data through its XML feed.
This module loads that feed, caches it locally, and converts the series into
simple proxy returns for backtesting.

Important caveat:
  The public feed provides yield-curve rates, not CUSIP-level traded prices.
  We therefore use a constant-maturity proxy price model when turning yields
  into daily returns. That is appropriate for a reproducible public backtest,
  but it is still an approximation of bond total return.
"""
from __future__ import annotations

import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "treasury"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TREASURY_XML_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
)


# Treasury feed field-name aliases. The feed has changed field casing over time,
# so we normalize aggressively and then map to human-readable maturity labels.
_ALIASES = {
    "bc1month": "1 Mo",
    "bc15month": "1.5 Mo",
    "bc2month": "2 Mo",
    "bc3month": "3 Mo",
    "bc4month": "4 Mo",
    "bc6month": "6 Mo",
    "bc1year": "1 Yr",
    "bc2year": "2 Yr",
    "bc3year": "3 Yr",
    "bc5year": "5 Yr",
    "bc7year": "7 Yr",
    "bc10year": "10 Yr",
    "bc20year": "20 Yr",
    "bc30year": "30 Yr",
    "1month": "1 Mo",
    "15month": "1.5 Mo",
    "2month": "2 Mo",
    "3month": "3 Mo",
    "4month": "4 Mo",
    "6month": "6 Mo",
    "1year": "1 Yr",
    "2year": "2 Yr",
    "3year": "3 Yr",
    "5year": "5 Yr",
    "7year": "7 Yr",
    "10year": "10 Yr",
    "20year": "20 Yr",
    "30year": "30 Yr",
    "newdate": "Date",
    "date": "Date",
}


DEFAULT_MATURITIES = ["1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]
RISK_FREE_TENOR = "3 Mo"


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _parse_entry(entry: ET.Element) -> dict[str, str | None]:
    """Extract leaf values from one OData entry."""
    row: dict[str, str | None] = {}
    props = entry.find(".//{*}properties")
    source = list(props) if props is not None else list(entry)
    for child in source:
        key = _normalize_key(child.tag.rsplit("}", 1)[-1])
        if key in row:
            continue
        text = child.text.strip() if child.text else None
        row[key] = text
    return row


def _fetch_year_xml(year: int, data_key: str = "daily_treasury_yield_curve") -> str:
    params = {
        "data": data_key,
        "field_tdr_date_value": str(year),
    }
    url = TREASURY_XML_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _fetch_all_years_xml(
    start_year: int,
    end_year: int,
    data_key: str = "daily_treasury_yield_curve",
) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    for year in range(start_year, end_year + 1):
        xml_text = _fetch_year_xml(year, data_key=data_key)
        root = ET.fromstring(xml_text)
        entries = root.findall(".//{*}entry")
        for entry in entries:
            row = _parse_entry(entry)
            if row:
                rows.append(row)
    return rows


def load_yield_curve(
    start: str,
    end: str,
    *,
    refresh: bool = False,
    cache_name: str = "daily_treasury_yield_curve.csv",
) -> pd.DataFrame:
    """Load Treasury daily yield-curve data for a date window.

    Results are cached under data/treasury/ for reproducibility and to avoid
    repeated network fetches.
    """
    cache_path = DATA_DIR / cache_name
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        return df.set_index("Date").sort_index().loc[start:end]

    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    rows = _fetch_all_years_xml(start_year, end_year)
    if not rows:
        raise ValueError("Treasury XML feed returned no rows")

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Treasury XML feed parsed into an empty frame")

    rename_map: dict[str, str] = {}
    for col in df.columns:
        norm = _normalize_key(col)
        if norm in _ALIASES:
            rename_map[col] = _ALIASES[norm]
    df = df.rename(columns=rename_map)

    date_col = next((c for c in df.columns if c.lower() == "date" or "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("could not locate a date column in Treasury feed")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "Date"

    for col in df.columns:
        if col == "Date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(cache_path)
    return df.loc[start:end]


def choose_available_maturities(df: pd.DataFrame, maturities: Iterable[str]) -> list[str]:
    """Return maturities that exist and are not all-missing in the frame."""
    out = []
    for m in maturities:
        if m in df.columns and df[m].notna().any():
            out.append(m)
    return out


def yields_to_proxy_prices(yields_df: pd.DataFrame, maturities: list[str]) -> pd.DataFrame:
    """Convert annualized Treasury yields into constant-maturity proxy prices.

    We use a simple zero-coupon style approximation:
        price_t = exp(-yield_t * duration_years)

    This is not a traded bond price, but it produces a stable return series for
    a public backtest.
    """
    duration_years = {
        "1 Mo": 1.0 / 12.0,
        "1.5 Mo": 1.5 / 12.0,
        "2 Mo": 2.0 / 12.0,
        "3 Mo": 3.0 / 12.0,
        "4 Mo": 4.0 / 12.0,
        "6 Mo": 6.0 / 12.0,
        "1 Yr": 1.0,
        "2 Yr": 2.0,
        "3 Yr": 3.0,
        "5 Yr": 5.0,
        "7 Yr": 7.0,
        "10 Yr": 10.0,
        "20 Yr": 20.0,
        "30 Yr": 30.0,
    }

    cols = choose_available_maturities(yields_df, maturities)
    if not cols:
        raise ValueError("no requested maturities available in Treasury frame")

    px = pd.DataFrame(index=yields_df.index)
    for m in cols:
        y = yields_df[m].astype(float) / 100.0
        px[m] = np.exp(-y * duration_years[m])
    return px.dropna(how="all")


def yields_to_proxy_returns(yields_df: pd.DataFrame, maturities: list[str]) -> pd.DataFrame:
    """Daily simple returns from the proxy price series."""
    px = yields_to_proxy_prices(yields_df, maturities)
    rets = px.pct_change().dropna(how="all")
    return rets.dropna(axis=1, how="all")


def daily_rate_from_annualized(yield_pct: float) -> float:
    """Convert an annualized percent yield to an approximate daily simple rate."""
    return float((1.0 + yield_pct / 100.0) ** (1.0 / 252.0) - 1.0)


def annualized_sharpe(daily_returns: np.ndarray, daily_rf: np.ndarray | float = 0.0) -> float | None:
    """Annualized Sharpe ratio on daily excess returns."""
    r = np.asarray(daily_returns, dtype=float)
    if np.isscalar(daily_rf):
        ex = r - float(daily_rf)
    else:
        ex = r - np.asarray(daily_rf, dtype=float)
    if ex.size < 2:
        return None
    vol = ex.std(ddof=1)
    if vol < 1e-12:
        return None
    return float(ex.mean() / vol * math.sqrt(252.0))
