# Price history

Two frozen CSVs, one per problem variant.

| File | Variant | Tickers | Days | Range | Size |
|---|---|---|---|---|---|
| `prices_daily.csv` | QUBO (binary selection) | 25 | 1255 | 2021-01-04 → 2025-12-31 | ~343 KB |
| `prices_miqp_daily.csv` | MIQP (continuous weights) | 50 | 1255 | 2021-01-04 → 2025-12-31 | ~670 KB |

Format (both files):

| Column | Meaning |
|---|---|
| `Date` | ISO date (`YYYY-MM-DD`) |
| Each remaining column | one ticker, daily auto-adjusted close (USD), 6-decimal float |

Source: `yfinance`, auto_adjust=True. No missing values.

## Universes

**QUBO universe (25, used by the binary-selection benchmarks)**

```
AAPL  MSFT  GOOG  AMZN  NVDA  META       (tech)
JPM   GS    BAC                          (financials)
JNJ   UNH   PFE                          (healthcare)
KO    PG    NKE   WMT                    (staples)
XOM   CVX                                (energy)
CAT   HON   UPS                          (industrials)
DIS   NFLX  TSLA  COST                   (discretionary)
```

**MIQP universe (50, used by `MIQP.ipynb`)**

```
AAPL MSFT AMZN GOOGL META BRK-B TSLA NVDA JPM   V
JNJ  WMT  PG   MA    XOM  UNH   HD   CVX  LLY   BAC
ABBV KO   AVGO PEP   COST ORCL  TMO  CSCO PFE   ABT
ADBE NKE  DIS  MCD   WFC  DHR   PM   AMD  TXN   VZ
NEE  HON  RTX  AMGN  INTC COP   LOW  IBM  GE    CAT
```

21 tickers overlap; the rest differ. Don't compare a QUBO result against an MIQP result on a per-ticker basis — they're different universes.

## Windows

| Window | Range |
|---|---|
| In-sample | 2022-01-01 .. 2024-12-31 |
| Out-of-sample | 2025-01-01 .. 2025-12-31 |
| Full | 2021-01-04 .. 2025-12-31 |

## How to load

```python
from benchmark_protocol import prices

# QUBO universe (default)
px = prices.load_prices(start="2022-01-01", end="2024-12-31")

# MIQP universe
px = prices.load_prices(start="2025-01-01", end="2025-12-31", universe="miqp")

# returns
rets = prices.load_returns(universe="miqp")
```

## Regenerating

```bash
python scripts/fetch_prices.py --universe qubo
python scripts/fetch_prices.py --universe miqp
```
