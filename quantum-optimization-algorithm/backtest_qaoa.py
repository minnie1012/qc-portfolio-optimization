import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import csv
import math
import numpy as np
import pandas as pd
from pathlib import Path
from benchmark_protocol import instances
from benchmark_protocol.prices import load_prices


# Prices come from the frozen CSV (benchmark_protocol.prices) instead of yfinance,
# so the backtest runs offline and deterministically. The end dates mirror the
# original yf.download(..., end=...) calls, whose `end` is exclusive: the last
# trading days included are 2024-12-30 (train) and 2025-12-30 (test).
TRAIN_WINDOW = ("2022-01-01", "2024-12-30")
TEST_WINDOW = ("2025-01-01", "2025-12-30")
RISK_FREE = 0.0456


# ── Michael's backtest function ────────────────────────────────────────────────
# DEPRECATED: divides the 2025 out-of-sample total return by the *in-sample*
# (2022-2024) annualized volatility, so numerator and denominator come from
# different periods and the result is not an out-of-sample Sharpe ratio.
# Kept for comparison with earlier results; use backtesting_oos_realized().

def backtesting(stocks, weights, risk_free=RISK_FREE):
    train = load_prices(*TRAIN_WINDOW, tickers=list(stocks))
    backtest_data = load_prices(*TEST_WINDOW, tickers=list(stocks))
    weight_arr = np.array(weights)
    ret = train.pct_change().dropna()
    cov = ret.cov() * 252
    risk = np.sqrt(np.dot(np.dot(weight_arr.T, cov), weight_arr))
    init = np.array([backtest_data[stocks[i]].iloc[0] for i in range(len(stocks))])
    final = np.array([backtest_data[stocks[i]].iloc[-1] for i in range(len(stocks))])
    returns = (final - init) / init
    profit = np.dot(returns, weight_arr)
    sharpe = (profit - risk_free) / risk
    return sharpe


# ── Corrected out-of-sample Sharpe ─────────────────────────────────────────────

def backtesting_oos_realized(stocks, weights, risk_free=RISK_FREE):
    """Sharpe ratio computed entirely from the realized 2025 OOS return series.

    The portfolio is bought at the first test-day close with `weights` and held
    (buy-and-hold, the same position as the deprecated function's numerator).
    Its realized daily returns r_t give both numerator and denominator:

        Sharpe = mean(r_t - rf_d) / std(r_t - rf_d) * sqrt(252),
        rf_d   = (1 + risk_free)^(1/252) - 1
    """
    px = load_prices(*TEST_WINDOW, tickers=list(stocks))
    w = np.asarray(weights, dtype=float)
    value = (px / px.iloc[0]).to_numpy() @ w
    daily = np.diff(value) / value[:-1]
    rf_d = (1.0 + risk_free) ** (1.0 / 252.0) - 1.0
    excess = daily - rf_d
    return float(excess.mean() / excess.std(ddof=1) * math.sqrt(252.0))


def main():
    # ── Load result files ──────────────────────────────────────────────────────

    results_dir = Path(__file__).resolve().parent.parent / "results"

    qaoa_files = sorted((results_dir / "qaoa").glob("qaoa_saksham__*.json"))
    warm_files = sorted((results_dir / "warm_start").glob("warm_start_qaoa_saksham__*.json"))

    all_files = list(qaoa_files) + list(warm_files)
    print(f"Found {len(qaoa_files)} QAOA results and {len(warm_files)} Warm Start results")
    print(f"Running backtest on {len(all_files)} total results...")
    print()

    # ── Run backtest on each result ────────────────────────────────────────────

    rows = []

    for path in all_files:
        with open(path) as f:
            r = json.load(f)

        instance_id = r["instance_id"]
        algorithm = r["algorithm"]
        p = r["hyperparameters"]["p"]

        try:
            inst = instances.load(instance_id)
        except Exception as e:
            print(f"Skipping {instance_id} - could not load instance: {e}")
            continue

        # Get selected stocks
        bitstring = r["bitstring"]
        selected_stocks = [inst.asset_tickers[i] for i, b in enumerate(bitstring) if b == 1]
        K = len(selected_stocks)

        if K == 0:
            print(f"Skipping {instance_id} - no stocks selected")
            continue

        # Equal weights since QAOA picks binary 0/1
        weights = [1/K] * K

        sharpe_old = float(backtesting(selected_stocks, weights))
        sharpe_new = backtesting_oos_realized(selected_stocks, weights)

        rows.append({
            "algorithm": algorithm,
            "instance_id": instance_id,
            "p": p,
            "selected_stocks": "+".join(selected_stocks),
            "K": K,
            "sharpe_oos_realized": round(sharpe_new, 4),
            "sharpe_deprecated_insample_vol": round(sharpe_old, 4),
            "feasible": r["feasible"],
            "approx_ratio": r.get("approx_ratio"),
        })

        print(f"{algorithm:<30} {instance_id:<15} p={p} | stocks: {selected_stocks} "
              f"| Sharpe OOS: {sharpe_new:.4f} (deprecated: {sharpe_old:.4f})")

    # ── Save to CSV ────────────────────────────────────────────────────────────
    # Written to a new file; the earlier qaoa_backtest_results*.csv (deprecated
    # metric, yfinance data) are left as they were.

    output_dir = results_dir / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qaoa_backtest_results_oos_sharpe.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved to: {output_path}")
    print()

    # ── Print summary table ────────────────────────────────────────────────────

    print("=" * 110)
    print(f"{'Algorithm':<30} {'Instance':<15} {'p':>3} {'Selected Stocks':<30} {'Sharpe':>8} {'Deprec.':>8} {'AR':>8}")
    print("=" * 110)

    for row in rows:
        ar_str = f"{row['approx_ratio']:.4f}" if row['approx_ratio'] is not None else "N/A"
        print(f"{row['algorithm']:<30} {row['instance_id']:<15} {row['p']:>3} {row['selected_stocks']:<30} "
              f"{row['sharpe_oos_realized']:>8.4f} {row['sharpe_deprecated_insample_vol']:>8.4f} {ar_str:>8}")

    print("=" * 110)
    print(f"\nTotal results: {len(rows)}")


if __name__ == "__main__":
    main()
