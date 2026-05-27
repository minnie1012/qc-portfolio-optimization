import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import csv
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from benchmark_protocol import instances


# ── Michael's backtest function ────────────────────────────────────────────────

def backtesting(stocks, weights, risk_free=0.0456/252):
    train = yf.download(stocks, start="2022-01-01", end="2024-12-31", auto_adjust=True, progress=False)
    test = yf.download(stocks, start="2025-01-01", end="2025-12-31", auto_adjust=True, progress=False)
    backtest_data = pd.DataFrame(test['Close'])
    train = pd.DataFrame(train['Close'])
    cov = train.cov()
    weight_arr = np.array(weights)
    risk = np.dot(np.dot(weight_arr.T, cov), weight_arr)
    returns = 0
    for i in range(len(stocks)):
        temp = weights[i] * (backtest_data[stocks[i]].iloc[-1] - backtest_data[stocks[i]].iloc[0])
        returns += temp
    sharpe = (returns - risk_free) / risk
    return sharpe


# ── Load result files ──────────────────────────────────────────────────────────

results_dir = Path(__file__).resolve().parent.parent / "results"

qaoa_files = sorted((results_dir / "qaoa").glob("qaoa_saksham__*.json"))
warm_files = sorted((results_dir / "warm_start").glob("warm_start_qaoa_saksham__*.json"))

all_files = list(qaoa_files) + list(warm_files)
print(f"Found {len(qaoa_files)} QAOA results and {len(warm_files)} Warm Start results")
print(f"Running backtest on {len(all_files)} total results...")
print()


# ── Run backtest on each result ────────────────────────────────────────────────

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
        print(f"Skipping {instance_id} — could not load instance: {e}")
        continue

    # Get selected stocks
    bitstring = r["bitstring"]
    selected_stocks = [inst.asset_tickers[i] for i, b in enumerate(bitstring) if b == 1]
    K = len(selected_stocks)

    if K == 0:
        print(f"Skipping {instance_id} — no stocks selected")
        continue

    # Equal weights since QAOA picks binary 0/1
    weights = [1/K] * K

    try:
        sharpe = backtesting(selected_stocks, weights)
        sharpe_val = float(sharpe)
    except Exception as e:
        print(f"Skipping {instance_id} p={p} ({algorithm}) — backtest error: {e}")
        sharpe_val = None

    rows.append({
        "algorithm": algorithm,
        "instance_id": instance_id,
        "p": p,
        "selected_stocks": "+".join(selected_stocks),
        "K": K,
        "sharpe_ratio": round(sharpe_val, 4) if sharpe_val is not None else None,
        "feasible": r["feasible"],
        "approx_ratio": r.get("approx_ratio"),
    })

    sharpe_str = f"{sharpe_val:.4f}" if sharpe_val is not None else "N/A"
    print(f"{algorithm:<30} {instance_id:<15} p={p} | stocks: {selected_stocks} | Sharpe: {sharpe_str}")


# ── Save to CSV ────────────────────────────────────────────────────────────────

output_dir = Path(__file__).resolve().parent.parent / "results" / "backtest"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "qaoa_backtest_results.csv"

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["algorithm", "instance_id", "p", "selected_stocks", "K", "sharpe_ratio", "feasible", "approx_ratio"])
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Saved to: {output_path}")
print()


# ── Print summary table ────────────────────────────────────────────────────────

print("=" * 100)
print(f"{'Algorithm':<30} {'Instance':<15} {'p':>3} {'Selected Stocks':<30} {'Sharpe':>8} {'AR':>8}")
print("=" * 100)

for row in rows:
    sharpe_str = f"{row['sharpe_ratio']:.4f}" if row['sharpe_ratio'] is not None else "N/A"
    ar_str = f"{row['approx_ratio']:.4f}" if row['approx_ratio'] is not None else "N/A"
    print(f"{row['algorithm']:<30} {row['instance_id']:<15} {row['p']:>3} {row['selected_stocks']:<30} {sharpe_str:>8} {ar_str:>8}")

print("=" * 100)
print(f"\nTotal results: {len(rows)}")