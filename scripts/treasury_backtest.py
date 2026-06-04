"""Rolling Treasury backtest with Sharpe-ratio evaluation.

This script uses the official U.S. Treasury daily yield-curve feed as a public
proxy for Treasury bond returns. It then:

  1. builds a rolling in-sample window,
  2. selects a small Treasury basket with the repo's QUBO/Sharpe machinery,
  3. allocates weights on the selected basket by maximizing in-sample Sharpe,
  4. evaluates the fixed-weight portfolio on the out-of-sample window,
  5. compares against an equal-weight Treasury baseline.

The feed contains yield curves, not CUSIP-level traded prices, so the return
series here is a constant-maturity proxy rather than a literal traded-bond
total-return series.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol.treasury import (  # noqa: E402
    DEFAULT_MATURITIES,
    RISK_FREE_TENOR,
    annualized_sharpe,
    daily_rate_from_annualized,
    load_yield_curve,
    yields_to_proxy_returns,
)
from problem_definition import ProblemInstance, allocate, build_qubo, brute_force_select  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

START_DATE = "2022-01-03"
END_DATE = "2025-12-31"
LOOKBACK_DAYS = 252
HOLD_DAYS = 21
REBALANCE_STEP = 21
SELECT_K = 4
RISK_AVERSION = 1.0
RISK_FREE_FALLBACK = 0.02

OUTDIR = ROOT / "results" / "treasury_backtest"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _annualized_from_daily(returns: np.ndarray) -> tuple[float, float]:
    if returns.size < 2:
        return float("nan"), float("nan")
    ann_ret = float(np.mean(returns) * 252.0)
    ann_vol = float(np.std(returns, ddof=1) * math.sqrt(252.0))
    return ann_ret, ann_vol


def _equal_weight_returns(test_returns: pd.DataFrame) -> np.ndarray:
    w = np.ones(test_returns.shape[1], dtype=float) / test_returns.shape[1]
    return test_returns.to_numpy() @ w


def rolling_backtest() -> tuple[list[dict], dict]:
    yields = load_yield_curve(START_DATE, END_DATE)
    asset_cols = [m for m in DEFAULT_MATURITIES if m in yields.columns]
    if len(asset_cols) < 2:
        raise ValueError("not enough Treasury maturities available for backtest")
    if RISK_FREE_TENOR not in yields.columns:
        raise ValueError(f"risk-free tenor {RISK_FREE_TENOR!r} missing from Treasury feed")

    returns = yields_to_proxy_returns(yields, asset_cols).dropna()
    yields = yields.loc[returns.index]
    rf_series = yields[RISK_FREE_TENOR].astype(float).dropna() / 100.0
    rf_series = rf_series.reindex(returns.index).ffill().bfill()

    if len(returns) < LOOKBACK_DAYS + HOLD_DAYS:
        raise ValueError("insufficient data for configured lookback/hold horizon")

    rows: list[dict] = []
    all_port_daily: list[np.ndarray] = []
    all_bench_daily: list[np.ndarray] = []
    all_rf_daily: list[np.ndarray] = []

    for start_ix in range(LOOKBACK_DAYS, len(returns) - HOLD_DAYS + 1, REBALANCE_STEP):
        train = returns.iloc[start_ix - LOOKBACK_DAYS : start_ix]
        test = returns.iloc[start_ix : start_ix + HOLD_DAYS]

        train_yields = yields.iloc[start_ix - LOOKBACK_DAYS : start_ix]
        test_yields = yields.iloc[start_ix : start_ix + HOLD_DAYS]

        mu = train.mean().to_numpy() * 252.0
        sigma = train.cov().to_numpy() * 252.0
        rf_annual = float(train_yields[RISK_FREE_TENOR].mean()) if not train_yields.empty else RISK_FREE_FALLBACK
        rf_annual /= 100.0

        inst = ProblemInstance(
            mu=mu,
            sigma=sigma,
            K=min(SELECT_K, len(asset_cols)),
            q=RISK_AVERSION,
            tickers=asset_cols,
        )

        Q = build_qubo(inst)
        x_star, _ = brute_force_select(Q, K=inst.K)
        selected = [i for i, b in enumerate(x_star) if b == 1]
        alloc = allocate(inst, selected, risk_free_rate=rf_annual)
        weights = np.asarray(alloc["weights"], dtype=float)

        test_sel = test.iloc[:, selected]
        port_daily = test_sel.to_numpy() @ weights
        bench_daily = _equal_weight_returns(test)
        rf_daily = np.array([daily_rate_from_annualized(v * 100.0) for v in rf_series.loc[test.index]])

        port_excess = port_daily - rf_daily
        bench_excess = bench_daily - rf_daily

        port_ann_ret, port_ann_vol = _annualized_from_daily(port_daily)
        bench_ann_ret, bench_ann_vol = _annualized_from_daily(bench_daily)
        port_sharpe = annualized_sharpe(port_daily, rf_daily)
        bench_sharpe = annualized_sharpe(bench_daily, rf_daily)

        equity = float(np.prod(1.0 + port_daily) - 1.0)
        bench_equity = float(np.prod(1.0 + bench_daily) - 1.0)

        rows.append(
            {
                "rebalance_date": str(test.index[0].date()),
                "train_end": str(train.index[-1].date()),
                "selected_maturities": ",".join(asset_cols[i] for i in selected),
                "selected_weights": ",".join(f"{w:.6f}" for w in weights),
                "in_sample_sharpe": float(alloc["sharpe_ratio"]),
                "out_sample_return": port_ann_ret,
                "out_sample_vol": port_ann_vol,
                "out_sample_sharpe": port_sharpe,
                "out_sample_total_return": equity,
                "baseline_return": bench_ann_ret,
                "baseline_vol": bench_ann_vol,
                "baseline_sharpe": bench_sharpe,
                "baseline_total_return": bench_equity,
                "rf_annual": rf_annual,
                "rf_daily_mean": float(np.mean(rf_daily)),
                "n_days": int(len(test)),
            }
        )

        all_port_daily.append(port_daily)
        all_bench_daily.append(bench_daily)
        all_rf_daily.append(rf_daily)

    port_concat = np.concatenate(all_port_daily)
    bench_concat = np.concatenate(all_bench_daily)
    rf_concat = np.concatenate(all_rf_daily)

    summary = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "lookback_days": LOOKBACK_DAYS,
        "hold_days": HOLD_DAYS,
        "rebalance_step": REBALANCE_STEP,
        "select_k": SELECT_K,
        "asset_universe": asset_cols,
        "risk_free_tenor": RISK_FREE_TENOR,
        "periods": len(rows),
        "portfolio": {
            "annual_return": _annualized_from_daily(port_concat)[0],
            "annual_vol": _annualized_from_daily(port_concat)[1],
            "sharpe": annualized_sharpe(port_concat, rf_concat),
            "total_return": float(np.prod(1.0 + port_concat) - 1.0),
        },
        "baseline": {
            "annual_return": _annualized_from_daily(bench_concat)[0],
            "annual_vol": _annualized_from_daily(bench_concat)[1],
            "sharpe": annualized_sharpe(bench_concat, rf_concat),
            "total_return": float(np.prod(1.0 + bench_concat) - 1.0),
        },
    }
    return rows, summary


def main() -> None:
    rows, summary = rolling_backtest()

    rows_csv = OUTDIR / "treasury_backtest_periods.csv"
    summary_json = OUTDIR / "treasury_backtest_summary.json"

    with open(rows_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"wrote {rows_csv}")
    print(f"wrote {summary_json}")
    print()
    print("Treasury backtest summary")
    print(f"  periods: {summary['periods']}")
    print(f"  portfolio Sharpe: {summary['portfolio']['sharpe']:.4f}")
    print(f"  baseline  Sharpe: {summary['baseline']['sharpe']:.4f}")
    print(f"  portfolio total return: {summary['portfolio']['total_return']:.4%}")
    print(f"  baseline  total return: {summary['baseline']['total_return']:.4%}")


if __name__ == "__main__":
    main()
