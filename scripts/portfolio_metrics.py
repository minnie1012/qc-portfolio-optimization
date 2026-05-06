"""Post-process every result in results/raw/ and compute portfolio metrics.

For each (algorithm, instance) result we add:

  Analytical (equal-weight on the selected K, from mu / sigma):
    annual_return, annual_vol, sharpe_ratio_analytical
  Analytical (max-Sharpe Stage-2 allocation):
    maxsharpe_return, maxsharpe_vol, maxsharpe_sharpe
  Out-of-sample backtest (real_* instances only, on the matching CSV):
    bt_total_return, bt_annual_return, bt_annual_vol, bt_sharpe, bt_max_drawdown

Outputs:
  results/summary_tables/portfolio_metrics.csv
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark_protocol import instances, prices  # noqa: E402
from problem_definition import optimize_sharpe  # noqa: E402

RAW = ROOT / "results" / "raw"
REAL_DIR = ROOT / "data" / "instances_real"
OUT_CSV = ROOT / "results" / "summary_tables" / "portfolio_metrics.csv"

RISK_FREE = 0.05
TRADING_DAYS = 252
BACKTEST_WINDOW = ("2025-01-01", "2025-12-31")  # out-of-sample


# ---------------------------------------------------------------------------

def load_instance_any(instance_id: str):
    if instance_id.startswith("real_"):
        path = REAL_DIR / f"{instance_id}.json"
    else:
        path = ROOT / "data" / "instances" / f"{instance_id}.json"
    with open(path) as f:
        d = json.load(f)
    return d


def selected_indices(bitstring: list[int]) -> list[int]:
    return [i for i, b in enumerate(bitstring) if b == 1]


def analytical_metrics(mu: np.ndarray, sigma: np.ndarray, sel: list[int]) -> dict:
    if not sel:
        return {"annual_return": None, "annual_vol": None, "sharpe_ratio_analytical": None}
    K = len(sel)
    w = np.ones(K) / K
    mu_S = mu[sel]
    sigma_S = sigma[np.ix_(sel, sel)]
    ret = float(mu_S @ w)
    vol = float(np.sqrt(w @ sigma_S @ w))
    sharpe = float((ret - RISK_FREE) / vol) if vol > 1e-12 else None
    return {"annual_return": ret, "annual_vol": vol, "sharpe_ratio_analytical": sharpe}


def maxsharpe_metrics(mu: np.ndarray, sigma: np.ndarray, sel: list[int], seed: int) -> dict:
    if not sel:
        return {"maxsharpe_return": None, "maxsharpe_vol": None, "maxsharpe_sharpe": None}
    mu_S = mu[sel]
    sigma_S = sigma[np.ix_(sel, sel)]
    w, sharpe = optimize_sharpe(mu_S, sigma_S, risk_free_rate=RISK_FREE, seed=seed)
    if w.size == 0:
        return {"maxsharpe_return": None, "maxsharpe_vol": None, "maxsharpe_sharpe": None}
    ret = float(mu_S @ w)
    vol = float(np.sqrt(w @ sigma_S @ w))
    return {"maxsharpe_return": ret, "maxsharpe_vol": vol, "maxsharpe_sharpe": float(sharpe)}


def backtest_metrics(instance_id: str, tickers_selected: list[str], universe: str | None = None) -> dict:
    """Equal-weight, daily rebalanced (i.e. fixed weights) on out-of-sample window."""
    if not tickers_selected:
        return _empty_bt()

    if universe is None:
        universe = "miqp" if "miqp" in instance_id else "qubo"
    try:
        rets = prices.load_returns(
            start=BACKTEST_WINDOW[0], end=BACKTEST_WINDOW[1],
            tickers=tickers_selected, universe=universe,
        )
    except Exception as exc:
        # fall back to the other universe if the requested one didn't have all tickers
        try:
            other = "miqp" if universe == "qubo" else "qubo"
            rets = prices.load_returns(
                start=BACKTEST_WINDOW[0], end=BACKTEST_WINDOW[1],
                tickers=tickers_selected, universe=other,
            )
        except Exception:
            return _empty_bt(error=str(exc))

    if rets.empty:
        return _empty_bt(error="empty backtest window")

    K = len(tickers_selected)
    w = np.ones(K) / K
    port_daily = rets.to_numpy() @ w  # (T,)
    cum = float(np.prod(1 + port_daily) - 1)
    ann_ret = float(port_daily.mean() * TRADING_DAYS)
    ann_vol = float(port_daily.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = float((ann_ret - RISK_FREE) / ann_vol) if ann_vol > 1e-12 else None
    equity = np.cumprod(1 + port_daily)
    peak = np.maximum.accumulate(equity)
    drawdown = float(((equity - peak) / peak).min()) if equity.size else None

    return {
        "bt_total_return": cum,
        "bt_annual_return": ann_ret,
        "bt_annual_vol": ann_vol,
        "bt_sharpe": sharpe,
        "bt_max_drawdown": drawdown,
        "bt_n_days": int(len(port_daily)),
    }


def _empty_bt(error: str | None = None) -> dict:
    out = {
        "bt_total_return": None, "bt_annual_return": None, "bt_annual_vol": None,
        "bt_sharpe": None, "bt_max_drawdown": None, "bt_n_days": None,
    }
    if error:
        out["bt_error"] = error
    return out


# ---------------------------------------------------------------------------

COLUMNS = [
    "algorithm", "instance_id", "bucket", "seed", "feasible",
    "objective_value", "approx_ratio", "wall_time_seconds",
    "selected_K", "selected_tickers",
    "annual_return", "annual_vol", "sharpe_ratio_analytical",
    "maxsharpe_return", "maxsharpe_vol", "maxsharpe_sharpe",
    "bt_total_return", "bt_annual_return", "bt_annual_vol",
    "bt_sharpe", "bt_max_drawdown", "bt_n_days",
]


def bucket_of(iid: str) -> str:
    try:
        return instances.bucket_of(iid)
    except ValueError:
        return "real" if iid.startswith("real_") else "unknown"


def main() -> None:
    rows: list[dict] = []
    for path in sorted(RAW.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        iid = r["instance_id"]
        try:
            inst = load_instance_any(iid)
        except FileNotFoundError:
            continue
        mu = np.asarray(inst["mu"], dtype=float)
        sigma = np.asarray(inst["sigma"], dtype=float)
        bits = list(map(int, r["bitstring"]))
        sel = selected_indices(bits)
        tickers_selected = [inst["asset_tickers"][i] for i in sel]

        am = analytical_metrics(mu, sigma, sel)
        ms = maxsharpe_metrics(mu, sigma, sel, seed=int(r["seed"]))
        # Backtest if tickers are real (all current instances now are; SYN_* would skip)
        all_real = tickers_selected and not any(t.startswith("SYN_") for t in tickers_selected)
        if all_real:
            universe_hint = inst.get("_source", {}).get("universe")
            bt = backtest_metrics(iid, tickers_selected, universe=universe_hint)
        else:
            bt = _empty_bt(error="synthetic_tickers")

        row = {
            "algorithm": r["algorithm"],
            "instance_id": iid,
            "bucket": bucket_of(iid),
            "seed": r["seed"],
            "feasible": r["feasible"],
            "objective_value": r["objective_value"],
            "approx_ratio": r.get("approx_ratio"),
            "wall_time_seconds": r["wall_time_seconds"],
            "selected_K": len(sel),
            "selected_tickers": ",".join(tickers_selected),
            **am, **ms, **bt,
        }
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    # quick console summary on real_* instances (where backtest is meaningful)
    real_rows = [r for r in rows if r["bucket"] == "real" and r["feasible"]]
    if real_rows:
        print()
        print(f"{'algorithm':<22} {'instance':<18} {'ann_ret':>9} {'ann_vol':>9} {'sharpe':>9} {'bt_ret':>10} {'bt_sharpe':>10}")
        print("-" * 96)
        for r in real_rows:
            ar = f"{r['annual_return']:.4f}" if r["annual_return"] is not None else "  n/a"
            av = f"{r['annual_vol']:.4f}" if r["annual_vol"] is not None else "  n/a"
            sh = f"{r['sharpe_ratio_analytical']:.4f}" if r["sharpe_ratio_analytical"] is not None else "  n/a"
            btret = f"{r['bt_total_return']:.4f}" if r["bt_total_return"] is not None else "   n/a"
            btsh = f"{r['bt_sharpe']:.4f}" if r["bt_sharpe"] is not None else "    n/a"
            print(f"{r['algorithm']:<22} {r['instance_id']:<18} {ar:>9} {av:>9} {sh:>9} {btret:>10} {btsh:>10}")


if __name__ == "__main__":
    main()
