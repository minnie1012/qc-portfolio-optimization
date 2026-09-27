"""Walk-forward equity backtest: QUBO-select + max-Sharpe vs. mean-variance and 1/N.

Universe : the 25-stock QUBO universe in data/prices/prices_daily.csv
           (daily simple returns from adjusted close).
Windows  : lookback 252 trading days, hold 21, rebalance every 21 days, over the
           full price history (the final holding period may be shorter).
No lookahead: at each rebalance date t, mu / Sigma / rf are estimated from data
           strictly before t (asserted every period).

Strategies (K = 5, q = 1.0):
    qubo_sample / qubo_lw : exact K-cardinality QUBO selection (build_qubo, all
                            C(25,5) subsets enumerated) + max-Sharpe weights in [5%, 50%]
    mv_sample / mv_lw     : long-only max-Sharpe over all 25 assets, weights in [0%, 50%]
    equal_weight          : 1/N over all 25 assets
    (*_sample uses the sample covariance, *_lw the Ledoit-Wolf covariance)

Risk-free rate: U.S. Treasury 3-Mo constant-maturity yield (data/treasury cache).
    - allocation at t uses the last 3-Mo yield observed strictly before t
    - OOS Sharpe uses the realized daily 3-Mo rate on each OOS day
    - constant 4% is used only if no Treasury observation is available

Accounting: weights drift buy-and-hold within each holding period. Turnover at a
rebalance is sum |w_target - w_drifted|; net returns deduct 10 bps x turnover at
each rebalance, including the initial buy-in (turnover 1.0). "avg_turnover" is the
mean over rebalances after the initial buy-in.

Outputs (results/walkforward_equity/): periods.csv, summary.json, equity_curves.png
"""
from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol.covariance import estimate_cov  # noqa: E402
from benchmark_protocol.prices import load_returns  # noqa: E402
from benchmark_protocol.treasury import (  # noqa: E402
    RISK_FREE_TENOR,
    annualized_sharpe,
    daily_rate_from_annualized,
    load_yield_curve,
)
from problem_definition import ProblemInstance, build_qubo, optimize_sharpe  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration (fixed before looking at out-of-sample results)
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = 252
HOLD_DAYS = 21
REBALANCE_STEP = 21
SELECT_K = 5
RISK_AVERSION = 1.0
QUBO_W_BOUNDS = (0.05, 0.50)
MV_W_BOUNDS = (0.00, 0.50)
RF_FALLBACK = 0.04
COST_BPS = 10.0
SEED = 42

STRATEGIES = ["qubo_sample", "qubo_lw", "mv_sample", "mv_lw", "equal_weight"]
COV_OF = {"qubo_sample": "sample", "qubo_lw": "ledoit_wolf",
          "mv_sample": "sample", "mv_lw": "ledoit_wolf", "equal_weight": None}

OUTDIR = ROOT / "results" / "walkforward_equity"


# ---------------------------------------------------------------------------
# Stage 1: exact K-cardinality QUBO solve
# ---------------------------------------------------------------------------

def exact_k_subset_select(Q: np.ndarray, K: int) -> tuple[np.ndarray, float]:
    """Exact minimizer of x^T Q x over bitstrings with sum(x) = K.

    Same search space and optimum as problem_definition.brute_force_select(Q, K),
    but enumerates only the C(N, K) feasible subsets (vectorized), which makes
    N = 25, K = 5 (53,130 subsets) tractable at every rebalance.
    """
    N = Q.shape[0]
    subsets = np.array(list(combinations(range(N), K)), dtype=int)
    costs = Q[subsets[:, :, None], subsets[:, None, :]].sum(axis=(1, 2))
    best = int(np.argmin(costs))
    x = np.zeros(N, dtype=int)
    x[subsets[best]] = 1
    return x, float(costs[best])


# ---------------------------------------------------------------------------
# Target weights per strategy
# ---------------------------------------------------------------------------

def target_weights(strategy: str, mu: np.ndarray, sigma: np.ndarray | None,
                   rf: float, tickers: list[str]) -> tuple[np.ndarray, float | None]:
    """Full-length (N,) target weight vector and in-sample Sharpe (None for 1/N)."""
    N = len(mu)
    if strategy == "equal_weight":
        return np.full(N, 1.0 / N), None

    if strategy.startswith("qubo"):
        inst = ProblemInstance(mu=mu, sigma=sigma, K=SELECT_K, q=RISK_AVERSION, tickers=tickers)
        x, _ = exact_k_subset_select(build_qubo(inst), SELECT_K)
        sel = np.flatnonzero(x)
        w_sel, is_sharpe = optimize_sharpe(
            mu[sel], sigma[np.ix_(sel, sel)], risk_free_rate=rf,
            w_min=QUBO_W_BOUNDS[0], w_max=QUBO_W_BOUNDS[1], seed=SEED,
        )
        w = np.zeros(N)
        w[sel] = w_sel
    elif strategy.startswith("mv"):
        w, is_sharpe = optimize_sharpe(
            mu, sigma, risk_free_rate=rf,
            w_min=MV_W_BOUNDS[0], w_max=MV_W_BOUNDS[1], seed=SEED,
        )
        w = np.asarray(w, dtype=float)
    else:
        raise ValueError(strategy)

    # SLSQP can leave tiny bound violations; clip and renormalize.
    lo = QUBO_W_BOUNDS[0] if strategy.startswith("qubo") else MV_W_BOUNDS[0]
    w = np.where(w > 0, np.clip(w, 0.0, None), 0.0)
    w[w < 1e-10] = 0.0
    w /= w.sum()
    assert abs(w.sum() - 1.0) < 1e-9 and w.max() <= MV_W_BOUNDS[1] + 1e-6
    if strategy.startswith("qubo"):
        assert (w[w > 0] >= lo - 1e-6).all()
    return w, float(is_sharpe)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def perf_metrics(r: np.ndarray, rf_daily: np.ndarray) -> dict:
    n = r.size
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return {
        "annual_return": float(equity[-1] ** (252.0 / n) - 1.0),   # CAGR
        "annual_vol": float(np.std(r, ddof=1) * math.sqrt(252.0)),
        "sharpe": annualized_sharpe(r, rf_daily),                   # realized OOS excess / OOS vol
        "max_drawdown": float((equity / peak - 1.0).min()),
        "total_return": float(equity[-1] - 1.0),
    }


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def load_rf_3mo(start: str, end: str) -> pd.Series:
    """3-Mo Treasury yield in percent, indexed by date."""
    y = load_yield_curve(start, end)
    return y[RISK_FREE_TENOR].astype(float).dropna().sort_index()


def run() -> tuple[list[dict], dict, pd.DataFrame]:
    returns = load_returns(universe="qubo")
    tickers = list(returns.columns)
    N = len(tickers)
    rf_pct = load_rf_3mo(str(returns.index[0].date()), str(returns.index[-1].date()))

    starts = list(range(LOOKBACK_DAYS, len(returns), REBALANCE_STEP))
    cost_rate = COST_BPS / 1e4

    rows: list[dict] = []
    gross: dict[str, list[np.ndarray]] = {s: [] for s in STRATEGIES}
    net: dict[str, list[np.ndarray]] = {s: [] for s in STRATEGIES}
    rf_chunks: list[np.ndarray] = []
    drifted = {s: np.zeros(N) for s in STRATEGIES}   # start fully in cash
    turnovers: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    deltas: list[float] = []
    rf_fallback_periods = 0

    for p, ix in enumerate(starts):
        train = returns.iloc[ix - LOOKBACK_DAYS: ix]
        test = returns.iloc[ix: ix + HOLD_DAYS]
        t0 = test.index.min()
        # No lookahead: every training observation precedes the first OOS day.
        assert train.index.max() < t0, f"lookahead in period {p}"

        # Point-in-time risk-free rate for allocation (strictly before t0).
        rf_hist = rf_pct.loc[rf_pct.index < t0]
        if rf_hist.empty:
            rf_alloc = RF_FALLBACK
            rf_fallback_periods += 1
        else:
            assert rf_hist.index.max() < t0
            rf_alloc = float(rf_hist.iloc[-1]) / 100.0

        # Realized daily rf on each OOS day (last observation on or before that day).
        rf_oos_pct = rf_pct.reindex(rf_pct.index.union(test.index)).ffill().loc[test.index]
        rf_oos_pct = rf_oos_pct.fillna(RF_FALLBACK * 100.0)
        rf_daily = np.array([daily_rate_from_annualized(v) for v in rf_oos_pct])
        rf_chunks.append(rf_daily)

        mu = train.mean().to_numpy() * 252.0
        covs: dict[str, tuple[np.ndarray, float | None]] = {
            m: estimate_cov(train, method=m, annualize=252) for m in ("sample", "ledoit_wolf")
        }
        delta = covs["ledoit_wolf"][1]
        deltas.append(delta)

        R = test.to_numpy()
        for s in STRATEGIES:
            cm = COV_OF[s]
            sigma = covs[cm][0] if cm else None
            w_target, is_sharpe = target_weights(s, mu, sigma, rf_alloc, tickers)

            turnover = float(np.abs(w_target - drifted[s]).sum())
            turnovers[s].append(turnover)

            # Buy-and-hold within the period.
            w = w_target.copy()
            g = np.empty(len(R))
            for d in range(len(R)):
                g[d] = float(w @ R[d])
                w = w * (1.0 + R[d]) / (1.0 + g[d])
            drifted[s] = w

            n_ = g.copy()
            n_[0] = (1.0 - cost_rate * turnover) * (1.0 + g[0]) - 1.0
            gross[s].append(g)
            net[s].append(n_)

            nz = np.flatnonzero(w_target > 0)
            rows.append({
                "period": p,
                "rebalance_date": str(t0.date()),
                "train_start": str(train.index.min().date()),
                "train_end": str(train.index.max().date()),
                "test_end": str(test.index.max().date()),
                "n_days": len(test),
                "strategy": s,
                "cov_method": cm or "",
                "lw_delta": delta if cm == "ledoit_wolf" else "",
                "rf_alloc": rf_alloc,
                "n_assets": len(nz),
                "weights": ";".join(f"{tickers[i]}:{w_target[i]:.4f}" for i in nz),
                "in_sample_sharpe": "" if is_sharpe is None else is_sharpe,
                "turnover": turnover,
                "cost": cost_rate * turnover,
                "gross_return": float(np.prod(1.0 + g) - 1.0),
                "net_return": float(np.prod(1.0 + n_) - 1.0),
            })

    rf_all = np.concatenate(rf_chunks)
    oos_index = returns.index[starts[0]:]
    curves = {}
    strat_summary = {}
    for s in STRATEGIES:
        g = np.concatenate(gross[s])
        n_ = np.concatenate(net[s])
        assert g.size == rf_all.size == len(oos_index)
        curves[f"{s}_gross"] = np.cumprod(1.0 + g)
        curves[f"{s}_net"] = np.cumprod(1.0 + n_)
        lw = COV_OF[s] == "ledoit_wolf"
        strat_summary[s] = {
            "cov_method": COV_OF[s],
            "gross": perf_metrics(g, rf_all),
            "net": perf_metrics(n_, rf_all),
            "avg_turnover": float(np.mean(turnovers[s][1:])),
            "initial_turnover": turnovers[s][0],
            "avg_lw_delta": float(np.mean(deltas)) if lw else None,
        }

    summary = {
        "universe": "qubo (25 stocks, data/prices/prices_daily.csv)",
        "tickers": tickers,
        "lookback_days": LOOKBACK_DAYS,
        "hold_days": HOLD_DAYS,
        "rebalance_step": REBALANCE_STEP,
        "K": SELECT_K,
        "q": RISK_AVERSION,
        "qubo_weight_bounds": list(QUBO_W_BOUNDS),
        "mv_weight_bounds": list(MV_W_BOUNDS),
        "cost_bps_per_unit_turnover": COST_BPS,
        "risk_free": f"Treasury {RISK_FREE_TENOR} (point-in-time for allocation, realized daily for Sharpe)",
        "rf_fallback_periods": rf_fallback_periods,
        "seed": SEED,
        "periods": len(starts),
        "oos_start": str(oos_index[0].date()),
        "oos_end": str(oos_index[-1].date()),
        "oos_days": len(oos_index),
        "avg_lw_delta_all_periods": float(np.mean(deltas)),
        "metric_definitions": {
            "annual_return": "CAGR of the concatenated daily OOS returns",
            "annual_vol": "std(daily OOS returns, ddof=1) * sqrt(252)",
            "sharpe": "mean/std of daily OOS excess returns over realized 3-Mo T-bill * sqrt(252)",
            "max_drawdown": "min over OOS days of equity / running peak - 1",
            "avg_turnover": "mean sum|w_target - w_drifted| per 21-day rebalance, excluding initial buy-in",
        },
        "strategies": strat_summary,
    }
    return rows, summary, pd.DataFrame(curves, index=oos_index)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

LABELS = {
    "qubo_sample": "QUBO-select K=5 + max-Sharpe (sample cov)",
    "qubo_lw": "QUBO-select K=5 + max-Sharpe (Ledoit-Wolf)",
    "mv_sample": "Mean-variance max-Sharpe, 25 assets (sample cov)",
    "mv_lw": "Mean-variance max-Sharpe, 25 assets (Ledoit-Wolf)",
    "equal_weight": "Equal weight 1/N",
}


def plot_curves(curves: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = dict(zip(STRATEGIES, ["#1f77b4", "#6baed6", "#d62728", "#fc9272", "#2ca02c"]))
    for s in STRATEGIES:
        ax.plot(curves.index, curves[f"{s}_net"], color=colors[s], lw=1.6, label=f"{LABELS[s]} (net)")
        ax.plot(curves.index, curves[f"{s}_gross"], color=colors[s], lw=0.8, ls="--", alpha=0.7)
    ax.set_title(f"Walk-forward OOS equity, 25-stock universe "
                 f"({curves.index[0].date()} to {curves.index[-1].date()}); "
                 f"solid = net of {COST_BPS:.0f} bps, dashed = gross")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_table(summary: dict) -> str:
    hdr = ("| Strategy | Ann. return (gross / net) | Ann. vol | Sharpe (gross / net) "
           "| Max DD (gross / net) | Avg turnover / rebalance | Avg LW δ |")
    lines = [hdr, "|" + "---|" * 7]
    for s in STRATEGIES:
        m = summary["strategies"][s]
        g, n = m["gross"], m["net"]
        d = "n/a" if m["avg_lw_delta"] is None else f"{m['avg_lw_delta']:.3f}"
        lines.append(
            f"| {s} | {g['annual_return']:.2%} / {n['annual_return']:.2%} "
            f"| {g['annual_vol']:.2%} | {g['sharpe']:.3f} / {n['sharpe']:.3f} "
            f"| {g['max_drawdown']:.2%} / {n['max_drawdown']:.2%} "
            f"| {m['avg_turnover']:.3f} | {d} |"
        )
    return "\n".join(lines)


def main() -> None:
    warnings.filterwarnings("ignore", message="Values in x were outside bounds")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows, summary, curves = run()

    with open(OUTDIR / "periods.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUTDIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    plot_curves(curves, OUTDIR / "equity_curves.png")

    print(f"periods: {summary['periods']}  OOS {summary['oos_start']} .. {summary['oos_end']} "
          f"({summary['oos_days']} days); rf fallback periods: {summary['rf_fallback_periods']}")
    print(summary_table(summary))
    print(f"\nwrote {OUTDIR / 'periods.csv'}\nwrote {OUTDIR / 'summary.json'}\n"
          f"wrote {OUTDIR / 'equity_curves.png'}")


if __name__ == "__main__":
    main()
