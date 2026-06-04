"""
ETF version of the sample CVaR-VQA experiment.

This keeps the same quantum workflow as the synthetic bond/ETF demo, but
replaces the artificial data with a small real ETF universe downloaded from
yfinance.

Pipeline:
  - download daily adjusted closes for a liquid ETF basket
  - build an in-sample QUBO portfolio-selection problem
  - solve it with CVaR-VQA using TwoLocal and BFCD-style ansatze
  - post-process the best samples with local search
  - evaluate the selected portfolio out of sample with Sharpe ratio
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from qiskit_aer.primitives import SamplerV2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_protocol.treasury import annualized_sharpe, daily_rate_from_annualized, load_yield_curve  # noqa: E402
from problem_definition import ProblemInstance, allocate, build_qubo, brute_force_select  # noqa: E402
from qvqa import bfcd_ansatz, cvar_aggregate, local_search, nft_optimize, sample_circuit, twolocal_ansatz  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ETF_UNIVERSE = [
    "SPY",  # US large-cap equities
    "QQQ",  # US growth / NASDAQ
    "IWM",  # US small caps
    "EFA",  # developed ex-US equities
    "EEM",  # emerging markets
    "AGG",  # US aggregate bonds
    "TLT",  # long-duration Treasuries
    "LQD",  # investment-grade corporates
    "HYG",  # high-yield corporates
    "GLD",  # gold
    "VNQ",  # REITs
    "XLE",  # energy
]

IN_SAMPLE = ("2022-01-01", "2024-12-31")
OUT_SAMPLE = ("2025-01-01", "2025-12-31")
DOWNLOAD_RANGE = ("2021-12-31", "2026-01-02")

N_ETFS = len(ETF_UNIVERSE)
K_SELECT = 4
REPS = 2
SHOTS = 1024
INIT_VALUE = np.pi / 3
NFT_EPOCHS = 4
SEED = 7
RISK_AVERSION = 1.0
RISK_FREE_TENOR = "3 Mo"

OUTDIR = ROOT / "results_etf"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PRICE_CACHE = CACHE_DIR / "etf_prices.csv"
ansatz_circ = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_etf_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closes from yfinance and normalize to a flat frame."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no ETF data")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        field = "Close" if "Close" in level0 else "Adj Close"
        px = raw[field].copy()
    else:
        if "Close" not in raw.columns and "Adj Close" not in raw.columns:
            raise RuntimeError("unexpected yfinance column layout")
        field = "Close" if "Close" in raw.columns else "Adj Close"
        px = raw[[field]].copy()

    px.index = pd.to_datetime(px.index).normalize()
    px = px.sort_index()
    px = px[tickers]
    px = px.ffill().dropna()
    px.index.name = "Date"
    px.to_csv(PRICE_CACHE)
    return px


def load_etf_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Load ETF prices from cache if possible, otherwise download them."""
    if PRICE_CACHE.exists():
        px = pd.read_csv(PRICE_CACHE, parse_dates=["Date"], index_col="Date")
        missing = [t for t in tickers if t not in px.columns]
        if not missing:
            return px.loc[start:end, tickers].dropna()
    px = download_etf_prices(tickers, start, end)
    return px.loc[start:end, tickers].dropna()


def split_in_out_sample(px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ins = px.loc[IN_SAMPLE[0] : IN_SAMPLE[1]]
    oos = px.loc[OUT_SAMPLE[0] : OUT_SAMPLE[1]]
    if ins.empty or oos.empty:
        raise RuntimeError("ETF price history does not cover the requested windows")
    return ins, oos


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="any")


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------


def build_etf_instance(insample_returns: pd.DataFrame) -> tuple[ProblemInstance, np.ndarray]:
    mu = insample_returns.mean().to_numpy() * 252.0
    sigma = insample_returns.cov().to_numpy() * 252.0
    inst = ProblemInstance(
        mu=mu,
        sigma=sigma,
        K=K_SELECT,
        q=RISK_AVERSION,
        tickers=ETF_UNIVERSE,
    )
    Q = build_qubo(inst)
    return inst, Q


def sample_cost_fn(Q: np.ndarray):
    return lambda x: float(x @ Q @ x)


def exact_reference(Q: np.ndarray, K: int):
    return brute_force_select(Q, K=K)


def run_cvar_vqa(
    inst: ProblemInstance,
    Q: np.ndarray,
    alpha: float,
    shots: int,
    n_epochs: int,
    sampler,
    seed: int = 0,
    label: str = "",
):
    """Run sampling-based CVaR-VQA on the ETF QUBO."""
    rng = np.random.default_rng(seed)
    if ansatz_circ is None:
        raise RuntimeError("ansatz_circ is not initialized")
    n_params = ansatz_circ.num_parameters
    cost_on_x = sample_cost_fn(Q)

    samples_log = []
    cvar_history = []
    best_cost_history = []

    def cvar_objective(theta):
        bitstrings, weights = sample_circuit(sampler, ansatz_circ, theta, shots)
        costs = np.array([cost_on_x(x) for x in bitstrings])
        return cvar_aggregate(costs, weights, alpha)

    def batch_cvar(theta_list):
        qc_template = ansatz_circ.copy()
        qc_template.measure_all()
        pubs = [(qc_template, [list(theta)], shots) for theta in theta_list]
        result = sampler.run(pubs).result()
        out = []
        for r in result:
            counts = r.data.meas.get_counts()
            costs = []
            wts = []
            for bs, ct in counts.items():
                x = np.array([int(c) for c in bs[::-1]], dtype=np.int8)
                costs.append(cost_on_x(x))
                wts.append(ct)
            out.append(cvar_aggregate(np.array(costs), np.array(wts), alpha))
        return out

    def callback(epoch, theta, f_val):
        bitstrings, weights = sample_circuit(sampler, ansatz_circ, theta, shots)
        costs = np.array([cost_on_x(x) for x in bitstrings])
        best_idx = int(np.argmin(costs))
        best_cost_history.append(float(costs[best_idx]))
        cvar_history.append(float(cvar_aggregate(costs, weights, alpha)))
        samples_log.append((epoch, bitstrings.copy(), weights.copy(), costs.copy()))
        print(f"  [{label}] epoch {epoch:>2}: CVaR={cvar_history[-1]:.4f}, best raw={best_cost_history[-1]:.4f}")

    x0 = np.full(n_params, INIT_VALUE)

    t0 = time.time()
    x_opt, _ = nft_optimize(
        cvar_objective,
        x0,
        n_epochs=n_epochs,
        callback=callback,
        rng=rng,
        batch_cost_fn=batch_cvar,
    )
    elapsed = time.time() - t0

    bitstrings, weights = sample_circuit(sampler, ansatz_circ, x_opt, shots)
    costs = np.array([cost_on_x(x) for x in bitstrings])
    best_idx = int(np.argmin(costs))
    best_x = bitstrings[best_idx]
    best_cost = float(costs[best_idx])

    selected = [i for i, b in enumerate(best_x) if b == 1]
    if selected:
        alloc = allocate(inst, selected, risk_free_rate=0.05)
    else:
        alloc = {"weights": [], "portfolio_return": None, "portfolio_volatility": None, "sharpe_ratio": None}

    return {
        "label": label,
        "alpha": alpha,
        "x_opt_params": x_opt.tolist(),
        "best_x": best_x.tolist(),
        "best_cost": best_cost,
        "cvar_history": cvar_history,
        "best_cost_history": best_cost_history,
        "samples_log": samples_log,
        "elapsed": elapsed,
        "allocation": alloc,
    }


def post_process_with_local_search(samples_log, cost_fn, last_k_epochs: int = 5):
    rng = np.random.default_rng(42)
    polished = []
    recent = samples_log[-last_k_epochs:] if len(samples_log) >= last_k_epochs else samples_log
    for (_, bitstrings, weights, _) in recent:
        order = np.argsort(-weights)[:50]
        for idx in order:
            x_polished, f_polished = local_search(bitstrings[idx], cost_fn, rng=rng)
            polished.append((f_polished, x_polished))
    return polished


def oos_metrics(px_oos: pd.DataFrame, selected: list[int], weights: np.ndarray, rf_oos: np.ndarray):
    if not selected or weights.size == 0:
        return {
            "annual_return": None,
            "annual_vol": None,
            "sharpe": None,
            "total_return": None,
        }

    port_daily = px_oos.iloc[:, selected].pct_change().dropna().to_numpy() @ weights
    rf = rf_oos[-len(port_daily) :]
    ann_ret = float(np.mean(port_daily) * 252.0)
    ann_vol = float(np.std(port_daily, ddof=1) * np.sqrt(252.0))
    sharpe = annualized_sharpe(port_daily, rf)
    total = float(np.prod(1.0 + port_daily) - 1.0)
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "total_return": total,
    }


def equal_weight_metrics(px_oos: pd.DataFrame, rf_oos: np.ndarray):
    rets = px_oos.pct_change().dropna().to_numpy()
    weights = np.ones(rets.shape[1]) / rets.shape[1]
    port_daily = rets @ weights
    rf = rf_oos[-len(port_daily) :]
    ann_ret = float(np.mean(port_daily) * 252.0)
    ann_vol = float(np.std(port_daily, ddof=1) * np.sqrt(252.0))
    sharpe = annualized_sharpe(port_daily, rf)
    total = float(np.prod(1.0 + port_daily) - 1.0)
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "total_return": total,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main():
    global ansatz_circ
    print("=" * 70)
    print("ETF CVaR-VQA experiment")
    print("=" * 70)
    print(f"Universe: {', '.join(ETF_UNIVERSE)}")

    px = load_etf_prices(ETF_UNIVERSE, DOWNLOAD_RANGE[0], DOWNLOAD_RANGE[1])
    ins_px, oos_px = split_in_out_sample(px)
    ins_rets = daily_returns(ins_px)
    oos_rets = daily_returns(oos_px)

    rf_yields = load_yield_curve(OUT_SAMPLE[0], OUT_SAMPLE[1])
    if RISK_FREE_TENOR not in rf_yields.columns:
        raise RuntimeError(f"risk-free tenor {RISK_FREE_TENOR} missing from Treasury feed")
    rf_series = rf_yields[RISK_FREE_TENOR].astype(float) / 100.0
    rf_series.index = pd.to_datetime(rf_series.index).normalize()
    rf_daily = rf_series.reindex(oos_rets.index).ffill().bfill().to_numpy()
    rf_daily = np.array([daily_rate_from_annualized(v * 100.0) for v in rf_daily], dtype=float)

    inst, Q = build_etf_instance(ins_rets)
    x_opt_true, cost_opt_true = exact_reference(Q, K_SELECT)
    selected_true = [i for i, b in enumerate(x_opt_true) if b == 1]
    print(f"Selected-asset exact optimum: cost={cost_opt_true:.6f}, selected={selected_true}")

    def gap(val):
        return 100.0 * abs(val - cost_opt_true) / abs(cost_opt_true)

    sampler = SamplerV2(default_shots=SHOTS, seed=42)

    # Experiment A: alpha sweep on TwoLocal bilinear
    print("\n" + "=" * 70)
    print("Experiment A: CVaR alpha sweep")
    print("=" * 70)
    twoloc = twolocal_ansatz(N_ETFS, reps=REPS, ent_map="bilinear")
    print(f"TwoLocal bilinear: {twoloc.num_parameters} params, depth {twoloc.decompose().depth()}")

    alpha_results = {}
    for alpha in [0.1, 0.2, 1.0]:
        print(f"\n--- alpha = {alpha} ---")
        ansatz_circ = twoloc
        res = run_cvar_vqa(
            inst,
            Q,
            alpha=alpha,
            shots=SHOTS,
            n_epochs=NFT_EPOCHS,
            sampler=sampler,
            seed=int(alpha * 100),
            label=f"TL-bi alpha={alpha}",
        )
        cost_fn = sample_cost_fn(Q)
        polished = post_process_with_local_search(res["samples_log"], cost_fn)
        best_polished = min(polished, key=lambda t: t[0])
        res["best_polished_cost"] = float(best_polished[0])
        res["best_polished_x"] = best_polished[1].tolist()
        res["raw_gap_pct"] = gap(res["best_cost"])
        res["polished_gap_pct"] = gap(res["best_polished_cost"])
        selected = [i for i, b in enumerate(res["best_polished_x"]) if b == 1]
        weights = np.asarray(allocate(inst, selected, risk_free_rate=0.05)["weights"], dtype=float) if selected else np.array([])
        res["oos_strategy"] = oos_metrics(oos_px, selected, weights, rf_daily)
        res["oos_equal_weight_all"] = equal_weight_metrics(oos_px, rf_daily)
        print(f"  raw best gap:      {res['raw_gap_pct']:.3f}%")
        print(f"  polished best gap: {res['polished_gap_pct']:.3f}%")
        print(f"  OOS strategy Sharpe: {res['oos_strategy']['sharpe']:.4f}")
        alpha_results[alpha] = res

    # Experiment B: ansatz comparison at alpha=0.1
    print("\n" + "=" * 70)
    print("Experiment B: ansatz comparison at alpha=0.1")
    print("=" * 70)

    ansatz_results = {"TwoLocal-bilinear": alpha_results[0.1]}
    for name, ans in [
        ("TwoLocal-colored", twolocal_ansatz(N_ETFS, REPS, "colored")),
        ("BFCD-bilinear", bfcd_ansatz(N_ETFS, REPS, "bilinear")),
    ]:
        print(f"\n--- {name}: {ans.num_parameters} params, depth {ans.decompose().depth()} ---")
        ansatz_circ = ans
        res = run_cvar_vqa(
            inst,
            Q,
            alpha=0.1,
            shots=SHOTS,
            n_epochs=NFT_EPOCHS,
            sampler=sampler,
            seed=hash(name) % 1000,
            label=name,
        )
        cost_fn = sample_cost_fn(Q)
        polished = post_process_with_local_search(res["samples_log"], cost_fn)
        best_polished = min(polished, key=lambda t: t[0])
        res["best_polished_cost"] = float(best_polished[0])
        res["best_polished_x"] = best_polished[1].tolist()
        res["raw_gap_pct"] = gap(res["best_cost"])
        res["polished_gap_pct"] = gap(res["best_polished_cost"])
        selected = [i for i, b in enumerate(res["best_polished_x"]) if b == 1]
        weights = np.asarray(allocate(inst, selected, risk_free_rate=0.05)["weights"], dtype=float) if selected else np.array([])
        res["oos_strategy"] = oos_metrics(oos_px, selected, weights, rf_daily)
        res["oos_equal_weight_all"] = equal_weight_metrics(oos_px, rf_daily)
        print(f"  raw best gap:      {res['raw_gap_pct']:.3f}%")
        print(f"  polished best gap: {res['polished_gap_pct']:.3f}%")
        print(f"  OOS strategy Sharpe: {res['oos_strategy']['sharpe']:.4f}")
        ansatz_results[name] = res

    summary = {
        "universe": ETF_UNIVERSE,
        "in_sample": IN_SAMPLE,
        "out_sample": OUT_SAMPLE,
        "reps": REPS,
        "shots": SHOTS,
        "K": K_SELECT,
        "risk_aversion": RISK_AVERSION,
        "exact_optimum": float(cost_opt_true),
        "exact_selected": selected_true,
        "alpha_sweep": {
            str(a): {
                "raw_gap": r["raw_gap_pct"],
                "polished_gap": r["polished_gap_pct"],
                "oos_strategy_sharpe": r["oos_strategy"]["sharpe"],
                "oos_strategy_total_return": r["oos_strategy"]["total_return"],
                "oos_equal_weight_all_sharpe": r["oos_equal_weight_all"]["sharpe"],
            }
            for a, r in alpha_results.items()
        },
        "ansatz_comparison": {
            n: {
                "raw_gap": r["raw_gap_pct"],
                "polished_gap": r["polished_gap_pct"],
                "elapsed_s": r["elapsed"],
                "oos_strategy_sharpe": r["oos_strategy"]["sharpe"],
                "oos_strategy_total_return": r["oos_strategy"]["total_return"],
            }
            for n, r in ansatz_results.items()
        },
    }

    with open(OUTDIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha, res in alpha_results.items():
        axes[0].plot(res["cvar_history"], marker="o", label=f"alpha={alpha}")
        axes[1].plot(res["best_cost_history"], marker="o", label=f"alpha={alpha}")
    axes[0].axhline(cost_opt_true, ls="--", color="red", alpha=0.5, label="optimum")
    axes[1].axhline(cost_opt_true, ls="--", color="red", alpha=0.5, label="optimum")
    axes[0].set_xlabel("NFT epoch")
    axes[0].set_ylabel("CVaR objective")
    axes[1].set_xlabel("NFT epoch")
    axes[1].set_ylabel("Best raw cost in epoch")
    axes[0].set_title("ETF CVaR(alpha) trajectory")
    axes[1].set_title("ETF best sample cost per epoch")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_alpha_sweep.png", dpi=120)
    plt.close()

    names = list(ansatz_results.keys())
    raw_gaps = [ansatz_results[n]["raw_gap_pct"] for n in names]
    polished_gaps = [ansatz_results[n]["polished_gap_pct"] for n in names]
    x = np.arange(len(names))
    w = 0.30
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, raw_gaps, w, label="Quantum (raw)", color="#4C72B0")
    ax.bar(x + w / 2, polished_gaps, w, label="Quantum + local search", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Optimization gap (%)")
    ax.set_title(f"ETF ansatz comparison @ alpha=0.1, n={N_ETFS}, K={K_SELECT}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_ansatz_comparison.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    lo = min(cost_opt_true * 0.95, cost_opt_true * 1.75)
    hi = max(cost_opt_true * 0.95, cost_opt_true * 1.75)
    bins = np.linspace(lo, hi, 40)
    for name in names:
        res = ansatz_results[name]
        last_samples = res["samples_log"][-1]
        _, _, weights, costs = last_samples
        ax.hist(costs, bins=bins, weights=weights, alpha=0.4, label=name, density=True)
    twoloc_init = twolocal_ansatz(N_ETFS, reps=REPS, ent_map="bilinear")
    ansatz_circ = twoloc_init
    bs0, w0 = sample_circuit(
        sampler,
        twoloc_init,
        np.full(twoloc_init.num_parameters, INIT_VALUE),
        SHOTS,
    )
    costs0 = np.array([float(x @ Q @ x) for x in bs0])
    ax.hist(costs0, bins=bins, weights=w0, alpha=0.3, label="Initial (pi/3)", color="gray", density=True)
    ax.axvline(cost_opt_true, ls="--", color="red", label="optimum")
    ax.set_xlabel("Cost f(x)")
    ax.set_ylabel("density")
    ax.set_title("ETF final sample distributions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_distributions.png", dpi=120)
    plt.close()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Exact optimum value: {cost_opt_true:.4f}\n")
    print(f"{'Method':<30s} {'Raw gap':>10s} {'+LS gap':>10s} {'OOS Sharpe':>12s}")
    print("-" * 70)
    for a in [0.1, 0.2, 1.0]:
        r = alpha_results[a]
        print(
            f"  TwoLocal-bi alpha={a:<5.2f}  {r['raw_gap_pct']:>7.3f}%  {r['polished_gap_pct']:>7.3f}%  "
            f"{r['oos_strategy']['sharpe']:>10.4f}"
        )
    for name in ["TwoLocal-colored", "BFCD-bilinear"]:
        r = ansatz_results[name]
        print(f"  {name:<28s} {r['raw_gap_pct']:>7.3f}%  {r['polished_gap_pct']:>7.3f}%  {r['oos_strategy']['sharpe']:>10.4f}")
    print("\nFigures + JSON in", OUTDIR)


if __name__ == "__main__":
    main()
