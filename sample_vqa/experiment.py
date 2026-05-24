"""
Reproduction of key results from Agliardi et al., arXiv:2508.13557:
"Portfolio construction using a sampling-based variational quantum scheme."

The exact bond data in the paper is proprietary, so this script uses the
synthetic bond-ETF instance in `sample_vqa/problem.py`, which preserves the
same algebraic structure:

  - clustered targets
  - cash budget constraint
  - per-cluster guardrails
  - binary selection variables

The script tests three claims inspired by the paper:

  1. CVaR alpha=0.1 outperforms alpha=0.2 and alpha=1.0
  2. Quantum + local search beats local search alone
  3. BFCD-style ansatz can match or beat TwoLocal at the same reps

Outputs:
  - PNG plots in `sample_vqa/results/`
  - JSON summary in `sample_vqa/results/summary.json`
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
from qiskit_aer.primitives import SamplerV2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problem import build_problem, brute_force_optimum  # noqa: E402
from qvqa import (  # noqa: E402
    bfcd_ansatz,
    cvar_aggregate,
    local_search,
    nft_optimize,
    sample_circuit,
    twolocal_ansatz,
)


# ---------- Configuration ----------

N_BONDS = 16
REPS = 2
SHOTS = 1024
INIT_VALUE = np.pi / 3
SEED_PROB = 7
NFT_EPOCHS = 4
OUTDIR = ROOT / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)


def run_cvar_vqa(
    prob,
    ansatz_circ,
    alpha,
    shots,
    n_epochs,
    sampler,
    s_vec,
    seed=0,
    label="",
):
    """Run sampling-based CVaR-VQA on a given ansatz."""
    rng = np.random.default_rng(seed)
    n_params = ansatz_circ.num_parameters

    def cost_on_x(x):
        return prob.penalized(x, s_vec)

    samples_log = []
    cvar_history = []
    best_cost_history = []

    def cvar_objective(theta):
        bitstrings, weights = sample_circuit(sampler, ansatz_circ, theta, shots)
        costs = np.array([cost_on_x(x) for x in bitstrings])
        return cvar_aggregate(costs, weights, alpha)

    def batch_cvar(theta_list):
        """Evaluate CVaR at multiple parameter points in one sampler.run call."""
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
        print(
            f"  [{label}] epoch {epoch:>2}: CVaR={cvar_history[-1]:.4f}, "
            f"best raw={best_cost_history[-1]:.4f}"
        )

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
    }


def post_process_with_local_search(samples_log, prob, s_vec, last_k_epochs=5):
    """Run local search on the top-weight samples from the last few epochs."""

    def cost_on_x(x):
        return prob.penalized(x, s_vec)

    rng = np.random.default_rng(42)
    polished = []
    recent = samples_log[-last_k_epochs:] if len(samples_log) >= last_k_epochs else samples_log
    for (_, bitstrings, weights, _) in recent:
        order = np.argsort(-weights)[:50]
        for idx in order:
            x_polished, f_polished = local_search(bitstrings[idx], cost_on_x, rng=rng)
            polished.append((f_polished, x_polished))
    return polished


def random_starts_local_search(prob, s_vec, n_starts=200, seed=42):
    """Pure classical baseline: local search from many random starts."""

    rng = np.random.default_rng(seed)

    def cost_on_x(x):
        return prob.penalized(x, s_vec)

    best_cost = np.inf
    best_x = None
    n_distinct = set()
    median_track = []
    for _ in range(n_starts):
        x0 = rng.integers(0, 2, size=prob.n).astype(np.int8)
        x, f = local_search(x0, cost_on_x, rng=rng)
        median_track.append(f)
        n_distinct.add(round(f, 4))
        if f < best_cost:
            best_cost = f
            best_x = x
    return best_x, best_cost, {
        "n_distinct_local_optima": len(n_distinct),
        "median_local_optimum": float(np.median(median_track)),
    }


def main():
    print("=" * 70)
    print("Reproducing Agliardi et al. arXiv:2508.13557 on a synthetic ETF")
    print("=" * 70)

    prob = build_problem(n_bonds=N_BONDS, seed=SEED_PROB)
    print(
        f"\nProblem: {prob.n} bonds, Q nnz={np.count_nonzero(prob.Q)}, "
        f"{prob.A.shape[0]} constraints"
    )

    typical_Q = np.abs(prob.Q).max()
    s_vec = np.full(prob.A.shape[0], 100.0 * typical_Q)

    print("Brute-forcing optimum...")
    x_opt_true, obj_opt_true, pen_opt_true = brute_force_optimum(prob, s_vec)
    print(f"True optimum: obj={obj_opt_true:.6f}, x sum={x_opt_true.sum()}")

    def gap(val):
        return 100.0 * abs(val - pen_opt_true) / abs(pen_opt_true)

    sampler = SamplerV2(default_shots=SHOTS, seed=42)

    # Experiment A: alpha sweep on TwoLocal bilinear
    print("\n" + "=" * 70)
    print("Experiment A: CVaR alpha sweep (paper Fig. 3a)")
    print("=" * 70)
    twoloc = twolocal_ansatz(N_BONDS, reps=REPS, ent_map="bilinear")
    print(f"TwoLocal bilinear: {twoloc.num_parameters} params, depth {twoloc.decompose().depth()}")

    alpha_results = {}
    for alpha in [0.1, 0.2, 1.0]:
        print(f"\n--- alpha = {alpha} ---")
        res = run_cvar_vqa(
            prob,
            twoloc,
            alpha=alpha,
            shots=SHOTS,
            n_epochs=NFT_EPOCHS,
            sampler=sampler,
            s_vec=s_vec,
            seed=int(alpha * 100),
            label=f"TL-bi alpha={alpha}",
        )
        polished = post_process_with_local_search(res["samples_log"], prob, s_vec)
        best_polished = min(polished, key=lambda t: t[0])
        res["best_polished_cost"] = float(best_polished[0])
        res["best_polished_x"] = best_polished[1].tolist()
        res["raw_gap_pct"] = gap(res["best_cost"])
        res["polished_gap_pct"] = gap(res["best_polished_cost"])
        print(f"  raw best gap:      {res['raw_gap_pct']:.3f}%")
        print(f"  polished best gap: {res['polished_gap_pct']:.3f}%")
        alpha_results[alpha] = res

    # Experiment B: ansatz comparison at alpha=0.1
    print("\n" + "=" * 70)
    print("Experiment B: ansatz comparison at alpha=0.1 (paper Fig. 3a vs 3b)")
    print("=" * 70)

    ansatz_results = {"TwoLocal-bilinear": alpha_results[0.1]}

    for name, ans in [
        ("TwoLocal-colored", twolocal_ansatz(N_BONDS, REPS, "colored")),
        ("BFCD-bilinear", bfcd_ansatz(N_BONDS, REPS, "bilinear")),
    ]:
        print(f"\n--- {name}: {ans.num_parameters} params, depth {ans.decompose().depth()} ---")
        res = run_cvar_vqa(
            prob,
            ans,
            alpha=0.1,
            shots=SHOTS,
            n_epochs=NFT_EPOCHS,
            sampler=sampler,
            s_vec=s_vec,
            seed=hash(name) % 1000,
            label=name,
        )
        polished = post_process_with_local_search(res["samples_log"], prob, s_vec)
        best_polished = min(polished, key=lambda t: t[0])
        res["best_polished_cost"] = float(best_polished[0])
        res["best_polished_x"] = best_polished[1].tolist()
        res["raw_gap_pct"] = gap(res["best_cost"])
        res["polished_gap_pct"] = gap(res["best_polished_cost"])
        print(f"  raw best gap:      {res['raw_gap_pct']:.3f}%")
        print(f"  polished best gap: {res['polished_gap_pct']:.3f}%")
        ansatz_results[name] = res

    # Experiment C: classical local-search baseline
    print("\n" + "=" * 70)
    print("Experiment C: classical local-search baseline (Sec IV-B claim)")
    print("=" * 70)
    LS_BUDGET = 250
    t0 = time.time()
    x_ls, cost_ls, ls_stats = random_starts_local_search(prob, s_vec, n_starts=LS_BUDGET)
    elapsed_ls = time.time() - t0
    classical_gap = gap(cost_ls)
    print(
        f"Pure local search ({LS_BUDGET} random starts): gap={classical_gap:.4f}%, "
        f"time={elapsed_ls:.2f}s"
    )
    print(f"  Distinct local optima:  {ls_stats['n_distinct_local_optima']}")
    print(
        f"  Median local optimum:   {ls_stats['median_local_optimum']:.4f} "
        f"(gap {gap(ls_stats['median_local_optimum']):.2f}%)"
    )

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha, res in alpha_results.items():
        axes[0].plot(res["cvar_history"], marker="o", label=f"alpha={alpha}")
        axes[1].plot(res["best_cost_history"], marker="o", label=f"alpha={alpha}")
    axes[0].axhline(pen_opt_true, ls="--", color="red", alpha=0.5, label="optimum")
    axes[1].axhline(pen_opt_true, ls="--", color="red", alpha=0.5, label="optimum")
    axes[0].set_xlabel("NFT epoch")
    axes[0].set_ylabel("CVaR objective")
    axes[1].set_xlabel("NFT epoch")
    axes[1].set_ylabel("Best raw cost in epoch")
    axes[0].set_title("CVaR(alpha) trajectory (TwoLocal bilinear)")
    axes[1].set_title("Best sample cost per epoch")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_alpha_sweep.png", dpi=120)
    plt.close()

    names = list(ansatz_results.keys())
    raw_gaps = [ansatz_results[n]["raw_gap_pct"] for n in names]
    polished_gaps = [ansatz_results[n]["polished_gap_pct"] for n in names]
    classical_gaps = [classical_gap] * len(names)

    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, raw_gaps, w, label="Quantum (raw)", color="#4C72B0")
    ax.bar(x, polished_gaps, w, label="Quantum + local search", color="#55A868")
    ax.bar(x + w, classical_gaps, w, label="Local search alone", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Optimization gap (%)")
    ax.set_title(f"Ansatz comparison @ alpha=0.1, n={N_BONDS}, reps={REPS}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_ansatz_comparison.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(pen_opt_true * 0.95, pen_opt_true * 4, 40)
    for name in names:
        res = ansatz_results[name]
        last_samples = res["samples_log"][-1]
        _, _, weights, costs = last_samples
        ax.hist(costs, bins=bins, weights=weights, alpha=0.4, label=name, density=True)
    twoloc_init = twolocal_ansatz(N_BONDS, reps=REPS, ent_map="bilinear")
    bs0, w0 = sample_circuit(
        sampler,
        twoloc_init,
        np.full(twoloc_init.num_parameters, INIT_VALUE),
        SHOTS,
    )
    costs0 = np.array([prob.penalized(x, s_vec) for x in bs0])
    ax.hist(costs0, bins=bins, weights=w0, alpha=0.3, label="Initial (pi/3)", color="gray", density=True)
    ax.axvline(pen_opt_true, ls="--", color="red", label="optimum")
    ax.set_xlabel("Cost f(x)")
    ax.set_ylabel("density")
    ax.set_title("Final sample distributions (paper Fig. 5)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_distributions.png", dpi=120)
    plt.close()

    summary = {
        "n_bonds": N_BONDS,
        "reps": REPS,
        "shots": SHOTS,
        "true_optimum": float(pen_opt_true),
        "true_x": x_opt_true.tolist(),
        "alpha_sweep": {
            str(a): {"raw_gap": r["raw_gap_pct"], "polished_gap": r["polished_gap_pct"]}
            for a, r in alpha_results.items()
        },
        "ansatz_comparison": {
            n: {
                "raw_gap": r["raw_gap_pct"],
                "polished_gap": r["polished_gap_pct"],
                "elapsed_s": r["elapsed"],
            }
            for n, r in ansatz_results.items()
        },
        "classical_baseline_gap": classical_gap,
        "classical_elapsed_s": elapsed_ls,
        "paper_target_gap_pct": 0.49,
    }

    with open(OUTDIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"True optimum value: {pen_opt_true:.4f}\n")

    print(f"{'Method':<30s} {'Raw gap':>10s} {'+LS gap':>10s}")
    print("-" * 52)
    for a in [0.1, 0.2, 1.0]:
        r = alpha_results[a]
        print(
            f"  TwoLocal-bi alpha={a:<5.2f}  {' ':<5s}    {r['raw_gap_pct']:>7.3f}%  "
            f"{r['polished_gap_pct']:>7.3f}%"
        )
    for name in ["TwoLocal-colored", "BFCD-bilinear"]:
        r = ansatz_results[name]
        print(f"  {name:<28s} {r['raw_gap_pct']:>7.3f}%  {r['polished_gap_pct']:>7.3f}%")
    print(f"\n  Classical local search alone:        {classical_gap:>7.3f}%")
    print(f"\nPaper's reported best (109q hardware): 0.49% gap")
    print(f"\nFigures + JSON in {OUTDIR}/")


if __name__ == "__main__":
    main()
