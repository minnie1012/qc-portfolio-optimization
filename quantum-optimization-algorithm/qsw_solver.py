"""QSW solver adapted to benchmark protocol template (Variant B — MIQP).

Usage:
    python qsw_solver.py --instance medium_0030
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from sklearn.preprocessing import normalize

from benchmark_protocol import instances
from benchmark_protocol.result_schema import BenchmarkResult, validate

ALGORITHM = "qsw"
ALGORITHM_VERSION = "0.1.0"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"

# ── core QSW functions ────────────────────────────────────────────────────────

def weight_matrix(SR, Sigma, alpha=1.0, beta=1.0, lam=1.0):
    SR = np.asarray(SR, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    W = np.exp(alpha * SR[np.newaxis, :] - beta * Sigma)
    np.fill_diagonal(W, np.exp(lam * SR))
    return W

def hamiltonian(SR, Sigma, gamma1=1, gamma2=1):
    SR = np.asarray(SR, dtype=float)
    H = gamma2 * Sigma.copy()
    np.fill_diagonal(H, -gamma1 * SR)
    return H

def qsw_solver(H, G, omega, dt=0.01, tol=1e-6, max_iter=10000):
    N = H.shape[0]
    gamma = np.sqrt(1 - np.exp(-omega * G * dt))
    Gamma = gamma ** 2
    U = expm(-1j * (1 - omega) * H * dt)
    col_sum = np.sum(Gamma, axis=0)
    sqrt_inner = np.diag(np.sqrt(np.clip(1 - col_sum, 0, None)))
    K0 = sqrt_inner @ U

    rho = np.eye(N, dtype=complex) / N

    for _ in range(max_iter):
        rho_old = rho.copy()
        rho_tilde = K0 @ rho @ K0.conj().T
        P = np.real(np.diag(rho_tilde))
        rho = rho_tilde - np.diag(P) + np.diag(Gamma @ P)
        rho = rho / np.trace(rho)
        if np.sum(np.abs(rho - rho_old)) < tol:
            break

    diag = np.real(np.diag(rho))
    w = diag / diag.sum()
    return w

def miqp_objective(weights, mu, sigma, lam):
    """Evaluate the MIQP objective: r^T x - lambda * x^T Q x"""
    return float(mu @ weights - lam * weights @ sigma @ weights)

# ── main solve function ───────────────────────────────────────────────────────

def solve(instance, seed: int = 42) -> BenchmarkResult:
    t0 = time.perf_counter()

    mu = np.array(instance.mu)
    sigma = np.array(instance.sigma)
    q = instance.q

    # derive per-asset sharpe from pre-computed mu and sigma
    risk_free_daily = 0.0456 / 252
    sr = (mu - risk_free_daily) / np.sqrt(np.diag(sigma))

    # build graph
    W = weight_matrix(sr, sigma)
    P = normalize(W, norm='l1', axis=1)
    n = len(sr)
    alpha_damp = 0.85
    G = alpha_damp * P + (1 - alpha_damp) * np.ones((n, n)) / n
    H = hamiltonian(sr, sigma)

    # run QSW to get continuous weights
    w = qsw_solver(H, G, omega=0.1)

    # w already sums to 1 and is fully invested — satisfies sum(x) = 1
    # enforce cardinality by zeroing out small weights and renormalizing
    # use instance.K as the target number of assets if available
    K = getattr(instance, 'K', None)
    if K is not None:
        # keep only top K weights
        threshold_idx = np.argsort(w)[::-1][K:]
        w[threshold_idx] = 0.0
        w = w / w.sum()

    # binary selection indicators
    v = (w > 0).astype(int)

    objective_value = miqp_objective(w, mu, sigma, q)
    feasible = bool(abs(w.sum() - 1.0) < 1e-6 and v.sum() >= 1)

    wall_time_seconds = time.perf_counter() - t0

    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=instance.instance_id,
        seed=seed,
        objective_value=float(objective_value),
        # template casts weights to ints for MIQP as per README
        bitstring=list(map(float, w)),
        feasible=feasible,
        approx_ratio=None,
        wall_time_seconds=float(wall_time_seconds),
        optimizer_iters=None,
        num_circuit_evaluations=None,
        convergence_history=[],
        qubit_count=None,
        circuit_depth=None,
        two_qubit_gate_count=None,
        t_gate_count=None,
        total_gate_count=None,
        shots=None,
        backend=None,
        transpile_optimization_level=None,
        hyperparameters={
            "problem_variant": "miqp",
            "omega": 0.1,
            "alpha_damp": 0.85,
        },
        initial_params=None,
        final_params=None,
    )


def output_path(algorithm: str, instance_id: str, seed: int) -> Path:
    return RESULTS_DIR / f"{algorithm}__{instance_id}__seed{seed}.json"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inst = instances.load(args.instance)
    result = solve(inst, seed=args.seed)

    errors = validate(result)
    if errors:
        raise SystemExit("schema validation failed:\n  " + "\n  ".join(errors))

    out = result.to_json(output_path(result.algorithm, result.instance_id, result.seed))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

import pandas as pd
import yfinance as yf
import numpy as np

df = yf.download(tickers, start="2020-01-01", end="2022-12-31")
real_qubo_full = pd.DataFrame(df['Close'])

split_index = int(0.8*len(real_qubo_full))

train, test = real_qubo_full.iloc[0:split_index], real_qubo_full.iloc[split_index:]

weights = qsw_total(train,8)
weights = dict(weights)
stocks = [ticker for ticker in weights]
percents = [weights[ticker] for ticker in weights]
percents = np.array(percents)
percents = percents*np.sum(percents)

returns = 0
for i in range(len(stocks)):
  temp = percents[i]*(test[stocks[i]].iloc[-1] - test[stocks[i]].iloc[0])
  returns += temp
print(returns)
