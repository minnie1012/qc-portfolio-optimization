"""Mean-Variance Optimization (MVO) — classical Markowitz solver.

Selects the K assets and continuous weights that maximise the
Sharpe ratio via scipy SLSQP (no quantum component).

Two-phase approach matching the repo's problem_definition.py design:
  Phase 1 — enumerate all C(N,K) subsets, score each by its
             maximum-Sharpe ratio (fast for N ≤ ~25).
  Phase 2 — for the winning subset, run a polished SLSQP solve
             and record final portfolio statistics.

The QUBO objective value is also recorded so results are directly
comparable with the quantum solvers on the benchmark leaderboard.
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol import instances          # noqa: E402
from benchmark_protocol.result_schema import BenchmarkResult  # noqa: E402
from solver_common import (                       # noqa: E402
    approx_ratio,
    build_qubo_from_inst,
    exact_reference,
    qubo_eval,
)

ALGORITHM = "mvo"
ALGORITHM_VERSION = "1.0.0"

RISK_FREE_RATE = 0.05   # 5 % annualised, adjust if needed
W_MIN, W_MAX   = 0.05, 0.60
N_STARTS       = 8      # random restarts for SLSQP


# ── helpers ────────────────────────────────────────────────────────────────

def _max_sharpe(
    mu_S: np.ndarray,
    sigma_S: np.ndarray,
    rfr: float = RISK_FREE_RATE,
    w_min: float = W_MIN,
    w_max: float = W_MAX,
    n_starts: int = N_STARTS,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Return (weights, sharpe) for the K-asset sub-problem."""
    K = len(mu_S)
    if K == 0:
        return np.array([]), -np.inf
    if K == 1:
        vol = float(np.sqrt(max(sigma_S[0, 0], 1e-12)))
        return np.array([1.0]), float((mu_S[0] - rfr) / vol)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(max(float(w @ sigma_S @ w), 1e-12)))
        return -(float(mu_S @ w) - rfr) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(w_min, w_max)] * K
    rng = np.random.default_rng(seed)
    best_w, best_val = np.full(K, 1.0 / K), np.inf

    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(K))
        w0 = np.clip(w0, w_min, w_max)
        w0 /= w0.sum()
        res = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res.fun < best_val:
            best_val, best_w = res.fun, res.x

    return best_w, float(-best_val)


# ── main solver ────────────────────────────────────────────────────────────

def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    risk_free_rate: float = RISK_FREE_RATE,
    w_min: float = W_MIN,
    w_max: float = W_MAX,
) -> BenchmarkResult:
    """
    Enumerate every C(N, K) subset, find the one with the highest
    Sharpe ratio, then record its QUBO objective for benchmarking.
    """
    Q   = build_qubo_from_inst(inst)
    mu  = np.asarray(inst.mu)
    sig = np.asarray(inst.sigma)
    N, K = inst.N, inst.K

    t0 = time.perf_counter()

    best_sharpe  = -np.inf
    best_subset: tuple[int, ...] = tuple(range(K))
    best_weights = np.full(K, 1.0 / K)
    history: list[float] = []

    for combo in combinations(range(N), K):
        idx = list(combo)
        mu_S    = mu[idx]
        sigma_S = sig[np.ix_(idx, idx)]
        w, sharpe = _max_sharpe(
            mu_S, sigma_S,
            rfr=risk_free_rate, w_min=w_min, w_max=w_max,
            n_starts=N_STARTS, seed=seed,
        )
        if sharpe > best_sharpe:
            best_sharpe  = sharpe
            best_subset  = tuple(idx)
            best_weights = w
        history.append(best_sharpe)

    # Build binary solution vector
    best_x = np.zeros(N, dtype=int)
    best_x[list(best_subset)] = 1
    best_cost = qubo_eval(Q, best_x)

    elapsed = time.perf_counter() - t0

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(best_cost, c_opt, c_wst)
    else:
        ar = None

    stride = max(1, len(history) // 200)
    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=float(best_cost),
        bitstring=best_x.tolist(),
        feasible=bool(int(best_x.sum()) == K),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed),
        optimizer_iters=len(history),       # one per subset
        num_circuit_evaluations=None,
        convergence_history=[float(v) for v in history[::stride]],
        qubit_count=None,
        circuit_depth=None,
        two_qubit_gate_count=None,
        t_gate_count=None,
        total_gate_count=None,
        shots=None,
        backend=None,
        transpile_optimization_level=None,
        hyperparameters={
            "problem_variant": "qubo",
            "method": "mean_variance_sharpe",
            "risk_free_rate": risk_free_rate,
            "w_min": w_min,
            "w_max": w_max,
            "n_starts": N_STARTS,
        },
        initial_params=None,
        final_params={
            "selected_indices": list(best_subset),
            "weights": best_weights.tolist(),
            "sharpe_ratio": float(best_sharpe),
        },
    )
