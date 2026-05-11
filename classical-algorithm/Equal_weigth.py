"""Equal-Weight (1/K) baseline solver.

Selects the K assets with the highest expected return (mu) and assigns
uniform weight 1/K to each. This is the simplest possible strategy and
serves as a no-intelligence baseline on the benchmark leaderboard.

Why top-mu selection? It is the greediest deterministic rule compatible
with a cardinality-K constraint and requires zero covariance information,
so it stress-tests whether risk-aware solvers (SA, Tabu, MVO, HRP, quantum)
actually add value over a naive return-chasing heuristic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

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

ALGORITHM = "equal_weight"
ALGORITHM_VERSION = "1.0.0"


def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,          # unused; kept for API consistency
) -> BenchmarkResult:
    """Pick top-K assets by expected return; weight uniformly."""
    Q   = build_qubo_from_inst(inst)
    mu  = np.asarray(inst.mu)
    sig = np.asarray(inst.sigma)
    N, K = inst.N, inst.K

    t0 = time.perf_counter()

    # Select K assets with the highest expected return
    selected = np.argsort(mu)[::-1][:K].tolist()   # descending mu
    selected = sorted(selected)

    best_x = np.zeros(N, dtype=int)
    best_x[selected] = 1
    best_cost = qubo_eval(Q, best_x)

    elapsed = time.perf_counter() - t0

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(best_cost, c_opt, c_wst)
    else:
        ar = None

    weights  = np.full(K, 1.0 / K)
    mu_S     = mu[selected]
    sigma_S  = sig[np.ix_(selected, selected)]
    port_ret = float(mu_S @ weights)
    port_vol = float(np.sqrt(weights @ sigma_S @ weights))
    sharpe   = (port_ret - 0.05) / port_vol if port_vol > 1e-12 else 0.0

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
        optimizer_iters=1,
        num_circuit_evaluations=None,
        convergence_history=[float(best_cost)],
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
            "method": "equal_weight_top_mu",
            "selection_rule": "top_K_by_expected_return",
            "weight_rule": "uniform_1_over_K",
        },
        initial_params=None,
        final_params={
            "selected_indices": selected,
            "weights": weights.tolist(),
            "portfolio_return": port_ret,
            "portfolio_volatility": port_vol,
            "sharpe_ratio": sharpe,
        },
    )
