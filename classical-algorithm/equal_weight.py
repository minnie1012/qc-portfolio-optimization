"""Equal-weight baseline: greedy top-K by per-asset Sharpe proxy mu_i/sqrt(sigma_ii)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol import instances  # noqa: E402
from benchmark_protocol.result_schema import BenchmarkResult  # noqa: E402

from solver_common import (  # noqa: E402
    approx_ratio,
    build_qubo_from_inst,
    exact_reference,
    qubo_eval,
)

ALGORITHM = "equal_weight"
ALGORITHM_VERSION = "1.0.0"
MAX_N = None


def solve(inst: instances.ProblemInstance, seed: int = 42) -> BenchmarkResult:
    Q = build_qubo_from_inst(inst)
    N, K = inst.N, inst.K

    t0 = time.perf_counter()
    vols = np.sqrt(np.maximum(np.diag(inst.sigma), 1e-12))
    score = inst.mu / vols
    top_k = np.argsort(-score)[:K]
    x = np.zeros(N, dtype=int)
    x[top_k] = 1
    cost = float(qubo_eval(Q, x))
    elapsed = time.perf_counter() - t0

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(cost, c_opt, c_wst)
    else:
        ar = None

    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=cost,
        bitstring=x.tolist(),
        feasible=bool(int(x.sum()) == K),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed),
        optimizer_iters=None,
        num_circuit_evaluations=None,
        convergence_history=[cost],
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
            "method": "equal_weight_topK_by_sharpe_proxy",
        },
        initial_params=None,
        final_params=None,
    )
