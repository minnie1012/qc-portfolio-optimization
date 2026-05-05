"""Simulated annealing on K-cardinality bitstrings (one swap = neighbour)."""
from __future__ import annotations

import math
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

ALGORITHM = "simulated_annealing"
ALGORITHM_VERSION = "1.0.0"


def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    n_sweeps: int = 2000,
    T0: float | None = None,
    Tend: float = 1e-3,
) -> BenchmarkResult:
    Q = build_qubo_from_inst(inst)
    N, K = inst.N, inst.K
    rng = np.random.default_rng(seed)

    sel = rng.choice(N, size=K, replace=False).tolist()
    x = np.zeros(N, dtype=int); x[sel] = 1
    cost = qubo_eval(Q, x)
    best_x, best_cost = x.copy(), float(cost)
    history: list[float] = [best_cost]

    if T0 is None:
        T0 = max(abs(cost) * 0.1, 1.0)
    schedule = np.geomspace(T0, Tend, n_sweeps)

    t0 = time.perf_counter()
    iters = 0
    for T in schedule:
        on = np.flatnonzero(x == 1)
        off = np.flatnonzero(x == 0)
        i_out = int(rng.choice(on))
        i_in = int(rng.choice(off))
        x_new = x.copy(); x_new[i_out] = 0; x_new[i_in] = 1
        c_new = qubo_eval(Q, x_new)
        d = c_new - cost
        if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
            x, cost = x_new, c_new
            if cost < best_cost:
                best_cost, best_x = float(cost), x.copy()
        history.append(best_cost)
        iters += 1
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
        optimizer_iters=iters,
        num_circuit_evaluations=None,
        convergence_history=[float(c) for c in history[::stride]],
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
            "method": "simulated_annealing",
            "n_sweeps": n_sweeps,
            "T0": float(T0),
            "Tend": float(Tend),
        },
        initial_params=None,
        final_params=None,
    )
