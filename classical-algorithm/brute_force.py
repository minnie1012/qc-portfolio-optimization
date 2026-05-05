"""Brute-force exact solver: enumerate every K-cardinality bitstring."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol import instances  # noqa: E402
from benchmark_protocol.result_schema import BenchmarkResult  # noqa: E402

from solver_common import build_qubo_from_inst, exact_reference  # noqa: E402

ALGORITHM = "brute_force"
ALGORITHM_VERSION = "1.0.0"
MAX_N = 17


def solve(inst: instances.ProblemInstance, seed: int = 42) -> BenchmarkResult:
    Q = build_qubo_from_inst(inst)
    t0 = time.perf_counter()
    x_opt, cost_opt, _ = exact_reference(Q, inst.K)
    elapsed = time.perf_counter() - t0
    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=cost_opt,
        bitstring=x_opt.tolist(),
        feasible=bool(int(x_opt.sum()) == inst.K),
        approx_ratio=1.0,
        wall_time_seconds=float(elapsed),
        optimizer_iters=None,
        num_circuit_evaluations=None,
        convergence_history=[cost_opt],
        qubit_count=None,
        circuit_depth=None,
        two_qubit_gate_count=None,
        t_gate_count=None,
        total_gate_count=None,
        shots=None,
        backend=None,
        transpile_optimization_level=None,
        hyperparameters={"problem_variant": "qubo", "method": "exhaustive_K_subsets"},
        initial_params=None,
        final_params=None,
    )
