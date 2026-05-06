"""Template for a new solver. Copy, rename, fill in the TODOs.

Usage:
    python -m benchmark_protocol.solver_template --algo my_solver --instance tiny_0000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmark_protocol import gate_metrics, instances
from benchmark_protocol.result_schema import BenchmarkResult, validate

ALGORITHM = "TODO_rename_me"
ALGORITHM_VERSION = "0.1.0"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"


def solve(instance, seed: int = 42) -> BenchmarkResult:
    """Run the solver on one instance and return a populated BenchmarkResult."""

    # ----- TODO: build your circuit / call your solver here --------------------
    # For a quantum solver:
    #   from qiskit import transpile
    #   circuit = build_my_ansatz(instance)
    #   transpiled = transpile(circuit, basis_gates=[...], optimization_level=1)
    #   metrics = gate_metrics.extract_with_t_count(transpiled)
    #
    # For a classical solver, leave the quantum fields as None.
    # ---------------------------------------------------------------------------

    t0 = time.perf_counter()

    # placeholder run (replace with your actual solve call)
    bitstring = [0] * instance.N
    objective_value = 0.0
    convergence_history: list[float] = []
    feasible = sum(bitstring) == instance.K
    initial_params: list[float] | None = None
    final_params: list[float] | None = None
    optimizer_iters: int | None = None
    num_circuit_evaluations: int | None = None

    # quantum metrics — fill these from gate_metrics.extract_with_t_count(...)
    qubit_count: int | None = None
    circuit_depth: int | None = None
    two_qubit_gate_count: int | None = None
    t_gate_count: int | None = None
    total_gate_count: int | None = None
    shots: int | str | None = None
    backend: str | None = None
    transpile_optimization_level: int | None = None

    wall_time_seconds = time.perf_counter() - t0

    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=instance.instance_id,
        seed=seed,
        objective_value=float(objective_value),
        bitstring=list(map(int, bitstring)),
        feasible=bool(feasible),
        approx_ratio=None,  # fill in if you have the exact reference
        wall_time_seconds=float(wall_time_seconds),
        optimizer_iters=optimizer_iters,
        num_circuit_evaluations=num_circuit_evaluations,
        convergence_history=convergence_history,
        qubit_count=qubit_count,
        circuit_depth=circuit_depth,
        two_qubit_gate_count=two_qubit_gate_count,
        t_gate_count=t_gate_count,
        total_gate_count=total_gate_count,
        shots=shots,
        backend=backend,
        transpile_optimization_level=transpile_optimization_level,
        hyperparameters={},  # e.g. {"alpha": 0.25, "p": 1, "ansatz_reps": 2}
        initial_params=initial_params,
        final_params=final_params,
    )


def output_path(algorithm: str, instance_id: str, seed: int) -> Path:
    return RESULTS_DIR / f"{algorithm}__{instance_id}__seed{seed}.json"


def main() -> None:
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
