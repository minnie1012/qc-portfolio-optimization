"""Linear-Ramp QAOA (Montanez-Barrera & Michielsen, npj QI 2025).

Fixed schedule, no optimization loop:
    gamma_k = Delta_gamma * (k+1)/p
    beta_k  = Delta_beta  * (1 - (k+1)/p)
Single forward pass; sweep over p in {5, 10, 20, 40} and keep the best p.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol import gate_metrics, instances  # noqa: E402
from benchmark_protocol.result_schema import BenchmarkResult  # noqa: E402

from solver_common import (  # noqa: E402
    approx_ratio,
    build_qubo_from_inst,
    exact_reference,
    qubo_eval,
    qubo_to_pauli_op,
)

ALGORITHM = "lr_qaoa"
ALGORITHM_VERSION = "1.0.0"
MAX_N = 12  # MPS-friendly upper bound; fixed schedule has no optimizer


def lr_schedule(p: int, delta_gamma: float, delta_beta: float) -> list[float]:
    params: list[float] = []
    for k in range(p):
        beta = delta_beta * (1.0 - (k + 1) / p)
        gamma = delta_gamma * (k + 1) / p
        params.extend([beta, gamma])
    return params


def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    p_sweep: tuple[int, ...] = (5, 10, 20),
    delta_gamma: float = 0.5,
    delta_beta: float = 0.5,
    shots: int = 4096,
) -> BenchmarkResult:
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_aer.primitives import SamplerV2

    Q = build_qubo_from_inst(inst)
    N, K = inst.N, inst.K
    H = qubo_to_pauli_op(Q)

    sampler = SamplerV2(default_shots=shots, seed=seed)

    overall_best_x: np.ndarray | None = None
    overall_best_cost = math.inf
    overall_best_any: np.ndarray | None = None
    last_metrics: dict | None = None
    last_transpiled = None
    last_params: list[float] = []
    energy_per_p: list[float] = []

    t0 = time.perf_counter()
    for p in p_sweep:
        ansatz = QAOAAnsatz(cost_operator=H, reps=p)
        ansatz.measure_all()
        transpiled = transpile(ansatz, basis_gates=["sx", "rz", "cx"], optimization_level=1)
        params = lr_schedule(p, delta_gamma, delta_beta)
        bound = transpiled.assign_parameters(params)
        job = sampler.run([(bound,)], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        local_best_x: np.ndarray | None = None
        local_best_cost = math.inf
        local_best_any_count = -1
        local_best_any: np.ndarray | None = None
        total_energy = 0.0; total_cnt = 0
        for bitstr, cnt in counts.items():
            bits = [int(b) for b in bitstr[::-1]]
            x = np.array(bits, dtype=int)
            if cnt > local_best_any_count:
                local_best_any_count = cnt
                local_best_any = x.copy()
            c = qubo_eval(Q, x)
            total_energy += c * cnt; total_cnt += cnt
            if int(x.sum()) == K and c < local_best_cost:
                local_best_cost = float(c); local_best_x = x.copy()
        mean_energy = total_energy / max(total_cnt, 1)
        energy_per_p.append(mean_energy)
        if local_best_x is not None and local_best_cost < overall_best_cost:
            overall_best_cost = local_best_cost; overall_best_x = local_best_x
        if local_best_any is not None and overall_best_any is None:
            overall_best_any = local_best_any
        last_metrics = gate_metrics.extract(transpiled)
        last_metrics["t_gate_count"] = 0
        last_transpiled = transpiled
        last_params = params
    elapsed = time.perf_counter() - t0

    feasible = overall_best_x is not None
    if not feasible:
        overall_best_x = overall_best_any if overall_best_any is not None else np.zeros(N, dtype=int)
        overall_best_cost = float(qubo_eval(Q, overall_best_x))

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(overall_best_cost, c_opt, c_wst)
    else:
        ar = None

    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=float(overall_best_cost),
        bitstring=overall_best_x.tolist(),
        feasible=bool(feasible),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed),
        optimizer_iters=0,
        num_circuit_evaluations=len(p_sweep),
        convergence_history=[float(e) for e in energy_per_p],
        qubit_count=int(last_metrics["qubit_count"]),
        circuit_depth=int(last_metrics["circuit_depth"]),
        two_qubit_gate_count=int(last_metrics["two_qubit_gate_count"]),
        t_gate_count=int(last_metrics["t_gate_count"]),
        total_gate_count=int(last_metrics["total_gate_count"]),
        shots=int(shots),
        backend="qiskit_aer.AerSimulator",
        transpile_optimization_level=1,
        hyperparameters={
            "problem_variant": "qubo",
            "ansatz": "LR-QAOA",
            "schedule": "linear_ramp",
            "Delta_gamma": float(delta_gamma),
            "Delta_beta": float(delta_beta),
            "p_sweep": list(p_sweep),
            "p_best_reported": int(max(p_sweep)),
        },
        initial_params=[float(v) for v in last_params],
        final_params=[float(v) for v in last_params],
    )
