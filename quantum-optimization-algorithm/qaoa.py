"""QAOA on the QUBO/Ising Hamiltonian. Aer-simulated; matched-subset only."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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

ALGORITHM = "qaoa"
ALGORITHM_VERSION = "1.0.0"
MAX_N = 8  # cap on simulated qubit count


def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    reps: int = 2,
    shots: int = 4096,
    optimizer_maxiter: int = 80,
) -> BenchmarkResult:
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_aer.primitives import EstimatorV2, SamplerV2

    Q = build_qubo_from_inst(inst)
    N, K = inst.N, inst.K
    H = qubo_to_pauli_op(Q)

    ansatz = QAOAAnsatz(cost_operator=H, reps=reps)
    ansatz.measure_all()
    transpiled_meas = transpile(
        ansatz, basis_gates=["sx", "rz", "cx"], optimization_level=1
    )
    metrics = gate_metrics.extract(transpiled_meas)
    # NISQ basis has no T gate; T-count is 0 by definition for {sx,rz,cx}.
    metrics["t_gate_count"] = 0

    rng = np.random.default_rng(seed)
    nparams = ansatz.num_parameters
    x0 = rng.uniform(0, 2 * np.pi, size=nparams).tolist()

    estimator = EstimatorV2(options={"run_options": {"shots": shots, "seed_simulator": seed}})
    sampler = SamplerV2(default_shots=shots, seed=seed)

    energy_no_meas = ansatz.remove_final_measurements(inplace=False)
    energy_no_meas = transpile(
        energy_no_meas, basis_gates=["sx", "rz", "cx"], optimization_level=1
    )
    H_t = H.apply_layout(energy_no_meas.layout) if energy_no_meas.layout is not None else H

    history: list[float] = []
    eval_count = {"n": 0}

    def cost_fn(params):
        pub = (energy_no_meas, [H_t], [list(params)])
        res = estimator.run([pub]).result()
        e = float(res[0].data.evs[0])
        history.append(e)
        eval_count["n"] += 1
        return e

    t0 = time.perf_counter()
    res = minimize(cost_fn, x0=x0, method="COBYLA",
                   options={"maxiter": optimizer_maxiter, "rhobeg": 0.5})
    bound = transpiled_meas.assign_parameters(list(res.x))
    job = sampler.run([(bound,)], shots=shots)
    counts = job.result()[0].data.meas.get_counts()
    elapsed = time.perf_counter() - t0

    best_x, best_cost = None, math.inf
    best_count = -1; best_x_any = None
    for bitstr, cnt in counts.items():
        bits = [int(b) for b in bitstr[::-1]]
        x = np.array(bits, dtype=int)
        if cnt > best_count:
            best_count, best_x_any = cnt, x.copy()
        if int(x.sum()) == K:
            c = qubo_eval(Q, x)
            if c < best_cost:
                best_cost, best_x = float(c), x.copy()
    feasible = best_x is not None
    if not feasible:
        best_x = best_x_any if best_x_any is not None else np.zeros(N, dtype=int)
        best_cost = float(qubo_eval(Q, best_x))

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(best_cost, c_opt, c_wst)
    else:
        ar = None

    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=float(best_cost),
        bitstring=best_x.tolist(),
        feasible=bool(feasible),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed),
        optimizer_iters=int(res.nit) if hasattr(res, "nit") else eval_count["n"],
        num_circuit_evaluations=eval_count["n"] + 1,
        convergence_history=[float(h) for h in history],
        qubit_count=int(metrics["qubit_count"]),
        circuit_depth=int(metrics["circuit_depth"]),
        two_qubit_gate_count=int(metrics["two_qubit_gate_count"]),
        t_gate_count=int(metrics["t_gate_count"]) if metrics["t_gate_count"] is not None else None,
        total_gate_count=int(metrics["total_gate_count"]),
        shots=int(shots),
        backend="qiskit_aer.AerSimulator",
        transpile_optimization_level=1,
        hyperparameters={
            "problem_variant": "qubo",
            "ansatz": "QAOAAnsatz",
            "reps": reps,
            "optimizer": "COBYLA",
            "optimizer_maxiter": optimizer_maxiter,
        },
        initial_params=[float(v) for v in x0],
        final_params=[float(v) for v in res.x],
    )
