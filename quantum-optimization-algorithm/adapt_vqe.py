"""ADAPT-VQE with a hardware-efficient operator pool.

Pool: single-qubit {Y_i, Z_i} plus entangling {Y_i X_{i+1}, X_i Y_{i+1}}.
Otherwise identical to adapt_qaoa: greedy gradient-driven layer growth,
re-optimize all parameters at each step.
"""
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

ALGORITHM = "adapt_vqe"
ALGORITHM_VERSION = "1.0.0"
MAX_N = 8


def _pool(N: int):
    from qiskit.quantum_info import SparsePauliOp
    ops = []
    for i in range(N):
        s = ["I"] * N; s[N - 1 - i] = "Y"
        ops.append(("Y", i, None, SparsePauliOp(["".join(s)], np.array([1.0]))))
        s = ["I"] * N; s[N - 1 - i] = "Z"
        ops.append(("Z", i, None, SparsePauliOp(["".join(s)], np.array([1.0]))))
    for i in range(N - 1):
        s = ["I"] * N; s[N - 1 - i] = "Y"; s[N - 1 - (i + 1)] = "X"
        ops.append(("YX", i, i + 1, SparsePauliOp(["".join(s)], np.array([1.0]))))
        s = ["I"] * N; s[N - 1 - i] = "X"; s[N - 1 - (i + 1)] = "Y"
        ops.append(("XY", i, i + 1, SparsePauliOp(["".join(s)], np.array([1.0]))))
    return ops


def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    max_layers: int = 6,
    grad_tol: float = 1e-3,
    shots: int = 4096,
    optimizer_maxiter: int = 60,
) -> BenchmarkResult:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit_aer.primitives import EstimatorV2, SamplerV2

    Q = build_qubo_from_inst(inst)
    N, K = inst.N, inst.K
    H = qubo_to_pauli_op(Q)
    pool = _pool(N)

    estimator = EstimatorV2(options={"run_options": {"shots": shots, "seed_simulator": seed}})
    sampler = SamplerV2(default_shots=shots, seed=seed)

    layers: list = []
    theta: list[float] = []
    history: list[float] = []
    eval_count = {"n": 0}

    def base_circuit(layers_list, thetas):
        qc = QuantumCircuit(N)
        qc.h(range(N))
        for (_, _, _, op), t in zip(layers_list, thetas):
            qc.append(PauliEvolutionGate(op, time=t), range(N))
        return qc

    # Filter pool: drop operators whose commutator with H simplifies to zero
    # (Z operators commute with the diagonal Z-Ising cost Hamiltonian).
    active_pool: list = []
    active_commutators = []
    for (kind, i, j, A) in pool:
        comm = (1j * (H @ A - A @ H)).simplify()
        if len(comm) > 0 and float(np.max(np.abs(comm.coeffs))) > 1e-12:
            active_pool.append((kind, i, j, A))
            active_commutators.append(comm)
    if not active_pool:
        active_pool = pool
        active_commutators = [(1j * (H @ A - A @ H)).simplify() for (_, _, _, A) in pool]

    t0 = time.perf_counter()
    for _ in range(max_layers):
        circ = base_circuit(layers, theta)
        circ_t = transpile(circ, basis_gates=["sx", "rz", "cx"], optimization_level=1)
        observables = [c.apply_layout(circ_t.layout) if circ_t.layout is not None else c for c in active_commutators]
        res = estimator.run([(circ_t, observables)]).result()
        evs = np.asarray(res[0].data.evs)
        grads = np.abs(evs)
        eval_count["n"] += 1

        best_idx = int(np.argmax(grads))
        best_grad = float(grads[best_idx])
        if best_grad < grad_tol and layers:
            break
        layers.append(active_pool[best_idx])
        theta.append(0.01)

        params_sym = [Parameter(f"t_{k}") for k in range(len(layers))]
        qc = QuantumCircuit(N); qc.h(range(N))
        for (_, _, _, op), p in zip(layers, params_sym):
            qc.append(PauliEvolutionGate(op, time=p), range(N))
        energy_circ = transpile(qc, basis_gates=["sx", "rz", "cx"], optimization_level=1)
        H_t = H.apply_layout(energy_circ.layout) if energy_circ.layout is not None else H

        def cost_fn(p):
            pub2 = (energy_circ, [H_t], [list(p)])
            r2 = estimator.run([pub2]).result()
            e = float(r2[0].data.evs[0])
            history.append(e); eval_count["n"] += 1
            return e

        r = minimize(cost_fn, x0=theta, method="COBYLA",
                     options={"maxiter": optimizer_maxiter, "rhobeg": 0.2})
        theta = list(r.x)

    qc_final = QuantumCircuit(N); qc_final.h(range(N))
    for (_, _, _, op), t in zip(layers, theta):
        qc_final.append(PauliEvolutionGate(op, time=t), range(N))
    qc_final.measure_all()
    transpiled_meas = transpile(qc_final, basis_gates=["sx", "rz", "cx"], optimization_level=1)
    metrics = gate_metrics.extract(transpiled_meas)
    metrics["t_gate_count"] = 0
    job = sampler.run([(transpiled_meas,)], shots=shots)
    counts = job.result()[0].data.meas.get_counts()
    elapsed = time.perf_counter() - t0

    best_x, best_cost = None, math.inf
    best_any = None; best_any_cnt = -1
    for bitstr, cnt in counts.items():
        bits = [int(b) for b in bitstr[::-1]]
        x = np.array(bits, dtype=int)
        if cnt > best_any_cnt:
            best_any_cnt, best_any = cnt, x.copy()
        if int(x.sum()) == K:
            c = qubo_eval(Q, x)
            if c < best_cost:
                best_cost, best_x = float(c), x.copy()
    feasible = best_x is not None
    if not feasible:
        best_x = best_any if best_any is not None else np.zeros(N, dtype=int)
        best_cost = float(qubo_eval(Q, best_x))

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(best_cost, c_opt, c_wst)
    else:
        ar = None

    pool_summary = [{"kind": k[0], "i": k[1], "j": k[2]} for k in layers]
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
        optimizer_iters=len(layers),
        num_circuit_evaluations=eval_count["n"] + 1,
        convergence_history=[float(h) for h in history],
        qubit_count=int(metrics["qubit_count"]),
        circuit_depth=int(metrics["circuit_depth"]),
        two_qubit_gate_count=int(metrics["two_qubit_gate_count"]),
        t_gate_count=int(metrics["t_gate_count"]),
        total_gate_count=int(metrics["total_gate_count"]),
        shots=int(shots),
        backend="qiskit_aer.AerSimulator",
        transpile_optimization_level=1,
        hyperparameters={
            "problem_variant": "qubo",
            "ansatz": "ADAPT-VQE",
            "pool": "Y_Z_YX_XY",
            "max_layers": int(max_layers),
            "grad_tol": float(grad_tol),
            "optimizer": "COBYLA",
            "optimizer_maxiter": optimizer_maxiter,
            "selected_layers": pool_summary,
        },
        initial_params=[0.01] * len(layers) if layers else [],
        final_params=[float(t) for t in theta],
    )
