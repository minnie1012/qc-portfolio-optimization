"""CVaR-VQE (Barkoutsos et al. 2020): VQE optimizing CVaR_alpha of energy."""
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
)

ALGORITHM = "cvar_vqe"
ALGORITHM_VERSION = "1.0.0"
MAX_N = 8


def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    alpha: float = 0.25,
    reps: int = 2,
    shots: int = 4096,
    optimizer_maxiter: int = 80,
) -> BenchmarkResult:
    from qiskit import transpile
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_aer.primitives import SamplerV2

    Q = build_qubo_from_inst(inst)
    N, K = inst.N, inst.K

    ansatz = RealAmplitudes(num_qubits=N, reps=reps, entanglement="linear")
    qc = ansatz.copy()
    qc.measure_all()
    transpiled = transpile(qc, basis_gates=["sx", "rz", "cx"], optimization_level=1)
    metrics = gate_metrics.extract(transpiled)
    metrics["t_gate_count"] = 0

    sampler = SamplerV2(default_shots=shots, seed=seed)
    rng = np.random.default_rng(seed)
    nparams = ansatz.num_parameters
    x0 = rng.uniform(-np.pi, np.pi, size=nparams).tolist()

    history: list[float] = []
    eval_count = {"n": 0}

    def cvar_energy(params):
        bound = transpiled.assign_parameters(list(params))
        job = sampler.run([(bound,)], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        pairs = []
        total = 0
        for bitstr, cnt in counts.items():
            bits = [int(b) for b in bitstr[::-1]]
            x = np.array(bits, dtype=int)
            e = qubo_eval(Q, x)
            pairs.append((e, cnt))
            total += cnt
        pairs.sort(key=lambda p: p[0])
        cutoff = max(1, int(alpha * total))
        accumulated = 0; weighted = 0.0
        for e, cnt in pairs:
            take = min(cnt, cutoff - accumulated)
            if take <= 0:
                break
            weighted += e * take
            accumulated += take
            if accumulated >= cutoff:
                break
        cvar = weighted / max(accumulated, 1)
        history.append(cvar)
        eval_count["n"] += 1
        return cvar

    t0 = time.perf_counter()
    res = minimize(cvar_energy, x0=x0, method="COBYLA",
                   options={"maxiter": optimizer_maxiter, "rhobeg": 0.5})
    bound = transpiled.assign_parameters(list(res.x))
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
            "ansatz": "RealAmplitudes_linear",
            "reps": reps,
            "alpha": alpha,
            "optimizer": "COBYLA",
            "optimizer_maxiter": optimizer_maxiter,
        },
        initial_params=[float(v) for v in x0],
        final_params=[float(v) for v in res.x],
    )
