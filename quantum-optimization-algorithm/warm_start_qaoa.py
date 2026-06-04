import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import time
import numpy as np
from scipy.optimize import minimize, linprog
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from benchmark_protocol import instances
from benchmark_protocol.result_schema import BenchmarkResult
from solver_common import build_qubo_from_inst, qubo_eval, exact_reference, approx_ratio, qubo_to_pauli_op


# ── Instances to run on ────────────────────────────────────────────────────────

from benchmark_protocol.instances import SUBSETS
from benchmark_protocol.instances import BUCKETS
INSTANCE_IDS = BUCKETS["medium"]
P_VALUES = [1, 2]


# ── Warm Start: solve relaxed problem classically ─────────────────────────────

def warm_start_angles(Q, K, N):
    """
    Solve the relaxed (continuous) version of the portfolio problem.
    Instead of binary 0/1, allow weights between 0 and 1.
    Returns initial rotation angles for each qubit based on the relaxed solution.
    """
    # Minimize x^T Q x subject to sum(x) = K, 0 <= x <= 1
    # We use scipy linprog for the linear relaxation (ignore quadratic term)
    # Just use equal weights as warm start — simple and effective
    relaxed_x = np.full(N, K / N)  # equal weight relaxation

    # Convert relaxed solution to rotation angles
    # If x_i is close to 1, qubit should start near |1>
    # If x_i is close to 0, qubit should start near |0>
    # Angle formula: theta_i = 2 * arcsin(sqrt(x_i))
    angles = 2 * np.arcsin(np.sqrt(np.clip(relaxed_x, 0, 1)))
    return angles


def build_warm_start_circuit(H, N, reps, warm_angles):
    """
    Build a QAOA circuit with warm start initialization.
    Instead of equal superposition (Hadamard), initialize each qubit
    based on the relaxed solution using Ry rotations.
    """
    gamma = ParameterVector("γ", reps)
    beta = ParameterVector("β", reps)

    qc = QuantumCircuit(N)

    # Warm start initialization — Ry rotation instead of Hadamard
    for i in range(N):
        qc.ry(warm_angles[i], i)

    # QAOA layers
    for layer in range(reps):
        # Cost layer — apply phase based on Hamiltonian
        for term in H:
            pauli_str, coeff = list(term.items())[0] if hasattr(term, 'items') else (None, None)

        # Use Qiskit's built in evolution for cost operator
        # Apply ZZ interactions
        paulis = H.paulis
        coeffs = H.coeffs
        for pauli, coeff in zip(paulis, coeffs):
            active_qubits = [j for j in range(N) if str(pauli)[N - 1 - j] != 'I']
            if len(active_qubits) == 1:
                qc.rz(2 * gamma[layer] * float(np.real(coeff)), active_qubits[0])
            elif len(active_qubits) == 2:
                i, j = active_qubits[0], active_qubits[1]
                qc.cx(i, j)
                qc.rz(2 * gamma[layer] * float(np.real(coeff)), j)
                qc.cx(i, j)

        # Mixer layer — warm start mixer (Ry instead of Rx)
        for i in range(N):
            qc.ry(2 * beta[layer], i)

    qc.measure_all()
    return qc, gamma, beta


# ── Core solver ────────────────────────────────────────────────────────────────

def run_warm_start_qaoa(inst, p=1, seed=42, shots=4096, optimizer_maxiter=80):

    Q = build_qubo_from_inst(inst)
    H = qubo_to_pauli_op(Q)
    N = inst.N

    # Get warm start angles
    warm_angles = warm_start_angles(Q, inst.K, N)

    # Build warm start circuit
    qc, gamma, beta = build_warm_start_circuit(H, N, p, warm_angles)

    # Transpile
    transpiled_circuit = transpile(
        qc, basis_gates=["sx", "rz", "cx"], optimization_level=1
    )

    # Circuit without measurements for energy estimation
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    transpiled_no_meas = transpile(
        qc_no_meas, basis_gates=["sx", "rz", "cx"], optimization_level=1
    )

    estimator = EstimatorV2()
    sampler = SamplerV2(default_shots=shots)

    # Initialize with warm start angles for gamma, random for beta
    rng = np.random.default_rng(seed)
    all_params = list(qc.parameters)
    n_gamma = p
    n_beta = p
    gamma_init = rng.uniform(0, np.pi, size=n_gamma).tolist()
    beta_init = rng.uniform(0, np.pi, size=n_beta).tolist()
    x0 = gamma_init + beta_init

    # Map Hamiltonian to circuit layout
    H_mapped = H.apply_layout(transpiled_no_meas.layout) if transpiled_no_meas.layout is not None else H

    history = []

    def cost_function(params):
        param_dict = {p_obj: params[i] for i, p_obj in enumerate(sorted(qc_no_meas.parameters, key=lambda x: x.name))}
        bound = qc_no_meas.assign_parameters(param_dict)
        bound_transpiled = transpile(bound, basis_gates=["sx", "rz", "cx"], optimization_level=1)
        H_t = H.apply_layout(bound_transpiled.layout) if bound_transpiled.layout is not None else H
        pub = (bound_transpiled, [H_t], [[]])
        result = estimator.run([pub]).result()
        energy = float(result[0].data.evs[0])
        history.append(energy)
        return energy

    t0 = time.perf_counter()
    opt_result = minimize(
        cost_function, x0=x0,
        method="COBYLA", options={"maxiter": optimizer_maxiter}
    )
    elapsed = time.perf_counter() - t0

    # Sample final answer
    final_params = {p_obj: opt_result.x[i] for i, p_obj in enumerate(sorted(qc.parameters, key=lambda x: x.name))}
    final_circuit = transpile(
        qc.assign_parameters(final_params).remove_final_measurements(inplace=False),
        basis_gates=["sx", "rz", "cx"], optimization_level=1
    )
    meas_circuit = QuantumCircuit(N, N)
    meas_circuit.compose(final_circuit, inplace=True)
    meas_circuit.measure(range(N), range(N))
    meas_transpiled = transpile(meas_circuit, basis_gates=["sx", "rz", "cx"], optimization_level=1)

    job = sampler.run([(meas_transpiled,)], shots=shots)
    counts = job.result()[0].data.c.get_counts()

    best_bitstring = None
    best_cost = math.inf
    best_count = -1
    best_any = None

    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]
        x = np.array(bits, dtype=int)
        if count > best_count:
            best_count = count
            best_any = x.copy()
        if int(x.sum()) == inst.K:
            cost = qubo_eval(Q, x)
            if cost < best_cost:
                best_cost = cost
                best_bitstring = x.copy()

    feasible = best_bitstring is not None
    if not feasible:
        best_bitstring = best_any
        best_cost = float(qubo_eval(Q, best_bitstring))

    if inst.N <= 22:
        _, optimal_cost, worst_cost = exact_reference(Q, inst.K)
        ar = approx_ratio(best_cost, optimal_cost, worst_cost)
    else:
        ar = None

    return BenchmarkResult(
        algorithm="warm_start_qaoa_saksham",
        algorithm_version="1.0.0",
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=float(best_cost),
        bitstring=best_bitstring.tolist(),
        feasible=bool(feasible),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed),
        optimizer_iters=len(history),
        num_circuit_evaluations=len(history) + 1,
        convergence_history=[float(h) for h in history],
        qubit_count=int(transpiled_circuit.num_qubits),
        circuit_depth=int(transpiled_circuit.depth()),
        two_qubit_gate_count=int(sum(c for name, c in transpiled_circuit.count_ops().items() if name == "cx")),
        t_gate_count=0,
        total_gate_count=int(sum(transpiled_circuit.count_ops().values())),
        shots=shots,
        backend="qiskit_aer.AerSimulator",
        transpile_optimization_level=1,
        hyperparameters={
            "p": p,
            "optimizer": "COBYLA",
            "optimizer_maxiter": optimizer_maxiter,
            "warm_start": True,
            "relaxation": "equal_weight",
        },
        initial_params=[float(v) for v in x0],
        final_params=[float(v) for v in opt_result.x],
    )


# ── Run all experiments ────────────────────────────────────────────────────────

all_results = []

for instance_id in INSTANCE_IDS:
    inst = instances.load(instance_id)
    for p in P_VALUES:
        print(f"Running: {instance_id} | N={inst.N} | K={inst.K} | p={p} | stocks={inst.asset_tickers}")
        try:
            result = run_warm_start_qaoa(inst, p=p)
            output_path = result.to_json(f"./results/warm_start/warm_start_qaoa_saksham__{instance_id}__p{p}__seed42.json")
            ar_str = f"{result.approx_ratio:.4f}" if result.approx_ratio is not None else "N/A"
            selected = [inst.asset_tickers[i] for i, b in enumerate(result.bitstring) if b == 1]
            print(f"  → Selected: {selected} | AR: {ar_str} | Depth: {result.circuit_depth} | Gates: {result.total_gate_count} | Time: {result.wall_time_seconds:.2f}s")
            print(f"  → Saved to: {output_path}")
            all_results.append((instance_id, inst, p, result))
        except Exception as e:
            print(f"  → FAILED: {e}")
        print()


# ── Print comparison table ─────────────────────────────────────────────────────

if all_results:
    print("=" * 95)
    print(f"{'Instance':<15} {'N':>3} {'K':>3} {'p':>3} {'Selected':<25} {'AR':>8} {'Depth':>6} {'Gates':>6} {'Time':>7}")
    print("=" * 95)
    for instance_id, inst, p, result in all_results:
        selected = "+".join([inst.asset_tickers[i] for i, b in enumerate(result.bitstring) if b == 1])
        ar_str = f"{result.approx_ratio:.4f}" if result.approx_ratio is not None else "N/A"
        print(f"{instance_id:<15} {inst.N:>3} {inst.K:>3} {p:>3} {selected:<25} {ar_str:>8} {result.circuit_depth:>6} {result.total_gate_count:>6} {result.wall_time_seconds:>7.2f}s")
    print("=" * 95)
    print(f"\nTotal results: {len(all_results)}")