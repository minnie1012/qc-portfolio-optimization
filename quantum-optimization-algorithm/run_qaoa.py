import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from benchmark_protocol import instances
from benchmark_protocol.result_schema import BenchmarkResult
from solver_common import build_qubo_from_inst, qubo_eval, exact_reference, approx_ratio, qubo_to_pauli_op


# ── Instances to run on ────────────────────────────────────────────────────────

from benchmark_protocol.instances import BUCKETS
INSTANCE_IDS = (
    BUCKETS["tiny"] +
    BUCKETS["small"] +
    BUCKETS["n7_gap"] +
    BUCKETS["medium"] +
    BUCKETS["large"]
)

P_VALUES = [1, 2]


# ── Core QAOA solver function ──────────────────────────────────────────────────

def run_qaoa(inst, p=1, seed=42, shots=4096, optimizer_maxiter=80):

    Q = build_qubo_from_inst(inst)
    H = qubo_to_pauli_op(Q)

    ansatz = QAOAAnsatz(cost_operator=H, reps=p)
    ansatz.measure_all()
    transpiled_circuit = transpile(
        ansatz, basis_gates=["sx", "rz", "cx"], optimization_level=1
    )

    estimator = EstimatorV2()
    sampler = SamplerV2(default_shots=shots)

    rng = np.random.default_rng(seed)
    starting_angles = rng.uniform(0, 2 * np.pi, size=ansatz.num_parameters).tolist()

    circuit_no_meas = ansatz.remove_final_measurements(inplace=False)
    circuit_no_meas = transpile(
        circuit_no_meas, basis_gates=["sx", "rz", "cx"], optimization_level=1
    )
    H_mapped = H.apply_layout(circuit_no_meas.layout) if circuit_no_meas.layout is not None else H

    history = []

    def cost_function(angles):
        pub = (circuit_no_meas, [H_mapped], [list(angles)])
        result = estimator.run([pub]).result()
        energy = float(result[0].data.evs[0])
        history.append(energy)
        return energy

    t0 = time.perf_counter()
    opt_result = minimize(
        cost_function, x0=starting_angles,
        method="COBYLA", options={"maxiter": optimizer_maxiter}
    )
    elapsed = time.perf_counter() - t0

    final_circuit = transpiled_circuit.assign_parameters(list(opt_result.x))
    counts = sampler.run([(final_circuit,)], shots=shots).result()[0].data.meas.get_counts()

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
        algorithm="qaoa_saksham",
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
        },
        initial_params=[float(v) for v in starting_angles],
        final_params=[float(v) for v in opt_result.x],
    )


# ── Run all experiments ────────────────────────────────────────────────────────

all_results = []

for instance_id in INSTANCE_IDS:
    inst = instances.load(instance_id)
    for p in P_VALUES:
        print(f"Running: {instance_id} | N={inst.N} | K={inst.K} | p={p} | stocks={inst.asset_tickers}")
        result = run_qaoa(inst, p=p)
        output_path = result.to_json(f"./results/qaoa/qaoa_saksham__{instance_id}__p{p}__seed42.json")
        ar_str = f"{result.approx_ratio:.4f}" if result.approx_ratio is not None else "N/A"
        selected = [inst.asset_tickers[i] for i, b in enumerate(result.bitstring) if b == 1]
        print(f"  → Selected: {selected} | AR: {ar_str} | Depth: {result.circuit_depth} | Gates: {result.total_gate_count} | Time: {result.wall_time_seconds:.2f}s")
        print(f"  → Saved to: {output_path}")
        print()
        all_results.append((instance_id, inst, p, result))


# ── Print comparison table ─────────────────────────────────────────────────────

print("=" * 90)
print(f"{'Instance':<15} {'N':>3} {'K':>3} {'p':>3} {'Selected':<25} {'AR':>8} {'Depth':>6} {'Gates':>6} {'Time':>7}")
print("=" * 90)

for instance_id, inst, p, result in all_results:
    selected = "+".join([inst.asset_tickers[i] for i, b in enumerate(result.bitstring) if b == 1])
    ar_str = f"{result.approx_ratio:.4f}" if result.approx_ratio is not None else "N/A"
    print(f"{instance_id:<15} {inst.N:>3} {inst.K:>3} {p:>3} {selected:<25} {ar_str:>8} {result.circuit_depth:>6} {result.total_gate_count:>6} {result.wall_time_seconds:>7.2f}s")

print("=" * 90)
print(f"\nTotal result files saved: {len(all_results)}")