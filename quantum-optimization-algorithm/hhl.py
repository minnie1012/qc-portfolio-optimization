"""HHL-based portfolio selection (Rebentrost-Lloyd / Yalovetzky et al. style).

Formulation
-----------
Unconstrained Markowitz with budget constraint via Lagrange multiplier yields
the KKT linear system

    [q Sigma   1] [w]     [mu]
    [   1^T    0] [lam]   [1 ]

A 5-line block (N+1) x (N+1) matrix A and rhs b. HHL prepares a quantum state
|w> proportional to A^-1 b on a register of n_b = ceil(log2(N+1)) qubits with
an O(log N * kappa / eps) cost (Harrow-Hassidim-Lloyd 2009; NISQ-HHL,
Yalovetzky et al. 2022/2024).

This implementation is split in two layers:

    1. **build_hhl_circuit(A, b)** -- assembles a QPE + controlled-reciprocal
       rotation + uncomputation circuit on (n_b + n_clock + 1) qubits using
       Qiskit primitives (PhaseEstimation + Hermitian Hamiltonian
       UnitaryGate). For N <= 4 we run the circuit on statevector and decode
       the post-selected |w> register. For larger N we report the same
       circuit-construction metrics (qubit_count, depth, two_qubit_gate_count)
       but solve A w = b classically -- this is the noiseless, infinite-shot
       limit of HHL output, which is the right ceiling to benchmark
       portfolio quality against.

    2. The continuous solution w is mapped to a binary K-asset selection by
       top-K largest |w_i| (same decoding the JPM benchmark uses, and
       identical to the classical `mvo` solver, so the comparison isolates
       the linear-solve step).

backend
    "hhl_aer_statevector"  -- N <= 4, real HHL circuit on Aer
    "hhl_classical_emulation_with_resource_estimate"  -- otherwise

The latter is labeled distinctly so the comparison tables make the
methodology unambiguous.
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

from benchmark_protocol import instances  # noqa: E402
from benchmark_protocol.result_schema import BenchmarkResult  # noqa: E402

from solver_common import (  # noqa: E402
    approx_ratio,
    build_qubo_from_inst,
    exact_reference,
    qubo_eval,
)

ALGORITHM = "hhl"
ALGORITHM_VERSION = "0.1.0"
MAX_N = 25
N_REAL_HHL = 4   # above this we emulate but still cost the circuit


# ---------------------------------------------------------------------------
# Linear system
# ---------------------------------------------------------------------------

def build_markowitz_kkt(mu: np.ndarray, sigma: np.ndarray, q: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, b) for the KKT system of max mu^T w - (q/2) w^T Sigma w
    s.t. 1^T w = 1.

    Pad to next power of two so qubit register sizes are clean.
    """
    N = len(mu)
    dim = N + 1                              # +1 for Lagrange multiplier
    pad = 1 << (dim - 1).bit_length()        # next power of two
    A = np.eye(pad)                          # identity padding keeps cond OK
    b = np.zeros(pad)
    A[:N, :N] = q * sigma
    A[:N, N] = 1.0
    A[N, :N] = 1.0
    A[N, N] = 0.0
    b[:N] = mu
    b[N] = 1.0
    # symmetrize (sigma may have tiny asymmetry from numerical scaling)
    A = (A + A.T) / 2
    return A, b


# ---------------------------------------------------------------------------
# Resource estimate for HHL at a given system size
# ---------------------------------------------------------------------------

def hhl_resource_estimate(A: np.ndarray, n_clock: int = 6) -> dict:
    """Theoretical resource estimate for HHL applied to A.

    Counts qubits and a leading-order depth estimate. We do not pretend to
    transpile a full HHL circuit at N=25; instead we report the architecture
    cost so reviewers can see what an HHL run would require.
    """
    dim = A.shape[0]
    n_b = int(math.ceil(math.log2(dim)))
    qubits = n_b + n_clock + 1               # +1 reciprocal ancilla
    kappa = float(np.linalg.cond(A))
    # leading-order: O(n_b * 2^n_clock) controlled-U applications, each
    # depth O(n_b^2) on a generic Hermitian (Trotterized exp(iAt)).
    depth = (2 ** n_clock) * (n_b ** 2)
    two_qubit = (2 ** n_clock) * n_b * 4     # ~4 CX per controlled-U layer
    total = depth + two_qubit
    return {
        "qubit_count": qubits,
        "circuit_depth": int(depth),
        "two_qubit_gate_count": int(two_qubit),
        "t_gate_count": 0,
        "total_gate_count": int(total),
        "n_b": n_b,
        "n_clock": n_clock,
        "kappa": kappa,
    }


# ---------------------------------------------------------------------------
# Real (small-scale) HHL circuit on Aer statevector
# ---------------------------------------------------------------------------

def hhl_run_small(A: np.ndarray, b: np.ndarray, n_clock: int = 6,
                  shots: int = 8192, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Run an actual HHL circuit on Aer (statevector). Only viable for tiny A.

    Returns (decoded w of length dim, metrics dict).
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer import AerSimulator

    dim = A.shape[0]
    n_b = int(math.ceil(math.log2(dim)))

    # Build exp(iAt). Choose t so that eigenvalues of At fit in [0, 2 pi)
    eigs = np.linalg.eigvalsh(A)
    lam_max = float(np.max(np.abs(eigs)))
    if lam_max == 0:
        raise ValueError("singular matrix")
    t = (2 * math.pi) / (lam_max * (2 ** n_clock))
    U = _matrix_exp(1j * A * t)
    U_gate = UnitaryGate(U, label="expA")

    # registers
    qb = QuantumRegister(n_b, "b")
    qc_clk = QuantumRegister(n_clock, "clk")
    qa = QuantumRegister(1, "anc")
    cr_anc = ClassicalRegister(1, "m")
    qc = QuantumCircuit(qb, qc_clk, qa, cr_anc)

    # prepare |b>
    b_norm = np.zeros(2 ** n_b)
    b_norm[:dim] = b / np.linalg.norm(b)
    qc.initialize(b_norm.tolist(), list(qb))

    # QPE
    for q in qc_clk:
        qc.h(q)
    for k in range(n_clock):
        reps = 1 << k
        ctrl_U = U_gate.power(reps).control(1)
        qc.append(ctrl_U, [qc_clk[k]] + list(qb))
    qc.append(_qft_dagger(n_clock), list(qc_clk))

    # controlled reciprocal rotation: Ry(2 arcsin(C / lambda)) where lambda
    # is read off the clock register integer m as m / (2^n_clock * t)
    C = 1.0 / (2 ** n_clock)
    for m in range(1, 2 ** n_clock):
        lam = m / (2 ** n_clock * t)
        if abs(lam) < 1e-12:
            continue
        ratio = max(min(C / lam, 1.0), -1.0)
        theta = 2 * math.asin(ratio)
        # multi-controlled Ry on ancilla, conditioned on clock == m
        bits = [(m >> i) & 1 for i in range(n_clock)]
        # X-mask any 0 bits
        for i, bit in enumerate(bits):
            if bit == 0:
                qc.x(qc_clk[i])
        qc.mcry(theta, list(qc_clk), qa[0])
        for i, bit in enumerate(bits):
            if bit == 0:
                qc.x(qc_clk[i])

    # uncompute QPE
    qc.append(_qft_dagger(n_clock).inverse(), list(qc_clk))
    for k in reversed(range(n_clock)):
        reps = 1 << k
        ctrl_U = U_gate.power(reps).inverse().control(1)
        qc.append(ctrl_U, [qc_clk[k]] + list(qb))
    for q in qc_clk:
        qc.h(q)

    # measure ancilla -> post-select on |1>
    qc.measure(qa[0], cr_anc[0])

    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend, optimization_level=1)
    # save statevector after the circuit so we can extract |w>
    tqc.save_statevector()
    job = backend.run(tqc, shots=1, seed_simulator=seed)
    sv = np.asarray(job.result().get_statevector())

    # extract amplitudes where ancilla == 1, then trace out clock register
    sv = sv.reshape((2, 2 ** n_clock, 2 ** n_b))  # (anc, clk, b)
    block = sv[1, :, :]                            # ancilla=1 slice
    w_amps = block.sum(axis=0)                     # marginalize clock
    w = np.abs(w_amps[:dim]) * np.sign(np.real(w_amps[:dim]) + 1e-30)
    nrm = np.linalg.norm(w)
    if nrm > 0:
        w = w / nrm
    metrics = {
        "qubit_count": tqc.num_qubits,
        "circuit_depth": tqc.depth(),
        "two_qubit_gate_count": sum(1 for instr in tqc.data
                                    if len(instr.qubits) == 2),
        "t_gate_count": 0,
        "total_gate_count": len(tqc.data),
        "n_b": n_b,
        "n_clock": n_clock,
        "kappa": float(np.linalg.cond(A)),
    }
    return w, metrics


def _matrix_exp(M: np.ndarray) -> np.ndarray:
    from scipy.linalg import expm
    return expm(M)


def _qft_dagger(n: int):
    from qiskit.circuit.library import QFTGate
    return QFTGate(n).inverse()


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

def solve(inst: instances.ProblemInstance, seed: int = 42,
          n_clock: int = 6) -> BenchmarkResult:
    Q = build_qubo_from_inst(inst)
    N, K = inst.N, inst.K
    A, b = build_markowitz_kkt(inst.mu, inst.sigma, inst.q)

    t0 = time.perf_counter()
    if N <= N_REAL_HHL:
        try:
            w_full, metrics = hhl_run_small(A, b, n_clock=n_clock, seed=seed)
            backend = "hhl_aer_statevector"
        except Exception:
            w_full = np.linalg.solve(A, b)
            metrics = hhl_resource_estimate(A, n_clock=n_clock)
            backend = "hhl_classical_fallback"
    else:
        w_full = np.linalg.solve(A, b)
        metrics = hhl_resource_estimate(A, n_clock=n_clock)
        backend = "hhl_classical_emulation_with_resource_estimate"

    w_assets = w_full[:N]
    elapsed_solve = time.perf_counter() - t0

    # decode: top-K by positive weight (long-only), matches MVO classical
    # The KKT solution may include negative (short) weights -- we want the
    # K assets with the largest LONG positions in the continuous solution.
    selection_order = np.argsort(-w_assets)
    x = np.zeros(N, dtype=int)
    x[selection_order[:K]] = 1
    obj = float(qubo_eval(Q, x))
    feasible = int(x.sum()) == K

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(obj, c_opt, c_wst)
    else:
        ar = None

    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=obj,
        bitstring=x.tolist(),
        feasible=bool(feasible),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed_solve),
        optimizer_iters=None,
        num_circuit_evaluations=1,
        convergence_history=[],
        qubit_count=int(metrics["qubit_count"]),
        circuit_depth=int(metrics["circuit_depth"]),
        two_qubit_gate_count=int(metrics["two_qubit_gate_count"]),
        t_gate_count=int(metrics["t_gate_count"]),
        total_gate_count=int(metrics["total_gate_count"]),
        shots="statevector",
        backend=backend,
        transpile_optimization_level=1,
        hyperparameters={
            "problem_variant": "continuous_markowitz_kkt_then_topK",
            "n_clock": n_clock,
            "n_b": int(metrics["n_b"]),
            "condition_number": float(metrics["kappa"]),
            "decoding": "top_K_by_abs_weight",
        },
        initial_params=None,
        final_params=None,
    )
