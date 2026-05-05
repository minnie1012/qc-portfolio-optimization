"""Shared helpers used by every solver: QUBO build, eval, exact reference,
approx-ratio normalization."""
from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problem_definition import build_qubo, eval_qubo, qubo_to_ising  # noqa: E402
from problem_definition import ProblemInstance as _PdInst  # noqa: E402
from benchmark_protocol import instances  # noqa: E402

RESULTS_DIR = ROOT / "results" / "raw"


def build_qubo_from_inst(inst: instances.ProblemInstance) -> np.ndarray:
    pd_inst = _PdInst(
        mu=inst.mu, sigma=inst.sigma, K=inst.K, q=inst.q,
        prev_selection=inst.prev_selection, tickers=inst.asset_tickers,
    )
    return build_qubo(pd_inst)


def qubo_eval(Q: np.ndarray, x: np.ndarray) -> float:
    return float(eval_qubo(Q, x))


def exact_reference(Q: np.ndarray, K: int) -> tuple[np.ndarray, float, float]:
    """Optimum and worst over K-cardinality bitstrings (for approx_ratio)."""
    N = Q.shape[0]
    best_x, best_cost = None, math.inf
    worst_cost = -math.inf
    for combo in combinations(range(N), K):
        x = np.zeros(N, dtype=int)
        x[list(combo)] = 1
        c = float(eval_qubo(Q, x))
        if c < best_cost:
            best_cost, best_x = c, x
        if c > worst_cost:
            worst_cost = c
    return best_x, float(best_cost), float(worst_cost)


def approx_ratio(found: float, optimal: float, worst: float) -> float | None:
    if optimal is None or worst is None:
        return None
    if math.isclose(worst, optimal):
        return 1.0
    r = (worst - found) / (worst - optimal)
    return float(max(0.0, min(1.0, r)))


def output_path(algorithm: str, instance_id: str, seed: int) -> Path:
    return RESULTS_DIR / f"{algorithm}__{instance_id}__seed{seed}.json"


def qubo_to_pauli_op(Q: np.ndarray):
    """Build SparsePauliOp H = offset I + sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j."""
    from qiskit.quantum_info import SparsePauliOp

    offset, h, J = qubo_to_ising(Q)
    N = Q.shape[0]
    paulis: list[tuple[str, float]] = [("I" * N, float(offset))]
    for i in range(N):
        if abs(h[i]) > 1e-12:
            s = ["I"] * N; s[N - 1 - i] = "Z"
            paulis.append(("".join(s), float(h[i])))
    for i in range(N):
        for j in range(i + 1, N):
            if abs(J[i, j]) > 1e-12:
                s = ["I"] * N; s[N - 1 - i] = "Z"; s[N - 1 - j] = "Z"
                paulis.append(("".join(s), float(J[i, j])))
    labels, coeffs = zip(*paulis)
    return SparsePauliOp(list(labels), np.array(coeffs))
