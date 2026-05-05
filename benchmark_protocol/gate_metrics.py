"""Extract qubit / depth / 2-qubit / T-gate counts from a Qiskit QuantumCircuit.

Always pass a *transpiled* circuit (with the backend's basis gates) so the
counts reflect what would actually execute. T-gate count requires a basis
that includes T (e.g. {h, t, tdg, cx} for Clifford+T). For NISQ backends
(basis like {sx, rz, cx, ecr}) T-gate count is reported as 0.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


TWO_QUBIT_NAMES = frozenset({"cx", "cz", "ecr", "iswap", "swap", "rzx", "rzz", "rxx", "ryy"})
T_GATE_NAMES = frozenset({"t", "tdg"})

CLIFFORD_T_BASIS = ["h", "t", "tdg", "s", "sdg", "x", "y", "z", "cx"]


def extract(circuit: "QuantumCircuit") -> dict:
    """Return metric dict for an already-transpiled circuit."""
    ops = circuit.count_ops()
    two_q = sum(c for name, c in ops.items() if name in TWO_QUBIT_NAMES)
    t_gates = sum(c for name, c in ops.items() if name in T_GATE_NAMES)
    total = sum(ops.values())
    return {
        "qubit_count": circuit.num_qubits,
        "circuit_depth": circuit.depth(),
        "two_qubit_gate_count": int(two_q),
        "t_gate_count": int(t_gates),
        "total_gate_count": int(total),
    }


def extract_with_t_count(circuit: "QuantumCircuit", optimization_level: int = 1) -> dict:
    """Re-transpile to Clifford+T basis to get a meaningful T-gate count.

    NISQ backends report t_gate_count=0 in extract() because T is not in the
    native basis. This helper transpiles a copy to {H, T, S, CX} and returns
    the T count alongside the regular metrics from the original circuit.
    """
    from qiskit import transpile

    orig = extract(circuit)
    try:
        clifford_t = transpile(circuit, basis_gates=CLIFFORD_T_BASIS, optimization_level=optimization_level)
        orig["t_gate_count"] = int(clifford_t.count_ops().get("t", 0) + clifford_t.count_ops().get("tdg", 0))
    except Exception as exc:
        orig["t_gate_count"] = None
        orig["_t_count_error"] = str(exc)
    return orig
