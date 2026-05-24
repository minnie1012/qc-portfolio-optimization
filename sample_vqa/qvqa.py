"""
Sampling-based variational quantum ansatz helpers for the sample VQA
experiment in this folder.

This module is intentionally standalone. Qiskit is only imported inside the
functions that need it so the file can still be inspected or reused in a
minimal environment.

Two ansatz families:
  - TwoLocal-style: RY rotations + CZ entanglement
  - BFCD-inspired: RY rotations + shared-parameter YZ/ZY entanglers

Two entanglement schedules:
  - bilinear: odd-indexed pairs, then even-indexed pairs on a chain
  - colored: a simple 3-pass scheduling heuristic over a chain-plus-skip graph

Optimizer:
  - NFT (Nakanishi-Fujii-Todo) coordinate updates using the 3-point cosine fit
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def _require_qiskit() -> tuple[object, object]:
    """Import Qiskit lazily and raise a clear error if it is unavailable."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "qiskit is required for sample_vqa.qvqa but is not installed"
        ) from exc
    return QuantumCircuit, ParameterVector


# ---------- Entanglement maps ----------


def bilinear_pairs(n: int) -> list[tuple[int, int]]:
    """Odd-then-even nearest-neighbour pairs on a line."""
    odd = [(i, i + 1) for i in range(0, n - 1, 2)]
    even = [(i, i + 1) for i in range(1, n - 1, 2)]
    return odd + even


def colored_pairs(n: int) -> list[tuple[int, int]]:
    """
    Simple 3-pass schedule over nearest- and next-nearest-neighbour edges.

    The original paper uses a heavy-hex-inspired coloring. In this sample
    module we keep the same spirit but on a chain-plus-skip graph so the
    structure stays easy to reason about in a simulator.
    """
    edges = [(i, i + 1) for i in range(n - 1)] + [(i, i + 2) for i in range(n - 2)]
    colored = [[], [], []]
    for k, edge in enumerate(edges):
        colored[k % 3].append(edge)
    return [edge for group in colored for edge in group]


# ---------- Ansatze ----------


def twolocal_ansatz(n: int, reps: int, ent_map: str = "bilinear"):
    """
    TwoLocal-style ansatz: RY rotations on every qubit, then CZ on selected
    pairs, repeated `reps` times, with a final rotation layer.
    """
    QuantumCircuit, ParameterVector = _require_qiskit()
    pairs_fn = bilinear_pairs if ent_map == "bilinear" else colored_pairs
    pairs = pairs_fn(n)

    n_params = (reps + 1) * n
    theta = ParameterVector("theta", n_params)
    qc = QuantumCircuit(n)
    p = 0
    for _ in range(reps):
        for q in range(n):
            qc.ry(theta[p], q)
            p += 1
        for a, b in pairs:
            qc.cz(a, b)
    for q in range(n):
        qc.ry(theta[p], q)
        p += 1
    return qc


def bfcd_ansatz(n: int, reps: int, ent_map: str = "bilinear"):
    """
    BFCD-inspired ansatz: RY rotations plus shared-parameter YZ/ZY entanglers.
    """
    QuantumCircuit, ParameterVector = _require_qiskit()
    pairs_fn = bilinear_pairs if ent_map == "bilinear" else colored_pairs
    pairs = pairs_fn(n)

    n_rot_params = (reps + 1) * n
    n_ent_params = reps * len(pairs)
    theta = ParameterVector("theta", n_rot_params + n_ent_params)

    qc = QuantumCircuit(n)
    p = 0
    for _ in range(reps):
        for q in range(n):
            qc.ry(theta[p], q)
            p += 1
        for a, b in pairs:
            phi = theta[p]
            p += 1

            # exp(-i phi/2 Y_a Z_b)
            qc.sdg(a)
            qc.h(a)
            qc.rzz(phi, a, b)
            qc.h(a)
            qc.s(a)

            # exp(-i phi/2 Z_a Y_b)
            qc.sdg(b)
            qc.h(b)
            qc.rzz(phi, a, b)
            qc.h(b)
            qc.s(b)
    for q in range(n):
        qc.ry(theta[p], q)
        p += 1
    return qc


# ---------- CVaR aggregation ----------


def cvar_aggregate(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """
    CVaR_alpha: expected value of the lower alpha-tail of a distribution.

    values: cost f(x_i) for each unique sampled bitstring
    weights: how many shots produced each sample
    alpha: in (0, 1]
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0 or weights.size == 0:
        raise ValueError("cvar_aggregate requires at least one sample")
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in the interval (0, 1]")

    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    total = float(w_sorted.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")

    cutoff = alpha * total
    cum = 0.0
    acc = 0.0
    for v, w in zip(v_sorted, w_sorted):
        take = min(float(w), cutoff - cum)
        if take <= 0.0:
            break
        acc += take * float(v)
        cum += take
        if cum >= cutoff:
            break
    return acc / cutoff


# ---------- Sampling helpers ----------


def sample_circuit(sampler, circuit_with_params, params_assigned, shots):
    """Run the sampler and return (bitstring_array_int, counts)."""
    qc = circuit_with_params.assign_parameters(params_assigned)
    if getattr(qc, "num_clbits", 0) == 0:
        qc.measure_all()

    pub = (qc, [], shots)
    result = sampler.run([pub]).result()
    counts = result[0].data.meas.get_counts()

    n = circuit_with_params.num_qubits
    ordered_items = sorted(counts.items(), key=lambda item: item[0])
    bitstrings = np.zeros((len(ordered_items), n), dtype=np.int8)
    weights = np.zeros(len(ordered_items), dtype=np.int64)
    for i, (bs, ct) in enumerate(ordered_items):
        for j, ch in enumerate(bs[::-1]):
            bitstrings[i, j] = int(ch)
        weights[i] = ct
    return bitstrings, weights


def evaluate_cost_on_samples(bitstrings, weights, cost_fn):
    """Vectorized: compute cost f(x) for each unique sample."""
    return np.array([cost_fn(x) for x in bitstrings]), weights


# ---------- NFT optimizer ----------


def nft_optimize(
    cost_fn: Callable[[np.ndarray], float],
    x0: Sequence[float],
    n_epochs: int,
    callback=None,
    rng=None,
    batch_cost_fn=None,
):
    """
    Nakanishi-Fujii-Todo (NFT) sequential coordinate optimizer.

    For each parameter, the cost as a function of that single parameter has the
    form A + B cos(theta + C). We sample at three points {theta, theta +/- pi/2}
    and solve for the minimum.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x0, dtype=float).copy()
    n_params = len(x)
    history = []

    f_current = cost_fn(x)
    history.append(float(f_current))
    if callback:
        callback(0, x, f_current)

    for epoch in range(n_epochs):
        order = rng.permutation(n_params)
        for i in order:
            x_plus = x.copy()
            x_plus[i] = x[i] + np.pi / 2
            x_minus = x.copy()
            x_minus[i] = x[i] - np.pi / 2
            if batch_cost_fn is not None:
                f_curr, f_plus, f_minus = batch_cost_fn([x, x_plus, x_minus])
            else:
                f_curr = cost_fn(x)
                f_plus = cost_fn(x_plus)
                f_minus = cost_fn(x_minus)

            b_sin_c = (f_minus - f_plus) / 2.0
            b_cos_c = f_curr - (f_plus + f_minus) / 2.0
            c = np.arctan2(b_sin_c, b_cos_c)
            b = np.hypot(b_sin_c, b_cos_c)
            if b < 1e-10:
                continue
            phi_star = np.pi - c
            x[i] = (x[i] + phi_star + np.pi) % (2 * np.pi) - np.pi

        f_current = cost_fn(x)
        history.append(float(f_current))
        if callback:
            callback(epoch + 1, x, f_current)

    return x, history


# ---------- Local search post-processing ----------


def local_search(x_start, cost_fn, rng=None):
    """Single-flip local search: stop when no flip improves the cost."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x_start, dtype=np.int8).copy()
    f = cost_fn(x)
    improved = True
    while improved:
        improved = False
        order = rng.permutation(len(x))
        for i in order:
            x_try = x.copy()
            x_try[i] = 1 - x_try[i]
            f_try = cost_fn(x_try)
            if f_try < f - 1e-12:
                x = x_try
                f = f_try
                improved = True
                break
    return x, f


if __name__ == "__main__":
    try:
        qc = twolocal_ansatz(8, reps=2, ent_map="bilinear")
        print(f"TwoLocal bilinear: {qc.num_parameters} params, depth {qc.decompose().depth()}")
        qc = twolocal_ansatz(8, reps=2, ent_map="colored")
        print(f"TwoLocal colored:  {qc.num_parameters} params, depth {qc.decompose().depth()}")
        qc = bfcd_ansatz(8, reps=2, ent_map="bilinear")
        print(f"BFCD bilinear:     {qc.num_parameters} params, depth {qc.decompose().depth()}")
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        print(exc)
