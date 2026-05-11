"""Tabu Search on K-cardinality bitstrings.

Neighbour = single swap (one stock in, one stock out).
A tabu list blocks recently reversed moves for `tabu_tenure` iterations,
preventing the search from immediately cycling back to where it came from.
This lets it escape local minima that trap pure greedy descent.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol import instances          # noqa: E402
from benchmark_protocol.result_schema import BenchmarkResult  # noqa: E402
from solver_common import (                       # noqa: E402
    approx_ratio,
    build_qubo_from_inst,
    exact_reference,
    qubo_eval,
)

ALGORITHM = "tabu_search"
ALGORITHM_VERSION = "1.0.0"


def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    n_iters: int = 2000,
    tabu_tenure: int = 10,
    n_candidates: int = 20,
) -> BenchmarkResult:
    """
    Parameters
    ----------
    inst          : problem instance (mu, sigma, K, q, …)
    seed          : RNG seed for reproducibility
    n_iters       : total number of swap iterations
    tabu_tenure   : how many iterations a (i_out, i_in) swap stays forbidden
    n_candidates  : how many random candidate swaps to evaluate per iteration
                    (full neighbourhood = N*(N-K) can be huge; sampling keeps
                    it fast while preserving solution quality)
    """
    Q = build_qubo_from_inst(inst)
    N, K = inst.N, inst.K
    rng = np.random.default_rng(seed)

    # ── Initialise with a random K-subset ──────────────────────────────────
    sel = rng.choice(N, size=K, replace=False).tolist()
    x = np.zeros(N, dtype=int)
    x[sel] = 1
    cost = qubo_eval(Q, x)

    best_x, best_cost = x.copy(), float(cost)
    history: list[float] = [best_cost]

    # tabu_list stores (i_out, i_in) moves that are currently forbidden.
    # We use a deque as a fixed-length queue; each element is (i_out, i_in).
    tabu_list: deque[tuple[int, int]] = deque()
    tabu_set: set[tuple[int, int]] = set()

    t0 = time.perf_counter()

    for _it in range(n_iters):
        on  = np.flatnonzero(x == 1)
        off = np.flatnonzero(x == 0)

        # Sample up to n_candidates (i_out, i_in) pairs
        n_cand = min(n_candidates, len(on) * len(off))
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        tries = 0
        while len(pairs) < n_cand and tries < n_cand * 4:
            io = int(rng.choice(on))
            ii = int(rng.choice(off))
            if (io, ii) not in seen:
                seen.add((io, ii))
                pairs.append((io, ii))
            tries += 1

        # Evaluate each candidate; pick the best non-tabu move
        # (aspiration: accept tabu move if it beats global best)
        best_move: tuple[int, int] | None = None
        best_move_cost = float("inf")

        for io, ii in pairs:
            x_new = x.copy()
            x_new[io] = 0
            x_new[ii] = 1
            c_new = qubo_eval(Q, x_new)

            is_tabu = (io, ii) in tabu_set
            aspiration = c_new < best_cost  # override tabu if new global best

            if (not is_tabu or aspiration) and c_new < best_move_cost:
                best_move_cost = c_new
                best_move = (io, ii)

        if best_move is None:
            # All candidates tabu and none beat global best → take least bad
            best_move = pairs[0]
            best_move_cost = qubo_eval(
                Q, (lambda xc: (xc.__setitem__(best_move[0], 0) or
                                xc.__setitem__(best_move[1], 1) or xc))(x.copy())
            )

        io, ii = best_move
        x[io] = 0
        x[ii] = 1
        cost = best_move_cost

        # Update global best
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()

        # Update tabu list (the reverse move is now forbidden)
        reverse = (ii, io)
        tabu_set.add(reverse)
        tabu_list.append(reverse)
        if len(tabu_list) > tabu_tenure:
            expired = tabu_list.popleft()
            tabu_set.discard(expired)

        history.append(best_cost)

    elapsed = time.perf_counter() - t0

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(best_cost, c_opt, c_wst)
    else:
        ar = None

    stride = max(1, len(history) // 200)
    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=float(best_cost),
        bitstring=best_x.tolist(),
        feasible=bool(int(best_x.sum()) == K),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed),
        optimizer_iters=n_iters,
        num_circuit_evaluations=None,
        convergence_history=[float(c) for c in history[::stride]],
        qubit_count=None,
        circuit_depth=None,
        two_qubit_gate_count=None,
        t_gate_count=None,
        total_gate_count=None,
        shots=None,
        backend=None,
        transpile_optimization_level=None,
        hyperparameters={
            "problem_variant": "qubo",
            "method": "tabu_search",
            "n_iters": n_iters,
            "tabu_tenure": tabu_tenure,
            "n_candidates": n_candidates,
        },
        initial_params=None,
        final_params=None,
    )
