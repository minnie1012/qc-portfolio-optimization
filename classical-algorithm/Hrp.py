"""Hierarchical Risk Parity (HRP) — Lopez de Prado 2016.

HRP selects K assets via agglomerative clustering on the correlation
matrix, then allocates weights with recursive bisection so that each
cluster contributes equally to total portfolio risk.

Because HRP doesn't have a cardinality constraint built in, we use a
two-step approach:
  1. Cluster the N assets with Ward linkage; cut the dendrogram to get
     exactly K clusters; pick the asset closest to each cluster centroid
     (in correlation space) as the cluster representative.
  2. Apply HRP-style inverse-variance weights over the K selected assets.
  3. Record the binary selection and QUBO objective for benchmarking.

Reference: Lopez de Prado, M. (2016). Building diversified portfolios that
outperform out of sample. Journal of Portfolio Management, 42(4), 59-69.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

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

ALGORITHM = "hrp"
ALGORITHM_VERSION = "1.0.0"


# ── HRP helpers ────────────────────────────────────────────────────────────

def _corr_to_dist(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix d_ij = sqrt(0.5*(1-rho_ij))."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return d


def _select_representatives(
    corr: np.ndarray,
    labels: np.ndarray,
    K: int,
) -> list[int]:
    """Pick one representative asset per cluster (closest to cluster centroid)."""
    selected: list[int] = []
    for k in range(1, K + 1):
        members = np.where(labels == k)[0]
        if len(members) == 1:
            selected.append(int(members[0]))
        else:
            # centroid in correlation-distance space
            sub_corr = corr[np.ix_(members, members)]
            avg_dist = (1.0 - sub_corr).mean(axis=1)   # mean distance to others
            rep = members[int(np.argmin(avg_dist))]
            selected.append(int(rep))
    return selected


def _hrp_weights(sigma_S: np.ndarray) -> np.ndarray:
    """Recursive bisection HRP weights on the K-asset sub-covariance."""
    K = sigma_S.shape[0]
    if K == 1:
        return np.array([1.0])

    # Build correlation matrix for clustering
    std = np.sqrt(np.diag(sigma_S))
    std = np.where(std < 1e-12, 1e-12, std)
    corr_S = sigma_S / np.outer(std, std)
    corr_S = np.clip(corr_S, -1.0, 1.0)
    np.fill_diagonal(corr_S, 1.0)

    dist = _corr_to_dist(corr_S)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")

    # Quasi-diagonalise: reorder assets by dendrogram leaf order
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(link)

    # Recursive bisection
    weights = np.ones(K)
    items = list(order)

    def _bisect(items: list[int]) -> None:
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]

        def _cluster_var(idx: list[int]) -> float:
            sub = sigma_S[np.ix_(idx, idx)]
            w_iv = 1.0 / np.diag(sub)
            w_iv /= w_iv.sum()
            return float(w_iv @ sub @ w_iv)

        v_l = _cluster_var(left)
        v_r = _cluster_var(right)
        total = v_l + v_r
        alpha = 1.0 - v_l / total if total > 1e-12 else 0.5

        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= (1.0 - alpha)

        _bisect(left)
        _bisect(right)

    _bisect(items)
    weights /= weights.sum()
    return weights


# ── main solver ────────────────────────────────────────────────────────────

def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
) -> BenchmarkResult:
    """
    1. Cluster N assets → K clusters via Ward linkage.
    2. Pick one representative per cluster.
    3. Compute HRP weights on the K selected assets.
    4. Record QUBO objective of the binary selection for benchmarking.
    """
    Q   = build_qubo_from_inst(inst)
    mu  = np.asarray(inst.mu)
    sig = np.asarray(inst.sigma)
    N, K = inst.N, inst.K

    t0 = time.perf_counter()

    # ── Step 1: correlation → distance → Ward linkage → K flat clusters ───
    std = np.sqrt(np.diag(sig))
    std = np.where(std < 1e-12, 1e-12, std)
    corr = sig / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    dist = _corr_to_dist(corr)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")
    labels = fcluster(link, t=K, criterion="maxclust")   # integer labels 1..K

    # ── Step 2: pick K representatives ────────────────────────────────────
    selected = _select_representatives(corr, labels, K)
    selected = sorted(set(selected))   # deduplicate (edge case: tiny N)
    # If dedup removed some, pad with highest-mu remaining assets
    if len(selected) < K:
        remaining = sorted(set(range(N)) - set(selected), key=lambda i: -mu[i])
        selected += remaining[: K - len(selected)]
    selected = sorted(selected[:K])

    # ── Step 3: HRP weights ───────────────────────────────────────────────
    sigma_S = sig[np.ix_(selected, selected)]
    weights = _hrp_weights(sigma_S)

    # ── Step 4: QUBO objective ────────────────────────────────────────────
    best_x = np.zeros(N, dtype=int)
    best_x[selected] = 1
    best_cost = qubo_eval(Q, best_x)

    elapsed = time.perf_counter() - t0

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(best_cost, c_opt, c_wst)
    else:
        ar = None

    mu_S   = mu[selected]
    port_ret = float(mu_S @ weights)
    port_vol = float(np.sqrt(weights @ sigma_S @ weights))
    sharpe   = (port_ret - 0.05) / port_vol if port_vol > 1e-12 else 0.0

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
        optimizer_iters=1,
        num_circuit_evaluations=None,
        convergence_history=[float(best_cost)],
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
            "method": "hrp_ward_bisection",
            "linkage": "ward",
            "cluster_criterion": "maxclust",
        },
        initial_params=None,
        final_params={
            "selected_indices": selected,
            "weights": weights.tolist(),
            "portfolio_return": port_ret,
            "portfolio_volatility": port_vol,
            "sharpe_ratio": sharpe,
        },
    )
