"""
Synthetic bond-ETF construction problem mirroring the structure in
Agliardi et al., "Portfolio construction using a sampling-based variational
quantum scheme" (arXiv:2508.13557).

The paper uses proprietary Vanguard data, so we generate a structurally
faithful synthetic instance:
  - N bonds split across D dimensions of clusters (e.g. risk rating, sector)
  - For each (cluster, metric) we have a target tau and weights w_{i,D,j}
  - Objective: minimize sum_{D,d,j} (tau_{d,j} - sum_{i in d} w_{i,D,j} delta_i x_i)^2
  - Linear constraints: cash budget (<=) and per-cluster guardrails (<=, >=)
  - Binary x_i: include bond i at fixed lot size c_i*delta_i, or not.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class BondETFProblem:
    """Holds Q, A, b for x^T Q x  s.t.  A x <= b, x in {0,1}^n."""
    Q: np.ndarray            # (n, n) symmetric
    A: np.ndarray            # (m, n)
    b: np.ndarray            # (m,)
    n: int                   # number of bonds (= qubits)
    constant: float          # constant term from expanding (tau - w·x)^2
    # Bookkeeping for diagnostics
    targets: list
    weights: list
    cluster_assignments: list

    def objective(self, x: np.ndarray) -> float:
        """Constrained QP objective (no penalty)."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.constant)

    def violation(self, x: np.ndarray) -> np.ndarray:
        """Per-constraint violation, max(0, A x - b)."""
        return np.maximum(0.0, self.A @ x - self.b)

    def penalized(self, x: np.ndarray, s: np.ndarray) -> float:
        """Eq. (4) of the paper:  x^T Q x + 1^T max{0, s ⊙ (A x - b)}."""
        return self.objective(x) + float(np.sum(s * self.violation(x)))


def build_problem(
    n_bonds: int = 16,
    n_dimensions: int = 3,
    n_metrics: int = 3,
    seed: int = 7,
    guardrail_tightness: float = 0.15,
) -> BondETFProblem:
    """
    Generate a synthetic instance with the same algebraic structure as the
    paper (Section II). For each dimension D, bonds are partitioned into
    clusters d. For each (D, d, j) triple we have a target tau_{D,d,j} and
    each bond carries a weight w_{i,D,j}. The objective penalizes squared
    deviation of cluster-aggregated weighted-holdings from targets.

    n_bonds=16 keeps things small enough to:
      (a) enumerate all 2^16 = 65k assignments and find the true optimum,
      (b) simulate the full quantum circuit without MPS truncation.
    """
    rng = np.random.default_rng(seed)

    # Bond properties
    delta = rng.uniform(0.8, 1.2, size=n_bonds)              # lot sizes
    c = np.ones(n_bonds)                                      # fixed multiplicities
    prices = rng.uniform(0.95, 1.05, size=n_bonds)            # normalized prices

    # Cluster assignments: each dimension partitions bonds into clusters
    cluster_assignments = []  # list over dims; each is array of cluster id per bond
    n_clusters_per_dim = []
    for D in range(n_dimensions):
        # 3-4 clusters per dimension
        n_clusters = rng.integers(3, 5)
        n_clusters_per_dim.append(int(n_clusters))
        cluster_assignments.append(rng.integers(0, n_clusters, size=n_bonds))

    # Weights w_{i,D,j} and targets tau_{D,d,j}
    weights = []   # weights[D][j] -> array of shape (n_bonds,)
    targets = []   # targets[D][j] -> array of shape (n_clusters_in_D,)
    for D in range(n_dimensions):
        weights.append([])
        targets.append([])
        for j in range(n_metrics):
            w_D_j = rng.uniform(0.5, 1.5, size=n_bonds)
            weights[D].append(w_D_j)
            # Set target as a fraction of what selecting ~half the bonds in
            # each cluster would yield - this gives a non-trivial optimum.
            n_clusters = n_clusters_per_dim[D]
            tau_D_j = np.zeros(n_clusters)
            for d in range(n_clusters):
                mask = cluster_assignments[D] == d
                if mask.sum() > 0:
                    full_sum = (w_D_j[mask] * delta[mask]).sum()
                    tau_D_j[d] = 0.5 * full_sum
            targets[D].append(tau_D_j)

    # Build Q matrix and constant from expanding sum_{D,d,j} (tau - w·delta·x)^2
    # = sum (tau^2 - 2 tau (w·delta·x) + (w·delta·x)^2)
    n = n_bonds
    Q = np.zeros((n, n))
    linear = np.zeros(n)
    constant = 0.0
    for D in range(n_dimensions):
        for j in range(n_metrics):
            for d in range(n_clusters_per_dim[D]):
                mask = cluster_assignments[D] == d
                if mask.sum() == 0:
                    continue
                tau = targets[D][j][d]
                v = weights[D][j] * delta * mask  # vector v_i nonzero only for i in cluster d
                # (tau - v·x)^2 = tau^2 - 2 tau (v·x) + (v·x)^2
                constant += tau * tau
                linear += -2.0 * tau * v
                Q += np.outer(v, v)

    # For binary x, x_i^2 = x_i so diagonal of (v·x)^2 contributes to linear too.
    # Fold linear into Q's diagonal: x^T Q x already gets diag terms from outer products.
    # Add 'linear' as additional diagonal contribution so the QP is x^T Q' x + const.
    Q = Q + np.diag(linear)
    Q = 0.5 * (Q + Q.T)  # symmetrize

    # Constraints A x <= b
    A_rows = []
    b_rows = []

    # 1) Cash budget: sum_i p_i delta_i x_i <= M
    cash_coeffs = prices * delta
    M = 0.55 * cash_coeffs.sum()  # restrictive enough to bind
    A_rows.append(cash_coeffs)
    b_rows.append(M)

    # 2) Per-cluster guardrails: a few (not all to keep things tractable)
    for D in range(n_dimensions):
        for j in range(n_metrics):
            for d in range(n_clusters_per_dim[D]):
                mask = cluster_assignments[D] == d
                if mask.sum() == 0:
                    continue
                v = weights[D][j] * delta * mask
                tau = targets[D][j][d]
                eps = guardrail_tightness * abs(tau) + 1e-3
                # v·x <= tau + eps
                A_rows.append(v.copy())
                b_rows.append(tau + eps)
                # -v·x <= -(tau - eps)  i.e.  v·x >= tau - eps
                A_rows.append(-v.copy())
                b_rows.append(-(tau - eps))

    A = np.array(A_rows)
    b = np.array(b_rows)

    return BondETFProblem(
        Q=Q, A=A, b=b, n=n, constant=constant,
        targets=targets, weights=weights,
        cluster_assignments=cluster_assignments,
    )


def brute_force_optimum(prob: BondETFProblem, s: np.ndarray):
    """Enumerate all 2^n bitstrings to find the true minimum of the
    *penalized* objective (Eq. 4). Tractable for n <= ~22."""
    n = prob.n
    if n > 22:
        raise ValueError(f"n={n} too large for brute force")
    best_val = np.inf
    best_x = None
    best_obj = None
    for k in range(1 << n):
        x = np.array([(k >> i) & 1 for i in range(n)])
        val = prob.penalized(x, s)
        if val < best_val:
            best_val = val
            best_x = x
            best_obj = prob.objective(x)
    return best_x, best_obj, best_val


if __name__ == "__main__":
    prob = build_problem(n_bonds=16, seed=7)
    print(f"n bonds: {prob.n}")
    print(f"Q shape: {prob.Q.shape}")
    print(f"# constraints: {prob.A.shape[0]}")
    print(f"constant term: {prob.constant:.4f}")

    # rescaling vector s for penalty - paper says "large enough"
    # We pick something proportional to typical Q magnitudes.
    typical_Q = np.abs(prob.Q).max()
    s = np.full(prob.A.shape[0], 100.0 * typical_Q)

    print("\nBrute-forcing optimum (2^16 = 65k bitstrings)...")
    x_opt, obj_opt, pen_opt = brute_force_optimum(prob, s)
    print(f"Optimal selection: {x_opt}")
    print(f"Optimal #selected: {x_opt.sum()}")
    print(f"Optimal objective (no penalty): {obj_opt:.6f}")
    print(f"Optimal penalized value:        {pen_opt:.6f}")
    print(f"Constraint violations at opt: {prob.violation(x_opt).sum():.6e}")
