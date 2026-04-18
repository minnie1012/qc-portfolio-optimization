"""Two-stage quantum portfolio optimization: problem definition.

Stage 1 (quantum) -- asset selection as QUBO / Ising:
    min_{x in {0,1}^N}  q * x^T Sigma x  -  mu^T x
                        + lambda * (sum_i x_i - K)^2
                        - delta * sum_i x_i^prev x_i

    Ising mapping: x_i = (1 - Z_i)/2
        H_C = offset + sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j

Stage 2 (classical) -- weight allocation on selected subset S:
    max_w  (mu_S^T w - R_f) / sqrt(w^T Sigma_S w)
    s.t.   sum(w) = 1,  w_min <= w_i <= w_max
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

@dataclass
class ProblemInstance:
    mu: np.ndarray                 # (N,) annualized expected returns
    sigma: np.ndarray              # (N, N) annualized covariance
    K: int                         # cardinality (assets to select)
    q: float = 1.0                 # risk aversion
    prev_selection: list[int] | None = None  # indices selected last period
    tickers: list[str] = field(default_factory=list)

    @property
    def N(self) -> int:
        return len(self.mu)


# ---------------------------------------------------------------------------
# Stage 1a: QUBO builder
# ---------------------------------------------------------------------------

def adaptive_lambda(inst: ProblemInstance) -> float:
    """Stopfer & Wagner 2025 heuristic: penalty ~ 1.5x objective scale."""
    obj_scale = abs(inst.q) * np.trace(inst.sigma) / inst.N + np.max(np.abs(inst.mu))
    return float(max(1.5 * obj_scale, 1e-6))


def build_qubo(
    inst: ProblemInstance,
    lam: float | None = None,
    delta: float = 0.0,
) -> np.ndarray:
    """Build N x N upper-triangular QUBO matrix Q such that cost = x^T Q x."""
    N, mu, sigma, K, q = inst.N, inst.mu, inst.sigma, inst.K, inst.q

    Q_obj = q * sigma.copy()
    np.fill_diagonal(Q_obj, Q_obj.diagonal() - mu)

    lam_eff = adaptive_lambda(inst) if lam is None else lam
    Q_pen = lam_eff * np.ones((N, N))
    np.fill_diagonal(Q_pen, Q_pen.diagonal() - 2 * K * lam_eff)

    Q_cont = np.zeros((N, N))
    if delta > 0 and inst.prev_selection:
        for i in inst.prev_selection:
            if 0 <= i < N:
                Q_cont[i, i] -= delta

    Q = Q_obj + Q_pen + Q_cont
    Q = (Q + Q.T) / 2
    Q = np.triu(Q, k=1) * 2 + np.diag(np.diag(Q))
    return Q


def eval_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x @ Q @ x)


# ---------------------------------------------------------------------------
# Stage 1b: Ising mapping  x_i = (1 - Z_i)/2
# ---------------------------------------------------------------------------

def qubo_to_ising(Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (offset, h, J) so that H_C = offset + sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j."""
    N = Q.shape[0]
    Q_sym = (Q + Q.T) / 2
    np.fill_diagonal(Q_sym, np.diag(Q))

    h = np.zeros(N)
    J = np.zeros((N, N))
    offset = 0.0

    for i in range(N):
        offset += 0.5 * Q_sym[i, i]
        h[i] -= 0.5 * Q_sym[i, i]

    for i in range(N):
        for j in range(i + 1, N):
            qij = Q_sym[i, j] + Q_sym[j, i]
            offset += 0.25 * qij
            h[i] -= 0.25 * qij
            h[j] -= 0.25 * qij
            J[i, j] += 0.25 * qij

    return offset, h, J


# ---------------------------------------------------------------------------
# Stage 2: Classical weight allocation
# ---------------------------------------------------------------------------

def optimize_sharpe(
    mu_S: np.ndarray,
    sigma_S: np.ndarray,
    risk_free_rate: float = 0.05,
    w_min: float = 0.05,
    w_max: float = 0.50,
    n_starts: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Maximum-Sharpe allocation via SLSQP with multiple random starts."""
    K = len(mu_S)
    if K == 0:
        return np.array([]), 0.0
    if K == 1:
        return np.array([1.0]), float((mu_S[0] - risk_free_rate) / np.sqrt(sigma_S[0, 0]))

    def neg_sharpe(w):
        vol = np.sqrt(w @ sigma_S @ w)
        if vol < 1e-12:
            return 1e10
        return -(mu_S @ w - risk_free_rate) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(w_min, w_max)] * K
    rng = np.random.default_rng(seed)

    best_w, best_val = None, np.inf
    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(K))
        w0 = np.clip(w0, w_min, w_max)
        w0 /= w0.sum()
        res = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res.fun < best_val:
            best_val, best_w = res.fun, res.x
    return best_w, -best_val


def allocate(
    inst: ProblemInstance,
    selected: list[int],
    risk_free_rate: float = 0.05,
) -> dict:
    sel = np.asarray(selected)
    mu_S = inst.mu[sel]
    sigma_S = inst.sigma[np.ix_(sel, sel)]
    w, sharpe = optimize_sharpe(mu_S, sigma_S, risk_free_rate=risk_free_rate)
    port_ret = float(mu_S @ w)
    port_vol = float(np.sqrt(w @ sigma_S @ w))
    return {
        "selected": sel.tolist(),
        "tickers": [inst.tickers[i] for i in sel] if inst.tickers else None,
        "weights": w.tolist(),
        "portfolio_return": port_ret,
        "portfolio_volatility": port_vol,
        "sharpe_ratio": sharpe,
    }


# ---------------------------------------------------------------------------
# Brute-force reference solver (validation for small N)
# ---------------------------------------------------------------------------

def brute_force_select(Q: np.ndarray, K: int | None = None) -> tuple[np.ndarray, float]:
    """Enumerate all 2^N bitstrings; return (x*, cost*). Optionally require sum(x)=K."""
    N = Q.shape[0]
    best_x, best_cost = None, np.inf
    for bits in product([0, 1], repeat=N):
        x = np.array(bits)
        if K is not None and x.sum() != K:
            continue
        c = eval_qubo(Q, x)
        if c < best_cost:
            best_cost, best_x = c, x
    return best_x, best_cost


# ---------------------------------------------------------------------------
# Synthetic data generator (Ledoit-Wolf-style PSD covariance)
# ---------------------------------------------------------------------------

def synthetic_instance(N: int, K: int, seed: int = 42) -> ProblemInstance:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.10, 0.05, size=N)          # annualized returns
    A = rng.normal(0, 1, size=(N, N))
    sigma = (A @ A.T) / N * 0.04 + np.eye(N) * 0.02  # PSD, ~20% vol
    tickers = [f"A{i}" for i in range(N)]
    return ProblemInstance(mu=mu, sigma=sigma, K=K, q=1.0, tickers=tickers)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo(N: int = 6, K: int = 2) -> dict:
    inst = synthetic_instance(N=N, K=K, seed=42)

    Q = build_qubo(inst)
    offset, h, J = qubo_to_ising(Q)
    x_star, cost_star = brute_force_select(Q)
    selected = [i for i, b in enumerate(x_star) if b == 1]
    alloc = allocate(inst, selected)

    print(f"Problem: N={N}, K={K}, q={inst.q}")
    print(f"lambda  = {adaptive_lambda(inst):.4f}")
    print(f"mu      = {np.round(inst.mu, 4)}")
    print(f"\nQUBO (rounded):\n{np.round(Q, 3)}")
    print(f"\nIsing offset = {offset:.4f}")
    print(f"Ising h      = {np.round(h, 4)}")
    print(f"\nBrute-force optimum:")
    print(f"  x*       = {x_star.tolist()}")
    print(f"  cost     = {cost_star:.4f}")
    print(f"  selected = {selected}")
    print(f"\nStage-2 allocation:")
    print(f"  tickers  = {alloc['tickers']}")
    print(f"  weights  = {[round(w, 3) for w in alloc['weights']]}")
    print(f"  return   = {alloc['portfolio_return']:.4f}")
    print(f"  vol      = {alloc['portfolio_volatility']:.4f}")
    print(f"  Sharpe   = {alloc['sharpe_ratio']:.4f}")

    return {
        "instance": {"N": N, "K": K, "q": inst.q, "mu": inst.mu.tolist()},
        "qubo": Q.tolist(),
        "ising": {"offset": offset, "h": h.tolist(), "J": J.tolist()},
        "optimum": {"x": x_star.tolist(), "cost": cost_star, "selected": selected},
        "allocation": alloc,
    }


if __name__ == "__main__":
    import json, os
    result = demo(N=6, K=2)
    out_path = os.path.join(os.path.dirname(__file__), "results", "problem_definition_demo.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved demo output to {out_path}")
