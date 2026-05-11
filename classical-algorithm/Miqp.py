"""Mixed-Integer Quadratic Program (MIQP) solver for portfolio optimization.

Converted from MIQP.ipynb to the benchmark_protocol solver contract.

MIQP formulation
----------------
The portfolio selection problem is solved directly as a MIQP rather than
being relaxed to a QUBO first.  This lets us hand the problem to a
branch-and-bound MIP solver (scipy / PuLP / HiGHS) that handles the
continuous weight variables and binary selection variables simultaneously:

    min_{x in {0,1}^N, w in R^N}
        q * w^T Sigma w  -  mu^T w

    subject to:
        sum_i x_i  = K              (cardinality)
        sum_i w_i  = 1              (weights sum to 1)
        w_min * x_i <= w_i <= w_max * x_i   (weights only on selected assets)
        w_i >= 0

Because scipy.optimize.milp / linprog cannot handle a QP objective directly,
we use one of two back-ends (tried in order):

  1. cvxpy  (preferred) — concise quadratic MIP formulation via ECOS_BB / GLPK_MI
  2. pulp   — LP relaxation fallback (approximates the QP with outer linearisation)

The QUBO objective value is also recorded post-hoc so results sit on the
same leaderboard axis as the other solvers.

Install (one of):
    pip install cvxpy          # recommended
    pip install pulp           # fallback

Place this file in:
    classical-algorithm/miqp.py
"""
from __future__ import annotations

import sys
import time
import warnings
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

ALGORITHM = "miqp"
ALGORITHM_VERSION = "1.0.0"

W_MIN = 0.05   # minimum weight per selected asset
W_MAX = 0.60   # maximum weight per selected asset


# ── back-end: cvxpy ────────────────────────────────────────────────────────

def _solve_cvxpy(
    mu: np.ndarray,
    sigma: np.ndarray,
    K: int,
    q: float,
    w_min: float,
    w_max: float,
    time_limit: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (x_binary, w_continuous, obj_val).
    Solves the MIQP exactly via CVXPY + ECOS_BB (or SCIP/GLPK_MI).
    """
    import cvxpy as cp

    N = len(mu)
    x = cp.Variable(N, boolean=True)   # binary: select asset i?
    w = cp.Variable(N, nonneg=True)    # continuous: weight of asset i

    objective = cp.Minimize(q * cp.quad_form(w, sigma) - mu @ w)
    constraints = [
        cp.sum(x) == K,
        cp.sum(w) == 1,
        w >= w_min * x,
        w <= w_max * x,
    ]

    prob = cp.Problem(objective, constraints)

    # Try solvers in preference order
    for solver in [cp.ECOS_BB, cp.SCIP, cp.GLPK_MI]:
        try:
            prob.solve(
                solver=solver,
                verbose=False,
                **({ "time_limit": time_limit } if solver == cp.SCIP else {}),
            )
            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                x_val = np.round(x.value).astype(int)
                w_val = np.clip(w.value, 0.0, 1.0)
                w_val /= max(w_val.sum(), 1e-12)
                return x_val, w_val, float(prob.value)
        except Exception:
            continue

    raise RuntimeError("cvxpy: no solver succeeded")


# ── back-end: pulp (LP relaxation fallback) ────────────────────────────────

def _solve_pulp(
    mu: np.ndarray,
    sigma: np.ndarray,
    K: int,
    q: float,
    w_min: float,
    w_max: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    LP relaxation of the MIQP using PuLP + CBC.
    The quadratic risk term is approximated by its diagonal (variance-only),
    which makes it a linear program. Selection is done by first solving for
    weights, then picking the K assets with the largest weights.

    This is a fallback when cvxpy is unavailable; accuracy is lower.
    """
    import pulp

    N = len(mu)
    prob = pulp.LpProblem("MIQP_LP_relax", pulp.LpMinimize)

    w = [pulp.LpVariable(f"w_{i}", lowBound=0.0, upBound=w_max) for i in range(N)]
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(N)]

    # Objective: approximate risk as sum q*sigma_ii*w_i^2 → linearise as q*sigma_ii*w_i
    # (diagonal approximation — ignores covariance terms)
    diag_var = np.diag(sigma)
    prob += pulp.lpSum(
        q * diag_var[i] * w[i] - mu[i] * w[i] for i in range(N)
    )

    # Constraints
    prob += pulp.lpSum(x) == K
    prob += pulp.lpSum(w) == 1
    for i in range(N):
        prob += w[i] >= w_min * x[i]
        prob += w[i] <= w_max * x[i]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

    x_val = np.array([round(pulp.value(x[i])) for i in range(N)], dtype=int)
    w_val = np.array([max(pulp.value(w[i]), 0.0) for i in range(N)])
    w_val /= max(w_val.sum(), 1e-12)
    obj   = float(pulp.value(prob.objective)) if prob.objective else 0.0
    return x_val, w_val, obj


# ── main solver ────────────────────────────────────────────────────────────

def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    q: float | None = None,
    w_min: float = W_MIN,
    w_max: float = W_MAX,
    time_limit: float = 120.0,
) -> BenchmarkResult:
    """
    Parameters
    ----------
    inst       : benchmark problem instance
    seed       : kept for API consistency (MIQP is deterministic)
    q          : risk aversion override (defaults to inst.q)
    w_min      : minimum weight for any selected asset
    w_max      : maximum weight for any selected asset
    time_limit : wall-clock limit passed to MIQP solver (seconds)
    """
    Q   = build_qubo_from_inst(inst)
    mu  = np.asarray(inst.mu, dtype=float)
    sig = np.asarray(inst.sigma, dtype=float)
    N, K = inst.N, inst.K
    q_eff = float(inst.q) if q is None else float(q)

    t0 = time.perf_counter()
    backend_used = "unknown"
    w_continuous = np.zeros(N)

    # ── Try cvxpy first, then pulp ─────────────────────────────────────────
    x_bin: np.ndarray | None = None
    miqp_obj = float("nan")

    try:
        import cvxpy  # noqa: F401
        x_bin, w_cont, miqp_obj = _solve_cvxpy(mu, sig, K, q_eff, w_min, w_max, time_limit)
        w_continuous = w_cont
        backend_used = "cvxpy"
    except Exception as e_cvx:
        try:
            import pulp  # noqa: F401
            x_bin, w_cont, miqp_obj = _solve_pulp(mu, sig, K, q_eff, w_min, w_max)
            w_continuous = w_cont
            backend_used = "pulp_lp_relax"
        except Exception as e_pulp:
            raise RuntimeError(
                f"MIQP: both solvers failed.\n"
                f"  cvxpy error : {e_cvx}\n"
                f"  pulp  error : {e_pulp}\n"
                "Install at least one: pip install cvxpy  OR  pip install pulp"
            ) from e_pulp

    # Repair feasibility (rounding may change cardinality)
    if x_bin.sum() != K:
        # Keep top-K by weight
        top_k = np.argsort(w_continuous)[::-1][:K]
        x_bin = np.zeros(N, dtype=int)
        x_bin[top_k] = 1

    elapsed = time.perf_counter() - t0

    # QUBO objective (for leaderboard comparison)
    best_cost = qubo_eval(Q, x_bin)

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q, K)
        ar = approx_ratio(best_cost, c_opt, c_wst)
    else:
        ar = None

    selected = [i for i, b in enumerate(x_bin) if b == 1]
    mu_S    = mu[selected]
    sigma_S = sig[np.ix_(selected, selected)]
    w_S     = w_continuous[selected]
    if w_S.sum() > 1e-12:
        w_S /= w_S.sum()
    else:
        w_S = np.full(K, 1.0 / K)

    port_ret = float(mu_S @ w_S)
    port_vol = float(np.sqrt(max(float(w_S @ sigma_S @ w_S), 1e-12)))
    sharpe   = (port_ret - 0.05) / port_vol if port_vol > 1e-12 else 0.0

    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=float(best_cost),
        bitstring=x_bin.tolist(),
        feasible=bool(int(x_bin.sum()) == K),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed),
        optimizer_iters=None,
        num_circuit_evaluations=None,
        convergence_history=[float(best_cost)],
        qubit_count=None,
        circuit_depth=None,
        two_qubit_gate_count=None,
        t_gate_count=None,
        total_gate_count=None,
        shots=None,
        backend=backend_used,
        transpile_optimization_level=None,
        hyperparameters={
            "problem_variant": "miqp",
            "method": "mixed_integer_quadratic_program",
            "backend": backend_used,
            "q": q_eff,
            "w_min": w_min,
            "w_max": w_max,
            "time_limit": time_limit,
        },
        initial_params=None,
        final_params={
            "selected_indices": selected,
            "weights": w_S.tolist(),
            "portfolio_return": port_ret,
            "portfolio_volatility": port_vol,
            "sharpe_ratio": sharpe,
            "miqp_objective": float(miqp_obj),
        },
    )
