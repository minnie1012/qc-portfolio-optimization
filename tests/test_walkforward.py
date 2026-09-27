"""Tests for the exact K-subset QUBO solver used by the walk-forward backtest."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from problem_definition import brute_force_select, build_qubo, synthetic_instance  # noqa: E402
from walkforward_equity_backtest import exact_k_subset_select  # noqa: E402


@pytest.mark.parametrize("seed", range(5))
def test_exact_k_subset_solver_matches_brute_force(seed):
    inst = synthetic_instance(N=8, K=3, seed=seed)
    Q = build_qubo(inst)
    x_bf, c_bf = brute_force_select(Q, K=3)
    x_ex, c_ex = exact_k_subset_select(Q, K=3)
    np.testing.assert_array_equal(x_bf, x_ex)
    assert c_ex == pytest.approx(c_bf, rel=1e-12, abs=1e-12)
    assert x_ex.sum() == 3
