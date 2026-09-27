"""Tests for benchmark_protocol.covariance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol.covariance import estimate_cov  # noqa: E402
from benchmark_protocol.prices import load_returns  # noqa: E402


@pytest.fixture(scope="module")
def real_returns():
    return load_returns(universe="qubo")


def _windows(rets):
    return {"full": rets, "last_252": rets.iloc[-252:]}


@pytest.mark.parametrize("window", ["full", "last_252"])
def test_ledoit_wolf_symmetric_psd_and_delta_in_unit_interval(real_returns, window):
    rets = _windows(real_returns)[window]
    sigma, delta = estimate_cov(rets, method="ledoit_wolf")
    assert sigma.shape == (rets.shape[1], rets.shape[1])
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-14)
    assert np.linalg.eigvalsh(sigma).min() >= -1e-12
    assert delta is not None and 0.0 <= delta <= 1.0


@pytest.mark.parametrize("window", ["full", "last_252"])
def test_ledoit_wolf_condition_number_not_worse_than_sample(real_returns, window):
    rets = _windows(real_returns)[window]
    s_lw, _ = estimate_cov(rets, method="ledoit_wolf")
    s_sample, delta = estimate_cov(rets, method="sample")
    assert delta is None
    assert np.linalg.cond(s_lw) <= np.linalg.cond(s_sample)


def test_sample_matches_pandas_cov(real_returns):
    rets = real_returns.iloc[:300]
    sigma, _ = estimate_cov(rets, method="sample", annualize=252)
    np.testing.assert_array_equal(sigma, rets.cov().to_numpy() * 252)


def test_unknown_method_raises(real_returns):
    with pytest.raises(ValueError):
        estimate_cov(real_returns, method="shrinkage")

