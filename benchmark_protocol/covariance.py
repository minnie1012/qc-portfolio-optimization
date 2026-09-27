"""Covariance estimators for the portfolio pipeline.

    estimate_cov(returns, method="sample")       -> (annualized sample covariance, None)
    estimate_cov(returns, method="ledoit_wolf")  -> (annualized LW covariance, shrinkage delta)

Ledoit-Wolf uses sklearn.covariance.LedoitWolf, which shrinks the sample
covariance toward a scaled identity target mu*I (mu = mean sample variance):

    Sigma_LW = (1 - delta) * S + delta * mu * I,   0 <= delta <= 1

Note sklearn's S is the maximum-likelihood (ddof=0) covariance, whereas the
"sample" method uses pandas' unbiased (ddof=1) estimator so that existing
results that call `returns.cov()` are reproduced exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("sample", "ledoit_wolf")


def estimate_cov(
    returns: pd.DataFrame,
    method: str = "sample",
    annualize: int = 252,
) -> tuple[np.ndarray, float | None]:
    """Annualized covariance of daily returns.

    returns   : (T, N) daily returns, one column per asset, no NaNs.
    method    : "sample" or "ledoit_wolf".
    annualize : periods per year used to scale the daily covariance.

    Returns (Sigma, delta) where delta is the Ledoit-Wolf shrinkage intensity,
    or None for the sample estimator.
    """
    if method == "sample":
        return returns.cov().to_numpy() * annualize, None
    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(np.asarray(returns, dtype=float))
        return lw.covariance_ * annualize, float(lw.shrinkage_)
    raise ValueError(f"unknown covariance method: {method!r} (use one of {METHODS})")
