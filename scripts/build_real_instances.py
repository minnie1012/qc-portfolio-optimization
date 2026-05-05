"""Build real-price-derived ProblemInstances from data/prices/*.csv.

Creates four ablation instances (prefix `real_`):

    real_qubo_full      N=25  K=8   uses prices_daily.csv      (classical-only)
    real_qubo_q7        N=7   K=3   first 7 of QUBO universe   (matched)
    real_miqp_full      N=50  K=10  uses prices_miqp_daily.csv (classical-only)
    real_miqp_q7        N=7   K=3   first 7 of MIQP universe   (matched)

Annualized mu and sigma are computed from the in-sample window
(2022-01-01..2024-12-31). q is set to 1.0 for stability.

Files are saved under data/instances_real/ (not the frozen set).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark_protocol import prices  # noqa: E402

OUT = ROOT / "data" / "instances_real"
TRADING_DAYS = 252
WINDOW = ("2022-01-01", "2024-12-31")


def annualize(returns) -> tuple[np.ndarray, np.ndarray]:
    mu = returns.mean().to_numpy() * TRADING_DAYS
    sigma = returns.cov().to_numpy() * TRADING_DAYS
    return mu, sigma


def make_instance(instance_id: str, tickers, mu, sigma, K, q=1.0):
    return {
        "instance_id": instance_id,
        "N": len(tickers),
        "K": int(K),
        "q": float(q),
        "asset_tickers": list(tickers),
        "date_range": list(WINDOW),
        "prev_selection": None,
        "mu": [float(x) for x in mu],
        "sigma": [[float(x) for x in row] for row in sigma],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    todo = []

    for universe in ("qubo", "miqp"):
        rets = prices.load_returns(start=WINDOW[0], end=WINDOW[1], universe=universe)
        mu, sigma = annualize(rets)
        tickers = list(rets.columns)
        N = len(tickers)
        K_full = 8 if universe == "qubo" else 10

        full_id = f"real_{universe}_full"
        todo.append((full_id, make_instance(full_id, tickers, mu, sigma, K_full)))

        sub = list(range(7))
        q7_id = f"real_{universe}_q7"
        todo.append((q7_id, make_instance(
            q7_id, [tickers[i] for i in sub], mu[sub],
            sigma[np.ix_(sub, sub)], K=3,
        )))

    for iid, payload in todo:
        path = OUT / f"{iid}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {path}  N={payload['N']} K={payload['K']}")


if __name__ == "__main__":
    main()
