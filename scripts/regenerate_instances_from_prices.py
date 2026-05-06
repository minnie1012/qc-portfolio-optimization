"""Replace data/instances/*.json with real-price-derived data.

For every existing JSON file we keep N, K, q, instance_id and prev_selection
unchanged, but replace mu, sigma, asset_tickers, date_range with values
computed from the in-sample window of the appropriate prices CSV.

Selection rules:
    N <= 25  -> sample from the 25-stock QUBO universe (prices_daily.csv)
    N >  25  -> sample from the 50-stock MIQP universe (prices_miqp_daily.csv)

The sample is deterministic via a per-instance seed derived from the id, so
the output is reproducible.

The original synthetic JSONs are moved to data/_synthetic_backup/ first.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark_protocol import prices  # noqa: E402

INSTANCE_DIR = ROOT / "data" / "instances"
BACKUP_DIR = ROOT / "data" / "_synthetic_backup"

WINDOW = ("2022-01-01", "2024-12-31")
TRADING_DAYS = 252


def seed_for(instance_id: str) -> int:
    return int(hashlib.sha1(instance_id.encode()).hexdigest()[:8], 16)


def annualize(rets):
    mu = rets.mean().to_numpy() * TRADING_DAYS
    sigma = rets.cov().to_numpy() * TRADING_DAYS
    return mu, sigma


def main() -> None:
    # 1. backup current JSONs (only on first run)
    BACKUP_DIR.mkdir(exist_ok=True)
    for src in sorted(INSTANCE_DIR.glob("*.json")):
        dst = BACKUP_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"backed up originals to {BACKUP_DIR} ({len(list(BACKUP_DIR.glob('*.json')))} files)")

    # 2. preload both universes once
    rets_qubo = prices.load_returns(start=WINDOW[0], end=WINDOW[1], universe="qubo")
    rets_miqp = prices.load_returns(start=WINDOW[0], end=WINDOW[1], universe="miqp")
    tickers_qubo = list(rets_qubo.columns)
    tickers_miqp = list(rets_miqp.columns)

    # 3. rewrite each instance
    n_done = 0
    for path in sorted(INSTANCE_DIR.glob("*.json")):
        with open(path) as f:
            d = json.load(f)
        N, K, q = d["N"], d["K"], d["q"]

        if N <= len(tickers_qubo):
            universe = "qubo"
            full_tickers = tickers_qubo
            full_rets = rets_qubo
        else:
            universe = "miqp"
            full_tickers = tickers_miqp
            full_rets = rets_miqp

        rng = np.random.default_rng(seed_for(d["instance_id"]))
        idx = rng.choice(len(full_tickers), size=N, replace=False)
        idx.sort()
        chosen_tickers = [full_tickers[i] for i in idx]
        sub_rets = full_rets[chosen_tickers]
        mu, sigma = annualize(sub_rets)

        new = {
            "instance_id": d["instance_id"],
            "N": int(N),
            "K": int(K),
            "q": float(q),
            "asset_tickers": chosen_tickers,
            "date_range": list(WINDOW),
            "prev_selection": d.get("prev_selection"),
            "mu": [float(x) for x in mu],
            "sigma": [[float(x) for x in row] for row in sigma],
            "_source": {"universe": universe, "window": list(WINDOW)},
        }

        with open(path, "w") as f:
            json.dump(new, f, indent=2)
        n_done += 1

    print(f"rewrote {n_done} instance files in {INSTANCE_DIR}")


if __name__ == "__main__":
    main()
