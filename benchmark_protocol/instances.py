"""Canonical frozen instance set. JSON-based, no pickle dependency.

Each instance is one JSON file in data/instances/. Open one in any text
editor to see exactly what's in it. To load programmatically:

    from benchmark_protocol import instances
    inst = instances.load("syn_tiny_0000")
    inst.mu, inst.sigma, inst.K, inst.q   # ready to feed a solver
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INSTANCE_DIR = Path(__file__).resolve().parent.parent / "data" / "instances"

BUCKETS: dict[str, list[str]] = {
    "tiny":    [f"syn_tiny_{i:04d}"   for i in range(0, 15)],
    "small":   [f"syn_small_{i:04d}"  for i in range(15, 30)],
    "medium":  [f"syn_medium_{i:04d}" for i in range(30, 42)],
    "large":   [f"syn_large_{i:04d}"  for i in range(42, 52)],
    "n7_gap":  [f"syn_n7_gap_{i:04d}" for i in range(0, 5)],
}

SUBSETS = {
    "matched_quantum_classical": BUCKETS["tiny"] + BUCKETS["small"] + BUCKETS["n7_gap"],
    "classical_only_extension":  BUCKETS["medium"] + BUCKETS["large"],
    "all":                       sum(BUCKETS.values(), []),
}


@dataclass
class ProblemInstance:
    instance_id: str
    N: int                      # candidate assets
    K: int                      # cardinality (assets to select)
    q: float                    # risk aversion
    mu: np.ndarray              # (N,) expected returns
    sigma: np.ndarray           # (N, N) covariance
    asset_tickers: list[str]
    date_range: tuple[str, str]
    prev_selection: list[int] | None = None


def all_ids() -> list[str]:
    return SUBSETS["all"]


def load(instance_id: str) -> ProblemInstance:
    path = INSTANCE_DIR / f"{instance_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"instance {instance_id} not found at {path}")
    with open(path) as f:
        d = json.load(f)
    return ProblemInstance(
        instance_id=d["instance_id"],
        N=d["N"],
        K=d["K"],
        q=d["q"],
        mu=np.asarray(d["mu"], dtype=float),
        sigma=np.asarray(d["sigma"], dtype=float),
        asset_tickers=list(d["asset_tickers"]),
        date_range=tuple(d["date_range"]),
        prev_selection=d.get("prev_selection"),
    )


def bucket_of(instance_id: str) -> str:
    for bucket, ids in BUCKETS.items():
        if instance_id in ids:
            return bucket
    raise ValueError(f"unknown instance_id: {instance_id}")
