"""Backtest driver: run all 4 solvers on the real-price-derived instances
under data/instances_real/. Results land in results/raw/ and are aggregated
together with the synthetic runs.

    python scripts/build_real_instances.py    # one-time
    python scripts/run_csv_backtest.py        # runs both csvs, all 4 algos
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark_protocol import instances  # noqa: E402
from benchmark_protocol.result_schema import validate  # noqa: E402
from solver_common import output_path  # noqa: E402


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


brute_force = _load(ROOT / "classical-algorithm" / "brute_force.py", "bf2")
sim_anneal = _load(ROOT / "classical-algorithm" / "simulated_annealing.py", "sa2")
qaoa = _load(ROOT / "quantum-optimization-algorithm" / "qaoa.py", "qaoa2")
cvar_vqe = _load(ROOT / "quantum-optimization-algorithm" / "cvar_vqe.py", "cv2")

REAL_DIR = ROOT / "data" / "instances_real"


def load_real(instance_id: str) -> instances.ProblemInstance:
    path = REAL_DIR / f"{instance_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run scripts/build_real_instances.py first)")
    with open(path) as f:
        d = json.load(f)
    return instances.ProblemInstance(
        instance_id=d["instance_id"],
        N=d["N"], K=d["K"], q=d["q"],
        mu=np.asarray(d["mu"], dtype=float),
        sigma=np.asarray(d["sigma"], dtype=float),
        asset_tickers=list(d["asset_tickers"]),
        date_range=tuple(d["date_range"]),
        prev_selection=d.get("prev_selection"),
    )


SOLVERS = [
    ("brute_force", brute_force.solve, brute_force.MAX_N),
    ("simulated_annealing", sim_anneal.solve, None),
    ("qaoa", qaoa.solve, qaoa.MAX_N),
    ("cvar_vqe", cvar_vqe.solve, cvar_vqe.MAX_N),
]

INSTANCE_IDS = [
    "real_qubo_full", "real_qubo_q7",
    "real_miqp_full", "real_miqp_q7",
]


def run_one(name, solver, max_n, inst, seed):
    if max_n is not None and inst.N > max_n:
        return "skip", f"skip (N={inst.N} > {max_n})"
    try:
        t0 = time.perf_counter()
        result = solver(inst, seed=seed)
        errors = validate(result)
        if errors:
            return "fail", "validation: " + "; ".join(errors)
        out = result.to_json(output_path(result.algorithm, result.instance_id, result.seed))
        return "ok", (
            f"{out.name}  obj={result.objective_value:.4f}  feas={result.feasible}  "
            f"ar={result.approx_ratio}  t={time.perf_counter() - t0:.2f}s"
        )
    except Exception:
        return "fail", traceback.format_exc(limit=4)


def main() -> None:
    counts = {"ok": 0, "skip": 0, "fail": 0}
    for iid in INSTANCE_IDS:
        inst = load_real(iid)
        print(f"=== {iid}  N={inst.N} K={inst.K} ===")
        for name, solver, max_n in SOLVERS:
            status, msg = run_one(name, solver, max_n, inst, seed=42)
            tag = {"ok": "OK ", "skip": "SKIP", "fail": "FAIL"}[status]
            print(f"  [{tag}] {name:<22} {msg}")
            counts[status] += 1
    print()
    print(f"done: {counts['ok']} ok / {counts['skip']} skipped / {counts['fail']} failed")


if __name__ == "__main__":
    main()
