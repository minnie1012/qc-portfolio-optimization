"""Driver: run all 4 solvers on the JSON instance set and write results/raw/.

Solvers live in the hyphenated subdirs (`classical-algorithm/`,
`quantum-optimization-algorithm/`); we load them by file path because the
hyphen blocks normal Python imports.

    python scripts/run_benchmarks.py            # full suite
    python scripts/run_benchmarks.py --quick    # smoke test (one instance / algo)
    python scripts/run_benchmarks.py --only classical
    python scripts/run_benchmarks.py --only quantum
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

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


brute_force = _load(ROOT / "classical-algorithm" / "brute_force.py", "brute_force_solver")
sim_anneal = _load(ROOT / "classical-algorithm" / "simulated_annealing.py", "sa_solver")
qaoa = _load(ROOT / "quantum-optimization-algorithm" / "qaoa.py", "qaoa_solver")
cvar_vqe = _load(ROOT / "quantum-optimization-algorithm" / "cvar_vqe.py", "cvar_vqe_solver")

CLASSICAL = [
    ("brute_force", brute_force.solve, "all", brute_force.MAX_N),
    ("simulated_annealing", sim_anneal.solve, "all", None),
]
QUANTUM = [
    ("qaoa", qaoa.solve, "matched_quantum_classical", qaoa.MAX_N),
    ("cvar_vqe", cvar_vqe.solve, "matched_quantum_classical", cvar_vqe.MAX_N),
]


def run_one(name, solver, max_n, inst, seed) -> tuple[str, str]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", choices=["classical", "quantum"], default=None)
    args = parser.parse_args()

    todo: list[tuple[str, callable, list[str], int | None]] = []
    if args.only != "quantum":
        for name, solver, scope, max_n in CLASSICAL:
            ids = instances.SUBSETS[scope] if scope != "all" else instances.all_ids()
            todo.append((name, solver, ids, max_n))
    if args.only != "classical":
        for name, solver, scope, max_n in QUANTUM:
            ids = instances.SUBSETS[scope]
            if args.quick:
                ids = ids[:2]
            todo.append((name, solver, ids, max_n))

    n_total = sum(len(ids) for _, _, ids, _ in todo)
    print(f"running {n_total} jobs across {len(todo)} algorithms...")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    for name, solver, ids, max_n in todo:
        for iid in ids:
            inst = instances.load(iid)
            status, msg = run_one(name, solver, max_n, inst, args.seed)
            tag = {"ok": "OK ", "skip": "SKIP", "fail": "FAIL"}[status]
            print(f"  [{tag}] {name:<22} {iid:<22} {msg}")
            counts[status] += 1
    print()
    print(f"done: {counts['ok']} ok / {counts['skip']} skipped / {counts['fail']} failed of {n_total}")


if __name__ == "__main__":
    main()
