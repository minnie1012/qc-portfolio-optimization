"""Section-6 priority metrics from the benchmark protocol.

    1. Approx ratio on matched_quantum_classical (tiny + small + n7_gap)
    2. Mean objective on medium + large
    3. Wall-time per |objective|  (proxy for §6.3 vs equal-weight baseline)
    5. Hardware feasibility proxy: two_qubit_gate_count * circuit_depth
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark_protocol import instances  # noqa: E402

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "summary_tables" / "comparison_metrics.json"


def load_all():
    out = []
    for p in sorted(RAW.glob("*.json")):
        with open(p) as f:
            out.append(json.load(f))
    return out


def bucket_of(iid):
    try:
        return instances.bucket_of(iid)
    except ValueError:
        return "real" if iid.startswith("real_") else "unknown"


def main():
    rows = load_all()
    print(f"loaded {len(rows)} results")

    # 1. approx ratio on matched subset
    matched = set(instances.SUBSETS["matched_quantum_classical"])
    ar_by_algo = defaultdict(list)
    for r in rows:
        if r["instance_id"] in matched and r.get("approx_ratio") is not None:
            ar_by_algo[r["algorithm"]].append(float(r["approx_ratio"]))

    # 2. mean objective on medium + large
    obj_by_algo_bucket = defaultdict(list)
    for r in rows:
        b = bucket_of(r["instance_id"])
        if b in ("medium", "large"):
            obj_by_algo_bucket[(r["algorithm"], b)].append(float(r["objective_value"]))

    # 3. wall-time per |obj|
    tpo_by_algo = defaultdict(list)
    for r in rows:
        obj = abs(float(r["objective_value"])) or 1e-9
        tpo_by_algo[r["algorithm"]].append(float(r["wall_time_seconds"]) / obj)

    # 5. hardware feasibility proxy (quantum only)
    hw_by_algo = defaultdict(list)
    for r in rows:
        if r.get("two_qubit_gate_count") is None or r.get("circuit_depth") is None:
            continue
        hw_by_algo[r["algorithm"]].append(int(r["two_qubit_gate_count"]) * int(r["circuit_depth"]))

    def stats(xs):
        if not xs:
            return None
        return {
            "n": len(xs),
            "mean": float(sum(xs) / len(xs)),
            "min": float(min(xs)),
            "max": float(max(xs)),
        }

    summary = {
        "approx_ratio_matched_subset": {a: stats(v) for a, v in ar_by_algo.items()},
        "mean_objective_medium_large": {f"{a}__{b}": stats(v) for (a, b), v in obj_by_algo_bucket.items()},
        "wall_time_per_abs_obj": {a: stats(v) for a, v in tpo_by_algo.items()},
        "hardware_feasibility_proxy_2qd_x_depth": {a: stats(v) for a, v in hw_by_algo.items()},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== 1. Approx ratio on matched subset (tiny+small+n7_gap) ===")
    for a, s in sorted(summary["approx_ratio_matched_subset"].items()):
        if s:
            print(f"  {a:<22} n={s['n']:>3}  mean={s['mean']:.4f}  [min {s['min']:.4f} .. max {s['max']:.4f}]")

    print()
    print("=== 2. Mean objective on medium + large (per bucket) ===")
    for k in sorted(summary["mean_objective_medium_large"]):
        s = summary["mean_objective_medium_large"][k]
        print(f"  {k:<32} n={s['n']:>3}  mean={s['mean']:>12.4f}  best={s['min']:>12.4f}")

    print()
    print("=== 3. Wall-time per |objective|  (lower = better) ===")
    for a, s in sorted(summary["wall_time_per_abs_obj"].items()):
        print(f"  {a:<22} n={s['n']:>3}  mean={s['mean']:.6f}s")

    print()
    print("=== 5. Hardware feasibility proxy: 2q_gate * depth (lower = better) ===")
    for a, s in sorted(summary["hardware_feasibility_proxy_2qd_x_depth"].items()):
        print(f"  {a:<22} n={s['n']:>3}  mean={s['mean']:.1f}  [min {s['min']} .. max {s['max']}]")

    print()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
