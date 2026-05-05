"""Aggregate every result JSON in results/raw/ into one comparison table.

    python -m benchmark_protocol.aggregate                # write CSV + print summary
    python -m benchmark_protocol.aggregate --validate     # only validate, no output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_protocol import instances
from benchmark_protocol.result_schema import BenchmarkResult, validate

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
OUT_CSV = Path(__file__).resolve().parent.parent / "results" / "summary_tables" / "all_runs.csv"

FLAT_COLS = [
    "algorithm", "algorithm_version", "instance_id", "bucket", "seed",
    "objective_value", "feasible", "approx_ratio",
    "wall_time_seconds", "optimizer_iters", "num_circuit_evaluations",
    "qubit_count", "circuit_depth", "two_qubit_gate_count", "t_gate_count",
    "total_gate_count", "shots", "backend", "transpile_optimization_level",
    "timestamp_utc",
]


def load_all() -> tuple[list[BenchmarkResult], dict[Path, list[str]]]:
    results: list[BenchmarkResult] = []
    bad: dict[Path, list[str]] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            r = BenchmarkResult(**data)
        except Exception as exc:
            bad[path] = [f"load error: {exc}"]
            continue
        errs = validate(r)
        if errs:
            bad[path] = errs
        else:
            results.append(r)
    return results, bad


def to_rows(results: list[BenchmarkResult]) -> list[dict]:
    rows = []
    for r in results:
        try:
            bucket = instances.bucket_of(r.instance_id)
        except ValueError:
            bucket = "unknown"
        row = {col: getattr(r, col, None) for col in FLAT_COLS if col != "bucket"}
        row["bucket"] = bucket
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FLAT_COLS})


def print_summary(rows: list[dict]) -> None:
    by_algo: dict[str, list[dict]] = {}
    for row in rows:
        by_algo.setdefault(row["algorithm"], []).append(row)
    print(f"{'algorithm':<25} {'n':>4} {'mean_obj':>12} {'mean_time_s':>12} {'feas_rate':>10}")
    print("-" * 67)
    for algo in sorted(by_algo):
        rs = by_algo[algo]
        n = len(rs)
        obj = [r["objective_value"] for r in rs if r["objective_value"] is not None]
        t = [r["wall_time_seconds"] for r in rs if r["wall_time_seconds"] is not None]
        feas = [bool(r["feasible"]) for r in rs]
        mean_obj = sum(obj) / len(obj) if obj else float("nan")
        mean_t = sum(t) / len(t) if t else float("nan")
        feas_rate = sum(feas) / len(feas) if feas else float("nan")
        print(f"{algo:<25} {n:>4} {mean_obj:>12.4f} {mean_t:>12.3f} {feas_rate:>10.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="only validate; no CSV")
    args = parser.parse_args()

    results, bad = load_all()
    print(f"loaded {len(results)} valid results from {RESULTS_DIR}")
    if bad:
        print(f"found {len(bad)} invalid file(s):")
        for path, errs in bad.items():
            print(f"  {path.name}: {'; '.join(errs)}")

    if args.validate:
        raise SystemExit(1 if bad else 0)

    rows = to_rows(results)
    write_csv(rows, OUT_CSV)
    print(f"wrote {OUT_CSV}")
    print()
    print_summary(rows)


if __name__ == "__main__":
    main()
