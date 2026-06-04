import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path
from benchmark_protocol import instances

# ── Load all result files ──────────────────────────────────────────────────────

results_dir = Path(__file__).resolve().parent.parent / "results" / "qaoa"
result_files = sorted(results_dir.glob("qaoa_saksham__*.json"))

if not result_files:
    print("No result files found.")
    exit()

print(f"Found {len(result_files)} result files.")
print()
print("=" * 95)
print(f"{'Instance':<15} {'N':>3} {'K':>3} {'p':>3} {'Selected':<25} {'AR':>8} {'Depth':>6} {'Gates':>6} {'Time':>7} {'Feasible':>9}")
print("=" * 95)

for path in result_files:
    with open(path) as f:
        r = json.load(f)

    instance_id = r["instance_id"]
    inst = instances.load(instance_id)
    p = r["hyperparameters"]["p"]
    ar = f"{r['approx_ratio']:.4f}" if r["approx_ratio"] is not None else "N/A"
    bitstring = r["bitstring"]
    selected = "+".join([inst.asset_tickers[i] for i, b in enumerate(bitstring) if b == 1])
    feasible = "YES" if r["feasible"] else "NO"

    print(f"{instance_id:<15} {inst.N:>3} {inst.K:>3} {p:>3} {selected:<25} {ar:>8} {r['circuit_depth']:>6} {r['total_gate_count']:>6} {r['wall_time_seconds']:>7.2f}s {feasible:>9}")

print("=" * 95)
print(f"\nTotal results: {len(result_files)}")