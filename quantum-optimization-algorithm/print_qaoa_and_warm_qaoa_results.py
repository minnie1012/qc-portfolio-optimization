import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse
from pathlib import Path
from benchmark_protocol import instances

# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Print QAOA and Warm Start QAOA results")
parser.add_argument("--tiny",   action="store_true", help="Show only tiny instances")
parser.add_argument("--small",  action="store_true", help="Show only small instances")
parser.add_argument("--medium", action="store_true", help="Show only medium instances")
parser.add_argument("--large",  action="store_true", help="Show only large instances")
parser.add_argument("--n7gap",  action="store_true", help="Show only n7_gap instances")
parser.add_argument("--diff",   action="store_true", help="Show only results where QAOA and Warm Start selected different stocks")
args = parser.parse_args()

# ── Load results ───────────────────────────────────────────────────────────────

results_dir = Path(__file__).resolve().parent.parent / "results"

qaoa_files = sorted((results_dir / "qaoa").glob("qaoa_saksham__*.json"))
warm_files = sorted((results_dir / "warm_start").glob("warm_start_qaoa_saksham__*.json"))

def load_results(files):
    results = {}
    for path in files:
        with open(path) as f:
            r = json.load(f)
        key = (r["instance_id"], int(r["hyperparameters"]["p"]))
        results[key] = r
    return results

qaoa_results = load_results(qaoa_files)
warm_results = load_results(warm_files)

# ── Filter instances by flag ───────────────────────────────────────────────────

all_instances = sorted(
    set(k[0] for k in list(qaoa_results.keys()) + list(warm_results.keys())),
    key=lambda x: (x.split('_')[0], int(x.split('_')[-1]))
)

# Apply size filter if any flag given
any_filter = args.tiny or args.small or args.medium or args.large or args.n7gap
if any_filter:
    filtered = []
    for inst_id in all_instances:
        if args.tiny   and inst_id.startswith("tiny"):
            filtered.append(inst_id)
        if args.small  and inst_id.startswith("small"):
            filtered.append(inst_id)
        if args.medium and inst_id.startswith("medium"):
            filtered.append(inst_id)
        if args.large  and inst_id.startswith("large"):
            filtered.append(inst_id)
        if args.n7gap  and inst_id.startswith("n7_gap"):
            filtered.append(inst_id)
    all_instances = filtered

# ── Print comparison table ─────────────────────────────────────────────────────

print(f"Standard QAOA results:   {len(qaoa_files)}")
print(f"Warm Start QAOA results: {len(warm_files)}")
if args.diff:
    print(f"Filter: showing only instances where QAOA and Warm Start differ")
if any_filter:
    active = [f for f in ["tiny", "small", "medium", "large", "n7gap"] if getattr(args, f)]
    print(f"Filter: showing only {', '.join(active)} instances")
print()
print("=" * 130)
print(f"{'Instance':<15} {'N':>3} {'K':>3} {'p':>3} | {'QAOA Selected':<22} {'QAOA AR':>8} {'Depth':>6} {'Gates':>6} | {'WS Selected':<22} {'WS AR':>8} {'Depth':>6} {'Gates':>6}")
print("=" * 130)

rows_printed = 0

for instance_id in all_instances:
    try:
        inst = instances.load(instance_id)
    except Exception:
        continue

    for p in [1, 2]:
        qaoa = qaoa_results.get((instance_id, p))
        warm = warm_results.get((instance_id, p))

        # Standard QAOA columns
        if qaoa:
            qaoa_selected = "+".join([inst.asset_tickers[i] for i, b in enumerate(qaoa["bitstring"]) if b == 1])
            qaoa_ar = f"{qaoa['approx_ratio']:.4f}" if qaoa["approx_ratio"] is not None else "N/A"
            qaoa_depth = str(qaoa["circuit_depth"])
            qaoa_gates = str(qaoa["total_gate_count"])
        else:
            qaoa_selected = qaoa_ar = qaoa_depth = qaoa_gates = "---"

        # Warm start columns
        if warm:
            warm_selected = "+".join([inst.asset_tickers[i] for i, b in enumerate(warm["bitstring"]) if b == 1])
            warm_ar = f"{warm['approx_ratio']:.4f}" if warm["approx_ratio"] is not None else "N/A"
            warm_depth = str(warm["circuit_depth"])
            warm_gates = str(warm["total_gate_count"])
        else:
            warm_selected = warm_ar = warm_depth = warm_gates = "---"

        # Apply --diff filter
        if args.diff:
            if qaoa_selected == "---" or warm_selected == "---":
                continue
            if sorted(qaoa_selected.split("+")) == sorted(warm_selected.split("+")):
                continue

        print(f"{instance_id:<15} {inst.N:>3} {inst.K:>3} {p:>3} | {qaoa_selected:<22} {qaoa_ar:>8} {qaoa_depth:>6} {qaoa_gates:>6} | {warm_selected:<22} {warm_ar:>8} {warm_depth:>6} {warm_gates:>6}")
        rows_printed += 1

print("=" * 130)
print(f"\nRows shown: {rows_printed} | Total: {len(qaoa_files)} QAOA results, {len(warm_files)} Warm Start results")