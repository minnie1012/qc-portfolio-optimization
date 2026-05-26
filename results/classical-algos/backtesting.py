# quick_test.py — run simple backtesting analysis from repo root
import json
import dataclasses
import importlib.util
from pathlib import Path
import numpy as np

# ── All Solvers (Looping feature) ──────────────────────────────────────────
SOLVER_FILES = [
    "Miqp.py",
    "Tabu_search.py",
    "Hrp.py",
    "mvo.py",
    "Equal_weigth.py",
    "simulated_annealing.py",
    "brute_force.py"
]

# ── ANSI colour helpers (no deps) ──────────────────────────────────────────
R  = "\033[0m"        # reset
B  = "\033[1m"        # bold
DIM= "\033[2m"        # dim
CY = "\033[96m"       # cyan
GR = "\033[92m"       # green
AM = "\033[93m"       # amber
RD = "\033[91m"       # red
BL = "\033[94m"       # blue
MG = "\033[95m"       # magenta
WH = "\033[97m"       # white

def hr(char="─", width=58, color=DIM):
    print(f"{color}{char * width}{R}")

def section(title: str):
    print()
    print(f"{DIM}┌{'─'*56}┐{R}")
    print(f"{DIM}│{R}  {B}{WH}{title.upper():<54}{R}  {DIM}│{R}")
    print(f"{DIM}└{'─'*56}┘{R}")

def row(label: str, value: str, color: str = WH, indent: int = 4):
    pad = " " * indent
    print(f"{pad}{DIM}{label:<24}{R}{color}{B}{value}{R}")

def bar(value: float, width: int = 24, color: str = BL) -> str:
    filled = round(value * width)
    filled = max(0, min(width, filled))
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{R}"

# ── Load solver ────────────────────────────────────────────────────────────
def load_solver(filename: str):
    solver_path = Path(__file__).parent / "classical-algorithm" / filename
    spec = importlib.util.spec_from_file_location("solver", solver_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve

# ── Load instance ──────────────────────────────────────────────────────────
from benchmark_protocol.instances import ProblemInstance  # noqa: E402

inst_path = next(Path("data/instances").glob("*.json"))
raw = json.loads(inst_path.read_text())
known_fields = {f.name for f in dataclasses.fields(ProblemInstance)}
filtered = {k: v for k, v in raw.items() if k in known_fields}
filtered["mu"]    = np.array(filtered["mu"],    dtype=float)
filtered["sigma"] = np.array(filtered["sigma"], dtype=float)
inst = ProblemInstance(**filtered)

# ── Loop over all solver files ─────────────────────────────────────────────
for SOLVER_FILE in SOLVER_FILES:
    solve = load_solver(SOLVER_FILE)

    # ── Run solver ─────────────────────────────────────────────────────────────
    result = solve(inst, seed=42)

    # ══════════════════════════════════════════════════════
    #  PRETTY PRINT
    # ══════════════════════════════════════════════════════
    print()
    print(f"  {B}{CY}◆ QC Portfolio Benchmark{R}  {DIM}quick_test.py{R}")
    hr("═")

    # Header
    print(f"  {DIM}instance   {R}{B}{WH}{inst_path.name}{R}")
    print(f"  {DIM}solver     {R}{B}{MG}{SOLVER_FILE}{R}  {DIM}v{result.algorithm_version}{R}")
    print(f"  {DIM}instance id{R}{B}{WH}{result.instance_id}{R}  {DIM}seed {result.seed}{R}")

    # Status pills
    feasible_str = f"{GR}{B}✔ feasible{R}"   if result.feasible else f"{RD}{B}✘ infeasible{R}"
    ar_val       = result.approx_ratio
    ar_color     = GR if ar_val is not None and ar_val >= 0.95 else AM if ar_val is not None and ar_val >= 0.75 else RD
    ar_str       = f"{ar_color}{B}{ar_val:.3f}{R}" if ar_val is not None else f"{DIM}n/a{R}"
    backend_str  = f"{BL}{result.backend or 'n/a'}{R}"
    print()
    print(f"  {feasible_str}   {DIM}approx ratio{R} {ar_str}   {DIM}backend{R} {backend_str}")

    # ── Metrics ────────────────────────────────────────────────────────────────
    section("metrics")
    fp = result.final_params or {}
    obj = result.objective_value
    ret = fp.get("portfolio_return",   None)
    vol = fp.get("portfolio_volatility", None)
    shr = fp.get("sharpe_ratio",       None)
    wt  = result.wall_time_seconds

    row("objective value",  f"{obj:+.6f}",                    CY)
    row("expected return",  f"{ret*100:.2f}%" if ret else "n/a", GR)
    row("volatility",       f"{vol*100:.2f}%" if vol else "n/a", AM)
    row("sharpe ratio",     f"{shr:.4f}"  if shr else "n/a",  MG)
    row("wall time",        f"{wt:.4f}s",                      WH)
    row("optimizer iters",  str(result.optimizer_iters or "n/a"), DIM)

    # ── Asset selection ────────────────────────────────────────────────────────
    section("asset selection")
    bits = result.bitstring
    n = len(bits)
    bit_row = "    "
    for i, b in enumerate(bits):
        if b:
            bit_row += f"{GR}{B}[{i}]{R}"
        else:
            bit_row += f"{DIM} {i} {R}"
    print(bit_row)
    print(f"\n    {DIM}selected indices:{R} {GR}{B}{fp.get('selected_indices', '?')}{R}")

    # ── Weights ────────────────────────────────────────────────────────────────
    section("portfolio weights")
    weights = fp.get("weights", [])
    sel_idx = fp.get("selected_indices", [])
    for idx, w in zip(sel_idx, weights):
        label = f"asset {idx}"
        pct   = w * 100
        b     = bar(w, width=28, color=BL)
        print(f"    {WH}{label:<10}{R}  {b}  {B}{pct:5.1f}%{R}")

    # ── Hyperparameters ────────────────────────────────────────────────────────
    section("hyperparameters")
    for k, v in (result.hyperparameters or {}).items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        row(k, val, DIM)

    # Footer
    print()
    hr("═")
    print(f"  {DIM}schema v{result.schema_version}  ·  {result.timestamp_utc}{R}")
    print()
