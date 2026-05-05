# Benchmark Protocol

Single source of truth for how every solver in this repo is tested. If your numbers were not produced under this protocol, they don't get compared.

The goal: everyone runs on the **same instances**, records the **same fields**, and writes results to the **same place**, so a single command collates everything into one comparison table.

---

## 1. Two problem variants

Solvers can target either of two formulations. The same instance data (`mu`, `sigma`, `K`, `q`) feeds both — only the objective and the variable type differ.

### Variant A — QUBO (binary selection)

See repo-root `problem_definition.py`.

```
min_{x in {0,1}^N}  q·xᵀΣx − μᵀx + λ·(Σxᵢ − K)²
```

Pick exactly K assets equally; binary decision. Maps cleanly to QAOA, VQE, CVaR-VQE, Ising annealers.

### Variant B — MIQP (continuous weights with cardinality)

See repo-root `MIQP.ipynb`.

```
max_{x ∈ [fmin, fmax]^N, v ∈ {0,1}^N}  rᵀx − λ·xᵀQx
s.t.  Σxᵢ = 1,   m ≤ Σvᵢ ≤ M,   fmin·vᵢ ≤ xᵢ ≤ fmax·vᵢ
```

Pick somewhere between m and M assets; allocate continuous weights. Solved here via cutting-plane MILP (PuLP/CBC). Pairs naturally with Q-CHOP, Grover-adaptive search, and continuous-relaxation hybrids.

**When recording results, set `hyperparameters["problem_variant"] = "qubo"` or `"miqp"` so the aggregator can group correctly.**

---

## 2. Frozen instance set

All benchmarks run on the 57 JSON files in `data/instances/`.

| Bucket | IDs | N range | Notes |
|---|---|---|---|
| tiny | `syn_tiny_0000` .. `syn_tiny_0014` | 4–5 | exact reachable |
| small | `syn_small_0015` .. `syn_small_0029` | 6–8 | exact + quantum overlap |
| n7_gap | `syn_n7_gap_0000` .. `syn_n7_gap_0004` | 7 | hard-gap stress cases |
| medium | `syn_medium_0030` .. `syn_medium_0041` | 9–17 | classical + MPS quantum |
| large | `syn_large_0042` .. `syn_large_0051` | 18–49 | classical-only |

Each file is plain JSON — open one in any text editor:

```json
{
  "instance_id": "syn_tiny_0000",
  "N": 4,
  "K": 2,
  "q": 1.046,
  "asset_tickers": ["SYN_000", "SYN_001", "SYN_002", "SYN_003"],
  "date_range": ["synthetic", "synthetic"],
  "mu": [0.189, 0.069, 0.245, 0.202],
  "sigma": [[...], [...], [...], [...]]
}
```

Load with:

```python
from benchmark_protocol import instances
inst = instances.load("syn_tiny_0000")
inst.mu        # numpy array, shape (N,)
inst.sigma     # numpy array, shape (N, N)
inst.K         # cardinality
inst.q         # risk aversion
```

Never regenerate the JSONs. New instances only for ablations, with a different prefix (e.g. `abl_*`).

---

## 3. Real price history (for backtests)

`data/prices/prices_daily.csv` — 25 stocks × 1255 trading days (2021-01-04 → 2025-12-31), no missing values.

```python
from benchmark_protocol import prices
px = prices.load_prices(start="2022-01-01", end="2024-12-31")  # in-sample
```

Used only by backtest experiments, not by the per-instance benchmark.

---

## 4. What every solver records (mandatory)

Every run produces one `BenchmarkResult` JSON. See `result_schema.py` for the dataclass.

**Identity** — `algorithm`, `algorithm_version`, `instance_id`, `seed`, `timestamp_utc`, `git_commit`

**Quality** — `objective_value`, `bitstring` (or weight vector cast to ints for MIQP), `feasible`, `approx_ratio`

**Runtime** — `wall_time_seconds`, `optimizer_iters`, `num_circuit_evaluations`, `convergence_history`

**Quantum-only** (set to null for classical) — `qubit_count`, `circuit_depth`, `two_qubit_gate_count`, `t_gate_count`, `total_gate_count`, `shots`, `backend`, `transpile_optimization_level`

**Parameters** — `hyperparameters` (must include `"problem_variant"`), `initial_params`, `final_params`

**Environment** — `env` (auto-filled with python/qiskit/numpy versions)

---

## 5. Output location

```
results/raw/<algorithm>__<instance_id>__seed<seed>.json
```

One file per run. The aggregator handles joining.

---

## 6. Comparison metrics, in priority order

1. **Approx ratio** on tiny + small + n7_gap (matched quantum/classical set)
2. **Mean objective** on medium + large (classical-comparable extension)
3. **Wall-time per Δobjective** vs Equal-Weight baseline
4. **Shot-noise robustness** — approx ratio at 1024 / 4096 / 8192 shots
5. **Hardware feasibility proxy** — `two_qubit_gate_count × circuit_depth` (lower is better)

Comparisons are made **within a problem variant** (qubo vs qubo, miqp vs miqp). Cross-variant comparisons require a separate justification.

---

## 7. Reproducibility

- Pin `seed` for every run.
- Treat `data/instances/*.json` and `data/prices/prices_daily.csv` as read-only.
- Re-run experiments if `benchmark_protocol/` changes between runs.

---

## 8. How to add a new solver

1. Copy `solver_template.py` next to your solver code.
2. Fill in `solve(instance) -> BenchmarkResult`. Set `hyperparameters["problem_variant"]`.
3. Loop over `instances.all_ids()`, write one JSON per run to `results/raw/`.
4. `python -m benchmark_protocol.aggregate --validate` — must pass before sharing.
5. `python -m benchmark_protocol.aggregate` — produces the comparison CSV.
