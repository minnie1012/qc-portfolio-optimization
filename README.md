# qc-portfolio-optimization
QCSD x TQT Quantum Finance Project

## Layout

```
benchmark_protocol/             benchmarking package: instance loader, result schema,
                                gate-metrics, prices loader, aggregator, solver template
classical-algorithm/            brute_force.py, simulated_annealing.py
quantum-optimization-algorithm/ qaoa.py, cvar_vqe.py
data/instances/                 57 instance JSONs (mu, sigma, K, q, asset_tickers)
                                annualized from real prices, in-sample 2022-01..2024-12
data/instances_real/            4 ablation instances built directly from each CSV
                                universe (real_qubo_full/q7, real_miqp_full/q7)
data/prices/                    daily auto-adjusted close CSVs
                                  prices_daily.csv      QUBO universe (25 stocks)
                                  prices_miqp_daily.csv MIQP universe (50 stocks)
problem_definition.py           QUBO builder, Ising mapping, max-Sharpe Stage-2
solver_common.py                shared helpers (build, eval, exact reference)
scripts/
  regenerate_instances_from_prices.py  rebuild data/instances/ from CSVs
  build_real_instances.py              rebuild data/instances_real/
  run_benchmarks.py                    run all 4 solvers on data/instances/
  run_csv_backtest.py                  run all 4 solvers on data/instances_real/
  compare_metrics.py                   §6 priority metrics summary
  portfolio_metrics.py                 annual return, Sharpe, OOS backtest
  fetch_prices.py                      pull fresh prices from yfinance
results/
  raw/                          one BenchmarkResult JSON per (algorithm, instance, seed)
  summary_tables/               all_runs.csv, comparison_metrics.json,
                                portfolio_metrics.csv
```

## Reproduce from scratch

```bash
python scripts/regenerate_instances_from_prices.py   # data/instances/
python scripts/build_real_instances.py               # data/instances_real/
python scripts/run_benchmarks.py                     # 184 jobs
python scripts/run_csv_backtest.py                   # 16 jobs
python -m benchmark_protocol.aggregate               # all_runs.csv
python scripts/compare_metrics.py                    # comparison_metrics.json
python scripts/portfolio_metrics.py                  # portfolio_metrics.csv
```

See `benchmark_protocol/README.md` for the full benchmark contract
(problem variants, recorded fields, output layout, comparison priority).
