◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   small_0024.json
  solver     Miqp.py  v1.0.0
  instance idsmall_0024  seed 42

  ✔ feasible   approx ratio 1.000   backend pulp_lp_relax

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -6.015956
    expected return         24.24%
    volatility              16.97%
    sharpe ratio            1.1336
    wall time               1.1746s
    optimizer iters         n/a

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
    [0] 1  2  3  4  5 [6][7] 8  9 

    selected indices: [0, 6, 7]

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘
    asset 0     █░░░░░░░░░░░░░░░░░░░░░░░░░░░    5.0%
    asset 6     █████████████████░░░░░░░░░░░   60.0%
    asset 7     ██████████░░░░░░░░░░░░░░░░░░   35.0%

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         miqp
    method                  mixed_integer_quadratic_program
    backend                 pulp_lp_relax
    q                       1.1158
    w_min                   0.0500
    w_max                   0.6000
    time_limit              120.0000

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T01:37:48.960345+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   small_0024.json
  solver     Tabu_search.py  v1.0.0
  instance idsmall_0024  seed 42

  ✔ feasible   approx ratio 1.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -6.015956
    expected return         n/a
    volatility              n/a
    sharpe ratio            n/a
    wall time               6.0391s
    optimizer iters         2000

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
    [0] 1  2  3  4  5 [6][7] 8  9 

    selected indices: ?

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         qubo
    method                  tabu_search
    n_iters                 2000
    tabu_tenure             10
    n_candidates            20

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T01:37:55.012665+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   small_0024.json
  solver     Hrp.py  v1.0.0
  instance idsmall_0024  seed 42

  ✔ feasible   approx ratio 0.512   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -5.504036
    expected return         6.58%
    volatility              15.08%
    sharpe ratio            0.1049
    wall time               0.0019s
    optimizer iters         1

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0 [1] 2 [3] 4 [5] 6  7  8  9 

    selected indices: [1, 3, 5]

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘
    asset 1     ███░░░░░░░░░░░░░░░░░░░░░░░░░    9.0%
    asset 3     █████░░░░░░░░░░░░░░░░░░░░░░░   17.5%
    asset 5     █████████████████████░░░░░░░   73.5%

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         qubo
    method                  hrp_ward_bisection
    linkage                 ward
    cluster_criterion       maxclust

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T01:37:55.057339+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   small_0024.json
  solver     mvo.py  v1.0.0
  instance idsmall_0024  seed 42

  ✔ feasible   approx ratio 0.985   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -6.000579
    expected return         n/a
    volatility              n/a
    sharpe ratio            1.1929
    wall time               3.2455s
    optimizer iters         120

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0  1 [2] 3  4  5 [6][7] 8  9 

    selected indices: [2, 6, 7]

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘
    asset 2     ███░░░░░░░░░░░░░░░░░░░░░░░░░   11.0%
    asset 6     ████████████████░░░░░░░░░░░░   56.9%
    asset 7     █████████░░░░░░░░░░░░░░░░░░░   32.0%

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         qubo
    method                  mean_variance_sharpe
    risk_free_rate          0.0500
    w_min                   0.0500
    w_max                   0.6000
    n_starts                8

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T01:37:58.306716+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   small_0024.json
  solver     Equal_weigth.py  v1.0.0
  instance idsmall_0024  seed 42

  ✔ feasible   approx ratio 0.558   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -5.552397
    expected return         27.19%
    volatility              30.28%
    sharpe ratio            0.7327
    wall time               0.0001s
    optimizer iters         1

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0  1 [2] 3  4  5  6 [7] 8 [9]

    selected indices: [2, 7, 9]

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘
    asset 2     █████████░░░░░░░░░░░░░░░░░░░   33.3%
    asset 7     █████████░░░░░░░░░░░░░░░░░░░   33.3%
    asset 9     █████████░░░░░░░░░░░░░░░░░░░   33.3%

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         qubo
    method                  equal_weight_top_mu
    selection_rule          top_K_by_expected_return
    weight_rule             uniform_1_over_K

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T01:37:58.315663+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   small_0024.json
  solver     simulated_annealing.py  v1.0.0
  instance idsmall_0024  seed 42

  ✔ feasible   approx ratio 1.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -6.015956
    expected return         n/a
    volatility              n/a
    sharpe ratio            n/a
    wall time               0.1802s
    optimizer iters         2000

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
    [0] 1  2  3  4  5 [6][7] 8  9 

    selected indices: ?

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         qubo
    method                  simulated_annealing
    n_sweeps                2000
    T0                      1.0000
    Tend                    0.0010

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T01:37:58.571296+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   small_0024.json
  solver     brute_force.py  v1.0.0
  instance idsmall_0024  seed 42

  ✔ feasible   approx ratio 1.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -6.015956
    expected return         n/a
    volatility              n/a
    sharpe ratio            n/a
    wall time               0.0012s
    optimizer iters         n/a

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
    [0] 1  2  3  4  5 [6][7] 8  9 

    selected indices: ?

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         qubo
    method                  exhaustive_K_subsets

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T01:37:58.574380+00:00
