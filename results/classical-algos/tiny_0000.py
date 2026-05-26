  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   tiny_0000.json
  solver     Miqp.py  v1.0.0
  instance idtiny_0000  seed 42

  ✔ feasible   approx ratio 0.293   backend pulp_lp_relax

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -2.691460
    expected return         24.17%
    volatility              38.53%
    sharpe ratio            0.4976
    wall time               0.6393s
    optimizer iters         n/a

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
    [0][1] 2  3 

    selected indices: [0, 1]

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘
    asset 0     ███████████░░░░░░░░░░░░░░░░░   40.0%
    asset 1     █████████████████░░░░░░░░░░░   60.0%

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         miqp
    method                  mixed_integer_quadratic_program
    backend                 pulp_lp_relax
    q                       1.0461
    w_min                   0.0500
    w_max                   0.6000
    time_limit              120.0000

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T04:11:56.214734+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   tiny_0000.json
  solver     Tabu_search.py  v1.0.0
  instance idtiny_0000  seed 42

  ✔ feasible   approx ratio 1.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -2.846828
    expected return         n/a
    volatility              n/a
    sharpe ratio            n/a
    wall time               0.4507s
    optimizer iters         2000

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0 [1][2] 3 

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
  schema v1.0  ·  2026-05-26T04:11:56.670685+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   tiny_0000.json
  solver     Hrp.py  v1.0.0
  instance idtiny_0000  seed 42

  ✔ feasible   approx ratio 1.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -2.846828
    expected return         7.73%
    volatility              17.31%
    sharpe ratio            0.1580
    wall time               0.0009s
    optimizer iters         1

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0 [1][2] 3 

    selected indices: [1, 2]

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘
    asset 1     ███░░░░░░░░░░░░░░░░░░░░░░░░░   11.6%
    asset 2     █████████████████████████░░░   88.4%

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         qubo
    method                  hrp_ward_bisection
    linkage                 ward
    cluster_criterion       maxclust

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T04:11:56.693670+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   tiny_0000.json
  solver     mvo.py  v1.0.0
  instance idtiny_0000  seed 42

  ✔ feasible   approx ratio 0.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -2.627166
    expected return         n/a
    volatility              n/a
    sharpe ratio            0.5620
    wall time               0.0586s
    optimizer iters         6

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0 [1] 2 [3]

    selected indices: [1, 3]

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘
    asset 1     █████████████████░░░░░░░░░░░   60.0%
    asset 3     ███████████░░░░░░░░░░░░░░░░░   40.0%

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
  schema v1.0  ·  2026-05-26T04:11:56.753480+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   tiny_0000.json
  solver     Equal_weigth.py  v1.0.0
  instance idtiny_0000  seed 42

  ✔ feasible   approx ratio 0.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -2.627166
    expected return         28.33%
    volatility              42.04%
    sharpe ratio            0.5548
    wall time               0.0016s
    optimizer iters         1

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0 [1] 2 [3]

    selected indices: [1, 3]

┌────────────────────────────────────────────────────────┐
│  PORTFOLIO WEIGHTS                                       │
└────────────────────────────────────────────────────────┘
    asset 1     ██████████████░░░░░░░░░░░░░░   50.0%
    asset 3     ██████████████░░░░░░░░░░░░░░   50.0%

┌────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS                                         │
└────────────────────────────────────────────────────────┘
    problem_variant         qubo
    method                  equal_weight_top_mu
    selection_rule          top_K_by_expected_return
    weight_rule             uniform_1_over_K

══════════════════════════════════════════════════════════
  schema v1.0  ·  2026-05-26T04:11:56.757255+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   tiny_0000.json
  solver     simulated_annealing.py  v1.0.0
  instance idtiny_0000  seed 42

  ✔ feasible   approx ratio 1.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -2.846828
    expected return         n/a
    volatility              n/a
    sharpe ratio            n/a
    wall time               0.0907s
    optimizer iters         2000

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0 [1][2] 3 

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
  schema v1.0  ·  2026-05-26T04:11:56.852840+00:00


  ◆ QC Portfolio Benchmark  quick_test.py
══════════════════════════════════════════════════════════
  instance   tiny_0000.json
  solver     brute_force.py  v1.0.0
  instance idtiny_0000  seed 42

  ✔ feasible   approx ratio 1.000   backend n/a

┌────────────────────────────────────────────────────────┐
│  METRICS                                                 │
└────────────────────────────────────────────────────────┘
    objective value         -2.846828
    expected return         n/a
    volatility              n/a
    sharpe ratio            n/a
    wall time               0.0000s
    optimizer iters         n/a

┌────────────────────────────────────────────────────────┐
│  ASSET SELECTION                                         │
└────────────────────────────────────────────────────────┘
     0 [1][2] 3 

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
  schema v1.0  ·  2026-05-26T04:11:56.854378+00:00
