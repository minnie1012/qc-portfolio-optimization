# Walk-forward equity backtest: findings

Produced by `python scripts/walkforward_equity_backtest.py` (deterministic: seed 42, frozen
price CSV, cached Treasury curve). Numbers are copied from `summary.json`; nothing was tuned
after the out-of-sample results were seen.

**Setup:**
- 25-stock universe (`data/prices/prices_daily.csv`).
- 252-day lookback, 21-day hold, rebalance every 21 trading days: 48 periods.
- OOS 2022-01-04 to 2025-12-31 (1002 trading days).
- mu, Sigma and the risk-free rate at each rebalance use only data strictly before that date (asserted every period).
- K = 5, q = 1.0.
- QUBO weights are bounded to 5–50%; the mean-variance baseline is bounded to 0–50%.
- Risk-free rate: 3-Mo Treasury, point-in-time.
- Net figures deduct 10 bps per unit of turnover, including the initial buy-in.
- Sharpe = realized OOS daily excess return / realized OOS daily volatility, annualized.

| Strategy | Ann. return (gross / net) | Ann. vol | Sharpe (gross / net) | Max DD (gross / net) | Avg turnover / rebalance | Avg LW δ |
|---|---|---|---|---|---|---|
| qubo_sample | 23.05% / 22.14% | 21.19% | 0.892 / 0.857 | -22.75% / -22.98% | 0.610 | n/a |
| qubo_lw | 23.72% / 22.80% | 21.67% | 0.902 / 0.868 | -22.62% / -22.82% | 0.607 | 0.068 |
| mv_sample | 17.56% / 16.70% | 20.37% | 0.695 / 0.659 | -23.45% / -23.67% | 0.602 | n/a |
| mv_lw | 18.32% / 17.49% | 20.60% | 0.721 / 0.686 | -22.56% / -22.77% | 0.576 | 0.068 |
| equal_weight | 13.64% / 13.53% | 17.69% | 0.580 / 0.574 | -24.27% / -24.32% | 0.060 | n/a |

**Column definitions:**
- **Strategy:** `qubo_*` = exact K=5 QUBO selection followed by max-Sharpe weights; `mv_*` = long-only max-Sharpe over all 25 stocks; `equal_weight` = 1/N. The `_sample` / `_lw` suffix is the covariance estimator.
- **Ann. return:** CAGR.
- **Avg turnover:** Σ|Δw| per rebalance, excluding the initial buy-in.

Equity curves: `equity_curves.png`. Per-period weights, turnover and returns: `periods.csv`.

## What the numbers say

**Ledoit–Wolf vs. sample covariance**
- LW gave slightly higher Sharpe than the sample covariance for both optimized strategies: QUBO 0.868 vs 0.857 net, mean-variance 0.686 vs 0.659 net.
- The effect is small. Average shrinkage was only δ = 0.068 (range 0.027–0.118 across periods), because 252 observations for 25 assets is already a well-conditioned problem. The QUBO step picked the identical 5-stock set under both estimators in 41 of the 48 periods.

**QUBO-select vs. the baselines**
- QUBO-select + max-Sharpe beat 1/N on return and Sharpe: 22.1–22.8% vs 13.5% CAGR net, and Sharpe 0.86–0.87 vs 0.57 net.
- It also beat unconstrained mean-variance: Sharpe 0.66–0.69 net.
- It did so with higher volatility (about 21–22% vs 17.7%) and about 10× the turnover of 1/N. Its maximum drawdown was only about 1.5 points shallower.
- Most of the gap opened in 2024–2025. At the end of 2023, net growth of $1 was 1.08–1.11 for QUBO vs 1.04 for 1/N; by the end of 2025 it was 2.22–2.26 vs 1.66.
- The most frequently selected names were WMT, NVDA, NFLX, KO and COST.

**Transaction costs**
- At 10 bps per unit of turnover, costs cut the optimized strategies' Sharpe by about 0.035 and 1/N's by about 0.005. They do not change the ranking.

**Statistical significance**
- None of these differences is statistically established. With about 4 years of OOS data, the standard error of an annualized Sharpe ratio is roughly 0.54–0.59 (Lo 2002, i.i.d. approximation: √((1 + SR²/2)/T)).
- That standard error is larger than every gap in the table. The LW-vs-sample gap (≈0.01–0.03) in particular is indistinguishable from noise.

## Limitations

- **Single universe.** One hand-picked 25-stock universe of large U.S. companies; no other universes or asset classes were tested.
- **Survivorship and look-ahead in the ticker list.** The tickers were chosen in 2025 from names that are large and liquid today, which favors strategies that concentrate in past winners (e.g. NVDA, NFLX, COST).
- **Short history.** 5 years of prices gives 4 years (48 periods) of OOS data, which covers essentially one market regime: the 2022 drawdown followed by a growth-led recovery.
- **Simplified costs and rebalancing.** Costs are a flat 10 bps per unit of turnover, with no market impact, borrowing, or taxes. Rebalancing happens at the close.
- **Risk-free proxy.** The 3-Mo constant-maturity Treasury yield serves as the risk-free rate.
- **Fixed parameters, no sensitivity analysis.** K, q, the weight bounds and the window lengths were fixed before the run and not varied, so the results say nothing about sensitivity to those choices.
- **Classical solver.** The Stage-1 QUBO is solved exactly by classical enumeration of all C(25,5) subsets. These results measure the QUBO *formulation* plus max-Sharpe allocation, not any quantum solver.

## Related results on this branch

- **Treasury proxy backtest** (`scripts/treasury_backtest.py --cov-method ledoit_wolf`):
  - LW gave a portfolio Sharpe of -1.019 vs -1.156 with sample covariance (mean δ = 0.011).
  - Both versions are below the equal-weight Treasury baseline (-0.736).
- **QAOA backtest Sharpe fix** (`results/backtest/qaoa_backtest_results_oos_sharpe.csv`, 188 QAOA / warm-start results):
  - Using the realized 2025 OOS volatility instead of the in-sample volatility lowers the mean Sharpe from 1.010 (deprecated metric) to 0.959.
