# Decentralized Finance Project: market-neutral pair trading on liquid staking tokens

Extension of the *Optimal Trading Technique* of Yang & Malik (2024),
"Optimal Market-Neutral Multivariate Pair Trading on the Cryptocurrency
Platform", to liquid staking tokens traded against WETH on Uniswap V3
(Ethereum mainnet), with 1-hour candles and realistic DEX frictions.

Two model extensions:
1. **Ornstein-Uhlenbeck spreads**: the OU half-life replaces the paper's
   empirical mean-reversion time; the OU model gives the expected gain and
   the risk of each pair.
2. **EWMA bands** instead of a fixed z-score.

## Installation

Python 3.10+. Use a fresh virtual environment (installing `cvxpy` in an
existing scientific environment can upgrade `numpy` and break other packages):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The whole pipeline runs in a few seconds and prints each step. Outputs go to
`results/` (`metrics.csv`, `cointegration.csv` and PNG figures).

The data are read from the CSV files in `data/`. If `data/` is empty they are
downloaded again from the public GeckoTerminal API (no key, ~6 s between
calls). **Keep the CSV files**: the free API only serves the last 180 days,
so a new download made later would cover a different window.

## Structure

```
config.py            all parameters (assets, dates, thresholds, lambda, fees, gas)
main.py              runs the pipeline in order
src/data.py          download, cache, outlier filter, alignment, 70/30 split
src/cointegration.py OLS spreads, ADF test, OU fit, OU expected gain and variance
src/signals.py       EWMA bands, entry and exit signals
src/optimizer.py     bi-objective allocation with cvxpy (paper eq. 6)
src/frictions.py     fee tiers, x*y=k slippage, stochastic gas
src/backtest.py      candle-by-candle simulation, buy & hold benchmark
src/metrics.py       Sharpe, win rate, gain/loss, max drawdown, figures
report/rapport.tex   5-page report
```

## Design choices

- **Universe**: wstETH, rETH, weETH vs WETH (Uniswap V3, 0.01% pools).
  cbETH was dropped: its best pool had no swap in 72% of the hours.
- **Window**: 1 June - 26 November 2026 (179 days, the free API limit),
  70% in-sample / 30% out-of-sample. All parameters are estimated in-sample.
- **Cleaning**: causal 1% outlier filter (a level is accepted after 3
  consecutive deviating candles); missing hours are forward-filled, but trades
  are only opened or closed on hours where both legs had a real swap.
- **Cointegration**: pairwise OLS on log-prices + ADF (constant and trend, p < 0.10),
  as in the paper.
- **Signals**: EWMA mean and std (half-life 48 h) of the spread up to t-1;
  entry at |z| > k, exit when z changes sign; k = 2 chosen on the in-sample.
  No new entry during the last H candles of a pair (H = OU half-life).
- **Optimisation**: max expected gain - lambda x risk (lambda = 1), bounds,
  ETH neutrality and a budget per asset (each LST starts with 10 ETH of
  inventory, no short selling). With frictions, fees (linear) and x*y=k
  slippage (quadratic) enter the objective, and a trade is opened only if
  the expected net gain exceeds the round-trip gas.
- **Frictions**: real fee tier of each pool; x*y=k slippage on virtual
  reserves TVL/2 (conservative, with a sensitivity at 10x and 50x);
  gas = OU process on log(Gwei) (median 1 Gwei, fixed seed), 150k gas per swap,
  4 swaps per trade.
- **Benchmarks**: the strategy without frictions and buy & hold of the
  initial inventories.
