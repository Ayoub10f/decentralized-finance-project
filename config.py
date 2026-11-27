"""
All project parameters in one place.

Units convention: every time quantity used by the model (half-life,
time to revert, expected return) is expressed in number of 1-hour candles.
"""

# ---------------------------------------------------------------------------
# Universe: liquid staking tokens (LST) priced in WETH, Ethereum mainnet.
# Pool addresses were found with the GeckoTerminal search endpoint
#   GET /api/v2/search/pools?query=<LST> weth&network=eth
# keeping, for each LST, a liquid pool with an explicit fee tier and the
# densest hourly trading history. The fee tier and TVL are NOT typed here:
# they are read from the API by src/data.py (see data/pools.csv).
# ---------------------------------------------------------------------------
ASSETS = {
    "wstETH": "0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa",  # Uniswap V3 0.01%
    "rETH":   "0x553e9c493678d8606d6a5ba284643db2110df823",  # Uniswap V3 0.01%
    "weETH":  "0x202a6012894ae5c288ea824cbc8a9bfb26a49b93",  # Uniswap V3 0.01%
}
# cbETH was dropped: its most liquid pool (Uniswap V3 0.05%,
# 0x840deeef2f115cf50da625f7368c24af6fe74410) had no swap in 72% of the hours.

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"
TIMEFRAME = "hour"            # 1-hour candles
# The free public API only serves the last 180 days of OHLCV, so the
# requested Q4 2025 - Q3 2026 window is not reachable. We take the longest
# available window and freeze it with a fixed end date (the CSV cache in
# data/ makes the run reproducible after that date).
END_DATE = "2026-11-27 00:00"     # UTC, excluded
HISTORY_DAYS = 179                # just under the 180-day API limit
API_PAUSE_S = 6.0                 # pause between calls (observed limit is stricter than 30/min)
IN_SAMPLE_FRAC = 0.70             # 70% estimation, 30% out-of-sample test

# Causal outlier filter: a close that moves more than OUTLIER_THRESHOLD
# (log-distance) from the last accepted close is rejected (treated as a
# candle without trade). If the deviation persists for OUTLIER_CONFIRM
# consecutive candles, the new level is accepted (a real move, e.g. a depeg).
OUTLIER_THRESHOLD = 0.01          # 1%
OUTLIER_CONFIRM = 3               # candles

# ---------------------------------------------------------------------------
# Cointegration (paper logic: OLS on log-prices + ADF test)
# ---------------------------------------------------------------------------
ADF_PVALUE = 0.10                 # same 10% level as the original code
ADF_REGRESSION = "ct"             # constant + trend, as in the original code

# ---------------------------------------------------------------------------
# Signals: EWMA bands
# ---------------------------------------------------------------------------
EWMA_HALFLIFE = 48                # candles (hours) for EWMA mean and std
ENTRY_K_GRID = [2.0, 2.5, 3.0]    # enter when |spread - mean| > k * std;
# k is chosen on the IN-SAMPLE only (best in-sample Sharpe of the strategy
# with frictions). Exit when the spread crosses back its EWMA mean.

# End-of-sample rule: no new entry on a pair during the last H candles of
# the backtest, H = OU half-life of that pair. The spread would not have the
# time to revert, and the position would only be force-closed on the last
# candle (paying the round trip for nothing).
NO_ENTRY_LAST_HALF_LIFE = True
# A dust-volume filter (swap must exceed 0.01 WETH) was tested and dropped:
# it removed 0.7% / 8.5% / 14.0% of the wstETH / rETH / weETH candles,
# above the 5% tolerance we set.

# ---------------------------------------------------------------------------
# Portfolio / optimisation (paper, eq. 6)
# ---------------------------------------------------------------------------
CAPITAL_PER_ASSET_ETH = 10.0      # initial inventory of each LST, valued in ETH
LAMBDA = 1.0                      # risk aversion (paper default)
RISK_FREE_RATE = 0.0              # for the Sharpe ratio (returns measured in ETH)

# ---------------------------------------------------------------------------
# Frictions
# ---------------------------------------------------------------------------
# Swap fee: real fee tier of each pool, read from the API (data/pools.csv).
# Slippage: constant product x*y=k with virtual reserves x = y = TVL/2 (in ETH).
# This is conservative: Uniswap V3 concentrated liquidity is much deeper
# around the current price than a full-range x*y=k pool of the same TVL.

# Gas: Ornstein-Uhlenbeck process on log(gas price in Gwei), simulated hourly:
#   x_{t+1} = x_t + KAPPA * (log(GAS_MEDIAN_GWEI) - x_t) + SIGMA * eps_t
# GAS_MEDIAN_GWEI ~ typical recent mainnet level (base fee + tip, low after
# the 2024-2025 blob / gas-limit upgrades); KAPPA = 0.10 per hour gives a
# half-life of ~7 h for gas spikes; SIGMA = 0.30 gives a stationary std of
# log-gas of 0.30/sqrt(1-0.9^2) ~ 0.69, i.e. spikes of 3-5x the median.
SLIPPAGE_MULTIPLIERS = [1, 10, 50]   # sensitivity: virtual reserves = m * TVL/2

GAS_MEDIAN_GWEI = 1.0
GAS_KAPPA = 0.10
GAS_SIGMA = 0.30
GAS_PER_SWAP = 150_000            # gas units of one Uniswap V3 swap
SWAPS_PER_TRADE = 4               # 2 swaps to open + 2 swaps to close
SEED = 42                         # random seed for the gas path
