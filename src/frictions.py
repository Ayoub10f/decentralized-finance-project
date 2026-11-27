"""
Market frictions of a DEX swap: fee tier, x*y=k slippage and stochastic gas.
All amounts are in ETH.
"""
import numpy as np
import pandas as pd

import config


def swap_out(value_in, fee, tvl_eth, reserve_mult=1.0):
    """ETH value received when swapping `value_in` ETH worth of one token.

    Uniswap takes the fee on the input, then the constant product x*y=k
    gives out = y * dx / (x + dx). We use virtual reserves
    x = y = R = reserve_mult * TVL/2 (both valued in ETH at the current
    price), so the price impact is dx / (R + dx). With reserve_mult = 1 this
    is conservative: Uniswap V3 concentrated liquidity (and Curve StableSwap)
    is much deeper near the peg than a full-range x*y=k pool with the same
    TVL; reserve_mult = 10 or 50 is used as a sensitivity analysis.
    Input: ETH value sold, fee tier (decimal), pool TVL in ETH, multiplier.
    Output: ETH value received.
    """
    reserve = reserve_mult * tvl_eth / 2
    dx = value_in * (1 - fee)
    return reserve * dx / (reserve + dx)


def cost_coefficients(long, short, pools, reserve_mult=1.0):
    """Round-trip cost coefficients of a pair trade (4 swaps, 2 per asset).

    Fees: F = 2 (fee_long + fee_short), linear in the traded value v.
    Slippage: a swap of v in reserve R loses v - R v / (R + v) ~ v^2 / R,
    so S = 2 (1/R_long + 1/R_short) and the cost is about S v^2.
    Input: asset names, pools table, reserve multiplier.
    Output: (F, S), S in 1/ETH.
    """
    r = {a: reserve_mult * pools.at[a, "tvl_eth"] / 2 for a in (long, short)}
    fees = 2 * (pools.at[long, "fee"] + pools.at[short, "fee"])
    slip = 2 * (1 / r[long] + 1 / r[short])
    return fees, slip


def simulate_gas(index, seed=config.SEED):
    """Hourly gas price path in Gwei: OU process on log(gas), fixed seed.

    x_{t+1} = x_t + KAPPA (log(median) - x_t) + SIGMA eps_t, gas = exp(x).
    Input: DatetimeIndex of the backtest. Output: Series of gas prices (Gwei).
    """
    rng = np.random.default_rng(seed)
    target = np.log(config.GAS_MEDIAN_GWEI)
    x = np.empty(len(index))
    x[0] = target
    for t in range(1, len(index)):
        x[t] = (x[t - 1] + config.GAS_KAPPA * (target - x[t - 1])
                + config.GAS_SIGMA * rng.standard_normal())
    return pd.Series(np.exp(x), index=index, name="gas_gwei")


def gas_cost(gas_gwei, n_swaps):
    """ETH paid in gas for `n_swaps` swaps at a given gas price.

    Input: gas price in Gwei, number of swaps. Output: cost in ETH.
    """
    return gas_gwei * 1e-9 * config.GAS_PER_SWAP * n_swaps
