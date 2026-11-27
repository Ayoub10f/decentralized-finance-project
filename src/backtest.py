"""
Candle-by-candle simulation of the strategy.

Portfolio: each LST starts with CAPITAL_PER_ASSET_ETH worth of tokens (as
the paper starts with capital in each currency) plus a WETH account that
pays the gas. Opening a pair = sell the short LST for WETH, then buy the
long LST with that WETH (2 swaps); closing = the reverse (2 swaps).
Positions are opened/closed only on candles where both legs had a real swap.
Everything is valued in ETH.
"""
import pandas as pd

import config
from src import frictions
from src.cointegration import ou_expected_gain, ou_variance
from src.optimizer import optimise_weights
from src.signals import entry_signal, exit_signal


def execute(value_eth, sell, buy, prices_t, pools, with_frictions, reserve_mult):
    """Swap `value_eth` worth of token `sell` into token `buy` through WETH.

    Input: ETH value sold, asset names, current prices, pools table,
           frictions flag, slippage reserve multiplier.
    Output: number of `buy` tokens received.
    """
    if with_frictions:
        for asset in (sell, buy):
            value_eth = frictions.swap_out(value_eth, pools.at[asset, "fee"],
                                           pools.at[asset, "tvl_eth"], reserve_mult)
    return value_eth / prices_t[buy]


def trading_weights(open_pos, init_tokens):
    """Net fraction of each asset's initial inventory sold by open positions.

    Input: open positions dict, initial token amounts. Output: dict asset -> tw.
    """
    tw = {a: 0.0 for a in init_tokens}
    for pos in open_pos.values():
        tw[pos["short"]] += pos["sold"] / init_tokens[pos["short"]]
        tw[pos["long"]] -= pos["bought"] / init_tokens[pos["long"]]
    return tw


def run_backtest(prices, traded, deviations, zscores, pairs, pools, k,
                 with_frictions, reserve_mult=1.0):
    """Simulate the strategy on one sample.

    Input: prices, traded flags, spread deviations from the EWMA mean and
           z-scores (one column per pair), pairs table (cointegrated pairs
           with OU parameters), pools table, entry threshold k, frictions
           flag, slippage reserve multiplier.
    Output: (equity Series in ETH, trades DataFrame, counters dict).
    """
    cap = config.CAPITAL_PER_ASSET_ETH
    init_tokens = {a: cap / prices.iloc[0][a] for a in prices.columns}
    holdings = dict(init_tokens)
    weth = 0.0                                   # pays the gas (can go < 0)
    gas = frictions.simulate_gas(prices.index)
    open_pos, trades, equity = {}, [], []
    count = {"entries_ignored_no_swap": 0, "exits_delayed_no_swap": 0,
             "entries_rejected_optimizer": 0, "entries_rejected_gas": 0,
             "entries_skipped_end": 0}
    gas_roundtrip = frictions.gas_cost(config.GAS_MEDIAN_GWEI, config.SWAPS_PER_TRADE)
    last_t = prices.index[-1]

    for i, t in enumerate(prices.index):
        px = prices.loc[t]
        candles_left = len(prices) - 1 - i
        gas_2swaps = frictions.gas_cost(gas[t], 2) if with_frictions else 0.0

        # 1) close positions whose spread is back to its EWMA mean
        for pair in list(open_pos):
            pos = open_pos[pair]
            if t != last_t:                      # force close on the last candle
                if not exit_signal(zscores.at[t, pair], pos["direction"]):
                    continue
                if not (traded.at[t, pos["long"]] and traded.at[t, pos["short"]]):
                    count["exits_delayed_no_swap"] += 1
                    continue
            back = execute(pos["bought"] * px[pos["long"]], pos["long"], pos["short"],
                           px, pools, with_frictions, reserve_mult)
            holdings[pos["long"]] -= pos["bought"]
            holdings[pos["short"]] += back
            weth -= gas_2swaps
            pnl = (back - pos["sold"]) * px[pos["short"]] - pos["gas"] - gas_2swaps
            trades.append({"pair": pair, "open": pos["time"], "close": t,
                           "direction": pos["direction"], "weight": pos["w"],
                           "size_eth": pos["size"], "pnl_eth": pnl})
            del open_pos[pair]

        # 2) collect new entry signals and their OU expected gain / risk
        candidates, gain, var, fee, slip = [], {}, {}, {}, {}
        for pair, row in pairs.iterrows():
            z = zscores.at[t, pair]
            if pair in open_pos or t == last_t or pd.isna(z):
                continue
            direction = entry_signal(z, k)
            if direction == 0:
                continue
            if config.NO_ENTRY_LAST_HALF_LIFE and candles_left <= row["half_life"]:
                count["entries_skipped_end"] += 1    # no time left to revert
                continue
            long, short = (row["y"], row["x"]) if direction == 1 else (row["x"], row["y"])
            if not (traded.at[t, long] and traded.at[t, short]):
                count["entries_ignored_no_swap"] += 1
                continue
            candidates.append((pair, long, short, direction))
            gain[pair] = ou_expected_gain(deviations.at[t, pair], row["theta"], row["half_life"])
            var[pair] = ou_variance(row["theta"], row["sigma"], row["half_life"])
            fee[pair], slip[pair] = frictions.cost_coefficients(long, short, pools, reserve_mult)

        # 3) size them with the optimiser (fees and slippage inside the
        #    objective in variant b), check the gas, open the positions
        if candidates:
            weights = optimise_weights([c[:3] for c in candidates], gain, var,
                                       trading_weights(open_pos, init_tokens),
                                       fee if with_frictions else None,
                                       slip if with_frictions else None)
            for pair, long, short, direction in candidates:
                w = weights[pair]
                if w < 1e-3:
                    count["entries_rejected_optimizer"] += 1
                    continue
                sold = min(w * cap / px[short], holdings[short])
                value = sold * px[short]
                net_gain = value * (gain[pair] - fee[pair]) - slip[pair] * value ** 2
                if with_frictions and net_gain <= gas_roundtrip:
                    count["entries_rejected_gas"] += 1
                    continue
                bought = execute(value, short, long, px, pools, with_frictions, reserve_mult)
                holdings[short] -= sold
                holdings[long] += bought
                weth -= gas_2swaps
                open_pos[pair] = {"long": long, "short": short, "direction": direction,
                                  "w": w, "size": value, "sold": sold, "bought": bought,
                                  "gas": gas_2swaps, "time": t}

        equity.append(sum(holdings[a] * px[a] for a in holdings) + weth)

    return pd.Series(equity, index=prices.index), pd.DataFrame(trades), count


def buy_and_hold(prices):
    """Value in ETH of simply holding the initial LST inventories.

    Input: prices. Output: equity Series in ETH.
    """
    tokens = config.CAPITAL_PER_ASSET_ETH / prices.iloc[0]
    return (prices * tokens).sum(axis=1)
