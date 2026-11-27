"""Run the whole pipeline in order: python main.py"""
import os

import pandas as pd

import config
from src import backtest, cointegration, data, metrics, signals


def spread_bands(prices, pairs):
    """EWMA deviation and z-score of every pair's spread.

    Input: prices (whole sample), pairs table with in-sample alpha/beta.
    Output: (deviations DataFrame, z-scores DataFrame), one column per pair.
    """
    dev, z = {}, {}
    for pair, r in pairs.iterrows():
        spread = cointegration.compute_spread(prices, r["x"], r["y"], r["alpha"], r["beta"])
        dev[pair], z[pair] = signals.ewma_bands(spread)
    return pd.DataFrame(dev), pd.DataFrame(z)


def run(prices, traded, dev, z, pairs, pools, k, with_frictions, reserve_mult=1.0):
    """Backtest on the dates of `prices` and return (metrics dict, equity, trades, counters)."""
    idx = prices.index
    equity, trades, count = backtest.run_backtest(
        prices, traded.loc[idx], dev.loc[idx], z.loc[idx], pairs, pools, k,
        with_frictions, reserve_mult)
    return metrics.summary(equity, trades), equity, trades, count


def main():
    """Run every step of the project and print a short summary of each."""
    pd.set_option("display.width", 160)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("1) Data")
    prices, traded, pools, report = data.load_data()
    insample, outsample = data.split_sample(prices)
    print(pools[["name", "fee", "tvl_eth"]])
    print(report.round(2))
    print(f"  period     : {prices.index[0]} -> {prices.index[-1]} ({len(prices)} candles)")
    print(f"  in-sample  : {insample.index[0]} -> {insample.index[-1]} ({len(insample)})")
    print(f"  out-sample : {outsample.index[0]} -> {outsample.index[-1]} ({len(outsample)})")

    print("\n2) Cointegration and OU (in-sample)")
    pairs = cointegration.find_pairs(insample)
    pairs.to_csv(os.path.join(config.RESULTS_DIR, "cointegration.csv"))
    print(pairs[["beta", "adf_pvalue", "cointegrated", "theta", "half_life"]].round(4))
    pairs = pairs[pairs["cointegrated"]]
    # EWMA bands on the whole sample: they only look backwards, so the
    # out-of-sample starts with a warm EWMA and no future information.
    dev, z = spread_bands(prices, pairs)

    print("\n3) Entry threshold k, chosen on the in-sample only")
    rows = []
    for k in config.ENTRY_K_GRID:
        res_a = run(insample, traded, dev, z, pairs, pools, k, False)[0]
        res_b = run(insample, traded, dev, z, pairs, pools, k, True)[0]
        rows.append({"k": k, "sharpe_no_frictions": res_a["sharpe"],
                     "trades_no_frictions": res_a["n_trades"],
                     "sharpe_frictions": res_b["sharpe"],
                     "trades_frictions": res_b["n_trades"]})
    grid = pd.DataFrame(rows).set_index("k")
    print(grid.round(3).to_string())
    # best in-sample Sharpe with frictions; ties broken by Sharpe without frictions
    k = grid.sort_values(["sharpe_frictions", "sharpe_no_frictions"]).index[-1]
    print(f"  retained k = {k}")

    print("\n4) Out-of-sample backtest")
    table, curves, trades = {}, {}, {}
    curves["buy_and_hold"] = backtest.buy_and_hold(outsample)
    table["buy_and_hold"] = metrics.summary(curves["buy_and_hold"], pd.DataFrame())
    variants = [("no_frictions", False, 1)] + [
        (f"frictions_R{m}", True, m) for m in config.SLIPPAGE_MULTIPLIERS]
    for name, with_frictions, m in variants:
        table[name], curves[name], trades[name], count = run(
            outsample, traded, dev, z, pairs, pools, k, with_frictions, m)
        print(f"  {name} counters: {count}")
    table = pd.DataFrame(table).T
    table.to_csv(os.path.join(config.RESULTS_DIR, "metrics.csv"))
    print(table.round(4).to_string())

    print("\n5) Figures")
    out = config.RESULTS_DIR
    metrics.plot_spreads(outsample, pairs, dev, z, trades["no_frictions"], k,
                         os.path.join(out, "spreads_no_frictions.png"))
    deepest = f"frictions_R{config.SLIPPAGE_MULTIPLIERS[-1]}"   # most trades with frictions
    metrics.plot_spreads(outsample, pairs, dev, z, trades[deepest], k,
                         os.path.join(out, f"spreads_{deepest}.png"))
    metrics.plot_equity(curves, os.path.join(out, "equity_curves.png"))
    metrics.plot_sensitivity(table, os.path.join(out, "slippage_sensitivity.png"))
    print(f"  saved PNG files in {out}/")


if __name__ == "__main__":
    main()
