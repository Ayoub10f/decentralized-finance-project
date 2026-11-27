"""
Out-of-sample performance metrics (on an equity curve in ETH) and figures.
"""
import matplotlib
matplotlib.use("Agg")                     # write PNG files, no window
import matplotlib.pyplot as plt
import numpy as np

import config
from src.cointegration import compute_spread

HOURS_PER_YEAR = 24 * 365
# Categorical colours (fixed order) + neutral grey for the reference series
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GREY, INK = "#8a8985", "#3d3d3a"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def sharpe_ratio(equity):
    """Annualised Sharpe ratio of hourly returns.

    Input: equity Series (one value per 1h candle).
    Output: Sharpe ratio (NaN if returns are constant).
    """
    rets = equity.pct_change().dropna()
    excess = rets - config.RISK_FREE_RATE / HOURS_PER_YEAR
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(HOURS_PER_YEAR)


def max_drawdown(equity):
    """Largest peak-to-trough loss, as a (negative) fraction.

    Input: equity Series. Output: max drawdown, e.g. -0.02 for -2%.
    """
    return (equity / equity.cummax() - 1).min()


def win_rate(trades):
    """Share of winning trades, in %.

    Input: trades DataFrame with a pnl_eth column. Output: % (NaN if no trade).
    """
    return np.nan if trades.empty else 100 * (trades["pnl_eth"] > 0).mean()


def gain_loss_ratio(trades):
    """Average gain of winning trades / average loss of losing trades.

    Input: trades DataFrame. Output: ratio (NaN = n/a, e.g. no losing trade).
    """
    if trades.empty:
        return np.nan
    wins = trades.loc[trades["pnl_eth"] > 0, "pnl_eth"]
    losses = trades.loc[trades["pnl_eth"] <= 0, "pnl_eth"]
    if wins.empty or losses.empty or losses.mean() == 0:
        return np.nan
    return wins.mean() / abs(losses.mean())


def summary(equity, trades):
    """All metrics of one strategy in a dict.

    Input: equity Series, trades DataFrame. Output: dict.
    """
    return {
        "final_value_eth": equity.iloc[-1],
        "total_return_pct": 100 * (equity.iloc[-1] / equity.iloc[0] - 1),
        "sharpe": sharpe_ratio(equity),
        "win_rate_pct": win_rate(trades),
        "avg_gain_over_avg_loss": gain_loss_ratio(trades),
        "max_drawdown_pct": 100 * max_drawdown(equity),
        "n_trades": len(trades),
        "avg_size_eth": trades["size_eth"].mean() if len(trades) else np.nan,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax):
    """Recessive grid and axes for a matplotlib Axes."""
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ae")
    ax.tick_params(colors=INK, labelsize=8)


def plot_spreads(prices, pairs, dev, z, trades, k, path):
    """Spread of each pair with its EWMA mean, +-k EWMA std bands and trades.

    Input: out-of-sample prices, pairs table, EWMA deviation and z-score
           DataFrames, trades DataFrame (one variant), k, output path.
    """
    fig, axes = plt.subplots(len(pairs), 1, figsize=(9, 2.6 * len(pairs)), sharex=True)
    for ax, (pair, r) in zip(np.atleast_1d(axes), pairs.iterrows()):
        s = 1e4 * compute_spread(prices, r["x"], r["y"], r["alpha"], r["beta"])
        d, zz = 1e4 * dev.loc[prices.index, pair], z.loc[prices.index, pair]
        mean, sd = s - d, d / zz
        ax.fill_between(s.index, mean - k * sd, mean + k * sd, color=GREY, alpha=0.15,
                        linewidth=0, label=f"EWMA mean ± {k:g} std")
        ax.plot(mean, color=GREY, linewidth=1)
        ax.plot(s, color=COLORS[0], linewidth=1, label="spread")
        tr = trades[trades["pair"] == pair] if len(trades) else trades
        if len(tr):
            ax.scatter(tr["open"], s.loc[tr["open"]], marker="^", s=40, color=COLORS[1],
                       edgecolor="white", linewidth=1, zorder=3, label="entry")
            ax.scatter(tr["close"], s.loc[tr["close"]], marker="o", s=30, color=COLORS[2],
                       edgecolor="white", linewidth=1, zorder=3, label="exit")
        ax.set_title(pair, loc="left", fontsize=9, color=INK)
        ax.set_ylabel("spread (bp)", fontsize=8, color=INK)
        _style(ax)
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=4, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity(curves, path):
    """Portfolio value of each variant, and its difference to buy & hold.

    Input: dict name -> equity Series (must contain 'buy_and_hold'), output path.
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1]})
    bh = curves["buy_and_hold"]
    others = [n for n in curves if n != "buy_and_hold"]
    for i, name in enumerate(others):
        top.plot(curves[name], color=COLORS[i], linewidth=1.5, label=name)
        bottom.plot(curves[name] - bh, color=COLORS[i], linewidth=1.5, label=name)
    # reference drawn last and dashed so it stays visible under overlapping lines
    top.plot(bh, color=INK, linewidth=1.2, linestyle="--", label="buy_and_hold")
    bottom.axhline(0, color=GREY, linewidth=1)
    top.set_ylabel("portfolio value (ETH)", fontsize=8, color=INK)
    bottom.set_ylabel("value - buy & hold (ETH)", fontsize=8, color=INK)
    top.legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    for ax in (top, bottom):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(table, path):
    """Excess value over buy & hold and number of trades vs slippage reserves.

    Input: metrics table (rows buy_and_hold, no_frictions, frictions_R<m>),
           output path.
    """
    rows = [r for r in table.index if r.startswith("frictions_R")]
    labels = [f"R × {r.split('_R')[1]}" for r in rows]
    bh = table.at["buy_and_hold", "final_value_eth"]
    excess = table.loc[rows, "final_value_eth"] - bh
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    left.bar(labels, excess, color=COLORS[1], width=0.5)
    left.axhline(table.at["no_frictions", "final_value_eth"] - bh, color=COLORS[0],
                 linewidth=1.5, linestyle="--", label="no frictions")
    left.axhline(0, color=GREY, linewidth=1)
    left.set_ylabel("final value - buy & hold (ETH)", fontsize=8, color=INK)
    left.legend(fontsize=7, frameon=False)
    right.bar(labels, table.loc[rows, "n_trades"], color=COLORS[1], width=0.5)
    right.axhline(table.at["no_frictions", "n_trades"], color=COLORS[0],
                  linewidth=1.5, linestyle="--", label="no frictions")
    right.set_ylabel("number of trades", fontsize=8, color=INK)
    right.legend(fontsize=7, frameon=False)
    for ax in (left, right):
        ax.set_xlabel("virtual reserves (multiple of TVL/2)", fontsize=8, color=INK)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
