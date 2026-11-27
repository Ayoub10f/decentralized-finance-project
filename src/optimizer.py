"""
Capital allocation: the paper's bi-objective problem (eq. 6) solved with cvxpy.

For every pair n that signals at candle t we choose a long weight
wl_n in [0, 1] (fraction of the long asset's initial capital bought) and a
short weight ws_n in [-1, 0] (fraction of the short asset's initial
inventory sold):

  max  sum_n wl_n E_n  -  lambda * sum_n wl_n^2 V_n
  s.t. 0 <= wl_n <= 1,  -1 <= ws_n <= 0
       wl_n = -ws_n                    market neutrality (equal ETH value,
                                       all assets start with the same capital)
       tw_c + sum(sold_c) - sum(bought_c) <= 1   for every asset c
                                       (cannot sell more than we hold)

Inputs come from the OU model of each spread (horizon H = half-life, in
candles, which replaces the paper's mean-reversion time mr):
  E_n = |s_t - m_t| (1 - exp(-theta H))           expected spread gain
  V_n = sigma^2 (1 - exp(-2 theta H)) / (2 theta)  OU variance over H
The paper builds E from the assets' drift and counts the short leg as
+|ws| r_short; we instead use the spread's own mean reversion, which is
what a pair trade earns.

With frictions (variant b) two trading costs enter the objective, per unit
of capital C (each leg trades v = w C ETH, 4 swaps per round trip):
  - fees, linear      : w * F_n,          F_n = 2 (fee_long + fee_short)
  - slippage, x*y=k   : w^2 * C * S_n,    S_n = 2 (1/R_long + 1/R_short)
    (a swap of v ETH in a pool of reserve R loses about v^2 / R)
Gas is a fixed cost per trade, so it is checked after the optimisation.
"""
import cvxpy as cp

import config


def optimise_weights(signals, exp_gain, variance, trading_weights,
                     fee=None, slip=None, lam=config.LAMBDA):
    """Solve the allocation problem for the pairs that signal at one candle.

    Input: signals = list of (pair, long asset, short asset);
           exp_gain, variance = dicts pair -> E_n, V_n;
           trading_weights = dict asset -> net fraction already sold;
           fee, slip = dicts pair -> F_n, S_n (None = no frictions).
    Output: dict pair -> weight w = wl = -ws (0 if not worth trading).
    """
    n = len(signals)
    wl = cp.Variable(n)
    ws = cp.Variable(n)
    ret = sum(wl[i] * exp_gain[s[0]] for i, s in enumerate(signals))
    risk = sum(cp.square(wl[i]) * variance[s[0]] for i, s in enumerate(signals))
    if fee is not None:
        cap = config.CAPITAL_PER_ASSET_ETH
        ret -= sum(wl[i] * fee[s[0]] + cp.square(wl[i]) * cap * slip[s[0]]
                   for i, s in enumerate(signals))

    cons = [wl >= 0, wl <= 1, ws >= -1, ws <= 0, wl == -ws]
    for asset, tw in trading_weights.items():
        if not any(asset in s[1:] for s in signals):
            continue                          # asset not traded at this candle
        sold = sum(-ws[i] for i, s in enumerate(signals) if s[2] == asset)
        bought = sum(wl[i] for i, s in enumerate(signals) if s[1] == asset)
        cons.append(tw + sold - bought <= 1)

    cp.Problem(cp.Maximize(ret - lam * risk), cons).solve()
    if wl.value is None:                      # solver failure: no trade
        return {s[0]: 0.0 for s in signals}
    return {s[0]: float(max(wl.value[i], 0)) for i, s in enumerate(signals)}
