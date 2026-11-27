"""
Cointegration testing and Ornstein-Uhlenbeck estimation of the spreads.

Paper logic (Engle-Granger, pairwise): regress log-prices on each other,
take the residual as the spread and keep it if the ADF test rejects a unit
root. Extension 1: fit an OU process on each spread to get its
mean-reversion speed and half-life. All parameters use the in-sample only.
Time unit: one 1-hour candle.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import config


def fit_pair(log_x, log_y):
    """OLS hedge regression log_y = alpha + beta * log_x (as in the paper).

    Input: two log-price Series (in-sample). Output: (alpha, beta).
    """
    res = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    return res.params.iloc[0], res.params.iloc[1]


def compute_spread(prices, x, y, alpha, beta):
    """Spread s_t = log P_y - beta * log P_x - alpha, for any sample.

    Uses only prices at time t and fixed in-sample parameters (no look-ahead).
    Input: prices DataFrame, asset names x and y, alpha, beta.
    Output: spread Series.
    """
    return np.log(prices[y]) - beta * np.log(prices[x]) - alpha


def fit_ou(spread):
    """Estimate an Ornstein-Uhlenbeck process ds = theta (mu - s) dt + sigma dW.

    Exact discretisation with dt = 1 candle is an AR(1):
        s_{t+1} = a + b s_t + e,  b = exp(-theta),  mu = a / (1 - b).
    Input: in-sample spread Series.
    Output: dict with theta (per candle), half-life (candles), mu, sigma
            (diffusion, per sqrt(candle)) and sigma_eq (stationary std).
    """
    s = spread.to_numpy()
    res = sm.OLS(s[1:], sm.add_constant(s[:-1])).fit()
    a, b = res.params
    if not 0 < b < 1:                  # no mean reversion in the data
        return {"theta": np.nan, "half_life": np.inf, "mu": np.nan,
                "sigma": np.nan, "sigma_eq": np.nan}
    theta = -np.log(b)
    resid_var = np.var(res.resid, ddof=2)          # = sigma^2 (1-b^2) / (2 theta)
    return {
        "theta": theta,
        # Extension 1: the OU half-life ln(2)/theta replaces the paper's
        # empirical "average time to revert to the mean" (mr). It is used
        # as the holding horizon H in the expected return and risk.
        "half_life": np.log(2) / theta,
        "mu": a / (1 - b),
        "sigma": np.sqrt(resid_var * 2 * theta / (1 - b ** 2)),
        "sigma_eq": np.sqrt(resid_var / (1 - b ** 2)),
    }


def ou_expected_gain(deviation, theta, horizon):
    """Expected spread move back towards its mean over `horizon` candles.

    E = |s_t - mu| (1 - exp(-theta H)); with H = half-life this is |s_t - mu| / 2.
    Input: deviation s_t - mu, theta, horizon H (candles).
    Output: expected gain in log-spread units.
    """
    return abs(deviation) * (1 - np.exp(-theta * horizon))


def ou_variance(theta, sigma, horizon):
    """Variance of an OU process after `horizon` candles (risk of a pair).

    Var = sigma^2 (1 - exp(-2 theta H)) / (2 theta).
    Input: theta, diffusion sigma, horizon H (candles). Output: variance.
    """
    return sigma ** 2 * (1 - np.exp(-2 * theta * horizon)) / (2 * theta)


def find_pairs(insample):
    """Test every pair for cointegration and fit OU on the in-sample.

    Input: in-sample prices DataFrame.
    Output: DataFrame (one row per pair) with alpha, beta, ADF p-value,
            cointegrated flag and OU parameters.
    """
    rows = []
    log_p = np.log(insample)
    for x, y in combinations(insample.columns, 2):
        alpha, beta = fit_pair(log_p[x], log_p[y])
        spread = compute_spread(insample, x, y, alpha, beta)
        pvalue = adfuller(spread, regression=config.ADF_REGRESSION)[1]
        rows.append({"pair": f"{y}:{x}", "x": x, "y": y, "alpha": alpha,
                     "beta": beta, "adf_pvalue": pvalue,
                     "cointegrated": pvalue < config.ADF_PVALUE,
                     **fit_ou(spread)})
    return pd.DataFrame(rows).set_index("pair")
