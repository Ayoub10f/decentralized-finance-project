"""
Extension 2: dynamic EWMA bands instead of the paper's fixed z-score.

At candle t the band centre and width are the exponentially weighted mean
and standard deviation of the spread up to t-1 (strictly past data):
    dev_t = s_t - m_{t-1},   z_t = dev_t / sd_{t-1}
Entry : |z_t| > k   (short the spread if z > k, long it if z < -k)
Exit  : z_t crosses back its EWMA mean (sign change of z).
"""
import config


def ewma_bands(spread, halflife=config.EWMA_HALFLIFE):
    """Deviation of the spread from its past EWMA mean, raw and in std units.

    Input: spread Series (whole sample, so the EWMA is warm at the start of
    the out-of-sample; only past values are used at each t).
    Output: (deviation Series, z-score Series), NaN on the first candles.
    """
    mean = spread.ewm(halflife=halflife, adjust=False).mean().shift(1)
    std = spread.ewm(halflife=halflife, adjust=False).std().shift(1)
    deviation = spread - mean
    return deviation, deviation / std


def entry_signal(z, k):
    """Desired new position on the spread s = log y - beta log x.

    Input: z-score value at t, band multiplier k.
    Output: +1 = long spread (buy y, sell x), -1 = short spread
            (sell y, buy x), 0 = no entry.
    """
    if z < -k:
        return 1
    if z > k:
        return -1
    return 0


def exit_signal(z, direction):
    """True when the spread has come back to its EWMA mean.

    Input: z-score at t, direction of the open position (+1 or -1).
    Output: bool.
    """
    return z * direction >= 0
