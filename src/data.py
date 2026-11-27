"""
Data acquisition and cleaning (public GeckoTerminal API, no key).

- fetch_pool_info : fee tier and TVL of a pool
- fetch_ohlcv     : hourly candles of a pool, paginated backwards
- load_data       : download once, cache in data/*.csv, return clean prices
- split_sample    : 70% in-sample / 30% out-of-sample
"""
import os
import re
import time

import numpy as np
import pandas as pd
import requests

import config

BASE_URL = "https://api.geckoterminal.com/api/v2/networks/eth/pools"


def api_get(url, params=None):
    """GET a GeckoTerminal endpoint, pausing between calls and retrying on 429.

    Input: url, query params. Output: decoded JSON (dict).
    """
    for _ in range(5):
        time.sleep(config.API_PAUSE_S)
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:          # rate limit hit: wait one minute
            time.sleep(60)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Too many 429 errors on {url}")


def fetch_pool_info(symbol, address):
    """Read name, DEX, fee tier and TVL of one pool.

    Input: LST symbol, pool address.
    Output: dict with fee (decimal, e.g. 0.0001) and TVL in USD and in ETH.
    """
    attr = api_get(f"{BASE_URL}/{address}")["data"]["attributes"]
    fee = re.search(r"([\d.]+)%", attr["name"])   # tier is in the pool name
    if fee is None:
        raise ValueError(f"No fee tier in pool name '{attr['name']}'")
    tvl_usd = float(attr["reserve_in_usd"])
    weth_usd = float(attr["quote_token_price_usd"])  # quote token = WETH
    return {
        "symbol": symbol,
        "address": address,
        "name": attr["name"],
        "fee": float(fee.group(1)) / 100,
        "tvl_usd": tvl_usd,
        "tvl_eth": tvl_usd / weth_usd,
        "snapshot_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
    }


def fetch_ohlcv(symbol, address, start, end):
    """Download hourly OHLCV candles of a pool between start and end.

    Prices are quoted in the quote token (WETH), i.e. 1 LST = close WETH.
    The API returns at most 1000 candles per call, newest first, so we walk
    backwards with `before_timestamp`.
    Input: symbol, pool address, start/end as UTC Timestamps.
    Output: DataFrame indexed by UTC time with open, high, low, close, volume.
    """
    rows, before = [], int(end.timestamp())
    while before > start.timestamp():
        params = {"aggregate": 1, "limit": 1000, "currency": "token",
                  "before_timestamp": before}
        js = api_get(f"{BASE_URL}/{address}/ohlcv/{config.TIMEFRAME}", params)
        base = js["meta"]["base"]["symbol"]
        if base.lower() != symbol.lower():          # price must be LST/WETH
            raise ValueError(f"Base token is {base}, expected {symbol}")
        candles = js["data"]["attributes"]["ohlcv_list"]
        if not candles:
            break
        rows += candles
        oldest = min(c[0] for c in candles)
        if oldest >= before:                         # no progress: stop
            break
        before = oldest                              # continue before oldest
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df[(df.index >= start) & (df.index < end)]


def filter_outliers(close, threshold=config.OUTLIER_THRESHOLD,
                    confirm=config.OUTLIER_CONFIRM):
    """Causal filter of aberrant prints (uses only past and current candles).

    A close more than `threshold` (log-distance) away from the last accepted
    close is rejected (set to NaN). If `confirm` consecutive real candles
    deviate, the latest one is accepted as the new level (a real move).
    Input: close Series (NaN = no candle). Output: filtered close Series.
    """
    clean = close.copy()
    last, streak = np.nan, 0
    for t, p in close.items():
        if np.isnan(p):
            continue
        if np.isnan(last) or abs(np.log(p / last)) <= threshold:
            last, streak = p, 0            # normal candle
        else:
            streak += 1
            if streak >= confirm:
                last, streak = p, 0        # deviation persisted: accept it
            else:
                clean[t] = np.nan          # rejected (for now)
    return clean


def clean_prices(raw, start, end):
    """Filter outliers and align all pools on a common hourly grid.

    GeckoTerminal skips hours with no swap: the price did not move, so the
    last close is carried forward (no future data is used). The `traded`
    table marks the hours where a real, accepted swap happened (volume > 0);
    the backtest only opens/closes positions on such hours.
    Input: dict symbol -> OHLCV DataFrame, start/end Timestamps.
    Output: (close prices, traded flags, report of missing/rejected hours).
    """
    grid = pd.date_range(start, end, freq="1h", inclusive="left")
    raw_close = pd.DataFrame({s: df["close"] for s, df in raw.items()}).reindex(grid)
    volume = pd.DataFrame({s: df["volume"] for s, df in raw.items()}).reindex(grid)
    closes = raw_close.apply(filter_outliers)
    traded = closes.notna() & (volume > 0)
    report = pd.DataFrame({
        "candles": raw_close.notna().sum(),
        "outliers_rejected": (raw_close.notna() & closes.isna()).sum(),
        "hours_without_trade_pct": 100 * (~traded).mean(),
    })
    closes = closes.ffill().dropna()   # drop hours before every pool has traded
    return closes, traded.loc[closes.index], report


def load_data():
    """Download (or read from cache) prices and pool info for all assets.

    Output: (prices DataFrame of LST/WETH closes, traded flags DataFrame,
             pools DataFrame, report DataFrame).
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    end = pd.Timestamp(config.END_DATE, tz="UTC")
    start = end - pd.Timedelta(days=config.HISTORY_DAYS)

    raw = {}
    for symbol, address in config.ASSETS.items():
        path = os.path.join(config.DATA_DIR, f"{symbol}_WETH_1h.csv")
        if not os.path.exists(path):
            print(f"  downloading {symbol} ...")
            fetch_ohlcv(symbol, address, start, end).to_csv(path)
        raw[symbol] = pd.read_csv(path, index_col="time", parse_dates=["time"])

    pools_path = os.path.join(config.DATA_DIR, "pools.csv")
    if not os.path.exists(pools_path):
        info = [fetch_pool_info(s, a) for s, a in config.ASSETS.items()]
        pd.DataFrame(info).to_csv(pools_path, index=False)
    pools = pd.read_csv(pools_path, index_col="symbol").loc[list(config.ASSETS)]

    prices, traded, report = clean_prices(raw, start, end)
    return prices, traded, pools, report


def split_sample(prices, frac=config.IN_SAMPLE_FRAC):
    """Split prices chronologically into in-sample and out-of-sample.

    Input: prices DataFrame, in-sample fraction. Output: (insample, outsample).
    """
    cut = int(len(prices) * frac)
    return prices.iloc[:cut], prices.iloc[cut:]
