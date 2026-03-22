"""
intraday_fetcher.py — Fetches 1-min / 5-min intraday bars for Vietnam stocks via vnstock.

Intraday data unlocks:
  - Realized volatility (much better than close-to-close vol)
  - Volume profile (VWAP deviation, volume-at-price clustering)
  - Order flow imbalance proxy (up-tick vs down-tick volume)
  - Intraday momentum (morning vs afternoon session return)
  - Gap behaviour (open vs previous close)

vnstock intraday API:
  stock.quote.intraday(symbol, page_size=N)   -> recent ticks / 1-min bars
  stock.quote.history(interval="1")           -> 1-min historical (limited window)
  stock.quote.history(interval="5")           -> 5-min historical
  stock.quote.history(interval="15")          -> 15-min historical
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("intraday_fetcher")

INTRADAY_CACHE_DIR = Path("data/intraday_cache")
INTRADAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# How many days of intraday history to keep (vnstock caps at ~30-60 days)
INTRADAY_LOOKBACK_DAYS = 30


def _cache_path(symbol: str, interval: str) -> Path:
    return INTRADAY_CACHE_DIR / f"{symbol}_{interval}.pkl"


def _load_cache(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    p = _cache_path(symbol, interval)
    if not p.exists():
        return None
    age_h = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600
    if age_h > config.CACHE_TTL_HOURS:
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _save_cache(symbol: str, interval: str, df: pd.DataFrame):
    with open(_cache_path(symbol, interval), "wb") as f:
        pickle.dump(df, f)


def fetch_intraday(
    symbol: str,
    interval: str = "5",   # "1", "5", "15", "30", "60"
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday OHLCV bars.
    Returns DataFrame indexed by datetime with columns: open, high, low, close, volume
    Returns None if unavailable (new stock, API limit, etc.)
    """
    if use_cache:
        cached = _load_cache(symbol, interval)
        if cached is not None:
            return cached

    try:
        from vnstock import Vnstock
        vn = Vnstock()
        stock = vn.stock(symbol=symbol, source="VCI")

        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=INTRADAY_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

        df = stock.quote.history(start=start, end=end, interval=interval)
        if df is None or df.empty:
            logger.warning(f"[intraday] {symbol} interval={interval}: empty response")
            return None

        df.columns = [c.lower() for c in df.columns]
        rename = {"time": "datetime", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        df = df.rename(columns=rename)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")

        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(set(df.columns)):
            logger.warning(f"[intraday] {symbol}: missing columns {needed - set(df.columns)}")
            return None

        df = df[list(needed)].astype(float).sort_index()
        logger.info(f"[intraday] {symbol} interval={interval}: {len(df)} bars fetched")
        _save_cache(symbol, interval, df)
        return df

    except Exception as e:
        logger.warning(f"[intraday] {symbol} interval={interval}: {e}")
        return None


def build_intraday_features(
    symbol: str,
    daily_index: pd.DatetimeIndex,
    interval: str = "5",
) -> pd.DataFrame:
    """
    Aggregate intraday bars into *daily* features that can be merged
    with the main daily feature matrix.

    Returns a DataFrame indexed by date with columns:
        rv_5min          – realized volatility from 5-min returns (annualised)
        rv_1min          – realized volatility from 1-min returns (if available)
        volume_imbalance – (up_volume - down_volume) / total_volume  [-1, 1]
        intraday_mom     – morning session return (open→12:00) relative to full day
        vwap_dev         – close deviation from intraday VWAP
        range_ratio      – (high-low) / open  — intraday range normalised
        tick_intensity   – avg number of bars per hour (proxy for activity)
        open_gap         – (open - prev_close) / prev_close
        afternoon_vol    – realized vol in afternoon session only
    """
    bars = fetch_intraday(symbol, interval=interval)
    if bars is None or bars.empty:
        # Return empty DataFrame with the right columns so merge still works
        cols = [
            "rv_5min", "volume_imbalance", "intraday_mom",
            "vwap_dev", "range_ratio", "tick_intensity", "open_gap",
        ]
        return pd.DataFrame(index=daily_index, columns=cols, dtype=float)

    results = []

    # Group bars by calendar date
    bars_by_date = bars.groupby(bars.index.date)

    for date, day_bars in bars_by_date:
        if len(day_bars) < 4:
            continue

        ret = day_bars["close"].pct_change().dropna()

        # Realized volatility (annualised, assuming 252 trading days, 5h session)
        bars_per_day = len(day_bars)
        if interval == "1":
            bars_per_year = 252 * 300   # 5h * 60min
        elif interval == "5":
            bars_per_year = 252 * 60    # 5h * 12 bars
        else:
            bars_per_year = 252 * 20
        rv = ret.std() * np.sqrt(bars_per_year) if len(ret) > 1 else np.nan

        # Volume imbalance: up-tick vs down-tick bars
        up_vol   = day_bars["volume"][day_bars["close"] >= day_bars["open"]].sum()
        down_vol = day_bars["volume"][day_bars["close"] <  day_bars["open"]].sum()
        total_vol = up_vol + down_vol
        vol_imbalance = (up_vol - down_vol) / total_vol if total_vol > 0 else 0.0

        # VWAP deviation
        tp   = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3
        vwap = (tp * day_bars["volume"]).sum() / day_bars["volume"].sum() if day_bars["volume"].sum() > 0 else np.nan
        vwap_dev = (day_bars["close"].iloc[-1] - vwap) / vwap if vwap and vwap != 0 else np.nan

        # Intraday momentum: return of first half vs full day
        mid_point = len(day_bars) // 2
        morning_ret = (day_bars["close"].iloc[mid_point] / day_bars["open"].iloc[0] - 1) if day_bars["open"].iloc[0] != 0 else np.nan

        # Range ratio
        range_ratio = (day_bars["high"].max() - day_bars["low"].min()) / day_bars["open"].iloc[0] if day_bars["open"].iloc[0] != 0 else np.nan

        # Tick intensity (bars per hour proxy)
        tick_intensity = len(day_bars) / 5.0  # 5h session

        # Open gap vs previous close (needs previous day's last close)
        open_gap = np.nan  # filled below via shift

        results.append({
            "date":             pd.Timestamp(date),
            "rv_5min":          rv,
            "volume_imbalance": vol_imbalance,
            "intraday_mom":     morning_ret,
            "vwap_dev":         vwap_dev,
            "range_ratio":      range_ratio,
            "tick_intensity":   tick_intensity,
        })

    if not results:
        cols = ["rv_5min", "volume_imbalance", "intraday_mom", "vwap_dev", "range_ratio", "tick_intensity"]
        return pd.DataFrame(index=daily_index, columns=cols, dtype=float)

    feat_df = pd.DataFrame(results).set_index("date")

    # Compute open_gap using daily-level data (close shifted by 1)
    daily_close = bars.groupby(bars.index.date)["close"].last()
    daily_open  = bars.groupby(bars.index.date)["open"].first()
    daily_close.index = pd.to_datetime(daily_close.index)
    daily_open.index  = pd.to_datetime(daily_open.index)
    feat_df["open_gap"] = (daily_open - daily_close.shift(1)) / daily_close.shift(1)

    # Align to daily_index
    feat_df = feat_df.reindex(daily_index)

    # Forward-fill up to 3 days for minor gaps (e.g. if intraday data missing one day)
    feat_df = feat_df.ffill(limit=3)

    return feat_df
