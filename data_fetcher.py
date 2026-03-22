"""
data_fetcher.py — Fetches real Vietnam stock data via vnstock + yfinance fallback.

Primary source : vnstock  (HOSE / HNX / UPCOM)
Fallback source: yfinance (suffixes .VN for Vietnamese tickers)
"""

import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("data_fetcher")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cache_path(symbol: str) -> Path:
    return Path(config.DATA_CACHE_DIR) / f"{symbol}.pkl"


def _load_from_cache(symbol: str) -> Optional[pd.DataFrame]:
    p = _cache_path(symbol)
    if not p.exists():
        return None
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    age_h = (datetime.now() - mtime).total_seconds() / 3600
    if age_h > config.CACHE_TTL_HOURS:
        logger.debug(f"Cache expired for {symbol} ({age_h:.1f}h old)")
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _save_to_cache(symbol: str, df: pd.DataFrame) -> None:
    _cache_path(symbol).write_bytes(pickle.dumps(df))


# ── vnstock Fetcher ────────────────────────────────────────────────────────────

def _fetch_vnstock(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV using vnstock library (preferred for Vietnam stocks)."""
    try:
        from vnstock import Vnstock           # pip install vnstock
        vn = Vnstock()
        stock = vn.stock(symbol=symbol, source="VCI")
        df = stock.quote.history(start=start, end=end, interval="1D")
        if df is None or df.empty:
            return None

        # Normalise column names
        df.columns = [c.lower() for c in df.columns]
        rename_map = {
            "time": "date", "o": "open", "h": "high",
            "l": "low",  "c": "close", "v": "volume",
        }
        df = df.rename(columns=rename_map)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return None

        df = df[list(required)].astype(float).sort_index()
        df.index.name = "date"
        logger.info(f"[vnstock] {symbol}: {len(df)} rows fetched")
        return df

    except ImportError:
        logger.warning("vnstock not installed; falling back to yfinance")
        return None
    except Exception as e:
        logger.warning(f"[vnstock] {symbol} error: {e}")
        return None


# ── yfinance Fallback ──────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fallback: fetch via yfinance using .VN suffix."""
    try:
        import yfinance as yf
        ticker = f"{symbol}.VN"
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df.sort_index()
        logger.info(f"[yfinance] {symbol}: {len(df)} rows fetched")
        return df
    except Exception as e:
        logger.warning(f"[yfinance] {symbol} error: {e}")
        return None


# ── Main Public API ────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    lookback_days: int = config.LOOKBACK_DAYS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns:
        open, high, low, close, volume
    Raises ValueError if data cannot be obtained.
    """
    if use_cache:
        cached = _load_from_cache(symbol)
        if cached is not None:
            logger.debug(f"[cache] {symbol}: loaded {len(cached)} rows")
            return cached

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days + 50)).strftime("%Y-%m-%d")

    df = _fetch_vnstock(symbol, start, end)
    if df is None or df.empty:
        df = _fetch_yfinance(symbol, start, end)

    if df is None or df.empty:
        raise ValueError(
            f"Could not fetch data for {symbol}. "
            "Make sure vnstock or yfinance is installed and the symbol is valid."
        )

    # Keep only last `lookback_days` rows
    df = df.tail(lookback_days)

    if len(df) < 150:
        logger.warning(
            f"{symbol}: only {len(df)} rows available. "
            "This symbol may be too new or illiquid — consider removing it from your watchlist."
        )

    _save_to_cache(symbol, df)
    return df


def fetch_multiple(
    symbols: List[str],
    lookback_days: int = config.LOOKBACK_DAYS,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple symbols, with retry on failure."""
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        for attempt in range(3):
            try:
                results[sym] = fetch_ohlcv(sym, lookback_days, use_cache)
                break
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to fetch {sym} after 3 attempts: {e}")
                else:
                    time.sleep(2 ** attempt)
    return results


def get_vnindex(lookback_days: int = config.LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """Fetch VN-Index (market benchmark) for macro feature."""
    for source_sym in ["VNINDEX", "^VNINDEX"]:
        try:
            return fetch_ohlcv(source_sym, lookback_days)
        except Exception:
            pass
    # yfinance fallback
    try:
        import yfinance as yf
        df = yf.download("^VNINDEX", period="2y", progress=False, auto_adjust=True)
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            return df[["open", "high", "low", "close", "volume"]].astype(float)
    except Exception:
        pass
    logger.warning("Could not fetch VN-Index; macro feature will be omitted.")
    return None
