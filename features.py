"""
features.py — Comprehensive feature engineering for Vietnam stocks.

All rolling windows are ADAPTIVE — they automatically shrink to fit
shorter-history stocks like newly listed ones (e.g. HPA with ~26 rows).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("features")


# ── Utility ────────────────────────────────────────────────────────────────────

def _w(n: int, max_n: int) -> int:
    """Clamp window n to at most max_n (available rows minus buffer)."""
    return min(n, max(2, max_n))

def _sma(s: pd.Series, n: int, cap: int) -> pd.Series:
    return s.rolling(_w(n, cap)).mean()

def _ema(s: pd.Series, n: int, cap: int) -> pd.Series:
    return s.ewm(span=_w(n, cap), adjust=False).mean()

def _std(s: pd.Series, n: int, cap: int) -> pd.Series:
    return s.rolling(_w(n, cap)).std()

def _atr(high, low, close, n: int, cap: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=_w(n, cap), adjust=False).mean()


# ── Individual Indicators ──────────────────────────────────────────────────────

def add_returns(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    c = df["close"]
    for n in [1, 3, 5, 10, 20]:
        if n < cap:
            df[f"ret_{n}d"] = c.pct_change(n)
    df["log_ret"] = np.log(c / c.shift(1))
    return df


def add_trend(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    c = df["close"]
    for n in [5, 10, 20, 50, 200]:
        if n >= cap:
            continue
        df[f"sma_{n}"]  = _sma(c, n, cap)
        df[f"ema_{n}"]  = _ema(c, n, cap)
        df[f"price_vs_sma{n}"] = (c - df[f"sma_{n}"]) / df[f"sma_{n}"]
        df[f"price_vs_ema{n}"] = (c - df[f"ema_{n}"]) / df[f"ema_{n}"]

    # Double EMA — needs at least 10 bars
    if cap >= 10:
        dema_n = _w(20, cap)
        df["dema"] = 2 * _ema(c, dema_n, cap) - _ema(_ema(c, dema_n, cap), dema_n, cap)

    # MACD — needs 26 bars minimum; use scaled-down version for small data
    fast, slow, sig = _w(12, cap), _w(26, cap), _w(9, cap)
    ema_fast = _ema(c, fast, cap)
    ema_slow = _ema(c, slow, cap)
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd"], sig, cap)
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_cross"]  = (np.sign(df["macd_hist"]) != np.sign(df["macd_hist"].shift(1))).astype(int)

    # ADX — uses ATR internally, always adaptive
    h, l, pc = df["high"], df["low"], c.shift(1)
    adx_n = _w(14, cap)
    plus_dm  = np.where((h - h.shift(1)) > (l.shift(1) - l), np.maximum(h - h.shift(1), 0), 0)
    minus_dm = np.where((l.shift(1) - l) > (h - h.shift(1)), np.maximum(l.shift(1) - l, 0), 0)
    atr14    = _atr(h, l, c, adx_n, cap)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=adx_n).mean() / atr14
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=adx_n).mean() / atr14
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"]      = dx.ewm(span=adx_n).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    return df


def add_momentum(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # RSI — adaptive periods
    for n in [7, 14, 21]:
        rsi_n = _w(n, cap)
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(span=rsi_n, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=rsi_n, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f"rsi_{n}"] = 100 - (100 / (1 + rs))

    # Stochastic
    stoch_n = _w(14, cap)
    lo = l.rolling(stoch_n).min()
    hi = h.rolling(stoch_n).max()
    df["stoch_k"] = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(_w(3, cap)).mean()

    # CCI
    cci_n = _w(20, cap)
    tp  = (h + l + c) / 3
    df["cci"] = (tp - _sma(tp, cci_n, cap)) / (
        0.015 * tp.rolling(cci_n).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
    )

    # Williams %R
    wr_n = _w(14, cap)
    hi_n = h.rolling(wr_n).max()
    lo_n = l.rolling(wr_n).min()
    df["williams_r"] = -100 * (hi_n - c) / (hi_n - lo_n).replace(0, np.nan)

    # Rate of Change
    for n in [5, 10, 20]:
        if n < cap:
            df[f"roc_{n}"] = c.pct_change(n) * 100

    # Momentum
    mom_n = _w(10, cap)
    df["mom"] = c - c.shift(mom_n)

    return df


def add_volatility(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Bollinger Bands
    bb_n = _w(20, cap)
    for nstd in [2, 1]:
        mid   = _sma(c, bb_n, cap)
        sig   = _std(c, bb_n, cap)
        upper = mid + nstd * sig
        lower = mid - nstd * sig
        df[f"bb_upper_{nstd}"] = upper
        df[f"bb_lower_{nstd}"] = lower
        df[f"bb_width_{nstd}"] = (upper - lower) / mid.replace(0, np.nan)
        df[f"bb_pct_{nstd}"]   = (c - lower) / (upper - lower).replace(0, np.nan)

    # ATR
    for n in [7, 14, 21]:
        atr_n = _w(n, cap)
        df[f"atr_{n}"] = _atr(h, l, c, atr_n, cap)
        df[f"atr_pct_{n}"] = df[f"atr_{n}"] / c.replace(0, np.nan)

    # Historical Volatility
    log_ret = np.log(c / c.shift(1))
    for n in [10, 20, 30]:
        hv_n = _w(n, cap)
        df[f"hv_{n}"] = log_ret.rolling(hv_n).std() * np.sqrt(252)

    # Keltner Channel
    kelt_n = _w(20, cap)
    ema_k  = _ema(c, kelt_n, cap)
    atr_k  = _atr(h, l, c, _w(14, cap), cap)
    df["keltner_upper"] = ema_k + 2 * atr_k
    df["keltner_lower"] = ema_k - 2 * atr_k
    df["keltner_pct"]   = (c - df["keltner_lower"]) / (
        df["keltner_upper"] - df["keltner_lower"]
    ).replace(0, np.nan)

    df["tr_norm"] = _atr(h, l, c, 1, cap) / c.replace(0, np.nan)
    return df


def add_volume(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    c   = df["close"]
    v   = df["volume"]
    h   = df["high"]
    l   = df["low"]

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"]        = obv
    obv_n = _w(20, cap)
    df["obv_ema"]    = _ema(obv, obv_n, cap)
    df["obv_vs_ema"] = obv / df["obv_ema"].replace(0, np.nan) - 1

    # Volume SMA ratios
    for n in [5, 10, 20]:
        if n < cap:
            df[f"vol_vs_sma{n}"] = v / _sma(v, n, cap).replace(0, np.nan)

    # MFI
    mfi_n = _w(14, cap)
    tp    = (h + l + c) / 3
    mf    = tp * v
    pos   = mf.where(tp > tp.shift(1), 0).rolling(mfi_n).sum()
    neg   = mf.where(tp < tp.shift(1), 0).rolling(mfi_n).sum()
    df["mfi"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    # CMF
    cmf_n = _w(20, cap)
    mfv   = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
    df["cmf"] = mfv.rolling(cmf_n).sum() / v.rolling(cmf_n).sum().replace(0, np.nan)

    # Force Index
    df["force_index"] = _ema(c.diff() * v, _w(13, cap), cap)

    # VWAP proxy
    vwap_n = _w(20, cap)
    df["vwap"] = (tp * v).rolling(vwap_n).sum() / v.rolling(vwap_n).sum().replace(0, np.nan)
    df["price_vs_vwap"] = c / df["vwap"].replace(0, np.nan) - 1

    return df


def add_high_importance_features(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """
    Features identified as high-importance from published research on 1-day horizon:

    Top additions not previously in feature set:
      yday_vol_log_ret   — log change of yesterday's volume  (#1 in chart)
      yday_open_log_ret  — log return of yesterday's open
      zscore_20          — price z-score vs 20-day window
      skew_5d/10d        — rolling return skewness
      abnormal_vol_20    — volume z-score (how unusual is today's volume)
      overnight_gap      — today open vs yesterday close
      kurt_10d           — kurtosis (tail risk)
      momentum_20d_log   — log 20-day momentum (more important than 5d)
      intraday_range_pct — (high-low)/open per bar
    """
    c = df["close"]
    v = df["volume"]
    o = df["open"]   if "open" in df.columns else c
    h = df["high"]   if "high" in df.columns else c
    l = df["low"]    if "low"  in df.columns else c

    # Yesterday's OHLCV log returns
    df["yday_vol_log_ret"]   = np.log((v / v.shift(1)).replace(0, np.nan))
    df["yday_open_log_ret"]  = np.log((o / o.shift(1)).replace(0, np.nan))
    df["yday_close_log_ret"] = np.log((c / c.shift(1)).replace(0, np.nan))
    df["yday_high_log_ret"]  = np.log((h / h.shift(1)).replace(0, np.nan))
    df["yday_low_log_ret"]   = np.log((l / l.shift(1)).replace(0, np.nan))

    # Z-Score
    for zw in [10, 20]:
        n = _w(zw, cap)
        rm = c.rolling(n).mean()
        rs = c.rolling(n).std().replace(0, np.nan)
        df[f"zscore_{zw}"] = (c - rm) / rs

    # Return skewness
    log_ret = np.log((c / c.shift(1)).replace(0, np.nan))
    for sw in [5, 10, 20]:
        n = _w(sw, cap)
        if n >= 3:
            df[f"skew_{sw}d"] = log_ret.rolling(n).skew()

    # Kurtosis
    for kw in [10, 20]:
        n = _w(kw, cap)
        if n >= 4:
            df[f"kurt_{kw}d"] = log_ret.rolling(n).kurt()

    # Abnormal volume (volume z-score)
    for vw in [5, 10, 20]:
        n = _w(vw, cap)
        vm = v.rolling(n).mean()
        vs = v.rolling(n).std().replace(0, np.nan)
        df[f"abnormal_vol_{vw}"] = (v - vm) / vs

    # Overnight gap
    if "open" in df.columns:
        df["overnight_gap"]     = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)
        df["overnight_gap_abs"] = df["overnight_gap"].abs()

    # Volume log returns
    df["vol_log_ret"]    = np.log((v / v.shift(1)).replace(0, np.nan))
    df["vol_log_ret_5d"] = np.log((v / v.shift(5)).replace(0, np.nan))

    # Momentum (log)
    for mn in [5, 10, 20]:
        if cap > mn:
            df[f"mom_log_{mn}d"] = np.log((c / c.shift(mn)).replace(0, np.nan))

    # Intraday range
    if "high" in df.columns and "low" in df.columns and "open" in df.columns:
        df["intraday_range_pct"] = (h - l) / o.replace(0, np.nan)
        df["intraday_range_ema"] = df["intraday_range_pct"].ewm(
            span=_w(10, cap), adjust=False).mean()

    return df


def add_insider_proxy(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """
    Insider trading proxy features for Vietnam stocks.

    True insider data (SSC filings) requires scraping from
    https://www.ssc.gov.vn or vnstock insider endpoints.
    These are volume/price anomaly proxies that approximate
    institutional accumulation and smart-money signals.

    To use actual insider data: add columns insider_shares,
    insider_buy_flag, insider_amount before calling build_features().
    """
    c = df["close"]
    v = df["volume"]
    h = df["high"] if "high" in df.columns else c
    l = df["low"]  if "low"  in df.columns else c
    cap_n = _w(20, cap)

    # Volume z-score
    vol_z  = (v - v.rolling(cap_n).mean()) / v.rolling(cap_n).std().replace(0, np.nan)
    pr_abs = c.pct_change().abs()
    pr_z   = (pr_abs - pr_abs.rolling(cap_n).mean()) / pr_abs.rolling(cap_n).std().replace(0, np.nan)

    # High volume + low price volatility = accumulation
    df["accum_signal"]    = (vol_z - pr_z).clip(-3, 3)
    df["vol_surge_flag"]  = (vol_z > 2.0).astype(int)

    # Directional volume ratio (smart money proxy)
    up_v = v.where(c > c.shift(1), 0).rolling(_w(10, cap)).sum()
    dn_v = v.where(c <= c.shift(1), 0).rolling(_w(10, cap)).sum()
    df["smart_money_ratio"] = up_v / (up_v + dn_v).replace(0, np.nan)

    # Chaikin A/D momentum
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    ad  = (clv * v).fillna(0).cumsum()
    df["ad_momentum"] = ad - ad.ewm(span=_w(10, cap), adjust=False).mean()

    return df


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    body      = (c - o).abs()
    rng       = (h - l).replace(0, np.nan)
    body_frac = body / rng

    df["is_green"]   = (c > o).astype(int)
    df["body_size"]  = body_frac
    df["upper_wick"] = (h - np.maximum(o, c)) / rng
    df["lower_wick"] = (np.minimum(o, c) - l) / rng
    df["doji"]           = (body_frac < 0.1).astype(int)
    df["hammer"]         = ((df["lower_wick"] > 0.6) & (df["upper_wick"] < 0.1)).astype(int)
    df["shooting_star"]  = ((df["upper_wick"] > 0.6) & (df["lower_wick"] < 0.1)).astype(int)
    prev_body = c.shift(1) - o.shift(1)
    curr_body = c - o
    df["bull_engulf"] = ((curr_body > 0) & (prev_body < 0) & (c > o.shift(1)) & (o < c.shift(1))).astype(int)
    df["bear_engulf"] = ((curr_body < 0) & (prev_body > 0) & (c < o.shift(1)) & (o > c.shift(1))).astype(int)
    df["gap_up"]   = (o > h.shift(1)).astype(int)
    df["gap_down"] = (o < l.shift(1)).astype(int)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["day_of_week"]    = idx.dayofweek
    df["week_of_year"]   = idx.isocalendar().week.astype(int)
    df["month"]          = idx.month
    df["quarter"]        = idx.quarter
    df["is_month_end"]   = idx.is_month_end.astype(int)
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)
    return df


def add_market_relative(df: pd.DataFrame, vnindex: Optional[pd.DataFrame], cap: int) -> pd.DataFrame:
    if vnindex is None:
        return df
    vnidx_ret = vnindex["close"].pct_change()
    stock_ret  = df["close"].pct_change()

    aligned = pd.concat(
        [stock_ret.rename("stock"), vnidx_ret.rename("vnidx")], axis=1
    ).dropna()

    beta_n = _w(20, cap)
    cov  = aligned["stock"].rolling(beta_n).cov(aligned["vnidx"])
    vvar = aligned["vnidx"].rolling(beta_n).var()
    df["beta"] = (cov / vvar.replace(0, np.nan)).reindex(df.index)

    rs_n = _w(20, cap)
    stock_cum = (1 + aligned["stock"]).rolling(rs_n).apply(np.prod, raw=True)
    vnidx_cum = (1 + aligned["vnidx"]).rolling(rs_n).apply(np.prod, raw=True)
    df["rel_strength"] = (stock_cum / vnidx_cum.replace(0, np.nan)).reindex(df.index)

    df["vnidx_ret_5d"] = vnindex["close"].pct_change(_w(5, cap)).reindex(df.index)
    vn_delta = vnindex["close"].diff()
    vn_gain  = vn_delta.clip(lower=0).ewm(_w(14, cap)).mean()
    vn_loss  = (-vn_delta.clip(upper=0)).ewm(_w(14, cap)).mean()
    df["vnidx_rsi"] = (100 - 100 / (1 + vn_gain / vn_loss.replace(0, np.nan))).reindex(df.index)
    return df


# ── Main Entrypoint ────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    vnindex: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
    intraday_features: Optional[pd.DataFrame] = None,
    sentiment_features: Optional[pd.DataFrame] = None,
    use_vol_adjusted_labels: bool = True,
) -> pd.DataFrame:
    """
    Build full feature matrix.

    Args:
        df:                     OHLCV DataFrame indexed by date
        vnindex:                VN-Index OHLCV for market-relative features
        symbol:                 Stock ticker (used for intraday fetch if not provided)
        intraday_features:      Pre-built intraday feature DataFrame (or None to skip)
        sentiment_features:     Pre-built sentiment feature DataFrame (or None to skip)
        use_vol_adjusted_labels: Use volatility-adjusted targets (reduces label noise)

    Returns:
        DataFrame with all features + target columns
    """
    from target_engineering import (
        make_volatility_adjusted_label,
        make_risk_adjusted_target,
        add_regime_features,
    )

    df = df.copy()
    cap = len(df)
    logger.info(f"  Building features with cap={cap} rows (adaptive windows)")

    df = add_returns(df, cap)
    df = add_trend(df, cap)
    df = add_momentum(df, cap)
    df = add_volatility(df, cap)
    df = add_volume(df, cap)
    df = add_candlestick_patterns(df)
    df = add_calendar_features(df)
    df = add_market_relative(df, vnindex, cap)

    # ── Regime features ───────────────────────────────────────────────────
    df = add_regime_features(df, vnindex)

    # ── Intraday features (merge on date index) ───────────────────────────
    if intraday_features is not None and not intraday_features.empty:
        intra = intraday_features.reindex(df.index)
        for col in intra.columns:
            df[f"intra_{col}"] = intra[col]
        n_intra = intra.notna().any(axis=1).sum()
        logger.info(f"  Intraday features: {len(intra.columns)} cols, {n_intra} non-null rows")
    elif symbol:
        # Try to fetch intraday on the fly
        try:
            from intraday_fetcher import build_intraday_features
            intra = build_intraday_features(symbol, df.index)
            if not intra.empty:
                for col in intra.columns:
                    df[f"intra_{col}"] = intra[col]
                logger.info(f"  Intraday features fetched: {len(intra.columns)} cols")
        except Exception as e:
            logger.debug(f"  Intraday fetch skipped: {e}")

    # ── Sentiment features ────────────────────────────────────────────────
    if sentiment_features is not None and not sentiment_features.empty:
        sent = sentiment_features.reindex(df.index)
        for col in sent.columns:
            df[f"sent_{col}"] = sent[col]
        logger.info(f"  Sentiment features: {len(sent.columns)} cols")

    # ── Labels ────────────────────────────────────────────────────────────
    for h in [1, 3, 5, 10]:
        fwd_ret = df["close"].pct_change(h).shift(-h)
        df[f"target_ret_{h}d"] = fwd_ret

        if use_vol_adjusted_labels and cap >= 25:
            vol_adj = make_volatility_adjusted_label(
                df["close"], horizon=h,
                vol_window=min(20, cap // 3),
                threshold_multiplier=0.3,
            )
            df[f"target_dir_{h}d"] = vol_adj
        else:
            df[f"target_dir_{h}d"] = (fwd_ret > 0).astype(int)

        # Risk-adjusted regression target
        if cap >= 25:
            df[f"target_sharpe_{h}d"] = make_risk_adjusted_target(
                df["close"], horizon=h, vol_window=min(20, cap // 3)
            )

    drop_cols = [c for c in ["open", "high", "low", "volume"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Drop rows where the primary target is NaN
    # (keep rows where vol-adjusted label is NaN — they'll be filtered per-target)
    df = df.dropna(subset=["close"])
    df = df[df.index.notna()]

    # Fill remaining NaN features (intraday/sentiment may have gaps)
    df = df.ffill(limit=5).bfill(limit=2)

    # Final dropna for truly unrecoverable rows
    core_cols = [c for c in df.columns if not c.startswith("target_") and not c.startswith("intra_") and not c.startswith("sent_")]
    df = df.dropna(subset=core_cols)

    logger.info(f"  Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def get_feature_cols(df: pd.DataFrame, horizon: int = 5) -> list:
    exclude = set()
    for h in [1, 3, 5, 10]:
        exclude.add(f"target_ret_{h}d")
        exclude.add(f"target_dir_{h}d")
        exclude.add(f"target_sharpe_{h}d")
    exclude.add("close")
    return [c for c in df.columns if c not in exclude]




# ── Utility ────────────────────────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _std(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).std()

def _atr(high, low, close, n=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


# ── Individual Indicators ──────────────────────────────────────────────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ret_1d"]  = c.pct_change(1)
    df["ret_3d"]  = c.pct_change(3)
    df["ret_5d"]  = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)
    df["ret_20d"] = c.pct_change(20)
    df["log_ret"] = np.log(c / c.shift(1))
    return df


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    for n in [5, 10, 20, 50, 200]:
        df[f"sma_{n}"]  = _sma(c, n)
        df[f"ema_{n}"]  = _ema(c, n)
        df[f"price_vs_sma{n}"] = (c - df[f"sma_{n}"]) / df[f"sma_{n}"]
        df[f"price_vs_ema{n}"] = (c - df[f"ema_{n}"]) / df[f"ema_{n}"]

    # Double EMA
    df["dema_20"] = 2 * _ema(c, 20) - _ema(_ema(c, 20), 20)

    # MACD
    ema12 = _ema(c, 12); ema26 = _ema(c, 26)
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_cross"]  = np.sign(df["macd_hist"]) != np.sign(df["macd_hist"].shift(1))

    # ADX
    h, l, pc = df["high"], df["low"], c.shift(1)
    plus_dm  = np.where((h - h.shift(1)) > (l.shift(1) - l), np.maximum(h - h.shift(1), 0), 0)
    minus_dm = np.where((l.shift(1) - l) > (h - h.shift(1)), np.maximum(l.shift(1) - l, 0), 0)
    atr14    = _atr(h, l, c, 14)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=14).mean() / atr14
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=14).mean() / atr14
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"]      = dx.ewm(span=14).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # RSI
    for n in [7, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(span=n, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=n, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f"rsi_{n}"] = 100 - (100 / (1 + rs))

    # Stochastic %K and %D
    for n in [14]:
        lo = l.rolling(n).min()
        hi = h.rolling(n).max()
        df[f"stoch_k_{n}"] = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
        df[f"stoch_d_{n}"] = df[f"stoch_k_{n}"].rolling(3).mean()

    # CCI
    tp  = (h + l + c) / 3
    df["cci_20"] = (tp - _sma(tp, 20)) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True))

    # Williams %R
    for n in [14]:
        hi_n = h.rolling(n).max()
        lo_n = l.rolling(n).min()
        df[f"williams_r_{n}"] = -100 * (hi_n - c) / (hi_n - lo_n).replace(0, np.nan)

    # Rate of Change
    for n in [5, 10, 20]:
        df[f"roc_{n}"] = c.pct_change(n) * 100

    # Momentum
    df["mom_10"] = c - c.shift(10)

    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Bollinger Bands
    for n, nstd in [(20, 2), (20, 1)]:
        mid  = _sma(c, n)
        sig  = _std(c, n)
        upper = mid + nstd * sig
        lower = mid - nstd * sig
        df[f"bb_upper_{n}_{nstd}"] = upper
        df[f"bb_lower_{n}_{nstd}"] = lower
        df[f"bb_width_{n}_{nstd}"] = (upper - lower) / mid
        df[f"bb_pct_{n}_{nstd}"]   = (c - lower) / (upper - lower).replace(0, np.nan)

    # ATR
    for n in [7, 14, 21]:
        df[f"atr_{n}"] = _atr(h, l, c, n)
        df[f"atr_pct_{n}"] = df[f"atr_{n}"] / c

    # Historical Volatility (annualised)
    log_ret = np.log(c / c.shift(1))
    for n in [10, 20, 30]:
        df[f"hv_{n}"] = log_ret.rolling(n).std() * np.sqrt(252)

    # Keltner Channel
    ema20 = _ema(c, 20)
    atr14 = _atr(h, l, c, 14)
    df["keltner_upper"] = ema20 + 2 * atr14
    df["keltner_lower"] = ema20 - 2 * atr14
    df["keltner_pct"]   = (c - df["keltner_lower"]) / (df["keltner_upper"] - df["keltner_lower"]).replace(0, np.nan)

    # True Range normalised
    df["tr_norm"] = _atr(h, l, c, 1) / c

    return df


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["close"]
    v   = df["volume"]
    h   = df["high"]
    l   = df["low"]

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"]       = obv
    df["obv_ema20"] = _ema(obv, 20)
    df["obv_vs_ema"] = obv / df["obv_ema20"].replace(0, np.nan) - 1

    # Volume SMA ratios
    for n in [5, 10, 20]:
        df[f"vol_vs_sma{n}"] = v / _sma(v, n).replace(0, np.nan)

    # MFI (Money Flow Index)
    tp   = (h + l + c) / 3
    mf   = tp * v
    pos  = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg  = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    # CMF (Chaikin Money Flow)
    mfv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
    df["cmf_20"] = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # Force Index
    df["force_index_13"] = _ema(c.diff() * v, 13)

    # VWAP (rolling daily proxy)
    df["vwap_20"] = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    df["price_vs_vwap"] = c / df["vwap_20"].replace(0, np.nan) - 1

    return df


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    body   = (c - o).abs()
    rng    = h - l
    body_frac = body / rng.replace(0, np.nan)

    df["is_green"]   = (c > o).astype(int)
    df["body_size"]  = body_frac
    df["upper_wick"] = (h - np.maximum(o, c)) / rng.replace(0, np.nan)
    df["lower_wick"] = (np.minimum(o, c) - l) / rng.replace(0, np.nan)

    # Doji
    df["doji"] = (body_frac < 0.1).astype(int)
    # Hammer / Shooting Star
    df["hammer"]        = ((df["lower_wick"] > 0.6) & (df["upper_wick"] < 0.1)).astype(int)
    df["shooting_star"] = ((df["upper_wick"] > 0.6) & (df["lower_wick"] < 0.1)).astype(int)
    # Engulfing
    prev_body = (c.shift(1) - o.shift(1))
    curr_body = (c - o)
    df["bull_engulf"] = ((curr_body > 0) & (prev_body < 0) & (c > o.shift(1)) & (o < c.shift(1))).astype(int)
    df["bear_engulf"] = ((curr_body < 0) & (prev_body > 0) & (c < o.shift(1)) & (o > c.shift(1))).astype(int)
    # Gap up / down
    df["gap_up"]   = (o > h.shift(1)).astype(int)
    df["gap_down"] = (o < l.shift(1)).astype(int)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["day_of_week"]   = idx.dayofweek          # 0=Mon … 4=Fri
    df["week_of_year"]  = idx.isocalendar().week.astype(int)
    df["month"]         = idx.month
    df["quarter"]       = idx.quarter
    df["is_month_end"]  = idx.is_month_end.astype(int)
    df["is_month_start"]= idx.is_month_start.astype(int)
    df["is_quarter_end"]= idx.is_quarter_end.astype(int)
    return df


def add_market_relative(df: pd.DataFrame, vnindex: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add beta, relative-strength, and correlation vs VN-Index."""
    if vnindex is None:
        return df
    vnidx_ret = vnindex["close"].pct_change()
    stock_ret  = df["close"].pct_change()

    aligned = pd.concat(
        [stock_ret.rename("stock"), vnidx_ret.rename("vnidx")], axis=1
    ).dropna()

    # Rolling 20-day beta
    cov  = aligned["stock"].rolling(20).cov(aligned["vnidx"])
    vvar = aligned["vnidx"].rolling(20).var()
    df["beta_20"] = (cov / vvar.replace(0, np.nan)).reindex(df.index)

    # Relative strength
    stock_cum  = (1 + aligned["stock"]).rolling(20).apply(np.prod, raw=True)
    vnidx_cum  = (1 + aligned["vnidx"]).rolling(20).apply(np.prod, raw=True)
    df["rel_strength_20"] = (stock_cum / vnidx_cum.replace(0, np.nan)).reindex(df.index)

    # VN-Index momentum as macro signal
    df["vnidx_ret_5d"]  = vnindex["close"].pct_change(5).reindex(df.index)
    df["vnidx_rsi_14"]  = (
        lambda s: 100 - 100 / (1 + s.clip(lower=0).ewm(14).mean() /
                                (-s.clip(upper=0)).ewm(14).mean().replace(0, np.nan))
    )(vnindex["close"].diff()).reindex(df.index)

    return df


def add_sector_features(
    df: pd.DataFrame,
    symbol: str,
    sector_peers: Optional[dict] = None,   # {sym: ohlcv_df} for sector peers
) -> pd.DataFrame:
    """
    Add sector-relative momentum features.
    These help distinguish stock-specific moves from sector-wide trends.
    """
    if not sector_peers or len(sector_peers) < 2:
        return df

    # Compute sector median return (equal-weight)
    peer_rets = []
    for sym, peer_ohlcv in sector_peers.items():
        if sym == symbol:
            continue
        r = peer_ohlcv["close"].pct_change(5).reindex(df.index)
        if r.notna().sum() > 20:
            peer_rets.append(r)

    if not peer_rets:
        return df

    sector_ret_5d = pd.concat(peer_rets, axis=1).median(axis=1)
    stock_ret_5d  = df["close"].pct_change(5)

    # Stock return MINUS sector return = stock-specific alpha
    extra = pd.DataFrame(index=df.index)
    extra["sector_alpha_5d"]    = stock_ret_5d - sector_ret_5d
    extra["sector_momentum_5d"] = sector_ret_5d   # sector tailwind/headwind
    extra["sector_rel_strength"]= (stock_ret_5d / sector_ret_5d.replace(0, np.nan)).clip(-5, 5)

    df = pd.concat([df, extra], axis=1)
    return df


def add_fear_greed_proxy(
    df: pd.DataFrame,
    vnindex: Optional[pd.DataFrame],
    vn30: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add a market fear/greed proxy using VN-Index and VN30 breadth signals.

    Fear/greed in Vietnam is strongly driven by:
    - VN-Index vs its 50-day MA (above = greed, below = fear)
    - VN-Index 20-day RSI
    - VN-Index 5d/20d return momentum
    - Spread between VN30 and VN-Index (large-cap vs broad market divergence)
    """
    if vnindex is None:
        return df

    extra = {}
    vn_c  = vnindex["close"].reindex(df.index)
    vn_ma50 = vn_c.ewm(span=50, adjust=False).mean()

    # Distance from MA50 (positive = greed zone)
    extra["vn_ma50_dist"] = (vn_c - vn_ma50) / vn_ma50.replace(0, np.nan)

    # VN-Index RSI as market sentiment
    vn_delta = vn_c.diff()
    vn_gain  = vn_delta.clip(lower=0).ewm(14).mean()
    vn_loss  = (-vn_delta.clip(upper=0)).ewm(14).mean()
    vn_rsi   = 100 - 100 / (1 + vn_gain / vn_loss.replace(0, np.nan))
    extra["vn_rsi_14"]     = vn_rsi
    extra["vn_rsi_overbought"] = (vn_rsi > 70).astype(int)
    extra["vn_rsi_oversold"]   = (vn_rsi < 30).astype(int)

    # VN-Index momentum at multiple horizons
    for h in [5, 20, 60]:
        extra[f"vn_ret_{h}d"] = vn_c.pct_change(h)

    # Volatility regime on VN-Index (high vol = fear)
    vn_vol_20 = vn_c.pct_change().rolling(20).std() * np.sqrt(252)
    vn_vol_60 = vn_c.pct_change().rolling(60).std() * np.sqrt(252)
    extra["vn_vol_regime"] = (vn_vol_20 > vn_vol_60).astype(int)
    extra["vn_vol_ratio"]  = vn_vol_20 / vn_vol_60.replace(0, np.nan)

    df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
    return df


# ── Main Entrypoint ────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    vnindex: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
    sector_peers: Optional[dict] = None,
    sentiment_features: Optional[pd.DataFrame] = None,
    use_vol_adjusted_labels: bool = False,
) -> pd.DataFrame:
    """
    Build full feature matrix.
    Input:  OHLCV DataFrame indexed by date
    Output: DataFrame with 80+ feature columns + 'target_*' labels
    """
    df = df.copy()
    cap = len(df)
    df = add_returns(df, cap)
    df = add_trend(df, cap)
    df = add_momentum(df, cap)
    df = add_volatility(df, cap)
    df = add_volume(df, cap)
    df = add_high_importance_features(df, cap)
    df = add_insider_proxy(df, cap)
    df = add_candlestick_patterns(df)
    df = add_calendar_features(df)
    df = add_market_relative(df, vnindex, cap)
    df = add_fear_greed_proxy(df, vnindex)

    if symbol and sector_peers:
        df = add_sector_features(df, symbol, sector_peers)

    if sentiment_features is not None and not sentiment_features.empty:
        sent = sentiment_features.reindex(df.index)
        sent_cols = {f"sent_{c}": sent[c] for c in sent.columns}
        df = pd.concat([df, pd.DataFrame(sent_cols, index=df.index)], axis=1)

    # ── Math model features ───────────────────────────────────────────────
    try:
        from math_models import build_math_model_features
        if "open" in df.columns and "high" in df.columns:
            math_df = build_math_model_features(
                df["high"], df["low"], df["close"], df["open"]
            )
            # Drop columns already in df to avoid duplicates
            new_cols = [c for c in math_df.columns if c not in df.columns]
            if new_cols:
                df = pd.concat([df, math_df[new_cols]], axis=1)
    except Exception as e:
        logger.debug(f"Math model features skipped: {e}")

    # ── Labels ────────────────────────────────────────────────────────────
    label_cols = {}
    for h in [1, 3, 5, 10]:
        fwd_ret = df["close"].pct_change(h).shift(-h)
        label_cols[f"target_ret_{h}d"] = fwd_ret

        if use_vol_adjusted_labels and len(df) >= 25:
            try:
                from target_engineering import make_volatility_adjusted_label
                vol_label = make_volatility_adjusted_label(
                    df["close"], horizon=h,
                    vol_window=min(20, len(df) // 3),
                    threshold_multiplier=0.3,
                )
                label_cols[f"target_dir_{h}d"] = vol_label
            except Exception:
                label_cols[f"target_dir_{h}d"] = (fwd_ret > 0).astype(int)
        else:
            label_cols[f"target_dir_{h}d"] = (fwd_ret > 0).astype(int)

    df = pd.concat([df, pd.DataFrame(label_cols, index=df.index)], axis=1)
    df = df.copy()   # defragment

    drop_cols = [c for c in ["open", "high", "low", "volume"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(how="all")

    logger.info(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def get_feature_cols(df: pd.DataFrame, horizon: int = 5) -> list:
    """Return feature columns (excludes target and raw price columns)."""
    exclude = {f"target_ret_{h}d" for h in [1, 3, 5, 10]}
    exclude |= {f"target_dir_{h}d" for h in [1, 3, 5, 10]}
    exclude.add("close")
    return [c for c in df.columns if c not in exclude]
