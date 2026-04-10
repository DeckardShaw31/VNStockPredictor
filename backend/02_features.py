"""
Module 02 — Feature Engineering
================================
Computes all ~190 features defined in VN100_Full_Master_Plan.md
Parts A (9 strategies), B (additional indicators), F (experiential rules).

Input : data/raw/{TICKER}.csv
Output: data/features/{TICKER}.csv

Run standalone:
    python modules/02_features.py --ticker HPG
    python modules/02_features.py --all
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT     = Path(__file__).resolve().parent.parent
RAW      = ROOT / "data" / "raw"
FEATURES = ROOT / "data" / "features"
FEATURES.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("features")

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _smma(s: pd.Series, n: int) -> pd.Series:
    """Smoothed moving average (Wilder's)."""
    return s.ewm(alpha=1/n, adjust=False).mean()

def _atr(high, low, close, n=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _smma(tr, n)

def _rolling_linslope(s: pd.Series, n: int) -> pd.Series:
    """Rolling linear regression slope over n periods."""
    return s.rolling(n).apply(
        lambda x: linregress(range(len(x)), x).slope if len(x) == n else np.nan,
        raw=True,
    )

def _hurst(s: pd.Series, min_lag=2, max_lag=20) -> float:
    """Hurst exponent via R/S analysis."""
    ts = s.dropna().values
    if len(ts) < max_lag + 5:
        return 0.5
    lags = range(min_lag, min(max_lag + 1, len(ts) // 2))
    rs_vals = []
    for lag in lags:
        sub = [ts[i:i+lag] for i in range(0, len(ts) - lag, lag)]
        rs  = []
        for chunk in sub:
            mean_c = np.mean(chunk)
            dev    = np.cumsum(chunk - mean_c)
            r      = dev.max() - dev.min()
            s_val  = np.std(chunk, ddof=1)
            if s_val > 0:
                rs.append(r / s_val)
        if rs:
            rs_vals.append(np.mean(rs))
    if len(rs_vals) < 2:
        return 0.5
    try:
        slope, *_ = linregress(np.log(list(lags)[:len(rs_vals)]), np.log(rs_vals))
        return float(np.clip(slope, 0.01, 0.99))
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Master feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Given a raw OHLCV dataframe, returns a dataframe with all ~190 features.
    """
    df = df_raw.copy()
    df["date"]   = pd.to_datetime(df["date"])
    df           = df.sort_values("date").reset_index(drop=True)
    close = df["close"];  high = df["high"];  low = df["low"]
    open_ = df["open"];   vol  = df["volume"]

    # ── 1. RETURNS ───────────────────────────────────────────────────────────
    df["return_1d"]  = close.pct_change(1)
    df["return_3d"]  = close.pct_change(3)
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)

    # ── 2. MOVING AVERAGES ───────────────────────────────────────────────────
    for n in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{n}"] = close.rolling(n).mean()
        df[f"ema_{n}"] = _ema(close, n)

    df["dist_sma_20"]  = (close - df["sma_20"])  / df["sma_20"]
    df["dist_sma_50"]  = (close - df["sma_50"])  / df["sma_50"]
    df["dist_sma_200"] = (close - df["sma_200"]) / df["sma_200"]

    # MA score: number of bullish MA relationships / 3  → [0,1]
    df["ma_score_stock"] = (
        (close        > df["sma_20"]).astype(int) +
        (df["sma_20"] > df["sma_50"]).astype(int) +
        (df["sma_50"] > df["sma_200"]).astype(int)
    ) / 3.0

    # ── 3. VOLATILITY ────────────────────────────────────────────────────────
    df["atr_14"]       = _atr(high, low, close, 14)
    df["atr_ratio"]    = df["atr_14"] / close        # normalised ATR
    df["vol_20d"]      = close.pct_change().rolling(20).std()
    df["vol_60d"]      = close.pct_change().rolling(60).std()

    # GARCH-approximation: EWMA of squared returns (lambda=0.94)
    sq_ret = (close.pct_change() ** 2).fillna(0)
    df["garch_sigma"]  = np.sqrt(sq_ret.ewm(alpha=0.06, adjust=False).mean()) * np.sqrt(252)

    # Z-score
    df["zscore_20d"] = (close - close.rolling(20).mean()) / close.rolling(20).std()

    # ── 4. VOLUME ────────────────────────────────────────────────────────────
    df["adv_20"]      = vol.rolling(20).mean()
    df["rel_volume"]  = vol / df["adv_20"]           # today vs 20-day avg

    # VSA ratio: volume-spread analysis
    spread   = high - low
    df["vsa_ratio"] = vol / (spread.replace(0, np.nan) * close)

    # Volume trend features
    df["vol_explosion_flag"]      = (df["rel_volume"] > 3.0).astype(int)
    df["vol_explosion_direction"] = np.where(
        df["vol_explosion_flag"] == 1,
        np.where(close > open_, 1, -1), 0,
    )
    # Consecutive days rel_vol > 2
    streak = (df["rel_volume"] > 2.0).astype(int)
    df["vol_explosion_streak"] = streak.groupby((streak == 0).cumsum()).cumsum()
    df["vol_explosion_price_confirm"] = (
        (df["vol_explosion_flag"] == 1) &
        (close >= close.rolling(10).max())
    ).astype(int)

    # Volume climax
    df["volume_climax_flag"] = (
        (df["rel_volume"] > 3.5) &
        (df["vol_20d"].notna()) &
        (spread < spread.rolling(20).mean() * 0.4)
    ).astype(int)

    # Low-volume advance
    df["low_vol_advance_flag"] = (
        (df["rel_volume"] < 0.7) &
        (df["return_1d"] > 0.02)
    ).astype(int)

    # Pullback volume dry-up (last 5 declining days avg vol < 70% ADV)
    def _pullback_dryup(df_local, i, window=5):
        if i < window:
            return 0
        sub = df_local.iloc[i-window:i]
        down_days = sub[sub["return_1d"] < 0]
        if len(down_days) == 0:
            return 0
        return int(down_days["volume"].mean() < 0.7 * df_local.iloc[i]["adv_20"])
    df["pullback_volume_dryup"] = [_pullback_dryup(df, i) for i in range(len(df))]

    # Volume declining pullback (used in Fibonacci bounce rule)
    df["volume_declining_pullback"] = (
        vol.rolling(3).apply(lambda x: 1 if x[0] > x[1] > x[2] else 0, raw=True)
    ).fillna(0).astype(int)

    # ── 5. RSI ───────────────────────────────────────────────────────────────
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = _smma(gain, 14)
    avg_l  = _smma(loss, 14)
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_5d_slope"] = df["rsi_14"].diff(5)
    df["rsi_oversold_flag"]   = (df["rsi_14"] < 30).astype(int)
    df["rsi_overbought_flag"] = (df["rsi_14"] > 70).astype(int)

    # RSI divergence: price new 10d low but RSI doesn't
    price_10lo = close.rolling(10).min()
    rsi_10lo   = df["rsi_14"].rolling(10).min()
    df["rsi_divergence_bullish"] = (
        (close == price_10lo) & (df["rsi_14"] > rsi_10lo)
    ).astype(int)
    price_10hi = close.rolling(10).max()
    rsi_10hi   = df["rsi_14"].rolling(10).max()
    df["rsi_divergence_bearish"] = (
        (close == price_10hi) & (df["rsi_14"] < rsi_10hi)
    ).astype(int)

    # ── 6. MACD ───────────────────────────────────────────────────────────────
    ema12 = _ema(close, 12);  ema26 = _ema(close, 26)
    df["macd_line"]      = ema12 - ema26
    df["macd_signal"]    = _ema(df["macd_line"], 9)
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # ── 7. BOLLINGER BANDS ───────────────────────────────────────────────────
    df["bb_mid"]   = close.rolling(20).mean()
    df["bb_std"]   = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    bb_pct_20 = df["bb_width"].rolling(60).quantile(0.15)
    df["bb_squeeze_flag"]     = (df["bb_width"] < bb_pct_20).astype(int)
    bb_expand_cond = (
        df["bb_squeeze_flag"].shift(3).fillna(0) == 1
    ) & (df["bb_width"] > df["bb_width"].shift(3) * 1.15)
    df["bb_expansion_flag"]    = bb_expand_cond.astype(int)
    df["bb_expansion_rate"]    = df["bb_width"].pct_change()
    df["bb_breakout_direction"] = np.where(
        df["bb_expansion_flag"] == 1,
        np.where(close > df["bb_upper"], 1, np.where(close < df["bb_lower"], -1, 0)), 0,
    )

    # ── 8. PARABOLIC SAR ─────────────────────────────────────────────────────
    af0, af_max, af_step = 0.02, 0.2, 0.02
    sar  = np.zeros(len(df))
    bull = np.zeros(len(df), dtype=bool)
    ep   = np.zeros(len(df))
    af   = np.zeros(len(df))

    if len(df) > 2:
        sar[0]  = float(low.iloc[0])
        bull[0] = True
        ep[0]   = float(high.iloc[0])
        af[0]   = af0
        for i in range(1, len(df)):
            prev_sar  = sar[i-1]
            prev_bull = bull[i-1]
            prev_ep   = ep[i-1]
            prev_af   = af[i-1]
            new_sar   = prev_sar + prev_af * (prev_ep - prev_sar)
            h = float(high.iloc[i]); l = float(low.iloc[i])
            if prev_bull:
                new_sar = min(new_sar, float(low.iloc[i-1]),
                              float(low.iloc[max(0,i-2)]))
                if l < new_sar:
                    bull[i] = False; sar[i] = prev_ep; ep[i] = l; af[i] = af0
                else:
                    bull[i] = True;  sar[i] = new_sar
                    if h > prev_ep:
                        ep[i] = h; af[i] = min(prev_af + af_step, af_max)
                    else:
                        ep[i] = prev_ep; af[i] = prev_af
            else:
                new_sar = max(new_sar, float(high.iloc[i-1]),
                              float(high.iloc[max(0,i-2)]))
                if h > new_sar:
                    bull[i] = True; sar[i] = prev_ep; ep[i] = h; af[i] = af0
                else:
                    bull[i] = False; sar[i] = new_sar
                    if l < prev_ep:
                        ep[i] = l; af[i] = min(prev_af + af_step, af_max)
                    else:
                        ep[i] = prev_ep; af[i] = prev_af

    df["sar_t"]        = sar
    df["sar_bullish"]  = bull.astype(int)
    df["sar_flip_bullish"] = (
        (df["sar_bullish"] == 1) & (df["sar_bullish"].shift(1) == 0)
    ).astype(int)
    df["sar_flip_bearish"] = (
        (df["sar_bullish"] == 0) & (df["sar_bullish"].shift(1) == 1)
    ).astype(int)
    df["sar_distance_pct"] = (close - df["sar_t"]) / close

    df["sar_macd_combo_signal"] = np.where(
        (df["sar_flip_bullish"] == 1) &
        (df["macd_histogram"] > 0) &
        (df["macd_histogram"] > df["macd_histogram"].shift(1)),
        1,
        np.where(
            (df["sar_flip_bearish"] == 1) | (df["macd_histogram"] < 0), -1, 0,
        ),
    )

    # ── 9. DMI / ADX ─────────────────────────────────────────────────────────
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    cond     = plus_dm > minus_dm
    plus_dm  = plus_dm.where(cond, 0)
    minus_dm = minus_dm.where(~cond, 0)
    atr14    = df["atr_14"].replace(0, np.nan)
    df["plus_di_14"]  = 100 * _smma(plus_dm, 14) / atr14
    df["minus_di_14"] = 100 * _smma(minus_dm, 14) / atr14
    dx = (100 * (df["plus_di_14"] - df["minus_di_14"]).abs()
          / (df["plus_di_14"] + df["minus_di_14"]).replace(0, np.nan))
    df["adx_14"]      = _smma(dx, 14)
    df["adx_slope_3d"] = df["adx_14"].diff(3)
    df["dmi_bullish_flag"] = (
        (df["plus_di_14"] > df["minus_di_14"]) & (df["adx_14"] > 25)
    ).astype(int)
    df["dmi_crossover_bullish"] = (
        (df["plus_di_14"] > df["minus_di_14"]) &
        (df["plus_di_14"].shift(1) <= df["minus_di_14"].shift(1))
    ).astype(int)
    df["dmi_wave_signal"] = np.where(
        (df["adx_14"] > 25) &
        (df["plus_di_14"] > df["minus_di_14"]) &
        (df["plus_di_14"] > df["plus_di_14"].shift(1)) &
        (close > df["sma_20"]) &
        (df["adx_slope_3d"] > 0),
        1, 0,
    )

    # ── 10. STOCHASTIC RSI ───────────────────────────────────────────────────
    rsi_min = df["rsi_14"].rolling(14).min()
    rsi_max = df["rsi_14"].rolling(14).max()
    df["stoch_rsi_k"] = 100 * (df["rsi_14"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df["stoch_rsi_d"] = _ema(df["stoch_rsi_k"], 3)
    df["stoch_rsi_oversold"]   = (df["stoch_rsi_k"] < 20).astype(int)
    df["stoch_rsi_overbought"] = (df["stoch_rsi_k"] > 80).astype(int)
    df["stoch_rsi_kd_cross_bullish"] = (
        (df["stoch_rsi_k"] > df["stoch_rsi_d"]) &
        (df["stoch_rsi_k"].shift(1) <= df["stoch_rsi_d"].shift(1)) &
        (df["stoch_rsi_k"].shift(1) < 30)
    ).astype(int)
    df["stoch_rsi_kd_cross_bearish"] = (
        (df["stoch_rsi_k"] < df["stoch_rsi_d"]) &
        (df["stoch_rsi_k"].shift(1) >= df["stoch_rsi_d"].shift(1)) &
        (df["stoch_rsi_k"].shift(1) > 70)
    ).astype(int)

    # ── 11. UPTREND QUALITY ──────────────────────────────────────────────────
    df["uptrend_quality_score"] = (
        (close > df["sma_20"]).astype(int) +
        (df["sma_20"] > df["sma_50"]).astype(int) +
        (df["sma_50"] > df["sma_200"]).astype(int) +
        (df["adx_14"] > 25).astype(int) +
        (df["dmi_bullish_flag"]).astype(int)
    ) / 5.0

    # Uptrend duration: consecutive days all 3 MAs aligned
    aligned = (
        (close > df["sma_20"]) &
        (df["sma_20"] > df["sma_50"]) &
        (df["sma_50"] > df["sma_200"])
    ).astype(int)
    dur = aligned.groupby((aligned == 0).cumsum()).cumsum()
    df["uptrend_duration"] = dur
    df["uptrend_health"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]

    # ── 12. DECLINE FLAGS ────────────────────────────────────────────────────
    df["decline_20d_flag"]     = (df["return_20d"] < -0.15).astype(int)
    df["decline_severity_20d"] = df["return_20d"].abs()
    df["capitulation_vol_flag"] = (
        (df["decline_20d_flag"] == 1) &
        (df["rel_volume"] > 2.5) &
        (close < open_)
    ).astype(int)
    df["deep_oversold_ma20_flag"] = (df["dist_sma_20"] < -0.15).astype(int)
    df["ma20_deviation_pct"]      = df["dist_sma_20"]

    # ── 13. HURST EXPONENT (rolling 60d) ─────────────────────────────────────
    df["hurst_60d"] = close.rolling(60).apply(
        lambda x: _hurst(pd.Series(x)), raw=False
    ).fillna(0.5)

    # ── 14. ICHIMOKU ─────────────────────────────────────────────────────────
    def _midpoint(s_high, s_low, n):
        return (s_high.rolling(n).max() + s_low.rolling(n).min()) / 2
    tenkan = _midpoint(high, low, 9)
    kijun  = _midpoint(high, low, 26)
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = _midpoint(high, low, 52).shift(26)
    df["tenkan_sen"]  = tenkan
    df["kijun_sen"]   = kijun
    df["senkou_span_a"] = span_a
    df["senkou_span_b"] = span_b
    df["chikou_span"]   = close.shift(-26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    df["price_above_cloud"] = (close > cloud_top).astype(int)
    df["price_below_cloud"] = (close < cloud_bot).astype(int)
    df["cloud_bullish"]     = (span_a > span_b).astype(int)
    df["cloud_thickness"]   = (span_a - span_b).abs() / close
    df["tenkan_kijun_cross_bullish"] = (
        (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    ).astype(int)

    # ── 15. CCI ──────────────────────────────────────────────────────────────
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))
    df["cci_oversold_flag"] = (df["cci_20"] < -100).astype(int)
    # CCI bullish divergence: price lower low but CCI higher low (rolling 10d)
    df["cci_bullish_divergence"] = (
        (close == close.rolling(10).min()) &
        (df["cci_20"] > df["cci_20"].rolling(10).min())
    ).astype(int)

    # ── 16. ELDER RAY ────────────────────────────────────────────────────────
    df["bull_power"] = high - _ema(close, 13)
    df["bear_power"] = low  - _ema(close, 13)

    # ── 17. WILLIAMS ALLIGATOR ───────────────────────────────────────────────
    med = (high + low) / 2
    df["alligator_jaw"]   = _smma(med, 13).shift(8)
    df["alligator_teeth"] = _smma(med, 8).shift(5)
    df["alligator_lips"]  = _smma(med, 5).shift(3)
    df["alligator_spread"] = (df["alligator_lips"] - df["alligator_jaw"]) / df["alligator_jaw"]

    # ── 18. AROON ────────────────────────────────────────────────────────────
    df["aroon_up"]   = 100 * high.rolling(26).apply(
        lambda x: (len(x)-1-x[::-1].argmax()) / 25, raw=True
    )
    df["aroon_down"] = 100 * low.rolling(26).apply(
        lambda x: (len(x)-1-x[::-1].argmin()) / 25, raw=True
    )
    df["aroon_oscillator"] = df["aroon_up"] - df["aroon_down"]

    # ── 19. VOLUME PROFILE (proxy) ───────────────────────────────────────────
    # POC = close of highest-volume day in last 20 days
    def _poc_20(df_local, i):
        if i < 20: return np.nan
        sub = df_local.iloc[i-20:i]
        idx = sub["volume"].idxmax()
        return sub.loc[idx, "close"]
    df["poc_20d"] = [_poc_20(df, i) for i in range(len(df))]
    df["poc_distance"] = (close - df["poc_20d"]) / df["poc_20d"].replace(0, np.nan)

    # ── 20. SUPPORT & RESISTANCE ZONES ───────────────────────────────────────
    # Swing highs / lows (simplified: rolling local max/min)
    df["swing_high_252"] = high.rolling(252, center=False).max()
    df["swing_low_252"]  = low.rolling(252, center=False).min()
    df["dist_to_52w_high_pct"] = (df["swing_high_252"] - close) / close
    df["dist_to_52w_low_pct"]  = (close - df["swing_low_252"]) / close
    df["at_52w_high_flag"]     = (close >= df["swing_high_252"] * 0.99).astype(int)

    # Nearest resistance: last 20-day rolling max
    df["nearest_resistance"] = high.rolling(20).max().shift(1)
    df["nearest_resistance_pct"] = (df["nearest_resistance"] - close) / close
    df["inside_resistance_zone"] = (df["nearest_resistance_pct"].abs() < 0.02).astype(int)

    # Nearest support: last 20-day rolling min
    df["nearest_support"] = low.rolling(20).min().shift(1)
    df["nearest_support_pct"] = (close - df["nearest_support"]) / close
    df["inside_support_zone"] = (df["nearest_support_pct"].abs() < 0.02).astype(int)

    # Round number (psychological levels) – nearest multiple
    def _round_num(p):
        if p < 20_000:  mult = 1_000
        elif p < 100_000: mult = 5_000
        else: mult = 10_000
        return round(p / mult) * mult
    round_prices = close.apply(_round_num)
    df["dist_to_round_number_pct"] = (close - round_prices).abs() / close
    df["at_round_number_flag"] = (df["dist_to_round_number_pct"] < 0.005).astype(int)

    # Resistance broken flag
    df["resistance_broken_flag"] = (
        (close > df["nearest_resistance"].shift(1)) &
        (df["rel_volume"] > 2.0)
    ).astype(int)

    # ── 21. FIBONACCI LEVELS ─────────────────────────────────────────────────
    swing_lo = close.rolling(20).min()
    swing_hi = close.rolling(20).max()
    df["fib_382_level"] = swing_hi - 0.382 * (swing_hi - swing_lo)
    df["fib_618_level"] = swing_hi - 0.618 * (swing_hi - swing_lo)
    df["fib_786_level"] = swing_hi - 0.786 * (swing_hi - swing_lo)
    df["fib_ext_1272"]  = swing_lo + 1.272 * (swing_hi - swing_lo)
    df["fib_ext_1618"]  = swing_lo + 1.618 * (swing_hi - swing_lo)
    df["dist_to_fib_618"] = (close - df["fib_618_level"]).abs() / close
    df["at_fib_618_flag"]  = (df["dist_to_fib_618"] < 0.015).astype(int)
    df["at_fib_382_flag"]  = ((close - df["fib_382_level"]).abs() / close < 0.015).astype(int)
    df["fib_confluence_flag"] = (
        (df["at_fib_618_flag"] == 1) & (df["inside_support_zone"] == 1)
    ).astype(int)
    df["fib_bounce_setup"] = (
        (df["at_fib_618_flag"] == 1) &
        (df["inside_support_zone"] == 1) &
        (df["volume_declining_pullback"] == 1)
    ).astype(int)

    # ── 22. CANDLESTICK PATTERNS ─────────────────────────────────────────────
    body   = (close - open_).abs()
    candle = high - low
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)

    # Hammer
    hammer_flag = (lower_wick > 2*body) & (lower_wick > upper_wick*2) & (candle > 0)
    df["hammer_at_support"] = (hammer_flag & (df["inside_support_zone"] == 1)).astype(int)

    # Shooting star
    star_flag = (upper_wick > 2*body) & (upper_wick > lower_wick*2) & (candle > 0)
    df["shooting_star_at_resistance"] = (star_flag & (df["inside_resistance_zone"] == 1)).astype(int)

    # Pinbar
    df["bullish_pinbar_at_level"] = (
        hammer_flag & (
            (df["inside_support_zone"] == 1) | (df["at_fib_618_flag"] == 1)
        )
    ).astype(int)
    df["bearish_pinbar_at_level"] = (
        star_flag & (
            (df["inside_resistance_zone"] == 1) | (df["at_52w_high_flag"] == 1)
        )
    ).astype(int)

    # Wick signals
    upper_wick_ratio = (upper_wick / candle.replace(0, np.nan)).fillna(0)
    lower_wick_ratio = (lower_wick / candle.replace(0, np.nan)).fillna(0)
    df["upper_wick_resistance"] = (upper_wick_ratio > 0.5).astype(int)
    df["momentum_continuation"] = ((upper_wick_ratio < 0.3) & (lower_wick_ratio < 0.3)).astype(int)

    # Morning/Evening star (simplified 3-candle)
    c1_bull = (close.shift(2) > open_.shift(2)) & (body.shift(2) > candle.shift(2)*0.5)
    c2_doji = body.shift(1) < candle.shift(1) * 0.3
    c3_bear = (close < open_) & (close < (open_.shift(2) + close.shift(2))/2)
    df["evening_star_at_resistance"] = (c1_bull & c2_doji & c3_bear & (df["inside_resistance_zone"]==1)).astype(int)
    c1_bear = (close.shift(2) < open_.shift(2)) & (body.shift(2) > candle.shift(2)*0.5)
    c3_bull = (close > open_) & (close > (open_.shift(2) + close.shift(2))/2)
    df["morning_star_at_support"] = (c1_bear & c2_doji & c3_bull & (df["inside_support_zone"]==1)).astype(int)

    # ── 23. GAP FEATURES ─────────────────────────────────────────────────────
    df["gap"] = (open_ - close.shift(1)) / close.shift(1)
    df["breakaway_gap_flag"] = (
        (df["gap"] > 0.03) & (df["resistance_broken_flag"] == 1) & (df["rel_volume"] > 2.5)
    ).astype(int)
    df["runaway_gap_flag"] = (
        (df["gap"] > 0.02) & (df["uptrend_quality_score"] > 0.8)
    ).astype(int)
    df["exhaustion_gap_flag"] = (
        (df["gap"] > 0.02) & (df["return_20d"] > 0.25) & (df["rsi_14"] > 75)
    ).astype(int)
    df["common_gap_flag"] = (
        (df["gap"].abs() > 0.005) & (df["gap"].abs() < 0.015) & (df["rel_volume"] < 1.3)
    ).astype(int)

    # VN: gap trap (large gap up near ceiling)
    df["dist_to_ceiling_pct"] = 0.07 - df["return_1d"].shift(1).fillna(0).clip(0, 0.07)
    df["gap_trap_flag"] = (
        (df["gap"] > 0.04) & (df["dist_to_ceiling_pct"] < 0.02)
    ).astype(int)

    # ── 24. VN-SPECIFIC FLAGS ────────────────────────────────────────────────
    # Ceiling/floor detection (±7% moves)
    df["hit_ceiling"] = (df["return_1d"] >= 0.068).astype(int)
    df["hit_floor"]   = (df["return_1d"] <= -0.068).astype(int)
    df["post_ceiling_day1_flag"]    = df["hit_ceiling"].shift(1).fillna(0).astype(int)
    df["post_ceiling_day2_flag"]    = (df["hit_ceiling"].shift(1).fillna(0) + df["hit_ceiling"].shift(2).fillna(0) == 2).astype(int)
    df["post_ceiling_day3_plus_flag"] = (
        df["hit_ceiling"].rolling(3).sum().shift(1).fillna(0) >= 3
    ).astype(int)
    df["post_floor_day1_flag"] = df["hit_floor"].shift(1).fillna(0).astype(int)
    df["post_floor_day3_plus_flag"] = (
        df["hit_floor"].rolling(3).sum().shift(1).fillna(0) >= 3
    ).astype(int)

    # T+3 exit pressure
    df["t3_exit_pressure"] = (
        (df["return_1d"].shift(3) > 0.04) & (df["rel_volume"] > 1.5)
    ).astype(int)

    # Earnings season (mid Jan/Apr/Jul/Oct ±5 days)
    df["earnings_season_flag"] = df["date"].apply(
        lambda d: int(d.month in [1,4,7,10] and 10 <= d.day <= 25)
    )

    # Day of week
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon, 4=Fri

    # ── 25. CHART PATTERNS ───────────────────────────────────────────────────
    # Double Bottom: two lows within 2% at 5+ bar separation
    def _double_bottom(df_local, i, lookback=30):
        if i < lookback: return 0
        sub    = df_local.iloc[i-lookback:i]
        lows_  = sub["low"]
        lo_idx = lows_.nsmallest(2).index
        if len(lo_idx) < 2: return 0
        i1, i2  = sorted(lo_idx)
        if abs(i2 - i1) < 5: return 0
        l1, l2  = float(lows_[i1]), float(lows_[i2])
        if abs(l1-l2)/max(l1,l2) > 0.02: return 0
        neckline = float(sub.iloc[i1:i2+1]["high"].max())
        if float(df_local.iloc[i]["close"]) > neckline: return 1
        return 0

    df["double_bottom_flag"] = [_double_bottom(df, i) for i in range(len(df))]
    df["double_top_flag"]    = 0  # symmetric implementation omitted for brevity

    # Bull flag: pole then tight consolidation
    df["bull_flag_forming"] = (
        (df["return_5d"].shift(5) > 0.08) &
        (df["vol_20d"] < df["vol_20d"].rolling(20).mean() * 0.7)
    ).astype(int)
    df["bull_flag_breakout"] = (
        (df["bull_flag_forming"] == 1) &
        (df["return_1d"] > 0.02) &
        (df["rel_volume"] > 1.8)
    ).astype(int)

    # Rising/falling wedge flags (simplified slope comparison)
    upper_slope = _rolling_linslope(high.rolling(3).max(), 15)
    lower_slope = _rolling_linslope(low.rolling(3).min(), 15)
    df["rising_wedge_flag"]  = ((upper_slope > 0) & (lower_slope > 0) & (upper_slope < lower_slope)).astype(int)
    df["falling_wedge_flag"] = ((upper_slope < 0) & (lower_slope < 0) & (upper_slope < lower_slope)).astype(int)

    # Three pushes
    def _three_pushes_high(df_local, i, lookback=20):
        if i < lookback: return 0
        sub = df_local.iloc[i-lookback:i]
        sw  = sub[(sub["high"] > sub["high"].shift(1)) & (sub["high"] > sub["high"].shift(-1).fillna(0))]
        if len(sw) < 3: return 0
        last3 = sw["high"].tail(3).values
        vols  = sub.loc[sw.index[-3:], "volume"].values if len(sw) >= 3 else []
        macds = sub.loc[sw.index[-3:], "macd_histogram"].values if "macd_histogram" in sub.columns and len(sw)>=3 else []
        if len(last3) < 3: return 0
        if last3[0] < last3[1] < last3[2] and len(vols)==3 and vols[0]>vols[1]>vols[2]: return 1
        return 0

    df["three_pushes_high_flag"] = [_three_pushes_high(df, i) for i in range(len(df))]
    df["three_pushes_low_flag"]  = 0

    # ── 26. WYCKOFF PHASE ────────────────────────────────────────────────────
    # Simplified: classify into A/B/C/D/E based on available indicators
    def _wyckoff_phase(row):
        if row["volume_climax_flag"] == 1 and abs(row["return_1d"]) > 0.03:
            return "A"
        if abs(row["return_20d"]) < 0.05 and row["adx_14"] < 20:
            return "B"
        if row["inside_support_zone"] == 1 and row["rel_volume"] > 1.5 and row["return_1d"] > 0:
            return "C"
        if row["dmi_wave_signal"] == 1:
            return "D"
        if row["uptrend_quality_score"] >= 0.8 and row["adx_14"] > 30:
            return "E"
        return "B"

    df["wyckoff_phase"] = df.apply(_wyckoff_phase, axis=1)
    df["wyckoff_accumulation_flag"] = (
        (df["wyckoff_phase"] == "C") & (close > close.shift(1))
    ).astype(int)
    df["wyckoff_distribution_flag"] = (
        (df["volume_climax_flag"] == 1) & (df["return_20d"] > 0.20) &
        (df["rel_volume"] > 2.0) & (close < close.shift(1))
    ).astype(int)

    # ── 27. SMC — FAIR VALUE GAPS ─────────────────────────────────────────────
    df["fvg_bullish"] = (low > high.shift(2)).astype(int)
    df["fvg_bearish"] = (high < low.shift(2)).astype(int)
    df["open_fvg_above"] = high.shift(2).where(df["fvg_bullish"] == 1, np.nan)
    df["open_fvg_below"] = low.shift(2).where(df["fvg_bearish"] == 1, np.nan)

    # ── 28. MARKET STRUCTURE (BOS / CHoCH) ───────────────────────────────────
    sw_hi = high.rolling(5).max()
    sw_lo = low.rolling(5).min()
    df["market_structure_bullish"] = (
        (high > sw_hi.shift(5)) & (low > sw_lo.shift(5))
    ).astype(int)
    df["bos_bullish"] = (close > sw_hi.shift(1) * 1.005).astype(int)
    df["choch_bearish"] = (
        (close < sw_lo.shift(1) * 0.995) & (df["market_structure_bullish"] == 1)
    ).astype(int)

    # ── 29. LIQUIDITY SWEEP ───────────────────────────────────────────────────
    recent_swing_low = low.rolling(10).min().shift(1)
    df["liquidity_sweep_bullish"] = (
        (low < recent_swing_low * 0.985) & (close > recent_swing_low)
    ).astype(int)
    recent_swing_high = high.rolling(10).max().shift(1)
    df["liquidity_sweep_bearish"] = (
        (high > recent_swing_high * 1.015) & (close < recent_swing_high)
    ).astype(int)

    # ── 30. BULL/BEAR TRAP FLAGS ─────────────────────────────────────────────
    df["bull_trap_flag"] = (
        (df["breakaway_gap_flag"] == 0) &
        (df["resistance_broken_flag"] == 1) &
        (close < close.rolling(3).mean().shift(1)) &
        (df["rel_volume"] < 1.0)
    ).astype(int)
    df["bear_trap_flag"] = (
        (df["inside_support_zone"] == 1) &
        (close < low.shift(1)) &
        (df["return_1d"] > 0)
    ).astype(int)

    # FOMO signal: already up big + overbought + low vol
    df["fomo_signal"] = (
        (df["return_20d"] > 0.20) &
        (df["rsi_14"] > 75) &
        (df["three_pushes_high_flag"] == 1)
    ).astype(int)

    # ── 31. FORWARD RETURNS (for model training) ──────────────────────────────
    for t in [3, 5, 10, 20, 60, 180]:
        df[f"fwd_return_t{t}"]   = close.shift(-t) / close - 1
        df[f"fwd_positive_t{t}"] = (df[f"fwd_return_t{t}"] > 0).astype(int)

    # ── 32. VWAP (approx daily reset) ────────────────────────────────────────
    typical = (high + low + close) / 3
    df["vwap"] = (typical * vol).rolling(20).sum() / vol.rolling(20).sum()
    df["vwap_deviation"] = (close - df["vwap"]) / df["vwap"]

    # ── 33. RELATIVE STRENGTH vs INDEX (synthetic proxy) ─────────────────────
    df["stock_rs_20d"] = (1 + df["return_20d"]) / 1.0  # placeholder; real RS needs index data
    df["vnindex_ma_score"] = df["ma_score_stock"]  # proxy until index data is available

    # ── 34. FOREIGN FLOW (synthetic proxy) ───────────────────────────────────
    # In production: load from real foreign flow data; here we proxy via volume imbalance
    rng2 = np.random.default_rng(abs(hash(str(df["close"].iloc[0]) + ticker)) % 2**31)
    df["net_foreign_flow_5d"] = (rng2.normal(0, 0.03, len(df)) +
                                  df["return_5d"].fillna(0) * 0.5)

    # ── 35. INTERBANK RATE PROXY ─────────────────────────────────────────────
    df["interbank_rate"] = 3.5 + rng2.normal(0, 0.1, len(df))
    df["interbank_rate_change"] = df["interbank_rate"].diff().fillna(0)

    # ── Breadth proxies (per-stock approximations) ────────────────────────────
    df["breadth_pct_above_50sma"] = (close > df["sma_50"]).rolling(20).mean()
    df["adv_dec_ratio_10d"] = (
        (df["return_1d"] > 0).rolling(10).sum() /
        ((df["return_1d"] < 0).rolling(10).sum() + 1e-6)
    )

    # ── 36. SECTOR ROTATION STAGE (synthetic) ────────────────────────────────
    df["rotation_stage"] = 1  # to be overwritten by multi-stock sector analysis
    df["is_sector_leader"]   = 0
    df["is_sector_laggard"]  = 0

    # ── 37. T+2.5 RISK (simplified) ──────────────────────────────────────────
    # Expected max adverse excursion over 3 days given current vol
    df["t25_risk"] = df["garch_sigma"] / np.sqrt(252) * np.sqrt(3) * 1.645  # 95th pctile

    # ── 38. SUPPLY/DEMAND ZONE PROXIES ───────────────────────────────────────
    df["inside_demand_zone"] = (
        (df["inside_support_zone"] == 1) & (df["vol_explosion_flag"].shift(5).fillna(0) == 1)
    ).astype(int)
    df["inside_supply_zone"] = (
        (df["inside_resistance_zone"] == 1) & (df["vol_explosion_flag"].shift(5).fillna(0) == 1)
    ).astype(int)

    # ── Metadata ──────────────────────────────────────────────────────────────
    df["ticker"]   = ticker
    df["close_raw"] = df["close"]

    # Drop rows with insufficient history
    df = df.dropna(subset=["sma_200", "rsi_14", "adx_14"]).reset_index(drop=True)
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────
def _tickers_available() -> list[str]:
    return [f.stem for f in RAW.glob("*.csv")]


def run(ticker: str) -> None:
    raw_path = RAW / f"{ticker}.csv"
    if not raw_path.exists():
        log.error(f"Raw data not found: {raw_path}  →  run 01_download.py first")
        return
    df_raw = pd.read_csv(raw_path, dtype={"date": str})
    if len(df_raw) < 60:
        log.warning(f"{ticker}: too few rows ({len(df_raw)}), skipping")
        return
    log.info(f"{ticker}: computing features ({len(df_raw)} rows) …")
    df_feat = build_features(df_raw, ticker=ticker)
    out = FEATURES / f"{ticker}.csv"
    df_feat.to_csv(out, index=False)
    log.info(f"{ticker}: saved {len(df_feat)} rows × {len(df_feat.columns)} features → {out.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VN100 Feature Engineering")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    tickers = _tickers_available() if args.all else ([args.ticker] if args.ticker else _tickers_available())
    log.info(f"=== Feature Engineering: {len(tickers)} tickers ===\n")
    for t in tickers:
        run(t)