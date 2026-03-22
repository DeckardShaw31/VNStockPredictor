"""
math_models.py — Classical mathematical trading models.

Each function takes OHLCV data and returns levels/signals that are used:
  1. As features in the ML model
  2. As entry/stop-loss/take-profit inputs in trade_signals.py
  3. As confirmation votes in the live signal engine

Models implemented:
  - Fibonacci Retracement & Extension
  - Pivot Points (Classic, Camarilla, Woodie)
  - Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)
  - Hull Moving Average (HMA)
  - Triple EMA (TEMA)
  - Parabolic SAR
  - Donchian Channel
  - VWAP with standard deviation bands
  - Supertrend
  - Elder Ray (Bull/Bear Power)
  - Chande Momentum Oscillator
  - Squeeze Momentum (Lazy Bear)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# ── Utility ────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=max(2, min(n, len(s)-1)), adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(max(2, min(n, len(s)-1))).mean()

def _wma(s: pd.Series, n: int) -> pd.Series:
    """Weighted moving average."""
    n = max(2, min(n, len(s)))
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)


# ── Hull Moving Average ────────────────────────────────────────────────────────

def hull_moving_average(close: pd.Series, n: int = 20) -> pd.Series:
    """
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    
    Much smoother than SMA/EMA with significantly less lag.
    Ideal for trend identification without the usual lag penalty.
    """
    half = max(2, n // 2)
    sqrt_n = max(2, int(np.sqrt(n)))
    hma = _wma(2 * _wma(close, half) - _wma(close, n), sqrt_n)
    return hma


# ── Triple EMA ────────────────────────────────────────────────────────────────

def triple_ema(close: pd.Series, n: int = 20) -> pd.Series:
    """
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    
    Near-zero lag trend indicator. Faster reaction than EMA or DEMA.
    """
    e1 = _ema(close, n)
    e2 = _ema(e1, n)
    e3 = _ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


# ── Fibonacci ─────────────────────────────────────────────────────────────────

FIBO_RETRACE_LEVELS  = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
FIBO_EXTEND_LEVELS   = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]

def fibonacci_levels(
    high: float,
    low: float,
    direction: str = "up",    # "up" = retracement from high, "down" = from low
) -> Dict[str, float]:
    """
    Compute Fibonacci retracement and extension levels.
    
    For an uptrend: swing_low + (swing_high - swing_low) * level
    For a downtrend: swing_high - (swing_high - swing_low) * level
    
    Returns dict of {"retrace_23.6": price, "extend_161.8": price, ...}
    """
    diff   = high - low
    levels = {}

    for lvl in FIBO_RETRACE_LEVELS:
        if direction == "up":
            price = high - diff * lvl
        else:
            price = low + diff * lvl
        levels[f"fib_retrace_{lvl*100:.1f}"] = round(price, 2)

    for lvl in FIBO_EXTEND_LEVELS:
        if direction == "up":
            price = low + diff * lvl
        else:
            price = high - diff * lvl
        levels[f"fib_extend_{lvl*100:.1f}"] = round(price, 2)

    return levels


def rolling_fibonacci(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    Rolling Fibonacci levels based on the highest high and lowest low
    over the past `window` bars.
    
    Returns DataFrame with fib levels as columns, indexed like input.
    Useful as features: price distance from each fib level.
    """
    roll_high = high.rolling(window).max()
    roll_low  = low.rolling(window).min()
    diff      = roll_high - roll_low

    df = pd.DataFrame(index=high.index)
    df["fib_swing_high"] = roll_high
    df["fib_swing_low"]  = roll_low

    for lvl in [0.236, 0.382, 0.500, 0.618, 0.786]:
        df[f"fib_{int(lvl*1000)}"] = roll_high - diff * lvl   # retrace level

    # Current price distance to 61.8 fib (most watched level)
    df["fib_dist_618"]   = (high.rolling(1).max() - df["fib_618"]) / df["fib_618"].replace(0, np.nan)

    return df


# ── Pivot Points ──────────────────────────────────────────────────────────────

def pivot_classic(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    """
    Classic floor trader pivot points.
    Computed from previous session's H/L/C.
    
    Most widely watched S/R levels by Vietnamese retail traders.
    """
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    r2 = pp + (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s1 = 2 * pp - prev_high
    s2 = pp - (prev_high - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


def pivot_camarilla(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    """
    Camarilla pivot points — tighter levels, better for intraday scalping.
    Uses multiplier 1.1/12 for closer levels.
    """
    diff = prev_high - prev_low
    r4 = prev_close + diff * 1.1 / 2
    r3 = prev_close + diff * 1.1 / 4
    r2 = prev_close + diff * 1.1 / 6
    r1 = prev_close + diff * 1.1 / 12
    s1 = prev_close - diff * 1.1 / 12
    s2 = prev_close - diff * 1.1 / 6
    s3 = prev_close - diff * 1.1 / 4
    s4 = prev_close - diff * 1.1 / 2
    return {"R4": r4, "R3": r3, "R2": r2, "R1": r1,
            "S1": s1, "S2": s2, "S3": s3, "S4": s4}


def pivot_woodie(prev_high: float, prev_low: float, prev_close: float, curr_open: float) -> Dict[str, float]:
    """
    Woodie pivot — gives extra weight to the opening price.
    Popular with open-auction market traders.
    """
    pp = (prev_high + prev_low + 2 * curr_open) / 4
    r1 = 2 * pp - prev_low
    r2 = pp + prev_high - prev_low
    s1 = 2 * pp - prev_high
    s2 = pp - prev_high + prev_low
    return {"PP": pp, "R1": r1, "R2": r2, "S1": s1, "S2": s2}


def rolling_pivots(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """
    Apply pivot calculations rolling across the full price history.
    Returns DataFrame with all pivot levels as columns.
    """
    ph = high.shift(1)
    pl = low.shift(1)
    pc = close.shift(1)
    diff = ph - pl

    df = pd.DataFrame(index=high.index)
    df["pivot_pp"] = (ph + pl + pc) / 3
    df["pivot_r1"] = 2 * df["pivot_pp"] - pl
    df["pivot_r2"] = df["pivot_pp"] + diff
    df["pivot_r3"] = ph + 2 * (df["pivot_pp"] - pl)
    df["pivot_s1"] = 2 * df["pivot_pp"] - ph
    df["pivot_s2"] = df["pivot_pp"] - diff
    df["pivot_s3"] = pl - 2 * (ph - df["pivot_pp"])

    # Camarilla
    df["cam_r3"] = pc + diff * 1.1 / 4
    df["cam_r4"] = pc + diff * 1.1 / 2
    df["cam_s3"] = pc - diff * 1.1 / 4
    df["cam_s4"] = pc - diff * 1.1 / 2

    # Distance of current price from PP (feature)
    df["dist_from_pp"] = (close - df["pivot_pp"]) / df["pivot_pp"].replace(0, np.nan)
    df["above_r1"]     = (close > df["pivot_r1"]).astype(int)
    df["below_s1"]     = (close < df["pivot_s1"]).astype(int)

    return df


# ── Ichimoku Cloud ─────────────────────────────────────────────────────────────

def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_n: int = 9,
    kijun_n: int = 26,
    senkou_b_n: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Full Ichimoku Kinko Hyo system.
    
    Components:
      Tenkan-sen (Conversion): (9H-high + 9H-low) / 2  — fast trend
      Kijun-sen (Base):        (26H-high + 26H-low) / 2  — slow trend / S/R
      Senkou Span A:           (Tenkan + Kijun) / 2, displaced +26  — cloud top/bottom
      Senkou Span B:           (52H-high + 52H-low) / 2, displaced +26
      Chikou Span:             Close displaced -26  — lagging confirmation
    
    Signal rules used in trade_signals.py:
      - Price above cloud         = bullish bias
      - Price below cloud         = bearish bias
      - Tenkan cross above Kijun  = bullish TK cross (buy signal)
      - Tenkan cross below Kijun  = bearish TK cross (sell signal)
      - Price above Kijun         = trend support confirmed
    """
    n_t = max(2, min(tenkan_n,   len(high)-1))
    n_k = max(2, min(kijun_n,    len(high)-1))
    n_b = max(2, min(senkou_b_n, len(high)-1))

    tenkan = (high.rolling(n_t).max() + low.rolling(n_t).min()) / 2
    kijun  = (high.rolling(n_k).max() + low.rolling(n_k).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(min(displacement, len(high)-1))
    senkou_b = ((high.rolling(n_b).max() + low.rolling(n_b).min()) / 2).shift(min(displacement, len(high)-1))
    chikou  = close.shift(-min(displacement, len(close)-1))

    df = pd.DataFrame({
        "ichi_tenkan":   tenkan,
        "ichi_kijun":    kijun,
        "ichi_senkou_a": senkou_a,
        "ichi_senkou_b": senkou_b,
        "ichi_chikou":   chikou,
    }, index=high.index)

    # Derived signals
    cloud_top    = df[["ichi_senkou_a", "ichi_senkou_b"]].max(axis=1)
    cloud_bottom = df[["ichi_senkou_a", "ichi_senkou_b"]].min(axis=1)

    df["ichi_above_cloud"]  = (close > cloud_top).astype(int)
    df["ichi_below_cloud"]  = (close < cloud_bottom).astype(int)
    df["ichi_in_cloud"]     = ((close >= cloud_bottom) & (close <= cloud_top)).astype(int)
    df["ichi_cloud_green"]  = (df["ichi_senkou_a"] > df["ichi_senkou_b"]).astype(int)

    # Tenkan / Kijun cross
    df["ichi_tk_cross_up"]   = ((tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))).astype(int)
    df["ichi_tk_cross_down"] = ((tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))).astype(int)

    # Price vs Kijun (trend support)
    df["ichi_above_kijun"] = (close > kijun).astype(int)

    return df


# ── Parabolic SAR ─────────────────────────────────────────────────────────────

def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    step: float = 0.02,
    max_step: float = 0.20,
) -> pd.DataFrame:
    """
    Parabolic Stop and Reverse (SAR).
    
    The SAR level acts as a trailing stop:
      - When price is above SAR → bullish trend, SAR is your stop-loss floor
      - When price crosses below SAR → trend reversal, flip to short
    
    Widely used in Vietnam retail trading as a trailing stop reference.
    """
    high_arr  = high.values
    low_arr   = low.values
    n         = len(high_arr)

    sar    = np.zeros(n)
    ep     = np.zeros(n)      # extreme point
    af     = np.zeros(n)      # acceleration factor
    trend  = np.zeros(n)      # 1 = up, -1 = down

    # Initialise
    trend[0] = 1
    sar[0]   = low_arr[0]
    ep[0]    = high_arr[0]
    af[0]    = step

    for i in range(1, n):
        prev_sar   = sar[i-1]
        prev_ep    = ep[i-1]
        prev_af    = af[i-1]
        prev_trend = trend[i-1]

        if prev_trend == 1:   # Uptrend
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            new_sar = min(new_sar, low_arr[i-1], low_arr[max(0, i-2)])

            if low_arr[i] < new_sar:   # Reversal to downtrend
                trend[i] = -1
                sar[i]   = prev_ep
                ep[i]    = low_arr[i]
                af[i]    = step
            else:
                trend[i] = 1
                sar[i]   = new_sar
                if high_arr[i] > prev_ep:
                    ep[i] = high_arr[i]
                    af[i] = min(prev_af + step, max_step)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af

        else:   # Downtrend
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            new_sar = max(new_sar, high_arr[i-1], high_arr[max(0, i-2)])

            if high_arr[i] > new_sar:   # Reversal to uptrend
                trend[i] = 1
                sar[i]   = prev_ep
                ep[i]    = high_arr[i]
                af[i]    = step
            else:
                trend[i] = -1
                sar[i]   = new_sar
                if low_arr[i] < prev_ep:
                    ep[i] = low_arr[i]
                    af[i] = min(prev_af + step, max_step)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af

    result = pd.DataFrame({
        "psar":           sar,
        "psar_trend":     trend,     # 1 = up, -1 = down
        "psar_dist":      (high.values - sar) / high.values,  # normalised distance
        "psar_cross_up":  ((trend == 1) & (np.roll(trend, 1) == -1)).astype(int),
        "psar_cross_down":((trend == -1) & (np.roll(trend, 1) == 1)).astype(int),
    }, index=high.index)

    return result


# ── Donchian Channel ──────────────────────────────────────────────────────────

def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 20,
) -> pd.DataFrame:
    """
    Donchian Channel — breakout detection.
    
    Upper: highest high over n periods
    Lower: lowest low over n periods
    Mid:   (upper + lower) / 2
    
    Breakout above upper = momentum buy signal
    Breakout below lower = momentum sell signal
    """
    n = max(2, min(n, len(high)-1))
    upper = high.rolling(n).max()
    lower = low.rolling(n).min()
    mid   = (upper + lower) / 2

    df = pd.DataFrame({
        "don_upper":    upper,
        "don_lower":    lower,
        "don_mid":      mid,
        "don_pct":      (close - lower) / (upper - lower).replace(0, np.nan),
        "don_breakout_up":   (close >= upper).astype(int),
        "don_breakout_down": (close <= lower).astype(int),
        "don_width":    (upper - lower) / mid.replace(0, np.nan),
    }, index=high.index)

    return df


# ── Supertrend ────────────────────────────────────────────────────────────────

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend indicator — combines ATR with trend direction.
    
    Very popular in Vietnamese retail communities (often called 'xu huong sieu').
    Provides clear buy/sell signals and a trailing stop level.
    
    Green (uptrend): price above supertrend line
    Red (downtrend): price below supertrend line
    """
    n = max(2, min(atr_period, len(high)-1))

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()

    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Final bands (with trend-preservation logic)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    supertrend_arr = np.zeros(len(close))
    trend_arr = np.ones(len(close))

    for i in range(1, len(close)):
        # Upper band
        if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        # Lower band
        if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

        # Trend
        if supertrend_arr[i-1] == final_upper.iloc[i-1]:
            if close.iloc[i] <= final_upper.iloc[i]:
                supertrend_arr[i] = final_upper.iloc[i]
                trend_arr[i] = -1
            else:
                supertrend_arr[i] = final_lower.iloc[i]
                trend_arr[i] = 1
        else:
            if close.iloc[i] >= final_lower.iloc[i]:
                supertrend_arr[i] = final_lower.iloc[i]
                trend_arr[i] = 1
            else:
                supertrend_arr[i] = final_upper.iloc[i]
                trend_arr[i] = -1

    supertrend_s = pd.Series(supertrend_arr, index=close.index)
    trend_s      = pd.Series(trend_arr,      index=close.index)

    return pd.DataFrame({
        "supertrend":        supertrend_s,
        "supertrend_trend":  trend_s,      # 1 = bullish, -1 = bearish
        "supertrend_dist":   (close - supertrend_s) / close.replace(0, np.nan),
        "supertrend_buy":    ((trend_s == 1) & (trend_s.shift(1) == -1)).astype(int),
        "supertrend_sell":   ((trend_s == -1) & (trend_s.shift(1) == 1)).astype(int),
    }, index=close.index)


# ── Elder Ray ─────────────────────────────────────────────────────────────────

def elder_ray(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 13,
) -> pd.DataFrame:
    """
    Elder Ray Index — Bull Power and Bear Power.
    
    Bull Power = High - EMA(n)   (buyer strength above average)
    Bear Power = Low  - EMA(n)   (seller pressure below average)
    
    Buy signal:  EMA trending up  + Bear Power negative but rising
    Sell signal: EMA trending down + Bull Power positive but falling
    """
    n = max(2, min(n, len(close)-1))
    ema = _ema(close, n)
    bull = high - ema
    bear = low  - ema

    return pd.DataFrame({
        "elder_bull":      bull,
        "elder_bear":      bear,
        "elder_bull_rise": (bull > bull.shift(1)).astype(int),
        "elder_bear_rise": (bear > bear.shift(1)).astype(int),
    }, index=close.index)


# ── Squeeze Momentum (LazyBear) ───────────────────────────────────────────────

def squeeze_momentum(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_n: int = 20,
    bb_mult: float = 2.0,
    kc_n: int = 20,
    kc_mult: float = 1.5,
    mom_n: int = 12,
) -> pd.DataFrame:
    """
    Squeeze Momentum indicator (LazyBear).
    
    Detects when Bollinger Bands squeeze inside Keltner Channel (low volatility
    coiling before a breakout) then measures momentum direction of the breakout.
    
    squeeze_on  = BB inside KC  (energy building)
    squeeze_off = BB outside KC (energy released — breakout signal)
    momentum > 0 and rising = bullish breakout
    momentum < 0 and falling = bearish breakout
    """
    n_bb = max(2, min(bb_n,  len(close)-1))
    n_kc = max(2, min(kc_n,  len(close)-1))
    n_m  = max(2, min(mom_n, len(close)-1))

    # Bollinger Bands
    basis  = _sma(close, n_bb)
    dev    = close.rolling(n_bb).std()
    bb_up  = basis + bb_mult * dev
    bb_low = basis - bb_mult * dev

    # Keltner Channel (ATR-based)
    tr  = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=n_kc, adjust=False).mean()
    kc_mid = _ema(close, n_kc)
    kc_up  = kc_mid + kc_mult * atr
    kc_low = kc_mid - kc_mult * atr

    # Squeeze detection
    squeeze_on  = ((bb_low > kc_low) & (bb_up < kc_up)).astype(int)
    squeeze_off = ((bb_low < kc_low) & (bb_up > kc_up)).astype(int)

    # Momentum (delta of highest high / lowest low midpoint vs SMA)
    highest_high = high.rolling(n_m).max()
    lowest_low   = low.rolling(n_m).min()
    delta = close - ((highest_high + lowest_low) / 2 + _sma(close, n_m)) / 2
    momentum = _ema(delta, n_m)

    return pd.DataFrame({
        "sqz_on":       squeeze_on,
        "sqz_off":      squeeze_off,
        "sqz_momentum": momentum,
        "sqz_mom_rising":(momentum > momentum.shift(1)).astype(int),
        "sqz_bullish":  ((squeeze_off == 1) & (momentum > 0)).astype(int),
        "sqz_bearish":  ((squeeze_off == 1) & (momentum < 0)).astype(int),
    }, index=close.index)


# ── Master Builder ────────────────────────────────────────────────────────────

def build_math_model_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build all math model features and return as a single DataFrame.
    All columns are prefixed with their model name for clarity.
    Used by:
      - features.py (as additional ML features)
      - trade_signals.py (as confirmation votes)
    """
    frames = []

    # Fibonacci (rolling 20-bar swing)
    frames.append(rolling_fibonacci(high, low, window=min(20, len(high)-1)))

    # Pivot Points
    frames.append(rolling_pivots(high, low, close))

    # Ichimoku
    frames.append(ichimoku(high, low, close))

    # HMA (20, 50)
    for n in [20, 50]:
        n_adj = min(n, len(close) - 1)
        hma = hull_moving_average(close, n_adj)
        col = pd.DataFrame({f"hma_{n}": hma,
                            f"price_vs_hma{n}": (close - hma) / hma.replace(0, np.nan)},
                           index=close.index)
        frames.append(col)

    # TEMA (20)
    tema_n = min(20, len(close) - 1)
    tema = triple_ema(close, tema_n)
    frames.append(pd.DataFrame({"tema_20": tema,
                                 "price_vs_tema20": (close - tema) / tema.replace(0, np.nan)},
                                index=close.index))

    # Parabolic SAR
    if len(high) >= 4:
        frames.append(parabolic_sar(high, low))

    # Donchian Channel
    frames.append(donchian_channel(high, low, close, n=min(20, len(high)-1)))

    # Supertrend
    if len(high) >= 4:
        frames.append(supertrend(high, low, close))

    # Elder Ray
    frames.append(elder_ray(high, low, close, n=min(13, len(close)-1)))

    # Squeeze Momentum
    if len(close) >= 10:
        frames.append(squeeze_momentum(high, low, close))

    combined = pd.concat(frames, axis=1)
    combined = combined.ffill(limit=3).bfill(limit=2)
    return combined


def get_math_signal_votes(
    current_close: float,
    math_df: pd.DataFrame,
    latest_idx: int = -1,
) -> Dict[str, int]:
    """
    Extract the latest row of math model signals and return a vote dict.
    Each model votes +1 (bullish), -1 (bearish), or 0 (neutral).
    Used by trade_signals.py to compute agreement score.
    """
    row = math_df.iloc[latest_idx]
    votes = {}

    # Ichimoku
    if "ichi_above_cloud" in row:
        if row.get("ichi_above_cloud", 0):
            votes["ichimoku"] = 1
        elif row.get("ichi_below_cloud", 0):
            votes["ichimoku"] = -1
        else:
            votes["ichimoku"] = 0

    # Supertrend
    if "supertrend_trend" in row:
        votes["supertrend"] = int(row["supertrend_trend"])  # 1 or -1

    # Parabolic SAR
    if "psar_trend" in row:
        votes["psar"] = int(row["psar_trend"])  # 1 or -1

    # HMA trend
    if "price_vs_hma20" in row:
        votes["hma20"] = 1 if row["price_vs_hma20"] > 0 else -1

    # Donchian breakout
    if "don_breakout_up" in row:
        if row["don_breakout_up"]:
            votes["donchian"] = 1
        elif row["don_breakout_down"]:
            votes["donchian"] = -1
        else:
            votes["don_pct"] = 1 if row.get("don_pct", 0.5) > 0.6 else (-1 if row.get("don_pct", 0.5) < 0.4 else 0)

    # Squeeze momentum
    if "sqz_bullish" in row:
        if row["sqz_bullish"]:
            votes["squeeze"] = 1
        elif row["sqz_bearish"]:
            votes["squeeze"] = -1
        else:
            votes["squeeze"] = 0

    # Pivot position
    if "above_r1" in row and "below_s1" in row:
        if row["above_r1"]:
            votes["pivot"] = 1
        elif row["below_s1"]:
            votes["pivot"] = -1
        else:
            votes["pivot"] = 0

    # Fibonacci position (price near 61.8% support?)
    if "fib_dist_618" in row:
        d = row["fib_dist_618"]
        if abs(d) < 0.01:        # within 1% of 61.8 fib
            votes["fibonacci"] = 1 if d >= 0 else -1

    return votes
