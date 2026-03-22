"""
trade_signals.py — BUY / SELL / HOLD signal generation with
entry price, stop-loss, take-profit, and position sizing.

Signal generation logic:
  1. AI ensemble confidence (XGB + LGBM + LSTM)
  2. Math model agreement score (Ichimoku, Supertrend, PSAR, HMA, Pivots, etc.)
  3. Combined confidence = 60% AI + 40% math models
  4. Filter by minimum R/R ratio before emitting
  5. Position size using half-Kelly fraction

Stop-loss rules (tightest wins):
  - ATR-based: entry - ATR_SL_MULT * ATR14
  - Support-based: just below nearest pivot S1 or Fibonacci level
  
Take-profit rules (nearest significant level):
  - ATR-based: entry + ATR_TP_MULT * ATR14
  - Resistance-based: nearest pivot R1, R2, or Fibonacci extension
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("trade_signals")

# ── Signal Parameters ──────────────────────────────────────────────────────────
MIN_CONFIDENCE    = 0.65    # Below this → HOLD (no trade)
MIN_RR_RATIO      = 1.5     # Below this R/R → suppress signal
ATR_SL_MULTIPLIER = 1.5     # Stop-loss = entry - ATR_SL_MULT * ATR14
ATR_TP_MULTIPLIER = 3.0     # Take-profit = entry + ATR_TP_MULT * ATR14
MAX_POSITION_PCT  = 0.15    # Max 15% of capital per position
KELLY_FRACTION    = 0.25    # Conservative quarter-Kelly sizing


@dataclass
class TradeSignal:
    symbol:          str
    signal:          str          # BUY / SELL / HOLD / EXIT
    confidence:      float        # 0–1
    ai_confidence:   float        # raw AI model confidence
    math_score:      float        # math model agreement (-1 to +1)
    math_votes:      dict         # individual model votes

    entry_price:     float        # suggested entry
    entry_low:       float        # entry zone lower bound
    entry_high:      float        # entry zone upper bound
    stop_loss:       float        # stop-loss price
    take_profit:     float        # take-profit price
    stop_loss_pct:   float        # stop-loss % from entry
    take_profit_pct: float        # take-profit % from entry
    rr_ratio:        float        # risk/reward ratio

    position_size_pct: float      # suggested % of capital
    position_basis:  str          # human-readable reasoning

    atr14:           float        # ATR(14) at signal time
    last_close:      float        # last close price
    horizon_days:    int          # prediction horizon

    pivot_pp:        float = 0.0
    pivot_s1:        float = 0.0
    pivot_r1:        float = 0.0
    fib_618:         float = 0.0
    fib_382:         float = 0.0

    generated_at:    str = field(default_factory=lambda: datetime.now().isoformat())
    model_auc:       float = 0.0


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> float:
    """Compute current ATR(14)."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=min(n, len(tr)-1), adjust=False).mean()
    return float(atr.iloc[-1]) if not atr.empty else 0.0


def _find_support_resistance(
    close: float,
    math_df: pd.DataFrame,
    direction: int,  # 1=long, -1=short
) -> Tuple[float, float, float, float]:
    """
    Find nearest support (for stop-loss) and resistance (for take-profit)
    from pivot and Fibonacci levels.
    
    Returns (support_price, resistance_price, fib_618, fib_382)
    """
    row = math_df.iloc[-1]

    # Collect all support levels (below current price)
    support_levels = []
    resist_levels  = []

    for col in ["pivot_s1", "pivot_s2", "pivot_s3", "cam_s3", "cam_s4"]:
        v = row.get(col, np.nan)
        if pd.notna(v) and v < close:
            support_levels.append(float(v))

    for col in ["pivot_r1", "pivot_r2", "pivot_r3", "cam_r3", "cam_r4"]:
        v = row.get(col, np.nan)
        if pd.notna(v) and v > close:
            resist_levels.append(float(v))

    # Fibonacci levels
    fib_cols_support = [c for c in math_df.columns if "fib_" in c and "extend" not in c]
    fib_cols_resist  = [c for c in math_df.columns if "fib_extend" in c]

    for col in fib_cols_support:
        v = row.get(col, np.nan)
        if pd.notna(v) and v < close:
            support_levels.append(float(v))

    for col in fib_cols_resist:
        v = row.get(col, np.nan)
        if pd.notna(v) and v > close:
            resist_levels.append(float(v))

    nearest_support  = max(support_levels) if support_levels else close * 0.97
    nearest_resist   = min(resist_levels)  if resist_levels  else close * 1.03

    fib_618 = float(row.get("fib_618", close * 0.97))
    fib_382 = float(row.get("fib_382", close * 0.985))

    return nearest_support, nearest_resist, fib_618, fib_382


def generate_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    ai_confidence: float,          # raw AI model output (0–1)
    math_df: pd.DataFrame,         # output of math_models.build_math_model_features()
    math_votes: Dict[str, int],    # output of math_models.get_math_signal_votes()
    horizon: int = 5,
    model_auc: float = 0.0,
    total_capital: float = 100_000_000,  # VND
) -> TradeSignal:
    """
    Generate a complete trade signal for one symbol.
    
    Combines AI confidence with math model agreement to produce
    a final BUY/SELL/HOLD signal with full trade parameters.
    """
    last_close = float(ohlcv["close"].iloc[-1])
    high  = ohlcv["high"]
    low   = ohlcv["low"]
    close = ohlcv["close"]

    atr14 = _compute_atr(high, low, close, n=14)

    # ── Math model agreement score ─────────────────────────────────────────
    if math_votes:
        vote_values = list(math_votes.values())
        bull_votes  = sum(1 for v in vote_values if v > 0)
        bear_votes  = sum(1 for v in vote_values if v < 0)
        total_votes = len(vote_values)
        math_score  = (bull_votes - bear_votes) / total_votes if total_votes > 0 else 0.0
        math_agreement = (math_score + 1) / 2   # normalise to [0, 1]
    else:
        math_score = 0.0
        math_agreement = 0.5

    # ── Combined confidence ────────────────────────────────────────────────
    combined_conf = 0.60 * ai_confidence + 0.40 * math_agreement

    # ── Direction determination ────────────────────────────────────────────
    # AI says UP if ai_confidence > 0.5, DOWN if < 0.5
    ai_direction   = 1 if ai_confidence >= 0.5 else -1
    math_direction = 1 if math_score >= 0 else -1
    direction      = 1 if (ai_direction + math_direction) >= 0 else -1

    # ── Signal classification ──────────────────────────────────────────────
    pivot_row = math_df.iloc[-1]
    above_ma200 = float(pivot_row.get("price_vs_hma20", 0)) > -0.05  # within 5% below HMA-20

    if combined_conf >= MIN_CONFIDENCE and direction == 1:
        raw_signal = "BUY"
    elif combined_conf >= MIN_CONFIDENCE and direction == -1:
        raw_signal = "SELL"
    else:
        raw_signal = "HOLD"

    # ── Stop-loss calculation ──────────────────────────────────────────────
    nearest_support, nearest_resist, fib_618, fib_382 = _find_support_resistance(
        last_close, math_df, direction
    )

    if raw_signal == "BUY":
        sl_atr     = last_close - ATR_SL_MULTIPLIER * atr14
        sl_support = nearest_support * 0.998    # just below support
        stop_loss  = max(sl_atr, sl_support)    # tightest stop wins
        stop_loss  = min(stop_loss, last_close * 0.96)  # hard cap: max 4% loss

        tp_atr     = last_close + ATR_TP_MULTIPLIER * atr14
        tp_resist  = nearest_resist
        take_profit = min(tp_atr, tp_resist * 1.002) if tp_resist > last_close else tp_atr
        take_profit = max(take_profit, last_close * 1.02)   # minimum 2% target

    elif raw_signal == "SELL":
        sl_atr     = last_close + ATR_SL_MULTIPLIER * atr14
        sl_support = nearest_resist * 1.002
        stop_loss  = min(sl_atr, sl_support)
        stop_loss  = max(stop_loss, last_close * 1.04)

        tp_atr     = last_close - ATR_TP_MULTIPLIER * atr14
        tp_support = nearest_support
        take_profit = max(tp_atr, tp_support * 0.998) if tp_support < last_close else tp_atr
        take_profit = min(take_profit, last_close * 0.98)

    else:  # HOLD — still compute levels for reference
        stop_loss   = last_close - ATR_SL_MULTIPLIER * atr14
        take_profit = last_close + ATR_TP_MULTIPLIER * atr14

    # ── Entry zone ────────────────────────────────────────────────────────
    entry_spread = atr14 * 0.3
    entry_price  = last_close
    if raw_signal == "BUY":
        entry_low  = last_close - entry_spread
        entry_high = last_close + entry_spread * 0.5
    elif raw_signal == "SELL":
        entry_low  = last_close - entry_spread * 0.5
        entry_high = last_close + entry_spread
    else:
        entry_low  = last_close - entry_spread
        entry_high = last_close + entry_spread

    # ── Risk / Reward ──────────────────────────────────────────────────────
    risk   = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    rr_ratio = reward / risk if risk > 0 else 0.0

    # Suppress signal if R/R too low
    signal = raw_signal
    if raw_signal in ("BUY", "SELL") and rr_ratio < MIN_RR_RATIO:
        signal = "HOLD"
        logger.info(f"  {symbol}: signal suppressed (R/R={rr_ratio:.2f} < {MIN_RR_RATIO})")

    # ── Stop-loss and take-profit percentages ─────────────────────────────
    sl_pct = (stop_loss - entry_price) / entry_price * 100
    tp_pct = (take_profit - entry_price) / entry_price * 100

    # ── Position sizing (half-Kelly) ──────────────────────────────────────
    # Kelly fraction = (win_prob * reward - loss_prob * risk) / reward
    # Using half-Kelly for safety
    win_prob  = combined_conf if direction == 1 else (1 - combined_conf)
    loss_prob = 1 - win_prob
    if reward > 0 and risk > 0:
        kelly = (win_prob * reward - loss_prob * risk) / reward
        kelly = max(0.0, min(kelly, 1.0))
        position_pct = min(kelly * KELLY_FRACTION, MAX_POSITION_PCT)
    else:
        position_pct = 0.0

    if signal == "HOLD":
        position_pct = 0.0

    # ── Basis string ──────────────────────────────────────────────────────
    confirming = [k for k, v in math_votes.items() if v == direction]
    basis_parts = []
    if ai_confidence > 0.60:
        basis_parts.append(f"AI conf={ai_confidence:.1%}")
    if confirming:
        basis_parts.append(f"Models: {', '.join(confirming[:4])}")
    basis_parts.append(f"ATR14={atr14:.0f}")
    if atr14 > 0:
        basis_parts.append(f"SL={ATR_SL_MULTIPLIER}xATR TP={ATR_TP_MULTIPLIER}xATR")
    basis = " | ".join(basis_parts) if basis_parts else "Insufficient signal"

    # ── Pivot levels for display ──────────────────────────────────────────
    pp = float(pivot_row.get("pivot_pp", 0))
    s1 = float(pivot_row.get("pivot_s1", 0))
    r1 = float(pivot_row.get("pivot_r1", 0))

    return TradeSignal(
        symbol=symbol,
        signal=signal,
        confidence=round(combined_conf, 4),
        ai_confidence=round(ai_confidence, 4),
        math_score=round(math_score, 4),
        math_votes=math_votes,
        entry_price=round(entry_price, 0),
        entry_low=round(entry_low, 0),
        entry_high=round(entry_high, 0),
        stop_loss=round(stop_loss, 0),
        take_profit=round(take_profit, 0),
        stop_loss_pct=round(sl_pct, 2),
        take_profit_pct=round(tp_pct, 2),
        rr_ratio=round(rr_ratio, 2),
        position_size_pct=round(position_pct * 100, 1),
        position_basis=basis,
        atr14=round(atr14, 0),
        last_close=round(last_close, 0),
        horizon_days=horizon,
        pivot_pp=round(pp, 0),
        pivot_s1=round(s1, 0),
        pivot_r1=round(r1, 0),
        fib_618=round(fib_618, 0),
        fib_382=round(fib_382, 0),
        model_auc=round(model_auc, 4),
    )


def format_signal(sig: TradeSignal) -> str:
    """Format a trade signal for console/log output."""
    arrow = "[UP]" if sig.signal == "BUY" else ("[DOWN]" if sig.signal == "SELL" else "[--]")
    lines = [
        f"{'='*52}",
        f"  {sig.symbol:6s}  {sig.signal:4s} {arrow}  Conf={sig.confidence:.1%}  AUC={sig.model_auc:.3f}",
        f"{'='*52}",
        f"  Entry zone : {sig.entry_low:>10,.0f} - {sig.entry_high:>10,.0f}",
        f"  Stop-loss  : {sig.stop_loss:>10,.0f}  ({sig.stop_loss_pct:+.1f}%)",
        f"  Take-profit: {sig.take_profit:>10,.0f}  ({sig.take_profit_pct:+.1f}%)",
        f"  R/R ratio  : 1 : {sig.rr_ratio:.2f}",
        f"  Position   : {sig.position_size_pct:.1f}% of capital",
        f"  Pivot PP   : {sig.pivot_pp:>10,.0f}  S1={sig.pivot_s1:,.0f}  R1={sig.pivot_r1:,.0f}",
        f"  Fib 61.8%  : {sig.fib_618:>10,.0f}  38.2%={sig.fib_382:,.0f}",
        f"  Basis      : {sig.position_basis}",
        f"  Math votes : {sig.math_votes}",
    ]
    return "\n".join(lines)


def save_signals(signals: List[TradeSignal], date_str: Optional[str] = None) -> Path:
    """Save signals to JSON file in results/."""
    date_str = date_str or datetime.now().strftime("%Y%m%d")
    out = Path(config.RESULTS_DIR) / f"signals_{date_str}.json"
    data = [asdict(s) for s in signals]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Signals saved to {out}")
    return out


def load_signals(date_str: Optional[str] = None) -> List[dict]:
    """Load most recent signal file."""
    results_dir = Path(config.RESULTS_DIR)
    if date_str:
        p = results_dir / f"signals_{date_str}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return []
    # Find most recent
    files = sorted(results_dir.glob("signals_*.json"), reverse=True)
    if not files:
        return []
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)
