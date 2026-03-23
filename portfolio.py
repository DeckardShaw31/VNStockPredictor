"""
portfolio.py — Portfolio storage and analysis.

Stores your holdings in data/portfolio.json:
  {
    "VNM": [
      {"shares": 1000, "buy_price": 65400, "buy_date": "2026-03-20", "note": ""},
      {"shares": 500,  "buy_price": 63200, "buy_date": "2026-02-14", "note": "averaging down"}
    ],
    ...
  }

Multiple lots per symbol are supported (averaging down, DCA, etc.).
Each lot tracks: shares, buy price, buy date, optional note.

Portfolio metrics computed:
  - Current value (live price)
  - Unrealised P&L (VND + %)
  - Cost basis (weighted average across lots)
  - Day change (today vs yesterday)
  - Portfolio weight (% of total)
  - AI signal alignment (does the AI agree with your position?)
  - Risk level (how far is price from stop-loss?)
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio")

PORTFOLIO_FILE = Path("data/portfolio.json")
PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Data model ─────────────────────────────────────────────────────────────────

def _empty_portfolio() -> dict:
    return {}


def load_portfolio() -> dict:
    """Load portfolio from JSON file."""
    if not PORTFOLIO_FILE.exists():
        return _empty_portfolio()
    try:
        with open(PORTFOLIO_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load portfolio: {e}")
        return _empty_portfolio()


def save_portfolio(portfolio: dict):
    """Save portfolio to JSON file."""
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False, default=str)


def add_position(
    symbol: str,
    shares: int,
    buy_price: float,
    buy_date: Optional[str] = None,
    note: str = "",
) -> dict:
    """Add a new lot to the portfolio. Returns updated portfolio."""
    portfolio = load_portfolio()
    if symbol not in portfolio:
        portfolio[symbol] = []

    portfolio[symbol].append({
        "shares":    int(shares),
        "buy_price": float(buy_price),
        "buy_date":  buy_date or datetime.now().strftime("%Y-%m-%d"),
        "note":      note,
        "added_at":  datetime.now().isoformat(),
    })

    save_portfolio(portfolio)
    logger.info(f"Added {shares} shares of {symbol} @ {buy_price:,.0f} on {buy_date}")
    return portfolio


def remove_lot(symbol: str, lot_index: int) -> dict:
    """Remove a specific lot by index. Returns updated portfolio."""
    portfolio = load_portfolio()
    if symbol in portfolio and 0 <= lot_index < len(portfolio[symbol]):
        removed = portfolio[symbol].pop(lot_index)
        if not portfolio[symbol]:
            del portfolio[symbol]
        save_portfolio(portfolio)
        logger.info(f"Removed lot {lot_index} of {symbol}: {removed}")
    return portfolio


def update_lot(symbol: str, lot_index: int, **kwargs) -> dict:
    """Update fields of a specific lot. Returns updated portfolio."""
    portfolio = load_portfolio()
    if symbol in portfolio and 0 <= lot_index < len(portfolio[symbol]):
        portfolio[symbol][lot_index].update(kwargs)
        save_portfolio(portfolio)
    return portfolio


# ── Analytics ──────────────────────────────────────────────────────────────────

def _normalize_price(current_price: float, buy_price: float) -> float:
    """
    vnstock returns some stocks (esp. low-price ones like SHS) in units of
    1,000 VND (e.g. 15.0 = 15,000 VND), while users naturally enter the
    full price (21,829 VND).

    If buy_price is more than 500x current_price, current_price is almost
    certainly in thousands — scale it up to match.
    """
    if buy_price > 0 and current_price > 0:
        ratio = buy_price / current_price
        if ratio > 500:       # e.g. 21829 / 15 = 1455  → scale up
            return current_price * 1000
        if ratio < 0.002:     # inverse: user entered in thousands, data in full VND
            return current_price / 1000
    return current_price


def compute_position_metrics(
    symbol: str,
    lots: List[dict],
    current_price: float,
    prev_close: float,
    signal: Optional[dict] = None,
) -> dict:
    """
    Compute full metrics for one symbol's position.
    Automatically normalises price units (vnstock returns some stocks in
    thousands of VND; we detect and correct the scale mismatch).
    """
    avg_buy = sum(l["shares"] * l["buy_price"] for l in lots) / sum(l["shares"] for l in lots) if lots else 0

    # Detect and fix unit mismatch (e.g. SHS: vnstock=15, user entered=21829)
    current_price = _normalize_price(current_price, avg_buy)
    prev_close    = _normalize_price(prev_close,    avg_buy)
    total_shares   = sum(l["shares"] for l in lots)
    total_cost     = sum(l["shares"] * l["buy_price"] for l in lots)
    avg_cost       = total_cost / total_shares if total_shares > 0 else 0
    current_value  = total_shares * current_price
    unrealised_pnl = current_value - total_cost
    unrealised_pct = (unrealised_pnl / total_cost * 100) if total_cost > 0 else 0

    day_change_vnd = total_shares * (current_price - prev_close)
    day_change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0

    # Earliest buy date
    try:
        earliest = min(l["buy_date"] for l in lots)
        days_held = (datetime.now().date() - datetime.strptime(earliest, "%Y-%m-%d").date()).days
    except Exception:
        days_held = 0

    # Signal alignment
    signal_dir = None
    signal_conf = 0.0
    ai_agrees   = None
    stop_loss   = None
    take_profit = None
    rr_ratio    = None

    if signal:
        signal_dir  = signal.get("signal", "HOLD")
        signal_conf = signal.get("confidence", 0.5)
        # Normalize SL/TP to the same scale as current_price
        raw_sl = signal.get("stop_loss")
        raw_tp = signal.get("take_profit")
        if raw_sl:
            stop_loss   = _normalize_price(float(raw_sl), avg_buy)
        if raw_tp:
            take_profit = _normalize_price(float(raw_tp), avg_buy)
        rr_ratio    = signal.get("rr_ratio")

        # Does AI agree with your long position?
        if signal_dir == "BUY":
            ai_agrees = True
        elif signal_dir == "SELL":
            ai_agrees = False
        else:
            ai_agrees = None   # HOLD — neutral

    # Distance to stop-loss (negative = price already below SL)
    if stop_loss and stop_loss > 0:
        dist_to_sl_pct = (current_price - stop_loss) / current_price * 100
        # positive = price is above SL (safe)
        # negative = price has blown through SL (danger)
    else:
        dist_to_sl_pct = None

    # Annualised return
    if days_held > 0 and avg_cost > 0:
        ann_return_pct = (unrealised_pct / 100 + 1) ** (365 / days_held) - 1
        ann_return_pct *= 100
    else:
        ann_return_pct = 0.0

    return {
        "symbol":          symbol,
        "lots":            len(lots),
        "total_shares":    total_shares,
        "avg_cost":        round(avg_cost, 0),
        "current_price":   round(current_price, 0),
        "current_value":   round(current_value, 0),
        "total_cost":      round(total_cost, 0),
        "unrealised_pnl":  round(unrealised_pnl, 0),
        "unrealised_pct":  round(unrealised_pct, 2),
        "day_change_vnd":  round(day_change_vnd, 0),
        "day_change_pct":  round(day_change_pct, 2),
        "days_held":       days_held,
        "ann_return_pct":  round(ann_return_pct, 1),
        "signal":          signal_dir,
        "signal_conf":     round(signal_conf, 4),
        "ai_agrees":       ai_agrees,
        "stop_loss":       stop_loss,
        "take_profit":     take_profit,
        "rr_ratio":        rr_ratio,
        "dist_to_sl_pct":  round(dist_to_sl_pct, 2) if dist_to_sl_pct is not None else None,
    }


def compute_portfolio_summary(
    positions: List[dict],
) -> dict:
    """Aggregate portfolio-level metrics from individual position metrics."""
    if not positions:
        return {}

    total_value      = sum(p["current_value"]  for p in positions)
    total_cost       = sum(p["total_cost"]      for p in positions)
    total_pnl        = sum(p["unrealised_pnl"]  for p in positions)
    total_day_change = sum(p["day_change_vnd"]  for p in positions)

    # Add portfolio weight to each position
    for p in positions:
        p["portfolio_weight"] = round(p["current_value"] / total_value * 100, 1) if total_value > 0 else 0

    pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    day_pct  = (total_day_change / (total_value - total_day_change) * 100) \
               if (total_value - total_day_change) > 0 else 0

    winners = [p for p in positions if p["unrealised_pnl"] > 0]
    losers  = [p for p in positions if p["unrealised_pnl"] < 0]

    best  = max(positions, key=lambda p: p["unrealised_pct"]) if positions else None
    worst = min(positions, key=lambda p: p["unrealised_pct"]) if positions else None

    ai_agree_count    = sum(1 for p in positions if p["ai_agrees"] is True)
    ai_disagree_count = sum(1 for p in positions if p["ai_agrees"] is False)

    # Risk: positions within 3% of stop-loss
    at_risk = [p for p in positions if p.get("dist_to_sl_pct") is not None
               and p["dist_to_sl_pct"] < 3.0]

    return {
        "total_value":        round(total_value, 0),
        "total_cost":         round(total_cost, 0),
        "total_pnl":          round(total_pnl, 0),
        "total_pnl_pct":      round(pnl_pct, 2),
        "day_change_vnd":     round(total_day_change, 0),
        "day_change_pct":     round(day_pct, 2),
        "n_positions":        len(positions),
        "n_winners":          len(winners),
        "n_losers":           len(losers),
        "win_rate":           round(len(winners) / len(positions) * 100, 1) if positions else 0,
        "best_performer":     best,
        "worst_performer":    worst,
        "ai_agree_count":     ai_agree_count,
        "ai_disagree_count":  ai_disagree_count,
        "at_risk_positions":  at_risk,
    }


def build_portfolio_report(
    ohlcv_data: dict,
    signals: List[dict],
) -> Tuple[List[dict], dict]:
    """
    Build complete portfolio analysis.

    Args:
        ohlcv_data: {symbol: DataFrame} from fetch_multiple
        signals:    list of signal dicts from load_signals()

    Returns:
        (positions, summary) — list of per-position metrics + portfolio totals
    """
    portfolio = load_portfolio()
    if not portfolio:
        return [], {}

    signals_map = {s["symbol"]: s for s in signals} if signals else {}
    positions = []

    for sym, lots in portfolio.items():
        if not lots:
            continue

        ohlcv = ohlcv_data.get(sym)
        if ohlcv is None or ohlcv.empty:
            logger.warning(f"No price data for {sym} — skipping portfolio position")
            continue

        current_price = float(ohlcv["close"].iloc[-1])
        prev_close    = float(ohlcv["close"].iloc[-2]) if len(ohlcv) > 1 else current_price

        metrics = compute_position_metrics(
            sym, lots, current_price, prev_close,
            signal=signals_map.get(sym),
        )
        positions.append(metrics)

    summary = compute_portfolio_summary(positions)
    return positions, summary
