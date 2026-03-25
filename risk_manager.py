"""
risk_manager.py — Portfolio-level risk analytics.

Computes:
  - Historical Value-at-Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall)
  - Position correlation matrix (avoid concentrated correlated bets)
  - Sector exposure limits (max 35% in one sector)
  - Portfolio volatility and Sharpe ratio
  - Maximum drawdown on current portfolio
  - Concentration risk (Herfindahl index)
  - Suggested position size adjustments based on portfolio-level risk

These metrics are displayed in the dashboard Portfolio tab and used
by the signal engine to scale down positions when portfolio risk is high.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("risk_manager")


# ── Value at Risk ──────────────────────────────────────────────────────────────

def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Historical simulation VaR.
    Returns the loss (as positive %) that will not be exceeded with
    `confidence` probability over `horizon_days`.

    e.g. VaR(95%, 1d) = 2.5% means there is a 5% chance of losing
    more than 2.5% in a single day based on historical data.
    """
    clean = returns.dropna()
    if len(clean) < 20:
        return np.nan
    var_1d = np.percentile(clean, (1 - confidence) * 100)
    return abs(var_1d) * np.sqrt(horizon_days)


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Conditional VaR (CVaR / Expected Shortfall).
    Average loss in the worst (1-confidence)% of cases.
    More conservative than VaR — tells you how bad it gets when VaR is breached.
    """
    clean = returns.dropna()
    if len(clean) < 20:
        return np.nan
    threshold = np.percentile(clean, (1 - confidence) * 100)
    tail_losses = clean[clean <= threshold]
    return abs(tail_losses.mean()) if len(tail_losses) > 0 else np.nan


def portfolio_var(
    positions: List[dict],
    ohlcv_data: Dict[str, pd.DataFrame],
    confidence: float = config.VAR_CONFIDENCE_LEVEL,
    lookback: int = config.VAR_LOOKBACK_DAYS,
) -> dict:
    """
    Compute portfolio-level VaR using historical simulation.

    Method: Compute daily portfolio returns using actual historical prices
    weighted by current position values, then apply historical VaR.

    Returns dict with var_1d, var_5d, cvar_1d, portfolio_vol, etc.
    """
    if not positions or not ohlcv_data:
        return {}

    total_value = sum(p["current_value"] for p in positions)
    if total_value <= 0:
        return {}

    # Build daily return series for each position
    returns_by_sym = {}
    for pos in positions:
        sym   = pos["symbol"]
        ohlcv = ohlcv_data.get(sym)
        if ohlcv is None or len(ohlcv) < 20:
            continue
        ret = ohlcv["close"].pct_change().dropna()
        returns_by_sym[sym] = ret.tail(lookback)

    if not returns_by_sym:
        return {}

    # Align all return series to common dates
    ret_df = pd.DataFrame(returns_by_sym).dropna()
    if ret_df.empty or len(ret_df) < 20:
        return {}

    # Portfolio weights
    weights = {}
    for pos in positions:
        sym = pos["symbol"]
        if sym in ret_df.columns:
            weights[sym] = pos["current_value"] / total_value

    if not weights:
        return {}

    # Weighted portfolio return
    w_series = pd.Series(weights)
    w_aligned = w_series.reindex(ret_df.columns).fillna(0)
    port_returns = ret_df.dot(w_aligned)

    var_1d  = historical_var(port_returns, confidence)
    var_5d  = historical_var(port_returns, confidence, horizon_days=5)
    cvar_1d = conditional_var(port_returns, confidence)

    # Portfolio volatility (annualised)
    port_vol = port_returns.std() * np.sqrt(252) * 100   # as %

    # Portfolio Sharpe (assuming 0% risk-free rate for simplicity)
    sharpe = (port_returns.mean() * 252) / (port_returns.std() * np.sqrt(252)) \
             if port_returns.std() > 0 else 0.0

    # Current drawdown
    cum_ret = (1 + port_returns).cumprod()
    peak    = cum_ret.cummax()
    dd      = ((cum_ret - peak) / peak)
    max_dd  = float(dd.min()) * 100

    return {
        "var_1d_pct":    round(var_1d * 100, 2),
        "var_5d_pct":    round(var_5d * 100, 2),
        "cvar_1d_pct":   round(cvar_1d * 100, 2) if cvar_1d else None,
        "portfolio_vol": round(port_vol, 2),
        "sharpe_ratio":  round(sharpe, 3),
        "max_drawdown":  round(max_dd, 2),
        "n_days_history":len(port_returns),
    }


# ── Correlation Matrix ─────────────────────────────────────────────────────────

def correlation_matrix(
    symbols: List[str],
    ohlcv_data: Dict[str, pd.DataFrame],
    lookback: int = 120,
) -> pd.DataFrame:
    """
    Rolling 120-day return correlation matrix.

    High correlation (>0.8) between two positions = concentrated risk.
    Used to warn when adding a new position that is highly correlated
    with existing holdings.
    """
    rets = {}
    for sym in symbols:
        ohlcv = ohlcv_data.get(sym)
        if ohlcv is not None and len(ohlcv) >= lookback:
            rets[sym] = ohlcv["close"].pct_change().tail(lookback)

    if len(rets) < 2:
        return pd.DataFrame()

    ret_df = pd.DataFrame(rets).dropna()
    return ret_df.corr().round(3)


# ── Sector Exposure ────────────────────────────────────────────────────────────

def sector_exposure(positions: List[dict]) -> pd.DataFrame:
    """
    Compute portfolio exposure by sector (% of total value).
    Flags sectors exceeding MAX_SECTOR_EXPOSURE_PCT.
    """
    total_value = sum(p["current_value"] for p in positions)
    if total_value <= 0:
        return pd.DataFrame()

    sector_values: Dict[str, float] = {}
    for pos in positions:
        sector = config.SECTOR_MAP.get(pos["symbol"], "Other")
        sector_values[sector] = sector_values.get(sector, 0) + pos["current_value"]

    rows = []
    for sector, value in sorted(sector_values.items(), key=lambda x: -x[1]):
        pct      = value / total_value * 100
        at_limit = pct >= config.MAX_SECTOR_EXPOSURE_PCT * 100
        rows.append({
            "Sector":     sector,
            "Value (M)":  round(value / 1e6, 2),
            "Weight %":   round(pct, 1),
            "At Limit":   at_limit,
        })

    return pd.DataFrame(rows)


# ── Concentration Risk ─────────────────────────────────────────────────────────

def herfindahl_index(positions: List[dict]) -> float:
    """
    Herfindahl-Hirschman Index (HHI) — concentration measure.
    HHI = sum of squared weights.
    0.0 = perfectly diversified (infinite positions, equal weight)
    1.0 = 100% in one position
    < 0.15 = well diversified
    0.15-0.25 = moderate concentration
    > 0.25 = high concentration
    """
    total = sum(p["current_value"] for p in positions)
    if total <= 0:
        return 1.0
    weights = [p["current_value"] / total for p in positions]
    return round(sum(w**2 for w in weights), 4)


# ── Position Size Adjustment ───────────────────────────────────────────────────

def adjust_position_for_risk(
    symbol: str,
    raw_position_pct: float,
    positions: List[dict],
    ohlcv_data: Dict[str, pd.DataFrame],
    signal_direction: int,   # 1=BUY, -1=SELL
) -> Tuple[float, str]:
    """
    Adjust suggested position size down based on portfolio-level risk.

    Reductions applied:
      - Portfolio VaR > 3%/day → reduce by 30%
      - High correlation (>0.7) with existing holding → reduce by 20%
      - Sector already at 35% limit → reduce to 0 (block trade)
      - HHI > 0.25 (concentrated) → reduce by 15%

    Returns (adjusted_pct, reason_string)
    """
    if raw_position_pct <= 0:
        return 0.0, ""

    adjusted = raw_position_pct
    reasons  = []

    # 1. Sector limit check (hard block)
    sector = config.SECTOR_MAP.get(symbol, "Other")
    sector_exp = sector_exposure(positions)
    if not sector_exp.empty:
        row = sector_exp[sector_exp["Sector"] == sector]
        if not row.empty and float(row["Weight %"].iloc[0]) >= config.MAX_SECTOR_EXPOSURE_PCT * 100:
            return 0.0, f"Sector limit: {sector} already at {row['Weight %'].iloc[0]:.0f}% of portfolio"

    # 2. VaR check
    var_data = portfolio_var(positions, ohlcv_data)
    if var_data.get("var_1d_pct", 0) > 3.0:
        adjusted *= 0.70
        reasons.append(f"High portfolio VaR ({var_data['var_1d_pct']:.1f}%/day) → -30% size")

    # 3. Correlation check
    existing_syms = [p["symbol"] for p in positions]
    if existing_syms and symbol in ohlcv_data:
        corr = correlation_matrix(existing_syms + [symbol], ohlcv_data)
        if not corr.empty and symbol in corr.columns:
            max_corr = corr[symbol].drop(symbol, errors="ignore").abs().max()
            if max_corr > 0.80:
                adjusted *= 0.70
                reasons.append(f"High correlation ({max_corr:.2f}) with existing position → -30% size")
            elif max_corr > 0.70:
                adjusted *= 0.85
                reasons.append(f"Moderate correlation ({max_corr:.2f}) → -15% size")

    # 4. Concentration check
    hhi = herfindahl_index(positions)
    if hhi > 0.25:
        adjusted *= 0.85
        reasons.append(f"Concentrated portfolio (HHI={hhi:.2f}) → -15% size")

    adjusted = round(min(adjusted, config.MAX_POSITION_PCT * 100), 1)
    return adjusted, " | ".join(reasons) if reasons else ""
