"""
Module 03 — Signal Detection
=============================
Implements all 9 strategy detectors and the Tier classification system
from VN100_Full_Master_Plan.md Part A & C.2.

Input : data/features/{TICKER}.csv
Output: data/signals/{TICKER}_signals.csv   (one row per signal event)
        data/signals/{TICKER}_latest.json   (latest signal card data)

Run:
    python modules/03_signals.py --ticker HPG
    python modules/03_signals.py --all
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
FEAT    = ROOT / "data" / "features"
SIG_DIR = ROOT / "data" / "signals"
SIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("signals")

# ── Strategy metadata ─────────────────────────────────────────────────────────
STRATEGIES = {
    "PRICE_DOWN_15_MA20": {
        "name":     "Giá −15% vs MA20",
        "hold_days": 180,
        "win_rate":  1.000,
        "type":      "position_buy",
        "regime":    "mean_revert",
    },
    "RSI_OVERSOLD": {
        "name":     "RSI Quá Bán",
        "hold_days": 60,
        "win_rate":  0.743,
        "type":      "swing_buy",
        "regime":    "mean_revert",
    },
    "PRICE_DOWN_15_20D": {
        "name":     "Giá −15% trong 20 phiên",
        "hold_days": 5,
        "win_rate":  0.792,
        "type":      "bounce_trade",
        "regime":    "mean_revert",
    },
    "DMI_WAVE": {
        "name":     "Lướt Sóng DMI",
        "hold_days": 10,
        "win_rate":  0.700,
        "type":      "momentum_trade",
        "regime":    "trending",
    },
    "SAR_MACD": {
        "name":     "SAR × MACD Histogram",
        "hold_days": 20,
        "win_rate":  0.706,
        "type":      "trend_trade",
        "regime":    "trending",
    },
    "BOLLINGER": {
        "name":     "Mở Band Bollinger",
        "hold_days": 5,
        "win_rate":  0.582,
        "type":      "breakout_trade",
        "regime":    "volatile",
    },
    "VOLUME_EXPLOSION": {
        "name":     "Bùng Nổ Khối Lượng",
        "hold_days": 180,
        "win_rate":  0.614,
        "type":      "position_buy",
        "regime":    "any",
    },
    "UPTREND": {
        "name":     "Uptrend",
        "hold_days": 180,
        "win_rate":  0.593,
        "type":      "position_hold",
        "regime":    "trending",
    },
    "STOCH_RSI": {
        "name":     "Giá Tăng + Stochastic RSI",
        "hold_days": 180,
        "win_rate":  0.532,
        "type":      "position_buy",
        "regime":    "trending",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Individual strategy detectors
# ─────────────────────────────────────────────────────────────────────────────

def _check_macro_filter(row: pd.Series) -> bool:
    """Step 1: macro gates from Decision Tree Chapter 12."""
    return (
        row.get("vnindex_ma_score", 0.5) > -0.33 and
        row.get("breadth_pct_above_50sma", 0.5) > 0.35 and
        row.get("interbank_rate_change", 0) < 1.0
    )


def _check_risk_gates(row: pd.Series) -> bool:
    """Step 7: risk gates — any True here BLOCKS the signal."""
    if row.get("bull_trap_flag", 0)              == 1: return False
    if row.get("wyckoff_distribution_flag", 0)   == 1: return False
    if row.get("exhaustion_gap_flag", 0)         == 1: return False
    if row.get("three_pushes_high_flag", 0)      == 1: return False
    if row.get("upper_wick_resistance", 0)       == 1 and row.get("inside_resistance_zone", 0) == 1: return False
    if row.get("t25_risk", 0)                    >= 0.12: return False
    if row.get("choch_bearish", 0)               == 1: return False
    return True


def detect_signals_for_row(row: pd.Series) -> dict:
    """
    Returns a dict of strategy_key → 1/0/−1 for every row.
    Uses regime filter: mean-revert strategies only fire when Hurst < 0.5,
    trend strategies only when Hurst > 0.55.
    """
    hurst   = row.get("hurst_60d", 0.5)
    is_mean_rev = hurst < 0.5
    is_trend    = hurst > 0.55

    results = {}

    # 1. PRICE_DOWN_15_MA20 — 100% win rate at T+180 (position trade)
    results["PRICE_DOWN_15_MA20"] = int(
        is_mean_rev and
        row.get("deep_oversold_ma20_flag", 0) == 1 and
        row.get("net_foreign_flow_5d", 0)     >= 0 and
        row.get("interbank_rate_change", 0)   < 0.5
    )

    # 2. RSI_OVERSOLD — 74.3% at T+60
    results["RSI_OVERSOLD"] = int(
        is_mean_rev and
        row.get("rsi_14", 50)           < 30 and
        row.get("rsi_5d_slope", 0)      > 0 and       # RSI turning up
        row.get("net_foreign_flow_5d", 0) > -0.05 and
        row.get("vnindex_ma_score", 0.5)  > -0.67
    )

    # 3. PRICE_DOWN_15_20D — 79.2% at T+5 (short bounce)
    results["PRICE_DOWN_15_20D"] = int(
        row.get("decline_20d_flag", 0)  == 1 and
        row.get("hurst_60d", 0.5)       < 0.5 and
        row.get("zscore_20d", 0)        < -1.5 and
        row.get("rel_volume", 1)        > 1.5
    )

    # 4. DMI_WAVE — 70% at T+10
    results["DMI_WAVE"] = int(
        is_trend and
        row.get("dmi_wave_signal", 0) == 1
    )

    # 5. SAR_MACD — 70.6% at T+20
    results["SAR_MACD"] = int(
        row.get("sar_macd_combo_signal", 0) == 1
    )

    # 6. BOLLINGER — 58.2% at T+5
    results["BOLLINGER"] = int(
        row.get("bb_expansion_flag", 0) == 1 and
        row.get("bb_breakout_direction", 0) == 1 and
        row.get("rel_volume", 1)        > 1.5
    )

    # 7. VOLUME_EXPLOSION — 61.4% at T+180
    results["VOLUME_EXPLOSION"] = int(
        row.get("vol_explosion_flag", 0)     == 1 and
        row.get("vol_explosion_direction", 0) == 1
    )

    # 8. UPTREND — 59.3% at T+180
    results["UPTREND"] = int(
        is_trend and
        row.get("uptrend_quality_score", 0) == 1.0 and
        row.get("hurst_60d", 0.5)          > 0.55
    )

    # 9. STOCH_RSI — 53.2% at T+180
    results["STOCH_RSI"] = int(
        is_trend and
        row.get("return_5d", 0)     > 0.03 and
        row.get("stoch_rsi_k", 50)  < 50 and
        row.get("stoch_rsi_k", 50)  > row.get("stoch_rsi_d", 50) and
        row.get("adx_14", 0)        > 20
    )

    return results


def classify_tier(n_signals: int) -> int:
    if n_signals >= 3: return 1
    if n_signals == 2: return 2
    if n_signals == 1: return 3
    return 0


def pick_primary_strategy(signals_dict: dict) -> str | None:
    """Pick highest win-rate strategy among active ones."""
    active = [k for k, v in signals_dict.items() if v == 1]
    if not active:
        return None
    return max(active, key=lambda k: STRATEGIES[k]["win_rate"])


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def _add_trading_days(start_date, n_days: int) -> str:
    d = pd.bdate_range(start=start_date, periods=n_days + 1)
    return str(d[-1].date())


def process_ticker(ticker: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      signal_events: DataFrame of all signal trigger rows
      latest_signal: dict for the most recent signal (for JSON card)
    """
    signal_rows = []

    for idx, row in df.iterrows():
        if not _check_macro_filter(row):
            continue
        if not _check_risk_gates(row):
            continue

        strats = detect_signals_for_row(row)
        n_sig  = sum(strats.values())
        tier   = classify_tier(n_sig)

        if tier == 0:
            continue

        primary = pick_primary_strategy(strats)
        strat_info = STRATEGIES[primary]

        close = float(row["close"])
        atr   = float(row.get("atr_14", close * 0.018))
        entry_lo = round(close * 0.982 / 100) * 100
        entry_hi = round(close * 1.008 / 100) * 100

        # Target: use Fibonacci extension or ATR multiple
        if primary in ("PRICE_DOWN_15_MA20", "RSI_OVERSOLD"):
            target = round((float(row.get("fib_ext_1272", close * 1.10))) / 100) * 100
        else:
            target = round(close * (1 + strat_info["win_rate"] * 0.12) / 100) * 100

        # Stop-loss: Kalman/ATR or Fibonacci support
        stop_loss = round((close - 1.5 * atr) / 100) * 100
        if stop_loss >= close:
            stop_loss = round(close * 0.942 / 100) * 100

        reward_pct = (target - close) / close * 100
        risk_pct   = (close - stop_loss) / close * 100
        rr_ratio   = reward_pct / max(risk_pct, 0.1)

        date_str = str(row["date"])[:10]
        exit_date = _add_trading_days(date_str, strat_info["hold_days"])

        # Direction probability from combined signal strength
        base_prob = strat_info["win_rate"]
        bonus     = min(0.15, (n_sig - 1) * 0.05)
        dir_prob  = round((base_prob + bonus) * 100, 1)

        t25_risk = float(row.get("t25_risk", 0.05))
        t25_label = "THẤP" if t25_risk < 0.04 else ("TRUNG BÌNH" if t25_risk < 0.08 else "CAO")

        record = {
            "ticker":       ticker,
            "date":         date_str,
            "signal":       "BUY",
            "tier":         tier,
            "primary_strategy": primary,
            "strategy_name": strat_info["name"],
            "strategies_active": [k for k, v in strats.items() if v == 1],
            "n_signals":    n_sig,
            "hold_days":    strat_info["hold_days"],
            "exit_date":    exit_date,
            "current_price": close,
            "entry_lo":     entry_lo,
            "entry_hi":     entry_hi,
            "target":       target,
            "stop_loss":    stop_loss,
            "reward_pct":   round(reward_pct, 2),
            "risk_pct":     round(risk_pct, 2),
            "rr_ratio":     round(rr_ratio, 2),
            "dir_prob":     dir_prob,
            "win_rate_hist": round(strat_info["win_rate"] * 100, 1),
            "rsi_14":       round(float(row.get("rsi_14", 50)), 1),
            "adx_14":       round(float(row.get("adx_14", 20)), 1),
            "macd_histogram": round(float(row.get("macd_histogram", 0)), 4),
            "rel_volume":   round(float(row.get("rel_volume", 1)), 2),
            "net_foreign_flow_5d": round(float(row.get("net_foreign_flow_5d", 0))*100, 2),
            "hurst_60d":    round(float(row.get("hurst_60d", 0.5)), 3),
            "t25_risk_val": round(t25_risk * 100, 2),
            "t25_risk_label": t25_label,
            "garch_sigma":  round(float(row.get("garch_sigma", 0.2)) * 100, 2),
            "vwap_deviation": round(float(row.get("vwap_deviation", 0)) * 100, 2),
        }

        # Add individual strategy flags to record
        for k, v in strats.items():
            record[f"strat_{k}"] = v

        signal_rows.append(record)

    if not signal_rows:
        return pd.DataFrame(), {}

    sig_df  = pd.DataFrame(signal_rows)
    latest  = sig_df.iloc[-1].to_dict()

    # Enrich latest with extra display fields
    latest["strategies_active_str"] = " + ".join(latest.get("strategies_active", []))
    tier_labels = {1: "TIER 1 — ĐỘ TIN CẬY CAO NHẤT",
                   2: "TIER 2 — ĐỘ TIN CẬY CAO",
                   3: "TIER 3 — CHỜ XÁC NHẬN"}
    latest["tier_label"] = tier_labels.get(latest["tier"], "TIER 3")

    return sig_df, latest


def run(ticker: str) -> None:
    feat_path = FEAT / f"{ticker}.csv"
    if not feat_path.exists():
        log.error(f"Feature file not found: {feat_path}  →  run 02_features.py first")
        return

    df = pd.read_csv(feat_path, low_memory=False)
    log.info(f"{ticker}: detecting signals ({len(df)} rows) …")

    sig_df, latest = process_ticker(ticker, df)

    if sig_df.empty:
        log.info(f"{ticker}: no signals detected")
        json_out = SIG_DIR / f"{ticker}_latest.json"
        json_out.write_text(json.dumps({"ticker": ticker, "signal": "NONE", "tier": 0}, indent=2))
        return

    sig_csv = SIG_DIR / f"{ticker}_signals.csv"
    sig_df.to_csv(sig_csv, index=False)
    log.info(f"{ticker}: {len(sig_df)} signal events → {sig_csv.name}")

    json_out = SIG_DIR / f"{ticker}_latest.json"
    json_out.write_text(json.dumps(latest, indent=2, default=str))
    log.info(f"{ticker}: latest signal → {json_out.name} (Tier {latest.get('tier')})")


def run_all(tickers: list[str]) -> None:
    summary = []
    for t in tickers:
        run(t)
        jf = SIG_DIR / f"{t}_latest.json"
        if jf.exists():
            with open(jf) as f:
                d = json.load(f)
            if d.get("tier", 0) > 0:
                summary.append({
                    "ticker": t,
                    "tier":   d.get("tier"),
                    "strat":  d.get("primary_strategy"),
                    "date":   d.get("date"),
                    "prob":   d.get("dir_prob"),
                })

    summary.sort(key=lambda x: (x["tier"], -float(x.get("prob", 0))))
    summary_path = SIG_DIR / "daily_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"\n=== {len(summary)} active signals found ===")
    for s in summary[:10]:
        log.info(f"  [{s['tier']}] {s['ticker']:6s}  {s['strat']}  {s['prob']}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VN100 Signal Detection")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    tickers = [f.stem for f in FEAT.glob("*.csv")] if args.all else (
        [args.ticker] if args.ticker else [f.stem for f in FEAT.glob("*.csv")]
    )

    if len(tickers) > 1:
        run_all(tickers)
    else:
        for t in tickers:
            run(t)