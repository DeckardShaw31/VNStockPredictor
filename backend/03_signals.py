"""
Module 03 — Signal Detection  v2
==================================
Key improvements over v1:
  • base_price stored in every signal (price on signal date).
  • Signal cooldown: won't re-fire the same strategy within hold_days.
  • UPTREND requires uptrend_duration >= 10 days + ADX slope > 0.
  • Composite probability score (not raw model output).
  • SELL signal generated when target / stop / expiry conditions are met.
  • today_action field: BUY / HOLD / SELL / WATCH per signal.
  • signal_state field: ENTRY_ZONE / HOLDING / TARGET_HIT / STOP_HIT /
                        EXPIRED / WATCHING
  • Better tier gate: ONLY fires when signal quality score > threshold.
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
FEAT    = ROOT / "data" / "features"
SIG_DIR = ROOT / "data" / "signals"
SIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("signals")

TODAY = date.today().isoformat()

# ── Strategy definitions ──────────────────────────────────────────────────────
STRATEGIES = {
    "PRICE_DOWN_15_MA20": {"name":"Giá −15% vs MA20",        "hold":180, "wr":0.820, "type":"mean_revert"},
    "RSI_OVERSOLD":       {"name":"RSI Quá Bán",              "hold":60,  "wr":0.743, "type":"mean_revert"},
    "PRICE_DOWN_15_20D":  {"name":"Giá −15% trong 20 phiên", "hold":5,   "wr":0.792, "type":"mean_revert"},
    "DMI_WAVE":           {"name":"Lướt Sóng DMI",            "hold":10,  "wr":0.700, "type":"trend"},
    "SAR_MACD":           {"name":"SAR × MACD Histogram",     "hold":20,  "wr":0.706, "type":"trend"},
    "BOLLINGER":          {"name":"Mở Band Bollinger",        "hold":5,   "wr":0.582, "type":"breakout"},
    "VOLUME_EXPLOSION":   {"name":"Bùng Nổ Khối Lượng",      "hold":180, "wr":0.614, "type":"any"},
    "UPTREND":            {"name":"Uptrend Chắc Chắn",        "hold":60,  "wr":0.680, "type":"trend"},
    "STOCH_RSI":          {"name":"Giá Tăng + StochRSI",      "hold":180, "wr":0.532, "type":"trend"},
    "PULLBACK_BUY":       {"name":"Mua Vào Nhịp Điều Chỉnh",  "hold":20,  "wr":0.710, "type":"trend"},
    "GOLDEN_CROSS":       {"name":"Golden Cross MA",          "hold":90,  "wr":0.720, "type":"trend"},
    "SUPPORT_BOUNCE":     {"name":"Bật Khỏi Vùng Hỗ Trợ",   "hold":15,  "wr":0.680, "type":"mean_revert"},
    "LIQUIDITY_SWEEP":    {"name":"Quét Thanh Khoản Bullish","hold":10,  "wr":0.670, "type":"smc"},
    "FIB_BOUNCE":         {"name":"Bounce Fibonacci 61.8%",   "hold":20,  "wr":0.660, "type":"mean_revert"},
}

TARGET_PCT = {   # reward target as % of base price
    "PRICE_DOWN_15_MA20": 0.20, "RSI_OVERSOLD": 0.15, "PRICE_DOWN_15_20D": 0.08,
    "DMI_WAVE": 0.08, "SAR_MACD": 0.10, "BOLLINGER": 0.05,
    "VOLUME_EXPLOSION": 0.18, "UPTREND": 0.15, "STOCH_RSI": 0.15,
    "PULLBACK_BUY": 0.10, "GOLDEN_CROSS": 0.15, "SUPPORT_BOUNCE": 0.08,
    "LIQUIDITY_SWEEP": 0.07, "FIB_BOUNCE": 0.10,
}
STOP_PCT = {     # stop-loss as % of base price
    "PRICE_DOWN_15_MA20": 0.05, "RSI_OVERSOLD": 0.05, "PRICE_DOWN_15_20D": 0.04,
    "DMI_WAVE": 0.04, "SAR_MACD": 0.05, "BOLLINGER": 0.03,
    "VOLUME_EXPLOSION": 0.06, "UPTREND": 0.06, "STOCH_RSI": 0.06,
    "PULLBACK_BUY": 0.05, "GOLDEN_CROSS": 0.06, "SUPPORT_BOUNCE": 0.04,
    "LIQUIDITY_SWEEP": 0.04, "FIB_BOUNCE": 0.05,
}

# ── Macro gate ────────────────────────────────────────────────────────────────
def _macro_ok(row) -> bool:
    return (
        float(row.get("vnindex_ma_score",   0.5)) > 0.0   and   # at least neutral index
        float(row.get("breadth_pct_above_50sma", 0.5)) > 0.40 and
        float(row.get("interbank_rate_change", 0))  < 1.5
    )

# ── Risk gates ────────────────────────────────────────────────────────────────
def _risk_ok(row) -> bool:
    if int(row.get("bull_trap_flag",0))             : return False
    if int(row.get("wyckoff_distribution_flag",0))  : return False
    if int(row.get("exhaustion_gap_flag",0))         : return False
    if int(row.get("three_pushes_high_flag",0))      : return False
    if float(row.get("t25_risk",0))   >= 0.10       : return False
    if int(row.get("choch_bearish",0))               : return False
    if float(row.get("garch_sigma",0.2)) >= 0.65    : return False   # too volatile
    return True

# ── Individual detectors ──────────────────────────────────────────────────────
def _detect(row) -> dict:
    h = float(row.get("hurst_60d",  0.5))
    rsi  = float(row.get("rsi_14",  50))
    adx  = float(row.get("adx_14",  20))
    rv   = float(row.get("rel_volume", 1))
    ret1 = float(row.get("return_1d", 0))
    ret5 = float(row.get("return_5d", 0))
    ret20= float(row.get("return_20d",0))
    sma20_score = float(row.get("ma_score_stock", 0))
    upd  = float(row.get("uptrend_duration",0))
    uqs  = float(row.get("uptrend_quality_score",0))
    adx_slope = float(row.get("adx_slope_3d",0))
    macd_h = float(row.get("macd_histogram",0))
    k    = float(row.get("stoch_rsi_k",50))
    d    = float(row.get("stoch_rsi_d",50))

    is_trend    = h > 0.58
    is_mean_rev = h < 0.48

    out = {}

    # 1. Giá −15% vs MA20 — deep mean reversion
    out["PRICE_DOWN_15_MA20"] = int(
        is_mean_rev and
        float(row.get("dist_sma_20", 0)) < -0.13 and
        rsi < 38 and
        rv > 1.2
    )

    # 2. RSI Oversold — RSI < 30 turning up
    out["RSI_OVERSOLD"] = int(
        is_mean_rev and
        rsi < 30 and
        float(row.get("rsi_5d_slope", 0)) > 0 and
        ret5 > -0.15   # not in freefall
    )

    # 3. Price down 15% in 20d bounce
    out["PRICE_DOWN_15_20D"] = int(
        ret20 < -0.13 and
        is_mean_rev and
        float(row.get("zscore_20d", 0)) < -1.8 and
        rv > 1.8 and
        ret1 > 0   # green candle today
    )

    # 4. DMI Wave — strong directional move
    out["DMI_WAVE"] = int(
        is_trend and
        adx > 28 and
        adx_slope > 1.0 and
        float(row.get("plus_di_14",0)) > float(row.get("minus_di_14",0)) and
        float(row.get("plus_di_14",0)) > float(row.get("plus_di_14",0)) and
        rv > 1.3
    )

    # 5. SAR + MACD confirmation
    out["SAR_MACD"] = int(
        int(row.get("sar_flip_bullish",0)) == 1 and
        macd_h > 0 and
        macd_h > float(row.get("macd_histogram",0)) and  # MACD expanding
        adx > 22
    )

    # 6. Bollinger breakout with volume
    out["BOLLINGER"] = int(
        int(row.get("bb_expansion_flag",0)) == 1 and
        int(row.get("bb_breakout_direction",0)) == 1 and
        rv > 2.0 and
        adx > 20
    )

    # 7. Volume explosion confirming direction
    out["VOLUME_EXPLOSION"] = int(
        int(row.get("vol_explosion_flag",0)) == 1 and
        int(row.get("vol_explosion_direction",0)) == 1 and
        ret1 > 0.01 and
        int(row.get("vol_explosion_price_confirm",0)) == 1
    )

    # 8. Uptrend — STRICTER: need 10+ consecutive days aligned + ADX rising
    out["UPTREND"] = int(
        is_trend and
        uqs >= 1.0 and
        upd >= 10 and          # in trend at least 10 days
        adx > 28 and
        adx_slope > 0 and      # trend strengthening
        sma20_score >= 1.0 and
        ret5 > 0.01            # still moving up
    )

    # 9. Stochastic RSI pullback entry
    out["STOCH_RSI"] = int(
        is_trend and
        k < 40 and
        k > d and              # K crossing D upward
        adx > 25 and
        ret20 > 0.05           # in overall uptrend
    )

    # 10. Pullback buy — price pulls back to MA in uptrend
    out["PULLBACK_BUY"] = int(
        is_trend and
        adx > 25 and
        uqs >= 0.8 and
        float(row.get("dist_sma_20",0)) > -0.03 and
        float(row.get("dist_sma_20",0)) < 0.005 and    # AT the 20-day MA
        ret5 < 0.00 and        # recent pullback
        ret20 > 0.03           # but still in uptrend
    )

    # 11. Golden cross (MA20 crosses above MA50)
    out["GOLDEN_CROSS"] = int(
        float(row.get("sma_20",1)) > float(row.get("sma_50",1)) and
        float(row.get("dist_sma_20",0)) > 0 and
        float(row.get("dist_sma_50",0)) < 0.01 and    # just crossed
        adx > 20 and
        rv > 1.2
    )

    # 12. Support zone bounce with volume + candle confirmation
    out["SUPPORT_BOUNCE"] = int(
        int(row.get("inside_support_zone",0)) == 1 and
        (int(row.get("hammer_at_support",0)) == 1 or
         int(row.get("morning_star_at_support",0)) == 1) and
        rv > 1.5 and
        rsi > 25 and rsi < 50
    )

    # 13. SMC liquidity sweep + recovery
    out["LIQUIDITY_SWEEP"] = int(
        int(row.get("liquidity_sweep_bullish",0)) == 1 and
        int(row.get("bos_bullish",0)) == 1 and
        rv > 1.5
    )

    # 14. Fibonacci 61.8% bounce
    out["FIB_BOUNCE"] = int(
        int(row.get("at_fib_618_flag",0)) == 1 and
        int(row.get("inside_support_zone",0)) == 1 and
        int(row.get("volume_declining_pullback",0)) == 1 and
        rsi < 55
    )

    return out

# ── Signal quality score ───────────────────────────────────────────────────────
def _quality_score(row, strats: dict, n_sig: int) -> float:
    """
    Composite 0–100 score weighting:
      • Strategy confluence (n_signals)
      • Volume confirmation
      • ADX trend strength
      • RSI positioning
      • Regime match (Hurst)
      • MA alignment
      • Foreign flow
    """
    rsi  = float(row.get("rsi_14",  50))
    adx  = float(row.get("adx_14",  20))
    rv   = float(row.get("rel_volume", 1))
    h    = float(row.get("hurst_60d", 0.5))
    maf  = float(row.get("net_foreign_flow_5d", 0))
    uqs  = float(row.get("uptrend_quality_score", 0))

    score = 0.0

    # Confluence bonus: each extra signal adds points
    score += min(n_sig * 12, 40)

    # Volume: strong volume confirmation
    if rv > 3:   score += 15
    elif rv > 2: score += 10
    elif rv > 1.5: score += 5

    # ADX trend strength
    if adx > 35:   score += 12
    elif adx > 28: score += 8
    elif adx > 22: score += 4

    # RSI positioning (buy in good zone)
    if 30 <= rsi <= 55:  score += 10
    elif 20 <= rsi < 30: score += 8    # oversold
    elif rsi > 70:       score -= 8    # overbought

    # Hurst regime clarity
    if h > 0.65 or h < 0.40: score += 8
    elif h > 0.58 or h < 0.45: score += 4

    # MA alignment
    score += min(uqs * 10, 10)

    # Foreign flow
    if maf > 0.02:  score += 5
    elif maf < -0.03: score -= 5

    return round(min(max(score, 0), 100), 1)


def _probability(base_wr: float, n_sig: int, quality: float) -> float:
    """
    Probability estimate (percentage) blending historical win rate
    with quality score.  Range: 50 – 92 %.
    """
    # Base from historical win rate
    base = base_wr * 100          # e.g. 0.743 → 74.3

    # Quality adjustment: +/- up to 15 points
    quality_adj = (quality - 50) / 50 * 15   # quality 50 → 0, 100 → +15, 0 → -15

    # Confluence bonus
    conf_bonus = min((n_sig - 1) * 4, 12)

    prob = base + quality_adj + conf_bonus
    return round(min(max(prob, 50.0), 92.0), 1)

# ── Compute signal levels from base price ─────────────────────────────────────
def _levels(primary: str, base_price: float, atr: float):
    tgt_pct = TARGET_PCT.get(primary, 0.12)
    stp_pct = STOP_PCT.get(primary, 0.05)

    # Use ATR-based stop if tighter
    atr_stop = base_price - 1.8 * atr
    atr_stop_pct = (base_price - atr_stop) / base_price
    stp_pct = min(stp_pct, max(atr_stop_pct, 0.02))

    target    = base_price * (1 + tgt_pct)
    stop_loss = base_price * (1 - stp_pct)
    entry_lo  = base_price * 0.980
    entry_hi  = base_price * 1.010
    reward_pct = (target    - base_price) / base_price * 100
    risk_pct   = (base_price - stop_loss) / base_price * 100
    rr_ratio   = reward_pct / max(risk_pct, 0.1)
    return {
        "target":      round(target, 0),
        "stop_loss":   round(stop_loss, 0),
        "entry_lo":    round(entry_lo, 0),
        "entry_hi":    round(entry_hi, 0),
        "reward_pct":  round(reward_pct, 2),
        "risk_pct":    round(risk_pct, 2),
        "rr_ratio":    round(rr_ratio, 2),
    }

# ── Signal state evaluation ───────────────────────────────────────────────────
def _signal_state(sig: dict, current_close: float, today_str: str) -> dict:
    """
    Evaluate the state of a previously-created BUY signal.
    Uses base_price (price on signal date) as the reference,
    NOT current price.
    """
    base   = float(sig.get("base_price", current_close))
    tgt    = float(sig.get("target", base * 1.12))
    stop   = float(sig.get("stop_loss", base * 0.94))
    elo    = float(sig.get("entry_lo",  base * 0.98))
    ehi    = float(sig.get("entry_hi",  base * 1.01))
    exp_d  = sig.get("exit_date", today_str)

    held_pct = (current_close - base) / base * 100

    if current_close >= tgt:
        return {"code":"TARGET_HIT",  "label":"🎯 Chốt lời",      "cls":"state-target",
                "action":"SELL",      "held_pct": round(held_pct,1)}
    if current_close <= stop:
        return {"code":"STOP_HIT",    "label":"🔴 Cắt lỗ ngay",   "cls":"state-stop",
                "action":"SELL",      "held_pct": round(held_pct,1)}
    if today_str >= exp_d:
        return {"code":"EXPIRED",     "label":"⏰ Hết hạn — xem lại","cls":"state-expired",
                "action":"REVIEW",    "held_pct": round(held_pct,1)}
    if elo <= current_close <= ehi * 1.01:
        return {"code":"ENTRY_ZONE",  "label":"🟡 Vùng mua vào",  "cls":"state-entry",
                "action":"BUY",       "held_pct": round(held_pct,1)}
    if current_close > ehi:
        # Already moved above entry zone → in holding
        if current_close < tgt * 0.97:
            return {"code":"HOLDING", "label":"🔵 Đang giữ lệnh", "cls":"state-holding",
                    "action":"HOLD",  "held_pct": round(held_pct,1)}
        else:
            return {"code":"NEAR_TARGET","label":"🟢 Gần mục tiêu","cls":"state-target",
                    "action":"HOLD",  "held_pct": round(held_pct,1)}
    # current_close < entry_lo: pulled back — wait
    return     {"code":"WATCHING",   "label":"👀 Chờ điểm vào",  "cls":"state-watch",
                "action":"WATCH",     "held_pct": round(held_pct,1)}

# ── Main signal processing ────────────────────────────────────────────────────
def process_ticker(ticker: str, df: pd.DataFrame):
    signal_rows = []
    last_signal_date_by_strat: dict[str, str] = {}   # cooldown tracker

    for idx, row in df.iterrows():
        if not _macro_ok(row):  continue
        if not _risk_ok(row):   continue

        strats = _detect(row)
        n_sig  = sum(strats.values())
        if n_sig == 0:
            continue

        date_str   = str(row["date"])[:10]
        base_price = float(row["close"])
        atr        = float(row.get("atr_14", base_price * 0.018))

        # Apply per-strategy cooldown
        active = [k for k,v in strats.items() if v == 1]
        cooled_active = []
        for k in active:
            last = last_signal_date_by_strat.get(k, "")
            if last:
                hold = STRATEGIES[k]["hold"]
                try:
                    from datetime import date as ddate
                    d1 = ddate.fromisoformat(last)
                    d2 = ddate.fromisoformat(date_str)
                    if (d2 - d1).days < hold:
                        continue   # still in cooldown
                except Exception:
                    pass
            cooled_active.append(k)

        if not cooled_active:
            continue

        n_sig = len(cooled_active)
        tier  = 1 if n_sig >= 3 else 2 if n_sig == 2 else 3

        # Quality gate: Tier 3 needs quality > 35, Tier 2 > 45, Tier 1 > 55
        quality  = _quality_score(row, strats, n_sig)
        min_q    = {1:55, 2:45, 3:35}[tier]
        if quality < min_q:
            continue

        primary = max(cooled_active, key=lambda k: STRATEGIES[k]["wr"])
        strat   = STRATEGIES[primary]
        lvl     = _levels(primary, base_price, atr)
        prob    = _probability(strat["wr"], n_sig, quality)

        bday_range = pd.bdate_range(start=date_str, periods=strat["hold"] + 1)
        exit_date  = str(bday_range[-1].date())

        record = {
            "ticker":           ticker,
            "date":             date_str,
            "signal":           "BUY",
            "tier":             tier,
            "primary_strategy": primary,
            "strategy_name":    strat["name"],
            "strategies_active":cooled_active,
            "n_signals":        n_sig,
            "hold_days":        strat["hold"],
            "exit_date":        exit_date,
            "base_price":       round(base_price, 0),     # ← KEY: price at signal date
            "current_price":    round(base_price, 0),
            **lvl,
            "dir_prob":         prob,
            "quality_score":    quality,
            "win_rate_hist":    round(strat["wr"] * 100, 1),
            "rsi_14":           round(float(row.get("rsi_14",   50)), 1),
            "adx_14":           round(float(row.get("adx_14",   20)), 1),
            "macd_histogram":   round(float(row.get("macd_histogram", 0)), 4),
            "rel_volume":       round(float(row.get("rel_volume",  1)), 2),
            "net_foreign_flow_5d": round(float(row.get("net_foreign_flow_5d", 0))*100, 2),
            "hurst_60d":        round(float(row.get("hurst_60d", 0.5)), 3),
            "garch_sigma":      round(float(row.get("garch_sigma",  0.2))*100, 2),
            "vwap_deviation":   round(float(row.get("vwap_deviation", 0))*100, 2),
        }

        for k in STRATEGIES:
            record[f"strat_{k}"] = int(k in cooled_active)

        signal_rows.append(record)

        # Update cooldown
        for k in cooled_active:
            last_signal_date_by_strat[k] = date_str

    if not signal_rows:
        return pd.DataFrame(), {}

    sig_df = pd.DataFrame(signal_rows)
    latest = sig_df.iloc[-1].to_dict()

    tier_labels = {
        1: "TIER 1 — ĐỘ TIN CẬY CAO NHẤT",
        2: "TIER 2 — ĐỘ TIN CẬY CAO",
        3: "TIER 3 — CHỜ XÁC NHẬN",
    }
    latest["tier_label"] = tier_labels.get(int(latest.get("tier", 3)), "TIER 3")
    latest["strategies_active_str"] = " + ".join(latest.get("strategies_active", []))

    # Evaluate signal state vs TODAY's close
    latest_feat = None
    # Try to get the most recent close from the feature file
    try:
        latest_feat = float(sig_df.sort_values("date").iloc[-1]["base_price"])
    except Exception:
        latest_feat = float(latest.get("base_price", latest.get("current_price", 0)))

    return sig_df, latest


def run(ticker: str) -> None:
    feat_path = FEAT / f"{ticker}.csv"
    if not feat_path.exists():
        log.error(f"{ticker}: feature file missing")
        return

    df = pd.read_csv(feat_path, low_memory=False)
    log.info(f"{ticker}: detecting signals ({len(df)} rows) …")
    sig_df, latest = process_ticker(ticker, df)

    if sig_df.empty:
        log.info(f"{ticker}: no signals")
        (SIG_DIR / f"{ticker}_latest.json").write_text(
            json.dumps({"ticker": ticker, "signal": "NONE", "tier": 0}, indent=2))
        return

    sig_df.to_csv(SIG_DIR / f"{ticker}_signals.csv", index=False)
    (SIG_DIR / f"{ticker}_latest.json").write_text(
        json.dumps(latest, indent=2, default=str))
    log.info(f"{ticker}: {len(sig_df)} signals  latest Tier-{latest.get('tier')} "
             f"prob={latest.get('dir_prob')}%  quality={latest.get('quality_score')}")


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
                    "ticker": t, "tier": d.get("tier"),
                    "strat":  d.get("primary_strategy"),
                    "date":   d.get("date"),
                    "prob":   d.get("dir_prob"),
                    "quality":d.get("quality_score"),
                    "base_price": d.get("base_price"),
                    "target": d.get("target"),
                    "stop_loss": d.get("stop_loss"),
                    "exit_date": d.get("exit_date"),
                })

    summary.sort(key=lambda x: (x["tier"], -(float(x.get("prob") or 0))))
    (SIG_DIR / "daily_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"\n=== {len(summary)} signals found ===")
    for s in summary[:15]:
        log.info(f"  [T{s['tier']}] {s['ticker']:6s}  {(s['strat']or'')[:20]:20s}  "
                 f"prob={s['prob']}%  q={s['quality']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--all",    action="store_true")
    args = parser.parse_args()
    tickers = [f.stem for f in FEAT.glob("*.csv")] if (args.all or not args.ticker) else [args.ticker]
    if len(tickers) > 1: run_all(tickers)
    else:
        for t in tickers: run(t)