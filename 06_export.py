"""
Module 06 — Export to JSON for HTML Visualization
====================================================
Reads all processed data and generates a single ui_data.json that the
HTML dashboard loads via fetch().

Output structure:
{
  "generated_at": "...",
  "tickers": {
    "HPG": {
      "info":     { name, sector, price, change, changePercent },
      "chart":    [ { date, open, high, low, close, volume, sma20, sma50,
                      bb_upper, bb_lower, kalman, upperBound, lowerBound,
                      foreignFlow, rsi, adx, macd } ... ],
      "signal":   { latest signal card data },
      "metrics":  { backtest performance metrics }
    },
    ...
  },
  "daily_summary": [ ... tier-1/2 signals today ],
  "backtest_summary": { overall stats }
}

Run:
    python modules/06_export.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
RAW     = ROOT / "data" / "raw"
FEAT    = ROOT / "data" / "features"
SIG_DIR = ROOT / "data" / "signals"
MDL     = ROOT / "data" / "models"
REP     = ROOT / "data" / "reports"
VIZ     = ROOT / "viz"
VIZ.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("export")

# ── Stock metadata ─────────────────────────────────────────────────────────────
STOCK_META = {
    "HPG":  {"name": "Tập đoàn Hòa Phát",       "sector": "Steel"},
    "VCB":  {"name": "Vietcombank",               "sector": "Banking"},
    "VHM":  {"name": "Vinhomes",                  "sector": "Real Estate"},
    "VIC":  {"name": "Vingroup",                  "sector": "Conglomerate"},
    "VNM":  {"name": "Vinamilk",                  "sector": "Consumer"},
    "MSN":  {"name": "Masan Group",               "sector": "Consumer"},
    "TCB":  {"name": "Techcombank",               "sector": "Banking"},
    "BID":  {"name": "BIDV",                      "sector": "Banking"},
    "CTG":  {"name": "VietinBank",                "sector": "Banking"},
    "MBB":  {"name": "MB Bank",                   "sector": "Banking"},
    "ACB":  {"name": "ACB",                       "sector": "Banking"},
    "FPT":  {"name": "FPT Corporation",           "sector": "Technology"},
    "MWG":  {"name": "Mobile World Group",        "sector": "Retail"},
    "SSI":  {"name": "SSI Securities",            "sector": "Securities"},
    "GAS":  {"name": "PetroVietnam Gas",          "sector": "Energy"},
    "PLX":  {"name": "Petrolimex",                "sector": "Energy"},
    "VRE":  {"name": "Vincom Retail",             "sector": "Real Estate"},
    "HDB":  {"name": "HD Bank",                   "sector": "Banking"},
    "VPB":  {"name": "VPBank",                    "sector": "Banking"},
    "STB":  {"name": "Sacombank",                 "sector": "Banking"},
}


def _safe_float(v, default=0.0) -> float:
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return default


def _safe_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def build_chart_series(df_feat: pd.DataFrame, n_rows: int = 252) -> list[dict]:
    """Convert the last n_rows of feature data into chart-friendly dicts."""
    tail = df_feat.tail(n_rows).copy()
    tail["date"] = pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d")

    # Kalman proxy: EMA(5) of close
    tail["kalman"] = tail["close"].ewm(span=5).mean()

    # Bounds: close ± GARCH_sigma * 1.96 * close
    sigma = tail["garch_sigma"].fillna(0.2)
    tail["upperBound"] = tail["close"] * (1 + sigma / np.sqrt(252) * 1.96)
    tail["lowerBound"] = tail["close"] * (1 - sigma / np.sqrt(252) * 1.96)

    chart = []
    for _, row in tail.iterrows():
        chart.append({
            "date":         row["date"],
            "open":         _safe_float(row.get("open", row["close"])),
            "high":         _safe_float(row.get("high", row["close"])),
            "low":          _safe_float(row.get("low", row["close"])),
            "close":        _safe_float(row["close"]),
            "volume":       _safe_int(row.get("volume", 0)),
            "sma20":        _safe_float(row.get("sma_20")),
            "sma50":        _safe_float(row.get("sma_50")),
            "sma200":       _safe_float(row.get("sma_200")),
            "bb_upper":     _safe_float(row.get("bb_upper")),
            "bb_lower":     _safe_float(row.get("bb_lower")),
            "kalman":       _safe_float(row["kalman"]),
            "upperBound":   _safe_float(row["upperBound"]),
            "lowerBound":   _safe_float(row["lowerBound"]),
            "foreignFlow":  _safe_int(row.get("net_foreign_flow_5d", 0) * row.get("close", 1) * 1e6),
            "rsi":          _safe_float(row.get("rsi_14")),
            "adx":          _safe_float(row.get("adx_14")),
            "macd":         _safe_float(row.get("macd_histogram")),
            "relVolume":    _safe_float(row.get("rel_volume")),
            "sar":          _safe_float(row.get("sar_t")),
            "sarBullish":   _safe_int(row.get("sar_bullish", 0)),
            "hurst":        _safe_float(row.get("hurst_60d", 0.5)),
        })
    return chart


def build_ticker_payload(ticker: str) -> dict | None:
    feat_path = FEAT / f"{ticker}.csv"
    if not feat_path.exists():
        return None

    df_feat = pd.read_csv(feat_path, low_memory=False)
    if df_feat.empty:
        return None

    last_row = df_feat.iloc[-1]
    prev_row = df_feat.iloc[-2] if len(df_feat) > 1 else last_row

    current_price  = _safe_float(last_row["close"])
    prev_price     = _safe_float(prev_row["close"])
    change         = current_price - prev_price
    change_pct     = (change / prev_price * 100) if prev_price != 0 else 0

    meta = STOCK_META.get(ticker, {"name": ticker, "sector": "Unknown"})

    # Chart data
    chart = build_chart_series(df_feat)

    # Signal
    sig_path = SIG_DIR / f"{ticker}_latest.json"
    if sig_path.exists():
        with open(sig_path) as f:
            signal = json.load(f)
    else:
        signal = {"ticker": ticker, "signal": "NONE", "tier": 0}

    # Backtest metrics
    stats_path = REP / "backtest_report.json"
    metrics    = {}
    if stats_path.exists():
        with open(stats_path) as f:
            bt = json.load(f)
        for s in bt.get("per_ticker", []):
            if s.get("ticker") == ticker:
                metrics = s
                break

    # Model probabilities
    proba_path = MDL / f"{ticker}_proba.csv"
    model_proba = {}
    if proba_path.exists():
        p_df = pd.read_csv(proba_path)
        if not p_df.empty:
            p_last = p_df.iloc[-1]
            for col in p_df.columns:
                if col.startswith("proba_"):
                    model_proba[col] = _safe_float(p_last[col])

    # Assembled signals section with model proba overlay
    signal_enriched = {**signal}
    if model_proba:
        # Use the most relevant proba for the detected strategy
        strat = signal.get("primary_strategy", "GENERAL")
        target_map = {
            "PRICE_DOWN_15_MA20": "proba_fwd_positive_t180",
            "RSI_OVERSOLD":       "proba_fwd_positive_t60",
            "PRICE_DOWN_15_20D":  "proba_fwd_positive_t5",
            "DMI_WAVE":           "proba_fwd_positive_t10",
            "SAR_MACD":           "proba_fwd_positive_t20",
            "BOLLINGER":          "proba_fwd_positive_t5",
            "VOLUME_EXPLOSION":   "proba_fwd_positive_t180",
            "UPTREND":            "proba_fwd_positive_t180",
            "STOCH_RSI":          "proba_fwd_positive_t180",
        }
        proba_key = target_map.get(strat, "proba_fwd_positive_t20")
        raw_model_p = model_proba.get(proba_key, None)
        hist_wr     = float(signal.get("win_rate_hist", 60))
        dir_prob    = float(signal.get("dir_prob", hist_wr))
        if raw_model_p is not None:
            # Model gives 0-1; floor at 80% of historical win rate to avoid garbage
            model_pct = float(raw_model_p) * 100
            signal_enriched["model_dir_prob"] = round(max(model_pct, hist_wr * 0.80), 1)
        else:
            signal_enriched["model_dir_prob"] = round(dir_prob, 1)

    # Last-row indicators for dashboard cards
    indicators = {
        "hurst":              _safe_float(last_row.get("hurst_60d", 0.5)),
        "regime":             "Trending" if _safe_float(last_row.get("hurst_60d", 0.5)) > 0.55 else (
                              "Mean Reverting" if _safe_float(last_row.get("hurst_60d", 0.5)) < 0.45 else "Transitioning"),
        "vnIndexScore":       _safe_float(last_row.get("vnindex_ma_score", 0.5)),
        "sectorRS":           _safe_float(last_row.get("stock_rs_20d", 1.0)),
        "rsi14":              _safe_float(last_row.get("rsi_14", 50)),
        "adx14":              _safe_float(last_row.get("adx_14", 20)),
        "macdHistogram":      _safe_float(last_row.get("macd_histogram", 0)),
        "relVolume":          _safe_float(last_row.get("rel_volume", 1)),
        "garchSigma":         _safe_float(last_row.get("garch_sigma", 0.2) * 100),
        "vsaRatio":           _safe_float(last_row.get("vsa_ratio", 0)),
        "vwapDeviation":      _safe_float(last_row.get("vwap_deviation", 0) * 100),
        "usdVndTrend":        0.5,
        "interbankRate":      _safe_float(last_row.get("interbank_rate", 3.5)),
        "breadthPctAbove50":  _safe_float(last_row.get("breadth_pct_above_50sma", 0.5) * 100),
        "adRatio10d":         _safe_float(last_row.get("adv_dec_ratio_10d", 1.0)),
        "bullTrapFlag":       bool(_safe_int(last_row.get("bull_trap_flag", 0))),
        "bearTrapFlag":       bool(_safe_int(last_row.get("bear_trap_flag", 0))),
        "fomoSignal":         bool(_safe_int(last_row.get("fomo_signal", 0))),
        "ceilingDemand":      _safe_float(last_row.get("post_ceiling_day1_flag", 0)),
        "t25Risk":            _safe_float(last_row.get("t25_risk", 0.05) * 100),
        "wyckoffPhase":       str(last_row.get("wyckoff_phase", "B")),
        "uptrend_quality":    _safe_float(last_row.get("uptrend_quality_score", 0)),
        "uptrend_duration":   _safe_int(last_row.get("uptrend_duration", 0)),
        "directionProbability": signal_enriched.get("model_dir_prob", signal.get("dir_prob", 50)),
        "action":             signal.get("signal", "WATCH"),
        "actionReason":       _build_action_reason(last_row, signal),
        "directionProb":      signal_enriched.get("model_dir_prob", signal.get("dir_prob", 50)),
    }

    # ── Signal state vs today's close ─────────────────────────────────────────
    sig_state = {}
    try:
        if signal_enriched.get("tier", 0) > 0 and signal_enriched.get("base_price"):
            base   = float(signal_enriched.get("base_price", current_price))
            tgt    = float(signal_enriched.get("target",    base * 1.12))
            stop   = float(signal_enriched.get("stop_loss", base * 0.94))
            elo    = float(signal_enriched.get("entry_lo",  base * 0.98))
            ehi    = float(signal_enriched.get("entry_hi",  base * 1.01))
            exp_d  = signal_enriched.get("exit_date", "9999-12-31")
            today_s = datetime.now().strftime("%Y-%m-%d")
            held_pct = round((current_price - base) / max(base, 1) * 100, 1)
            if current_price >= tgt:
                sig_state = {"code":"TARGET_HIT","label":"🎯 Chốt lời","cls":"state-target","action":"SELL","held_pct":held_pct}
            elif current_price <= stop:
                sig_state = {"code":"STOP_HIT","label":"🔴 Cắt lỗ ngay","cls":"state-stop","action":"SELL","held_pct":held_pct}
            elif today_s >= exp_d:
                sig_state = {"code":"EXPIRED","label":"⏰ Hết hạn","cls":"state-expired","action":"REVIEW","held_pct":held_pct}
            elif elo * 0.99 <= current_price <= ehi * 1.01:
                sig_state = {"code":"ENTRY_ZONE","label":"🟡 Vùng mua vào","cls":"state-entry","action":"BUY","held_pct":held_pct}
            elif current_price > ehi:
                sig_state = {"code":"HOLDING","label":"🔵 Đang giữ lệnh","cls":"state-holding","action":"HOLD","held_pct":held_pct}
            else:
                sig_state = {"code":"WATCHING","label":"👀 Chờ điểm vào","cls":"state-watch","action":"WATCH","held_pct":held_pct}
    except Exception:
        pass
    signal_enriched["signal_state"] = sig_state
    signal_enriched["today_action"] = sig_state.get("action", "WATCH") if sig_state else "WATCH"

    # ── Backtest equity curve + trades ───────────────────────────────────────────
    backtest_equity = []
    backtest_trades = []
    eq_path = REP / f"{ticker}_equity.csv"
    if eq_path.exists():
        eq_df = pd.read_csv(eq_path)
        backtest_equity = eq_df.to_dict("records")
    if stats_path.exists():
        with open(stats_path) as f:
            bt_all = json.load(f)
        ticker_trades = bt_all.get("trades", {}).get(ticker, [])
        backtest_trades = ticker_trades if isinstance(ticker_trades, list) else []

    return {
        "ticker":     ticker,
        "name":       meta["name"],
        "sector":     meta["sector"],
        "price":      current_price,
        "change":     round(change, 0),
        "changePercent": round(change_pct, 2),
        "chart":      chart,
        "signal":     signal_enriched,
        "indicators": indicators,
        "metrics":    metrics,
        "modelProba": model_proba,
        "backtest":   {"equity": backtest_equity, "trades": backtest_trades},
    }


def _build_action_reason(row: pd.Series, signal: dict) -> str:
    parts = []
    hurst = _safe_float(row.get("hurst_60d", 0.5))
    rsi   = _safe_float(row.get("rsi_14", 50))
    adx   = _safe_float(row.get("adx_14", 20))

    if hurst > 0.55:
        parts.append(f"Hurst={hurst:.2f} (trending regime)")
    elif hurst < 0.45:
        parts.append(f"Hurst={hurst:.2f} (mean-reverting)")

    if rsi < 30:
        parts.append(f"RSI={rsi:.1f} (oversold)")
    elif rsi > 70:
        parts.append(f"RSI={rsi:.1f} (overbought)")

    if adx > 25:
        parts.append(f"ADX={adx:.1f} (strong trend)")

    tier = signal.get("tier", 0)
    if tier > 0:
        parts.append(f"Tier-{tier} signal: {signal.get('primary_strategy', '')}")

    return " · ".join(parts) if parts else "No active signal"


# ── Main export ───────────────────────────────────────────────────────────────

def run_export(tickers: list[str] | None = None) -> None:
    if tickers is None:
        tickers = [f.stem for f in FEAT.glob("*.csv")]

    output = {
        "generated_at": datetime.now().isoformat(),
        "tickers":      {},
    }

    for t in tickers:
        log.info(f"Exporting {t} …")
        payload = build_ticker_payload(t)
        if payload:
            output["tickers"][t] = payload
        else:
            log.warning(f"  {t}: skipped (no data)")

    # Daily summary
    sum_path = SIG_DIR / "daily_summary.json"
    if sum_path.exists():
        with open(sum_path) as f:
            output["daily_summary"] = json.load(f)
    else:
        output["daily_summary"] = []

    # Backtest summary
    bt_path = REP / "backtest_report.json"
    if bt_path.exists():
        with open(bt_path) as f:
            bt = json.load(f)
        output["backtest_summary"] = bt.get("summary", {})
    else:
        output["backtest_summary"] = {}

    out_path = VIZ / "ui_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"), default=str)

    size_kb = out_path.stat().st_size // 1024
    log.info(f"\n✓ Exported {len(output['tickers'])} tickers → {out_path}  ({size_kb} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VN100 Export to JSON")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()
    run_export(args.tickers)