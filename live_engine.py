"""
live_engine.py — Live intraday fine-tuning engine.

Runs during Vietnam market hours (09:00–14:45 ICT).
Every LIVE_INTERVAL_MIN minutes:
  1. Fetch latest 5-min bars
  2. Recompute intraday features (realized vol, volume imbalance, VWAP dev)
  3. Re-score AI models on updated feature vector (no refit — instant)
  4. Recalculate all math models (pivots, Fibonacci, Ichimoku, SAR, etc.)
  5. Combine AI + math model scores
  6. Emit updated BUY/SELL/HOLD signal if confidence changed > SIGNAL_CHANGE_THRESHOLD
  7. Check live price vs stop-loss / take-profit triggers
  8. Write to results/live_signals_YYYYMMDD.json

Architecture principle:
  - XGBoost and LightGBM are INFERENCE-ONLY during market hours (fast, <1s/symbol)
  - LSTM is frozen during market hours (too slow to retrain intraday)
  - Math models are always fully recalculated (stateless, instant)
  - Full retrain happens post-market at 15:00 ICT
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz

import config
from data_fetcher import fetch_ohlcv, get_vnindex
from intraday_fetcher import fetch_intraday, build_intraday_features
from math_models import build_math_model_features, get_math_signal_votes
from trade_signals import generate_signal, format_signal, save_signals, TradeSignal

logger = logging.getLogger("live_engine")

VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

# Live engine parameters
LIVE_INTERVAL_MIN         = 15      # minutes between updates
SIGNAL_CHANGE_THRESHOLD   = 0.05    # minimum confidence change to re-emit signal
MARKET_OPEN_HOUR          = 9
MARKET_OPEN_MINUTE        = 0
MARKET_CLOSE_HOUR         = 14
MARKET_CLOSE_MINUTE       = 45
MORNING_BREAK_START_HOUR  = 11
MORNING_BREAK_START_MIN   = 30
MORNING_BREAK_END_HOUR    = 13
MORNING_BREAK_END_MIN     = 0


def _is_market_open(now: Optional[datetime] = None) -> bool:
    """Check if HOSE is currently open (accounts for lunch break)."""
    now = now or datetime.now(VN_TZ)
    if now.weekday() >= 5:   # Saturday, Sunday
        return False
    t = (now.hour, now.minute)
    morning = (MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    lunch_start = (MORNING_BREAK_START_HOUR, MORNING_BREAK_START_MIN)
    lunch_end   = (MORNING_BREAK_END_HOUR, MORNING_BREAK_END_MIN)
    close = (MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    return (morning <= t < lunch_start) or (lunch_end <= t < close)


def _minutes_to_next_update(now: datetime) -> int:
    """How many seconds to sleep before the next 15-min tick."""
    mins = now.minute % LIVE_INTERVAL_MIN
    wait = (LIVE_INTERVAL_MIN - mins) * 60 - now.second
    return max(10, wait)


class LiveTradingEngine:
    """
    Manages intraday live signal updates for all tracked symbols.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        horizon: int = config.PREDICTION_HORIZON,
        total_capital: float = 100_000_000,
    ):
        self.symbols        = symbols or config.DEFAULT_SYMBOLS
        self.horizon        = horizon
        self.total_capital  = total_capital
        self.today_str      = datetime.now(VN_TZ).strftime("%Y%m%d")

        # Caches
        self._models: dict          = {}   # {sym: EnsembleModel}
        self._meta: dict            = {}   # {sym: meta_json}
        self._daily_ohlcv: dict     = {}   # {sym: DataFrame}
        self._math_df: dict         = {}   # {sym: DataFrame}
        self._last_signals: dict    = {}   # {sym: TradeSignal}
        self._live_signals: list    = []   # all signals emitted today
        self._vnindex               = None

        self._load_models()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_models(self):
        """Load trained ensemble models from disk."""
        from models import EnsembleModel
        import json
        for sym in self.symbols:
            meta_path = Path(f"{config.MODEL_DIR}/{sym}_h{self.horizon}_meta.json")
            if not meta_path.exists():
                logger.warning(f"[live] No model for {sym} — run train first")
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    self._meta[sym] = json.load(f)
                self._models[sym] = EnsembleModel.load(sym, self.horizon)
                logger.info(f"[live] Loaded model for {sym}")
            except Exception as e:
                logger.error(f"[live] Failed to load model for {sym}: {e}")

    # ── Data refresh ───────────────────────────────────────────────────────

    def _refresh_daily_data(self):
        """Refresh daily OHLCV (uses cache, TTL=6h)."""
        for sym in self.symbols:
            try:
                self._daily_ohlcv[sym] = fetch_ohlcv(sym)
            except Exception as e:
                logger.warning(f"[live] Daily data refresh failed for {sym}: {e}")
        try:
            self._vnindex = get_vnindex()
        except Exception:
            pass

    def _get_latest_intraday_bar(self, sym: str) -> Optional[pd.DataFrame]:
        """Fetch latest 5-min bars for a symbol."""
        try:
            return fetch_intraday(sym, interval="5", use_cache=False)
        except Exception as e:
            logger.debug(f"[live] Intraday fetch failed for {sym}: {e}")
            return None

    # ── Feature update ────────────────────────────────────────────────────

    def _build_live_features(self, sym: str) -> Optional[np.ndarray]:
        """
        Build the latest feature vector for a symbol using:
          - Daily OHLCV (already loaded)
          - Latest intraday bars (freshly fetched)
          - Math model features (recomputed)
        
        Returns a 1-row feature array aligned to the model's feature list.
        """
        if sym not in self._daily_ohlcv or sym not in self._meta:
            return None

        ohlcv     = self._daily_ohlcv[sym]
        feat_cols = self._meta[sym].get("features", [])
        if not feat_cols:
            return None

        # Build features — try extended signature, fall back to basic for older features.py
        from features import build_features, get_feature_cols
        try:
            feat_df = build_features(
                ohlcv, self._vnindex,
                symbol=sym,
                use_vol_adjusted_labels=False,
            )
        except TypeError:
            feat_df = build_features(ohlcv, self._vnindex)
        except Exception as e:
            logger.warning(f"[live] Feature build failed for {sym}: {e}")
            return None

        # Align to model's expected feature columns
        available = set(feat_df.columns)
        safe_cols = [c for c in feat_cols if c in available]
        if len(safe_cols) < len(feat_cols) * 0.8:
            logger.warning(f"[live] {sym}: only {len(safe_cols)}/{len(feat_cols)} features available")

        if not safe_cols:
            return None

        # Take the last row (most recent trading day)
        X_latest = feat_df[safe_cols].iloc[[-1]].values
        if np.any(np.isnan(X_latest)):
            X_latest = np.nan_to_num(X_latest, nan=0.0)

        return X_latest

    def _build_math_df(self, sym: str) -> Optional[pd.DataFrame]:
        """Build full math model feature DataFrame for a symbol."""
        if sym not in self._daily_ohlcv:
            return None
        ohlcv = self._daily_ohlcv[sym]
        try:
            math_df = build_math_model_features(
                ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"]
            )
            self._math_df[sym] = math_df
            return math_df
        except Exception as e:
            logger.warning(f"[live] Math model build failed for {sym}: {e}")
            return None

    # ── Signal generation ─────────────────────────────────────────────────

    def _score_symbol(self, sym: str) -> Optional[TradeSignal]:
        """Generate a fresh trade signal for one symbol."""
        if sym not in self._models:
            return None

        X = self._build_live_features(sym)
        if X is None:
            return None

        math_df = self._build_math_df(sym)
        if math_df is None or math_df.empty:
            return None

        # Get AI probability
        try:
            proba = self._models[sym].predict_proba(X)
            valid = ~np.isnan(proba)
            if not valid.any():
                return None
            ai_conf = float(proba[valid][-1])
        except Exception as e:
            logger.warning(f"[live] Model inference failed for {sym}: {e}")
            return None

        # Get math model votes
        last_close = float(self._daily_ohlcv[sym]["close"].iloc[-1])
        math_votes = get_math_signal_votes(last_close, math_df)

        # Generate full signal
        sig = generate_signal(
            symbol=sym,
            ohlcv=self._daily_ohlcv[sym],
            ai_confidence=ai_conf,
            math_df=math_df,
            math_votes=math_votes,
            horizon=self.horizon,
            model_auc=self._meta[sym].get("auc", 0),
            total_capital=self.total_capital,
        )
        return sig

    # ── Stop-loss / take-profit monitoring ───────────────────────────────

    def _check_price_triggers(self, sym: str) -> Optional[str]:
        """
        Check if the live price has hit stop-loss or take-profit.
        Returns "STOP", "TARGET", or None.
        """
        if sym not in self._last_signals or sym not in self._daily_ohlcv:
            return None

        prev_sig   = self._last_signals[sym]
        if prev_sig.signal not in ("BUY", "SELL"):
            return None

        current_price = float(self._daily_ohlcv[sym]["close"].iloc[-1])

        if prev_sig.signal == "BUY":
            if current_price <= prev_sig.stop_loss:
                logger.warning(f"[live] {sym}: STOP-LOSS triggered at {current_price:,.0f} (SL={prev_sig.stop_loss:,.0f})")
                return "STOP"
            if current_price >= prev_sig.take_profit:
                logger.info(f"[live] {sym}: TAKE-PROFIT triggered at {current_price:,.0f} (TP={prev_sig.take_profit:,.0f})")
                return "TARGET"

        elif prev_sig.signal == "SELL":
            if current_price >= prev_sig.stop_loss:
                logger.warning(f"[live] {sym}: STOP-LOSS triggered at {current_price:,.0f} (SL={prev_sig.stop_loss:,.0f})")
                return "STOP"
            if current_price <= prev_sig.take_profit:
                logger.info(f"[live] {sym}: TAKE-PROFIT triggered at {current_price:,.0f} (TP={prev_sig.take_profit:,.0f})")
                return "TARGET"

        return None

    # ── Main update loop ──────────────────────────────────────────────────

    def update_all(self) -> List[TradeSignal]:
        """
        Run one update cycle for all symbols.
        Returns list of signals that changed significantly.
        """
        self._refresh_daily_data()
        changed_signals = []
        now_str = datetime.now(VN_TZ).strftime("%H:%M ICT")

        logger.info(f"[live] Update cycle at {now_str} | {len(self.symbols)} symbols")

        for sym in self.symbols:
            # Check stop-loss / take-profit triggers first
            trigger = self._check_price_triggers(sym)
            if trigger:
                prev = self._last_signals[sym]
                exit_sig = TradeSignal(
                    **{**prev.__dict__,
                       "signal": f"EXIT ({trigger})",
                       "confidence": 1.0,
                       "generated_at": datetime.now().isoformat()}
                )
                changed_signals.append(exit_sig)
                self._last_signals[sym] = exit_sig
                logger.info(format_signal(exit_sig))
                continue

            # Generate new signal
            new_sig = self._score_symbol(sym)
            if new_sig is None:
                continue

            prev_sig = self._last_signals.get(sym)

            # Check if signal changed enough to re-emit
            is_new       = prev_sig is None
            conf_changed = prev_sig and abs(new_sig.confidence - prev_sig.confidence) >= SIGNAL_CHANGE_THRESHOLD
            dir_changed  = prev_sig and new_sig.signal != prev_sig.signal

            if is_new or conf_changed or dir_changed:
                self._last_signals[sym] = new_sig
                changed_signals.append(new_sig)
                self._live_signals.append(new_sig)
                logger.info(f"\n{format_signal(new_sig)}")
            else:
                logger.debug(f"[live] {sym}: no change (conf={new_sig.confidence:.1%})")

        # Save live signals
        if changed_signals:
            save_signals(self._live_signals, self.today_str)

        return changed_signals

    def run_until_close(self):
        """
        Block until market close, updating every LIVE_INTERVAL_MIN minutes.
        Called from scheduler.py during market hours.
        """
        logger.info("=" * 60)
        logger.info("Live Trading Engine STARTED")
        logger.info(f"Symbols : {', '.join(self.symbols)}")
        logger.info(f"Interval: every {LIVE_INTERVAL_MIN} minutes")
        logger.info("=" * 60)

        # Initial update immediately
        self.update_all()

        while True:
            now = datetime.now(VN_TZ)

            if not _is_market_open(now):
                logger.info(f"[live] Market closed at {now.strftime('%H:%M ICT')} — live engine stopping")
                break

            sleep_secs = _minutes_to_next_update(now)
            logger.debug(f"[live] Next update in {sleep_secs}s")
            time.sleep(sleep_secs)

            now = datetime.now(VN_TZ)
            if _is_market_open(now):
                self.update_all()

        # Final update after close
        logger.info("[live] Running final post-close update...")
        self.update_all()
        logger.info("[live] Live engine finished")

    def get_current_signals(self) -> Dict[str, TradeSignal]:
        """Return the latest signal for each symbol."""
        return dict(self._last_signals)
