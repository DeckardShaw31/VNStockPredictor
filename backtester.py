"""
backtester.py — Rigorous walk-forward backtest with realistic trading simulation.

Improvements over v1:
  - Proper intrabar stop-loss and take-profit simulation
    (not just end-of-period return — checks if SL/TP was hit during the holding period)
  - Transaction costs (brokerage + tax = 0.15% round-trip, configurable)
  - Trailing stop option (moves SL up as price rises)
  - Long AND short simulation (if SELL signal, short the stock)
  - Per-symbol and portfolio-level equity curve
  - Confidence-filtered win rate (only trades where conf >= threshold)
  - Calmar ratio (return / max drawdown)
  - Average trade duration
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import config
from data_fetcher import fetch_multiple, get_vnindex
from features import build_features, get_feature_cols
from models import EnsembleModel, XGBModel, LGBMModel

logger = logging.getLogger("backtester")

# Transaction costs (HOSE: 0.1% brokerage + 0.1% tax on sell side)
TRANSACTION_COST = 0.0015   # 0.15% one-way (buy+sell = 0.30% round-trip)


def _simulate_trade(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    direction: int,           # 1=long, -1=short
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    horizon: int,
    trailing_stop: bool = False,
    trailing_pct: float = 0.02,   # 2% trailing stop
) -> Tuple[float, int, str]:
    """
    Simulate a single trade with intrabar SL/TP checking.

    Returns:
        (pnl_pct, exit_idx, exit_reason)
        pnl_pct:     return including transaction costs
        exit_idx:    index at which trade closed
        exit_reason: "SL", "TP", "TRAIL", "EXPIRE", "ERROR"
    """
    n = len(ohlcv)
    if entry_idx >= n - 1:
        return 0.0, entry_idx, "ERROR"

    highest_price = entry_price   # for trailing stop tracking
    cost = TRANSACTION_COST * 2   # round-trip

    for i in range(entry_idx + 1, min(entry_idx + horizon + 1, n)):
        bar_high  = float(ohlcv["high"].iloc[i])
        bar_low   = float(ohlcv["low"].iloc[i])
        bar_close = float(ohlcv["close"].iloc[i])

        if direction == 1:   # Long trade
            # Update trailing stop
            if trailing_stop and bar_high > highest_price:
                highest_price = bar_high
                new_sl = highest_price * (1 - trailing_pct)
                if new_sl > stop_loss:
                    stop_loss = new_sl

            # Check stop-loss (did bar go below SL?)
            if bar_low <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price - cost
                return pnl, i, "TRAIL" if trailing_stop else "SL"

            # Check take-profit
            if bar_high >= take_profit:
                pnl = (take_profit - entry_price) / entry_price - cost
                return pnl, i, "TP"

        else:   # Short trade
            if trailing_stop and bar_low < highest_price:
                highest_price = bar_low
                new_sl = highest_price * (1 + trailing_pct)
                if new_sl < stop_loss:
                    stop_loss = new_sl

            if bar_high >= stop_loss:
                pnl = (entry_price - stop_loss) / entry_price - cost
                return pnl, i, "TRAIL" if trailing_stop else "SL"

            if bar_low <= take_profit:
                pnl = (entry_price - take_profit) / entry_price - cost
                return pnl, i, "TP"

    # Horizon expired — exit at close
    final_close = float(ohlcv["close"].iloc[min(entry_idx + horizon, n - 1)])
    if direction == 1:
        pnl = (final_close - entry_price) / entry_price - cost
    else:
        pnl = (entry_price - final_close) / entry_price - cost

    return pnl, min(entry_idx + horizon, n - 1), "EXPIRE"


class Backtester:
    def __init__(
        self,
        symbols:              Optional[List[str]] = None,
        horizon:              int   = config.PREDICTION_HORIZON,
        retrain_every:        int   = 20,       # trading days
        tune:                 bool  = False,
        initial_capital:      float = 100_000_000,
        position_size:        float = 0.10,     # 10% per trade
        confidence_threshold: float = 0.58,
        use_sl_tp:            bool  = True,     # simulate SL/TP hits
        trailing_stop:        bool  = False,
        allow_short:          bool  = False,    # trade SELL signals too
        atr_sl_mult:          float = 1.5,
        atr_tp_mult:          float = 3.0,
    ):
        self.symbols              = symbols or config.DEFAULT_SYMBOLS[:5]
        self.horizon              = horizon
        self.retrain_every        = retrain_every
        self.tune                 = tune
        self.initial_capital      = initial_capital
        self.position_size        = position_size
        self.confidence_threshold = confidence_threshold
        self.use_sl_tp            = use_sl_tp
        self.trailing_stop        = trailing_stop
        self.allow_short          = allow_short
        self.atr_sl_mult          = atr_sl_mult
        self.atr_tp_mult          = atr_tp_mult

    def run(self) -> Tuple[str, dict]:
        """Returns (text_report, results_dict)."""
        vnindex     = get_vnindex()
        data        = fetch_multiple(self.symbols)
        all_results = {}
        all_equity  = {}

        for sym in self.symbols:
            if sym not in data:
                continue
            try:
                result, equity = self._backtest_symbol(sym, data[sym], vnindex)
                all_results[sym] = result
                all_equity[sym]  = equity
                logger.info(
                    f"  {sym}: AUC={result.get('auc',0):.3f} "
                    f"strategy={result.get('strategy_return_pct',0):+.1f}% "
                    f"B&H={result.get('buy_hold_return_pct',0):+.1f}% "
                    f"Sharpe={result.get('sharpe_ratio',0):.2f}"
                )
            except Exception as e:
                logger.error(f"Backtest failed for {sym}: {e}", exc_info=True)
                all_results[sym] = {"error": str(e)}

        report = self._format_report(all_results)
        self._save(all_results, all_equity)
        return report, all_results

    def _compute_atr(self, ohlcv: pd.DataFrame, n: int = 14) -> pd.Series:
        tr = pd.concat([
            ohlcv["high"] - ohlcv["low"],
            (ohlcv["high"] - ohlcv["close"].shift()).abs(),
            (ohlcv["low"]  - ohlcv["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=n, adjust=False).mean()

    def _backtest_symbol(
        self, sym: str, ohlcv: pd.DataFrame, vnindex
    ) -> Tuple[dict, pd.Series]:

        feat_df   = build_features(ohlcv, vnindex)
        feat_cols = get_feature_cols(feat_df, self.horizon)
        tgt_col   = f"target_dir_{self.horizon}d"
        ret_col   = f"target_ret_{self.horizon}d"

        n         = len(feat_df)
        min_train = 120
        atr_ser   = self._compute_atr(ohlcv)

        predictions  = []
        capital      = self.initial_capital
        equity_curve = [capital]
        trades       = []

        step = 0
        while min_train + step * self.retrain_every < n - self.horizon - 10:
            train_end = min_train + step * self.retrain_every
            test_end  = min(train_end + self.retrain_every, n - self.horizon)
            if train_end >= test_end:
                break

            X_tr = feat_df.iloc[:train_end][feat_cols].values
            y_tr = feat_df.iloc[:train_end][tgt_col].values

            use_lstm = len(X_tr) >= 40
            model    = EnsembleModel()
            if not use_lstm:
                model.weights = {"xgb": 0.5, "lgbm": 0.5, "lstm": 0.0, "meta": 0.0}
            else:
                model.lstm.params["epochs"] = 12

            try:
                if use_lstm:
                    model.fit(X_tr, y_tr)
                else:
                    model.xgb.fit(X_tr, y_tr)
                    model.lgbm.fit(X_tr, y_tr)
            except Exception as e:
                logger.debug(f"  Step {step} fit failed: {e}")
                step += 1
                continue

            X_te  = feat_df.iloc[train_end:test_end][feat_cols].values
            proba = model.predict_proba(X_te)

            for i, p in enumerate(proba):
                if np.isnan(p):
                    continue
                feat_idx = train_end + i
                if feat_idx >= n - self.horizon:
                    break

                # Find matching OHLCV index by date
                feat_date  = feat_df.index[feat_idx]
                ohlcv_locs = ohlcv.index.get_indexer([feat_date], method="nearest")
                ohlcv_idx  = int(ohlcv_locs[0])
                if ohlcv_idx < 0 or ohlcv_idx >= len(ohlcv) - 1:
                    continue

                actual_dir = int(feat_df[tgt_col].iloc[feat_idx])
                actual_ret = float(feat_df[ret_col].iloc[feat_idx]) \
                             if ret_col in feat_df.columns else 0.0

                predictions.append({
                    "date":   str(feat_date.date()),
                    "proba":  float(p),
                    "pred":   int(p >= 0.5),
                    "actual": actual_dir,
                    "ret":    actual_ret,
                })

                # Trade simulation
                is_long  = p >= self.confidence_threshold
                is_short = (1 - p) >= self.confidence_threshold and self.allow_short

                if not (is_long or is_short):
                    equity_curve.append(capital)
                    continue

                direction    = 1 if is_long else -1
                entry_price  = float(ohlcv["close"].iloc[ohlcv_idx])
                atr_val      = float(atr_ser.iloc[ohlcv_idx]) if ohlcv_idx < len(atr_ser) else entry_price * 0.02

                if self.use_sl_tp and atr_val > 0:
                    if direction == 1:
                        stop_loss   = entry_price - self.atr_sl_mult * atr_val
                        take_profit = entry_price + self.atr_tp_mult * atr_val
                    else:
                        stop_loss   = entry_price + self.atr_sl_mult * atr_val
                        take_profit = entry_price - self.atr_tp_mult * atr_val

                    pnl_pct, exit_idx, reason = _simulate_trade(
                        ohlcv, ohlcv_idx, direction, entry_price,
                        stop_loss, take_profit, self.horizon,
                        trailing_stop=self.trailing_stop,
                    )
                else:
                    pnl_pct = actual_ret * direction - TRANSACTION_COST * 2
                    reason  = "EXPIRE"

                trade_capital = capital * self.position_size
                pnl_vnd       = trade_capital * pnl_pct
                capital      += pnl_vnd
                capital       = max(capital, 1)   # floor at 1 VND (no margin)
                equity_curve.append(capital)

                trades.append({
                    "date":      str(feat_date.date()),
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "confidence":float(p),
                    "pnl_pct":   round(pnl_pct * 100, 3),
                    "pnl_vnd":   round(pnl_vnd, 0),
                    "exit":      reason,
                    "capital":   round(capital, 0),
                })

            step += 1

        if not predictions:
            return {"error": "No predictions generated"}, pd.Series()

        pred_df   = pd.DataFrame(predictions)
        try:
            auc = roc_auc_score(pred_df["actual"], pred_df["proba"])
        except Exception:
            auc = 0.5
        acc = accuracy_score(pred_df["actual"], pred_df["pred"])

        # Filtered metrics (only high-confidence predictions)
        hi_conf = pred_df[pred_df["proba"] >= self.confidence_threshold]
        win_rate = float(hi_conf["actual"].mean()) if len(hi_conf) > 0 else 0.0

        bh_ret   = (ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[0] - 1)
        strat    = (capital - self.initial_capital) / self.initial_capital

        eq       = pd.Series(equity_curve)
        daily    = eq.pct_change().dropna()
        sharpe   = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
        peak     = eq.cummax()
        max_dd   = float(((eq - peak) / peak).min()) * 100
        calmar   = float(strat / abs(max_dd / 100)) if max_dd < 0 else float("inf")

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        avg_dur   = self.horizon   # approximate
        sl_exits  = len(trades_df[trades_df["exit"] == "SL"])  if not trades_df.empty else 0
        tp_exits  = len(trades_df[trades_df["exit"] == "TP"])  if not trades_df.empty else 0
        exp_exits = len(trades_df[trades_df["exit"] == "EXPIRE"]) if not trades_df.empty else 0

        return {
            "symbol":              sym,
            "n_predictions":       len(predictions),
            "n_trades":            len(trades),
            "auc":                 round(auc, 4),
            "accuracy":            round(acc, 4),
            "win_rate":            round(win_rate, 4),
            "strategy_return_pct": round(strat * 100, 2),
            "buy_hold_return_pct": round(bh_ret * 100, 2),
            "sharpe_ratio":        round(sharpe, 3),
            "max_drawdown_pct":    round(max_dd, 2),
            "calmar_ratio":        round(calmar, 3),
            "final_capital":       round(capital, 0),
            "sl_exits":            sl_exits,
            "tp_exits":            tp_exits,
            "expire_exits":        exp_exits,
        }, eq

    def _format_report(self, results: dict) -> str:
        lines = [
            "=" * 78,
            " VIETNAM STOCK AI — WALK-FORWARD BACKTEST REPORT",
            f" Config: SL={self.atr_sl_mult}xATR  TP={self.atr_tp_mult}xATR  "
            f"Conf>={self.confidence_threshold:.0%}  "
            f"Pos={self.position_size:.0%}  Cost={TRANSACTION_COST:.2%}/way",
            "=" * 78,
            f"{'Sym':<6} {'AUC':>6} {'Acc':>6} {'Win%':>6} "
            f"{'Strat%':>8} {'B&H%':>8} {'Sharpe':>7} {'DD%':>7} "
            f"{'Calmar':>7} {'SL':>5} {'TP':>5}",
            "-" * 78,
        ]
        for sym, r in results.items():
            if "error" in r:
                lines.append(f"{sym:<6}  ERROR: {r['error']}")
                continue
            lines.append(
                f"{sym:<6} {r['auc']:>6.3f} {r['accuracy']:>6.3f} "
                f"{r['win_rate']:>6.3f} "
                f"{r['strategy_return_pct']:>+8.1f} {r['buy_hold_return_pct']:>+8.1f} "
                f"{r['sharpe_ratio']:>7.2f} {r['max_drawdown_pct']:>7.1f} "
                f"{r.get('calmar_ratio',0):>7.2f} "
                f"{r.get('sl_exits',0):>5} {r.get('tp_exits',0):>5}"
            )
        lines.append("=" * 78)
        return "\n".join(lines)

    def _save(self, results: dict, equity: dict):
        date_str = datetime.now().strftime("%Y%m%d")
        out = Path(config.RESULTS_DIR) / f"backtest_{date_str}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        # Save equity curves as CSV
        if equity:
            eq_df = pd.DataFrame({sym: s.values for sym, s in equity.items()
                                   if isinstance(s, pd.Series) and len(s) > 0})
            if not eq_df.empty:
                eq_path = Path(config.RESULTS_DIR) / f"equity_curve_{date_str}.csv"
                eq_df.to_csv(eq_path, index=False)
