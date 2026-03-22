"""
backtester.py — Walk-forward backtest to evaluate prediction quality over time.

Simulates how the model would have performed if retrained monthly and
used for trading decisions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import config
from data_fetcher import fetch_multiple, get_vnindex
from features import build_features, get_feature_cols
from models import EnsembleModel, XGBModel, LGBMModel

logger = logging.getLogger("backtester")


class Backtester:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        horizon: int = config.PREDICTION_HORIZON,
        retrain_every: int = 20,      # trading days
        tune: bool = False,           # fast backtest without Optuna
        initial_capital: float = 100_000_000,  # VND
        position_size: float = 0.1,            # 10% of capital per trade
        confidence_threshold: float = 0.60,    # only trade if confidence > 60%
    ):
        self.symbols    = symbols or config.DEFAULT_SYMBOLS[:5]  # backtest on subset
        self.horizon    = horizon
        self.retrain_every      = retrain_every
        self.tune               = tune
        self.initial_capital    = initial_capital
        self.position_size      = position_size
        self.confidence_threshold = confidence_threshold

    def run(self) -> str:
        vnindex = get_vnindex()
        data    = fetch_multiple(self.symbols)
        all_results = {}

        for sym in self.symbols:
            if sym not in data:
                continue
            try:
                result = self._backtest_symbol(sym, data[sym], vnindex)
                all_results[sym] = result
            except Exception as e:
                logger.error(f"Backtest failed for {sym}: {e}", exc_info=True)

        return self._format_report(all_results)

    def _backtest_symbol(self, sym: str, ohlcv: pd.DataFrame, vnindex) -> dict:
        logger.info(f"Backtesting {sym}…")
        feat_df   = build_features(ohlcv, vnindex)
        feat_cols = get_feature_cols(feat_df, self.horizon)
        tgt_col   = f"target_dir_{self.horizon}d"
        ret_col   = f"target_ret_{self.horizon}d"

        n = len(feat_df)
        min_train = 120  # minimum rows before first trade

        predictions = []
        capital = self.initial_capital
        portfolio_values = [capital]
        trades = []

        step = 0
        while min_train + step * self.retrain_every < n - self.horizon - 10:
            train_end = min_train + step * self.retrain_every
            test_end  = min(train_end + self.retrain_every, n - self.horizon)

            if train_end >= test_end:
                break

            train_df = feat_df.iloc[:train_end]
            test_df  = feat_df.iloc[train_end:test_end]

            X_tr = train_df[feat_cols].values
            y_tr = train_df[tgt_col].values

            # Quick fit (no Optuna in backtest by default)
            # Disable LSTM when data chunks are too small for sequence modelling
            LSTM_SEQ_LEN = 30
            use_lstm = len(X_tr) >= LSTM_SEQ_LEN + 10

            model = EnsembleModel()
            if not use_lstm:
                model.weights = {"xgb": 0.5, "lgbm": 0.5, "lstm": 0.0}
            else:
                model.lstm.params["epochs"] = 15  # fast LSTM for backtest

            try:
                if use_lstm:
                    model.fit(X_tr, y_tr)
                else:
                    model.xgb.fit(X_tr, y_tr)
                    model.lgbm.fit(X_tr, y_tr)
            except Exception as e:
                logger.warning(f"  Walk-forward fit failed at step {step}: {e} — using XGB+LGBM only")
                try:
                    model.xgb.fit(X_tr, y_tr)
                    model.lgbm.fit(X_tr, y_tr)
                    model.weights = {"xgb": 0.5, "lgbm": 0.5, "lstm": 0.0}
                except Exception as e2:
                    logger.warning(f"  Skipping step {step}: {e2}")
                    step += 1
                    continue

            X_te = test_df[feat_cols].values
            proba = model.predict_proba(X_te)
            valid = ~np.isnan(proba)

            for i, (is_valid, p) in enumerate(zip(valid, proba)):
                if not is_valid:
                    continue
                idx = train_end + i
                if idx >= len(feat_df) - self.horizon:
                    break

                row = feat_df.iloc[idx]
                actual_dir = int(feat_df[tgt_col].iloc[idx])
                actual_ret = float(feat_df[ret_col].iloc[idx]) if ret_col in feat_df.columns else 0.0

                predictions.append({
                    "date": str(feat_df.index[idx].date()),
                    "proba": p,
                    "pred": int(p >= 0.5),
                    "actual": actual_dir,
                    "ret": actual_ret,
                })

                # Simple trading simulation
                if p >= self.confidence_threshold:
                    trade_capital = capital * self.position_size
                    pnl = trade_capital * actual_ret
                    capital += pnl
                    trades.append({
                        "date": str(feat_df.index[idx].date()),
                        "direction": "LONG" if p >= 0.5 else "SHORT",
                        "confidence": p,
                        "pnl": pnl,
                        "capital_after": capital,
                    })

                portfolio_values.append(capital)

            step += 1

        if not predictions:
            return {"error": "No predictions generated"}

        pred_df = pd.DataFrame(predictions)
        auc     = roc_auc_score(pred_df["actual"], pred_df["proba"])
        acc     = accuracy_score(pred_df["actual"], pred_df["pred"])

        # Strategy vs Buy-and-Hold
        bh_return = (ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[0] - 1)
        strat_return = (capital - self.initial_capital) / self.initial_capital

        # Sharpe ratio (rough daily)
        daily_rets = pd.Series(portfolio_values).pct_change().dropna()
        sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

        # Max drawdown
        pv = pd.Series(portfolio_values)
        peak = pv.cummax()
        drawdown = (pv - peak) / peak
        max_dd = float(drawdown.min())

        win_rate = float(pred_df[pred_df["proba"] >= self.confidence_threshold]["actual"].mean()) \
                   if len(pred_df[pred_df["proba"] >= self.confidence_threshold]) > 0 else 0

        return {
            "symbol": sym,
            "n_predictions": len(predictions),
            "n_trades": len(trades),
            "auc": auc,
            "accuracy": acc,
            "win_rate": win_rate,
            "strategy_return_pct": strat_return * 100,
            "buy_hold_return_pct": bh_return * 100,
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": max_dd * 100,
            "final_capital": capital,
        }

    def _format_report(self, results: dict) -> str:
        lines = [
            "=" * 70,
            " VIETNAM STOCK AI — WALK-FORWARD BACKTEST REPORT",
            "=" * 70,
            f"{'Symbol':<8} {'AUC':>6} {'Acc':>6} {'WinR':>6} {'Strat%':>8} {'B&H%':>8} {'Sharpe':>7} {'MaxDD%':>8}",
            "-" * 70,
        ]
        for sym, r in results.items():
            if "error" in r:
                lines.append(f"{sym:<8}  ERROR: {r['error']}")
                continue
            lines.append(
                f"{sym:<8} {r['auc']:>6.3f} {r['accuracy']:>6.3f} {r['win_rate']:>6.3f} "
                f"{r['strategy_return_pct']:>+8.1f} {r['buy_hold_return_pct']:>+8.1f} "
                f"{r['sharpe_ratio']:>7.2f} {r['max_drawdown_pct']:>8.1f}"
            )
        lines.append("=" * 70)
        report = "\n".join(lines)

        # Save JSON
        out = Path(config.RESULTS_DIR) / f"backtest_{datetime.now().strftime('%Y%m%d')}.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return report
