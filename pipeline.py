"""
pipeline.py — End-to-end training and prediction pipelines.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import config
from data_fetcher import fetch_multiple, get_vnindex
from features import build_features, get_feature_cols
from models import XGBModel, LGBMModel, LSTMModel, EnsembleModel
from tuner import tune_xgb, tune_lgbm, tune_lstm, tune_ensemble_weights

logger = logging.getLogger("pipeline")


def _time_split(df: pd.DataFrame, test_ratio: float = 0.15, val_ratio: float = 0.15):
    n = len(df)

    # For very small datasets guarantee at least 2 rows per split
    min_split = 2
    n_test = max(int(n * test_ratio), min_split)
    n_val  = max(int(n * val_ratio),  min_split)

    # If data is so small that train would be empty, shrink test/val to 1 row each
    if n - n_test - n_val < min_split:
        n_test = max(1, n // 3)
        n_val  = max(1, n // 3)

    n_tr = n - n_test - n_val
    # Final safety: never allow empty train
    if n_tr < 1:
        n_tr, n_val, n_test = max(1, n - 2), 1, 1

    train_df = df.iloc[:n_tr]
    val_df   = df.iloc[n_tr: n_tr + n_val]
    test_df  = df.iloc[n_tr + n_val:]
    return train_df, val_df, test_df


def _evaluate(model, X, y_true, prefix="") -> dict:
    prob  = model.predict_proba(X)
    valid = ~np.isnan(prob)
    prob  = prob[valid]; y_v = np.asarray(y_true)[valid]
    pred  = (prob >= 0.5).astype(int)
    acc   = accuracy_score(y_v, pred) if len(y_v) > 0 else 0.0
    try:
        auc = roc_auc_score(y_v, prob) if len(set(y_v)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    logger.info(f"{prefix}  AUC={auc:.4f}  Acc={acc:.4f}" if not np.isnan(auc)
                else f"{prefix}  AUC=N/A (single class in test)  Acc={acc:.4f}")
    return {"auc": auc, "accuracy": acc}


# ══════════════════════════════════════════════════════════════════════════════
# Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TrainingPipeline:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        horizon: int = config.PREDICTION_HORIZON,
        tune: bool = True,
    ):
        self.symbols = symbols or config.DEFAULT_SYMBOLS
        self.horizon = horizon
        self.tune    = tune

    def run(self):
        logger.info(f"=== Training Pipeline | horizon={self.horizon}d | {len(self.symbols)} symbols ===")
        vnindex = get_vnindex()
        data    = fetch_multiple(self.symbols)
        summary = {}

        for sym in self.symbols:
            if sym not in data:
                logger.warning(f"Skipping {sym}: no data")
                continue
            try:
                metrics = self._train_symbol(sym, data[sym], vnindex)
                summary[sym] = metrics
            except Exception as e:
                logger.error(f"Training failed for {sym}: {e}", exc_info=True)

        self._save_summary(summary)
        return summary

    def _train_symbol(self, sym: str, ohlcv: pd.DataFrame, vnindex) -> dict:
        logger.info(f"--- {sym} -----------------------------------")

        # ── Optional: sentiment features via Claude LLM ────────────────────
        sentiment_feat = None
        import os
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from sentiment import load_or_build_sentiment
                dummy_index = pd.DatetimeIndex([])
                sent_map = load_or_build_sentiment(
                    [sym], dummy_index, api_key=os.environ["ANTHROPIC_API_KEY"]
                )
                sentiment_feat = sent_map.get(sym)
            except Exception as e:
                logger.debug(f"Sentiment skipped for {sym}: {e}")

        feat_df = build_features(
            ohlcv, vnindex,
            symbol=sym,
            sentiment_features=sentiment_feat,
            use_vol_adjusted_labels=True,
        )
        feat_cols = get_feature_cols(feat_df, self.horizon)
        tgt_col   = f"target_dir_{self.horizon}d"

        # ── Minimum data guard ─────────────────────────────────────────────
        MIN_ROWS = 10
        if len(feat_df) < MIN_ROWS:
            raise ValueError(
                f"{sym} has only {len(feat_df)} usable rows (minimum: {MIN_ROWS})."
            )

        if tgt_col not in feat_df.columns:
            raise KeyError(f"Target column {tgt_col} missing")

        # ── Vol-adjusted labels may have NaN in dead zone — drop those rows ─
        feat_df_clean = feat_df.dropna(subset=[tgt_col])
        if len(feat_df_clean) < MIN_ROWS:
            logger.warning(
                f"{sym}: only {len(feat_df_clean)} clean-label rows after vol-adjustment. "
                "Falling back to simple binary labels."
            )
            fwd_ret = ohlcv["close"].pct_change(self.horizon).shift(-self.horizon)
            feat_df[tgt_col] = (fwd_ret > 0).astype(int).reindex(feat_df.index)
            feat_df_clean = feat_df.dropna(subset=[tgt_col])
        feat_df = feat_df_clean

        train_df, val_df, test_df = _time_split(feat_df)
        X_tr = train_df[feat_cols].values;  y_tr = train_df[tgt_col].values.astype(int)
        X_val= val_df[feat_cols].values;    y_val= val_df[tgt_col].values.astype(int)
        X_te = test_df[feat_cols].values;   y_te = test_df[tgt_col].values.astype(int)

        for split_name, X_s, y_s in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]:
            if len(X_s) == 0:
                raise ValueError(f"{sym}: {split_name} split is empty.")
            if len(set(y_s)) < 2:
                logger.warning(f"{sym}: {split_name} has only one class — AUC may be unreliable.")

        logger.info(f"  Splits: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

        # ── Feature selection: keep top-40 most predictive features ────────
        from target_engineering import select_top_features
        if len(feat_cols) > 40 and len(X_tr) >= 50:
            feat_cols = select_top_features(X_tr, y_tr, feat_cols, top_n=40)
            feat_idx  = [get_feature_cols(feat_df, self.horizon).index(f) for f in feat_cols]
            X_tr  = X_tr[:, feat_idx]
            X_val = X_val[:, feat_idx]
            X_te  = X_te[:, feat_idx]

        # ── Tune or use defaults ───────────────────────────────────────────
        if self.tune:
            xgb_params  = tune_xgb(X_tr, y_tr, X_val, y_val)
            lgbm_params = tune_lgbm(X_tr, y_tr, X_val, y_val)
            lstm_params = tune_lstm(X_tr, y_tr, X_val, y_val)
        else:
            xgb_params = lgbm_params = lstm_params = None

        # ── Final fit on train+val ─────────────────────────────────────────
        X_full = np.concatenate([X_tr, X_val])
        y_full = np.concatenate([y_tr, y_val])

        ensemble = EnsembleModel()
        if xgb_params:
            ensemble.xgb  = XGBModel(xgb_params)
        if lgbm_params:
            ensemble.lgbm = LGBMModel(lgbm_params)
        if lstm_params:
            ensemble.lstm = LSTMModel(params=lstm_params)
        ensemble.models = {"xgb": ensemble.xgb, "lgbm": ensemble.lgbm, "lstm": ensemble.lstm}

        ensemble.fit(X_tr, y_tr, X_val, y_val)

        # ── Tune ensemble weights on validation set ────────────────────────
        if self.tune:
            xgb_val  = ensemble.xgb.predict_proba(X_val)
            lgbm_val = ensemble.lgbm.predict_proba(X_val)
            lstm_val = ensemble.lstm.predict_proba(X_val)
            ensemble.weights = tune_ensemble_weights(xgb_val, lgbm_val, lstm_val, y_val)

        # ── Evaluate on test ───────────────────────────────────────────────
        metrics = _evaluate(ensemble, X_te, y_te, prefix=f"  TEST {sym}")
        metrics["trained_at"] = datetime.now().isoformat()
        metrics["n_train"]    = int(len(X_tr))
        metrics["n_test"]     = int(len(X_te))
        metrics["features"]   = feat_cols
        metrics["weights"]    = ensemble.weights

        # ── Save models ────────────────────────────────────────────────────
        ensemble.save(sym, self.horizon)

        # Save feature list for inference
        with open(f"{config.MODEL_DIR}/{sym}_h{self.horizon}_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "symbol": sym, "horizon": self.horizon,
                "features": feat_cols, "weights": ensemble.weights,
                "trained_at": metrics["trained_at"],
                "auc": metrics["auc"], "accuracy": metrics["accuracy"],
            }, f, indent=2)

        return metrics

    def _save_summary(self, summary: dict):
        out = Path(config.RESULTS_DIR) / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Prediction Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class PredictionPipeline:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        horizon: int = config.PREDICTION_HORIZON,
    ):
        self.symbols = symbols or config.DEFAULT_SYMBOLS
        self.horizon = horizon

    def run(self) -> Dict[str, dict]:
        try:
            vnindex = get_vnindex()
        except Exception:
            vnindex = None

        data    = fetch_multiple(self.symbols)
        results = {}

        for sym in self.symbols:
            if sym not in data:
                continue
            try:
                pred = self._predict_symbol(sym, data[sym], vnindex)
                results[sym] = pred
            except Exception as e:
                logger.error(f"Prediction failed for {sym}: {e}")

        self._save_predictions(results)
        return results

    def _predict_symbol(self, sym: str, ohlcv: pd.DataFrame, vnindex) -> dict:
        meta_path = Path(f"{config.MODEL_DIR}/{sym}_h{self.horizon}_meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"No trained model for {sym}. Run --mode train first.")

        with open(meta_path) as f:
            meta = json.load(f)

        feat_df = build_features(ohlcv, vnindex)
        feat_cols = meta["features"]
        X = feat_df[feat_cols].values

        ensemble = EnsembleModel.load(sym, self.horizon)
        ensemble.weights = meta["weights"]

        proba = ensemble.predict_proba(X)
        latest_proba = float(proba[~np.isnan(proba)][-1])
        direction    = 1 if latest_proba >= 0.5 else 0

        last_close   = float(ohlcv["close"].iloc[-1])
        # Expected return based on historical win-rate calibration (simplified)
        exp_ret_up   = float(feat_df[f"target_ret_{self.horizon}d"].clip(lower=0).mean()) if f"target_ret_{self.horizon}d" in feat_df.columns else 0.015
        exp_ret_down = float(feat_df[f"target_ret_{self.horizon}d"].clip(upper=0).mean()) if f"target_ret_{self.horizon}d" in feat_df.columns else -0.015

        exp_ret    = exp_ret_up * latest_proba + exp_ret_down * (1 - latest_proba)
        target_px  = last_close * (1 + exp_ret)

        return {
            "symbol":       sym,
            "date":         str(ohlcv.index[-1].date()),
            "last_close":   last_close,
            "confidence":   latest_proba,
            "direction":    direction,
            "return_pct":   exp_ret * 100,
            "target_price": target_px,
            "horizon_days": self.horizon,
            "model_auc":    meta.get("auc", 0),
            "trained_at":   meta.get("trained_at", ""),
        }

    def _save_predictions(self, results: dict):
        out = Path(config.RESULTS_DIR) / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Predictions saved to {out}")
