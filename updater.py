"""
updater.py — Incremental model update with today's (or recent) data.

Purpose:
  After the market closes each day, this ingests the new price bar and
  refits the ensemble models WITHOUT running Optuna from scratch.

  The existing hyperparameters (already tuned) are reused.
  Only the model weights are updated on the expanded dataset.

  This is ~30x faster than a full retrain:
    Full retrain (with Optuna):  30-60 min per symbol
    Incremental update:           15-30 seconds per symbol

Strategy per model:
  XGBoost   — refit from scratch on full data using saved best_params
              (XGB is fast enough; refitting takes ~1-2s)
  LightGBM  — same: refit on full data with saved params
  LSTM      — warm-start: continue training for N extra epochs on new data
              (avoids full retrain; preserves learned weights)
  Transformer — fine-tune for 5 epochs on the latest 60-day window

When to use:
  --mode update   after each market close (scheduled automatically by daemon)
  --mode train    weekly or when you want full Optuna re-optimisation

Usage:
  python main.py --mode update
  python main.py --mode update --symbols VNM HPG FPT
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from data_fetcher import fetch_multiple, get_vnindex, fetch_ohlcv
from features import build_features, get_feature_cols
from models import XGBModel, LGBMModel, LSTMModel, EnsembleModel

logger = logging.getLogger("updater")


def _load_meta(sym: str, horizon: int) -> Optional[dict]:
    path = Path(f"{config.MODEL_DIR}/{sym}_h{horizon}_meta.json")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_meta(sym: str, horizon: int, meta: dict):
    path = Path(f"{config.MODEL_DIR}/{sym}_h{horizon}_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


class IncrementalUpdater:
    """
    Updates trained models with the latest market data.
    Reuses existing hyperparameters — no Optuna needed.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        horizon: int = config.PREDICTION_HORIZON,
        lstm_warmup_epochs: int = 5,
        transformer_finetune_epochs: int = 5,
    ):
        self.symbols = symbols or config.DEFAULT_SYMBOLS
        self.horizon = horizon
        self.lstm_warmup_epochs = lstm_warmup_epochs
        self.transformer_finetune_epochs = transformer_finetune_epochs

    def run(self) -> Dict[str, dict]:
        logger.info(f"=== Incremental Update | {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
        logger.info(f"Symbols: {', '.join(self.symbols)} | Horizon: {self.horizon}d")
        logger.info("Using saved hyperparameters — no Optuna (fast update)")

        # Invalidate cache so we get today's close
        for sym in self.symbols:
            cache_path = Path(config.DATA_CACHE_DIR) / f"{sym}.pkl"
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"  Cleared cache for {sym}")

        vnindex = get_vnindex()
        data    = fetch_multiple(self.symbols)
        summary = {}

        for sym in self.symbols:
            if sym not in data:
                logger.warning(f"  {sym}: no data fetched — skipping")
                continue
            meta = _load_meta(sym, self.horizon)
            if meta is None:
                logger.warning(f"  {sym}: no trained model found — run --mode train first")
                continue
            try:
                result = self._update_symbol(sym, data[sym], vnindex, meta)
                summary[sym] = result
                logger.info(
                    f"  {sym}: updated | new_rows={result['new_rows']} | "
                    f"total_rows={result['total_rows']} | "
                    f"val_auc={result.get('val_auc', 0):.4f}"
                )
            except Exception as e:
                logger.error(f"  {sym}: update failed — {e}", exc_info=True)
                summary[sym] = {"error": str(e)}

        # Save update summary
        out = Path(config.RESULTS_DIR) / f"update_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Update summary saved to {out}")
        return summary

    def _update_symbol(self, sym: str, ohlcv: pd.DataFrame, vnindex, meta: dict) -> dict:
        logger.info(f"  Updating {sym}...")

        # Build feature matrix with all available data (including today)
        try:
            feat_df = build_features(
                ohlcv, vnindex,
                symbol=sym,
                use_vol_adjusted_labels=True,
            )
        except TypeError:
            feat_df = build_features(ohlcv, vnindex)

        tgt_col   = f"target_dir_{self.horizon}d"
        feat_cols = meta.get("features", [])

        # Filter to only the features this model was trained on
        available = set(feat_df.columns)
        feat_cols = [f for f in feat_cols if f in available]
        if not feat_cols:
            raise ValueError(f"No matching features found for {sym}")

        # Drop rows where target is NaN
        feat_df = feat_df.dropna(subset=[tgt_col])

        if len(feat_df) < 20:
            raise ValueError(f"Only {len(feat_df)} rows after cleaning — not enough to update")

        X = feat_df[feat_cols].values
        y = feat_df[tgt_col].values.astype(int)

        # Use last 15% as validation to check model quality after update
        n_val = max(int(len(X) * 0.15), 10)
        X_tr, y_tr = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        logger.info(f"    Data: total={len(X)}, train={len(X_tr)}, val={len(X_val)}")

        # ── Load existing models ──────────────────────────────────────────────
        try:
            ensemble = EnsembleModel.load(sym, self.horizon)
        except Exception as e:
            raise ValueError(f"Could not load existing model: {e}")

        # ── XGBoost — refit on full data with saved params ────────────────────
        logger.info(f"    Refitting XGBoost...")
        xgb_params = self._extract_xgb_params(ensemble.xgb)
        new_xgb = XGBModel(xgb_params)
        new_xgb.fit(X_tr, y_tr, X_val, y_val)

        # ── LightGBM — refit on full data with saved params ───────────────────
        logger.info(f"    Refitting LightGBM...")
        lgbm_params = self._extract_lgbm_params(ensemble.lgbm)
        new_lgbm = LGBMModel(lgbm_params)
        new_lgbm.fit(X_tr, y_tr, X_val, y_val)

        # ── LSTM — warm-start: a few more epochs on full data ─────────────────
        logger.info(f"    Warm-starting LSTM ({self.lstm_warmup_epochs} epochs)...")
        try:
            lstm_params = dict(ensemble.lstm.params)
            lstm_params["epochs"] = self.lstm_warmup_epochs
            new_lstm = LSTMModel(params=lstm_params)
            # Copy existing weights before continuing training
            new_lstm.scaler = ensemble.lstm.scaler
            new_lstm.model  = ensemble.lstm.model
            new_lstm.fit(X_tr, y_tr, X_val, y_val)
        except Exception as e:
            logger.warning(f"    LSTM warm-start failed ({e}) — keeping existing weights")
            new_lstm = ensemble.lstm

        # ── Transformer — fine-tune on latest window ──────────────────────────
        logger.info(f"    Fine-tuning transformer ({self.transformer_finetune_epochs} epochs)...")
        try:
            from transformer_pipeline import TransformerPipeline
            tp = TransformerPipeline(horizon=self.horizon)
            tp._pretrained_model = tp._load_meta   # loads pretrained if available
            tp.finetune_symbol(sym, ohlcv, epochs=self.transformer_finetune_epochs)
        except Exception as e:
            logger.debug(f"    Transformer fine-tune skipped: {e}")

        # ── Rebuild ensemble with updated models ──────────────────────────────
        ensemble.xgb  = new_xgb
        ensemble.lgbm = new_lgbm
        ensemble.lstm = new_lstm
        ensemble.models = {"xgb": new_xgb, "lgbm": new_lgbm, "lstm": new_lstm}
        ensemble.weights = meta.get("weights", config.ENSEMBLE_WEIGHTS)

        # ── Validate ──────────────────────────────────────────────────────────
        from sklearn.metrics import roc_auc_score, accuracy_score
        proba = ensemble.predict_proba(X_val)
        valid = ~np.isnan(proba)
        val_auc = float(roc_auc_score(y_val[valid], proba[valid])) \
                  if valid.sum() > 0 and len(set(y_val[valid])) > 1 else 0.5
        val_acc = float(accuracy_score(y_val[valid], (proba[valid] >= 0.5).astype(int))) \
                  if valid.sum() > 0 else 0.5
        logger.info(f"    Post-update validation: AUC={val_auc:.4f}  Acc={val_acc:.4f}")

        # ── Save updated models + meta ────────────────────────────────────────
        ensemble.save(sym, self.horizon)
        meta.update({
            "updated_at":  datetime.now().isoformat(),
            "total_rows":  int(len(X)),
            "last_close":  float(ohlcv["close"].iloc[-1]),
            "last_date":   str(ohlcv.index[-1].date()),
            "val_auc_after_update": val_auc,
        })
        _save_meta(sym, self.horizon, meta)

        return {
            "new_rows":   int(len(X)) - meta.get("n_train", 0),
            "total_rows": int(len(X)),
            "val_auc":    val_auc,
            "val_acc":    val_acc,
            "last_date":  str(ohlcv.index[-1].date()),
        }

    def _extract_xgb_params(self, xgb_model: XGBModel) -> dict:
        """Extract fitted XGBoost hyperparameters."""
        if xgb_model.model is None:
            return xgb_model.params
        try:
            p = xgb_model.model.get_params()
            p.pop("use_label_encoder", None)
            return p
        except Exception:
            return xgb_model.params

    def _extract_lgbm_params(self, lgbm_model: LGBMModel) -> dict:
        """Extract fitted LightGBM hyperparameters."""
        if lgbm_model.model is None:
            return lgbm_model.params
        try:
            return lgbm_model.model.get_params()
        except Exception:
            return lgbm_model.params
