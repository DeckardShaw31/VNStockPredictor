"""
transformer_pipeline.py — Two-stage training for the Pattern Transformer.

Stage 1 — Pre-training on full market corpus:
  Train on ALL symbols combined (~16 symbols × 1300 days = ~20,000 sequences).
  The model learns general Vietnam stock market pattern grammar.
  Like training GPT on the whole internet before specialising.

Stage 2 — Per-symbol fine-tuning:
  Fine-tune the pre-trained model on each individual symbol's data.
  Transfer learning: the model keeps its pattern knowledge and adapts
  to each stock's personality (VNM behaves differently from SSI).

The fine-tuned per-symbol models then slot into the ensemble alongside
XGBoost and LightGBM as the 4th model component.

Usage:
  from transformer_pipeline import TransformerPipeline
  tp = TransformerPipeline()
  tp.pretrain(data_dict)               # Stage 1
  tp.finetune_symbol("VNM", ohlcv)     # Stage 2
  prob = tp.predict("VNM", ohlcv)      # inference
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from pattern_transformer import PatternTransformerModel, tune_transformer_objective
from pattern_dataset import (
    build_corpus,
    build_symbol_finetune_data,
    time_split_corpus,
    save_corpus,
    load_corpus,
)

logger = logging.getLogger("transformer_pipeline")

PRETRAINED_MODEL_PATH   = Path(config.MODEL_DIR) / "pretrained_transformer"
FINETUNED_MODEL_DIR     = Path(config.MODEL_DIR)
TRANSFORMER_META_PATH   = Path(config.MODEL_DIR) / "transformer_meta.json"

# Training config
PRETRAIN_SEQ_LEN   = 60     # input sequence length (60 trading days = ~3 months)
PRETRAIN_EPOCHS    = 50
PRETRAIN_BATCH     = 64
FINETUNE_EPOCHS    = 20     # fewer epochs for fine-tuning (avoid overfitting)
FINETUNE_BATCH     = 16
OPTUNA_TRIALS      = 20     # transformer Optuna trials (architecture search)


class TransformerPipeline:

    def __init__(
        self,
        seq_len: int = PRETRAIN_SEQ_LEN,
        horizon: int = config.PREDICTION_HORIZON,
    ):
        self.seq_len  = seq_len
        self.horizon  = horizon
        self._pretrained_model: Optional[PatternTransformerModel] = None
        self._finetuned_models: Dict[str, PatternTransformerModel] = {}
        self._meta: dict = {}

    # ── Stage 1: Pre-training ─────────────────────────────────────────────

    def pretrain(
        self,
        data: Dict[str, pd.DataFrame],
        tune: bool = False,
        force_rebuild: bool = False,
    ) -> float:
        """
        Pre-train transformer on the full multi-symbol corpus.
        Returns validation AUC.
        """
        # Try loading existing pre-trained model
        if not force_rebuild and PRETRAINED_MODEL_PATH.with_suffix(".keras").exists():
            logger.info("[transformer] Loading existing pre-trained model...")
            try:
                self._pretrained_model = PatternTransformerModel.load(
                    str(PRETRAINED_MODEL_PATH)
                )
                meta = self._load_meta()
                return meta.get("pretrain_auc", 0.0)
            except Exception as e:
                logger.warning(f"[transformer] Could not load pre-trained model: {e}")

        logger.info("[transformer] === Stage 1: Pre-training on full market corpus ===")

        # Build corpus
        X, y, weights = build_corpus(data, seq_len=self.seq_len, horizon=self.horizon)
        if len(X) == 0:
            logger.error("[transformer] Empty corpus — cannot pre-train")
            return 0.0

        save_corpus(X, y, weights)

        # Time-split
        splits = time_split_corpus(X, y, weights)
        X_tr, y_tr, w_tr = splits[0], splits[1], splits[2]
        X_val, y_val      = splits[3], splits[4]
        X_te, y_te        = splits[6], splits[7]

        logger.info(
            f"[transformer] Corpus split: train={len(X_tr)}, "
            f"val={len(X_val)}, test={len(X_te)}"
        )

        # Optuna architecture search (optional)
        best_params = None
        if tune and len(X_tr) >= 200:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            logger.info(f"[transformer] Optuna tuning ({OPTUNA_TRIALS} trials)...")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda t: tune_transformer_objective(t, X_tr, y_tr, X_val, y_val),
                n_trials=OPTUNA_TRIALS,
                timeout=1200,
                catch=(Exception,),
            )
            best_params = study.best_params
            logger.info(f"[transformer] Best params: {best_params}  AUC={study.best_value:.4f}")

        # Build and train the model
        params = best_params or {
            "embed_dim":    64,
            "num_heads":    4,
            "num_blocks":   3,
            "ff_dim":       128,
            "dropout_rate": 0.1,
            "learning_rate":1e-3,
            "batch_size":   PRETRAIN_BATCH,
            "epochs":       PRETRAIN_EPOCHS,
        }

        self._pretrained_model = PatternTransformerModel(
            seq_len=self.seq_len,
            params=params,
        )
        self._pretrained_model.fit(X_tr, y_tr, X_val, y_val)

        # Evaluate
        from sklearn.metrics import roc_auc_score, accuracy_score
        prob  = self._pretrained_model.predict_proba(X_te)
        valid = ~np.isnan(prob)
        auc   = roc_auc_score(y_te[valid], prob[valid]) if valid.sum() > 0 and len(set(y_te[valid])) > 1 else 0.5
        acc   = accuracy_score(y_te[valid], (prob[valid] >= 0.5).astype(int)) if valid.sum() > 0 else 0.5
        logger.info(f"[transformer] Pre-train TEST: AUC={auc:.4f}  Acc={acc:.4f}")

        # Save
        self._pretrained_model.save(str(PRETRAINED_MODEL_PATH))
        self._save_meta({
            "pretrain_auc": auc,
            "pretrain_acc": acc,
            "pretrained_at": datetime.now().isoformat(),
            "n_train": int(len(X_tr)),
            "params": params,
        })
        logger.info(f"[transformer] Pre-trained model saved to {PRETRAINED_MODEL_PATH}")
        return auc

    # ── Stage 2: Per-symbol fine-tuning ──────────────────────────────────

    def finetune_symbol(
        self,
        sym: str,
        ohlcv: pd.DataFrame,
        epochs: int = FINETUNE_EPOCHS,
    ) -> float:
        """
        Fine-tune the pre-trained model on a single symbol.
        Uses transfer learning — keeps the transformer's learned pattern
        representations and adapts the classification head.
        Returns test AUC.
        """
        if self._pretrained_model is None:
            logger.warning(f"[transformer] No pre-trained model — {sym} will train from scratch")

        logger.info(f"[transformer] Fine-tuning on {sym}...")

        X, y = build_symbol_finetune_data(ohlcv, seq_len=self.seq_len, horizon=self.horizon)
        if len(X) < 20:
            logger.warning(f"[transformer] {sym}: only {len(X)} sequences — skipping fine-tune")
            self._finetuned_models[sym] = self._pretrained_model
            return 0.5

        # Time split
        n      = len(X)
        n_test = max(int(n * 0.15), 5)
        n_val  = max(int(n * 0.15), 5)
        n_tr   = n - n_test - n_val

        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_val, y_val = X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val]
        X_te, y_te   = X[n_tr+n_val:], y[n_tr+n_val:]

        # Clone pre-trained model for fine-tuning (PyTorch-based)
        if self._pretrained_model and self._pretrained_model.model is not None:
            # Transfer pre-trained weights (PyTorch state_dict)
            ft_model = PatternTransformerModel(
                seq_len=self.seq_len,
                params={**self._pretrained_model.params,
                        "epochs": epochs,
                        "batch_size": FINETUNE_BATCH,
                        "learning_rate": 3e-4},   # lower LR for fine-tuning
            )
            # Initialise architecture by doing a dummy forward pass
            ft_model.fit(X_tr[:2], y_tr[:2])
            if ft_model.model is not None:
                try:
                    import torch
                    ft_model.model.load_state_dict(
                        self._pretrained_model.model.state_dict(), strict=False
                    )
                    logger.info(f"[transformer] {sym}: transferred pre-trained weights")
                except Exception as e:
                    logger.debug(f"[transformer] Weight transfer skipped: {e}")
            # Fine-tune with lower LR
            ft_model.fit(X_tr, y_tr, X_val, y_val)
        else:
            # Train from scratch for this symbol
            ft_model = PatternTransformerModel(
                seq_len=self.seq_len,
                params={"embed_dim":64, "num_heads":4, "num_blocks":2, "ff_dim":128,
                        "dropout_rate":0.15, "learning_rate":1e-3,
                        "batch_size":FINETUNE_BATCH, "epochs":epochs},
            )
            ft_model.fit(X_tr, y_tr, X_val, y_val)

        self._finetuned_models[sym] = ft_model

        # Evaluate
        from sklearn.metrics import roc_auc_score, accuracy_score
        prob  = ft_model.predict_proba(X_te)
        valid = ~np.isnan(prob)
        if valid.sum() < 3 or len(set(y_te[valid])) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_te[valid], prob[valid]))
        acc = float(accuracy_score(y_te[valid], (prob[valid] >= 0.5).astype(int))) if valid.sum() > 0 else 0.5
        logger.info(f"[transformer] {sym} fine-tune TEST: AUC={auc:.4f}  Acc={acc:.4f}")

        # Save
        save_path = str(FINETUNED_MODEL_DIR / f"{sym}_h{self.horizon}_transformer")
        ft_model.save(save_path)

        return auc

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, sym: str, ohlcv: pd.DataFrame) -> float:
        """
        Return the probability of an UP move for the most recent sequence.
        Uses the fine-tuned symbol model if available, else pre-trained.
        """
        from price_tokenizer import PriceTokenizer

        model = self._finetuned_models.get(sym) or self._pretrained_model
        if model is None:
            # Try loading from disk
            model = self._load_symbol_model(sym)
        if model is None:
            return 0.5

        tokenizer = PriceTokenizer()
        tokens    = tokenizer.encode(ohlcv)
        if len(tokens) < self.seq_len:
            # Pad with repeating pattern for very short histories
            tokens = tokens + tokens * (self.seq_len // len(tokens) + 1)
        seq   = np.array([tokens[-self.seq_len:]], dtype=np.int32)
        prob  = model.predict_proba(seq)
        return float(prob[0]) if not np.isnan(prob[0]) else 0.5

    def _load_symbol_model(self, sym: str) -> Optional[PatternTransformerModel]:
        """Load fine-tuned model from disk."""
        path = FINETUNED_MODEL_DIR / f"{sym}_h{self.horizon}_transformer_config.pkl"
        if path.exists():
            try:
                model = PatternTransformerModel.load(
                    str(FINETUNED_MODEL_DIR / f"{sym}_h{self.horizon}_transformer")
                )
                self._finetuned_models[sym] = model
                return model
            except Exception as e:
                logger.debug(f"[transformer] Could not load {sym} model: {e}")
        return None

    # ── Meta persistence ──────────────────────────────────────────────────

    def _save_meta(self, data: dict):
        self._meta.update(data)
        TRANSFORMER_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRANSFORMER_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, default=str)

    def _load_meta(self) -> dict:
        if TRANSFORMER_META_PATH.exists():
            with open(TRANSFORMER_META_PATH, encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta
