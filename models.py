"""
models.py — XGBoost, LightGBM and LSTM wrappers with a unified API.

Each model exposes:
  .fit(X_train, y_train, X_val, y_val)
  .predict_proba(X)  -> array of shape (n,) in [0,1]
  .save(path)
  .load(path)        (classmethod)
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Silence TensorFlow oneDNN & retracing noise before any TF import
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # suppress C++ INFO logs

logger = logging.getLogger("models")


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost Classifier
# ══════════════════════════════════════════════════════════════════════════════

class XGBModel:
    name = "xgb"

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from xgboost import XGBClassifier
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model = XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "XGBModel":
        obj = cls()
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# LightGBM Classifier
# ══════════════════════════════════════════════════════════════════════════════

class LGBMModel:
    name = "lgbm"

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {
            "n_estimators": 300,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        eval_set  = [(X_val, y_val)] if X_val is not None else None
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if eval_set else None,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "LGBMModel":
        obj = cls()
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# LSTM Classifier  (TensorFlow / Keras)
# ══════════════════════════════════════════════════════════════════════════════

class LSTMModel:
    name = "lstm"

    def __init__(
        self,
        seq_len: int = 30,
        params: Optional[dict] = None,
    ):
        self.seq_len = seq_len
        self.params  = params or {
            "units_1": 64,
            "units_2": 32,
            "dropout": 0.2,
            "learning_rate": 5e-4,
            "batch_size": 32,
            "epochs": 40,
        }
        self.model   = None
        self.scaler  = None   # sklearn StandardScaler fitted on training data

    # ── Data prep ──────────────────────────────────────────────────────────

    def _make_sequences(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Roll X into overlapping windows of length seq_len.
        Returns None if X is too short to form even one sequence."""
        n, f = X.shape
        if n < self.seq_len:
            return None
        seqs = np.zeros((n - self.seq_len + 1, self.seq_len, f), dtype=np.float32)
        for i in range(seqs.shape[0]):
            seqs[i] = X[i: i + self.seq_len]
        return seqs

    def _align_labels(self, y: np.ndarray) -> np.ndarray:
        """Labels aligned to the last timestep of each sequence."""
        return y[self.seq_len - 1:]

    # ── Build ───────────────────────────────────────────────────────────────

    def _build(self, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers, Model, Input

        p   = self.params
        inp = Input(shape=(self.seq_len, n_features))
        x   = layers.LSTM(p["units_1"], return_sequences=True)(inp)
        x   = layers.Dropout(p["dropout"])(x)
        x   = layers.LSTM(p["units_2"], return_sequences=False)(x)
        x   = layers.Dropout(p["dropout"])(x)
        x   = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(p["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
            jit_compile=False,   # avoids repeated tf.function retracing on CPU
        )
        return model

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.preprocessing import StandardScaler
        import tensorflow as tf

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        X_seq = self._make_sequences(X_scaled)
        y_seq = self._align_labels(np.asarray(y_train))

        val_data = None
        if X_val is not None:
            Xv = self.scaler.transform(X_val)
            Xv_seq = self._make_sequences(Xv)
            yv_seq = self._align_labels(np.asarray(y_val))
            val_data = (Xv_seq, yv_seq)

        self.model = self._build(X_train.shape[1])
        p = self.params
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_data else "loss",
                patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if val_data else "loss",
                factor=0.5, patience=5, verbose=0
            ),
        ]
        self.model.fit(
            X_seq, y_seq,
            validation_data=val_data,
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )
        return self

    # ── Predict ─────────────────────────────────────────────────────────────

    def predict_proba(self, X) -> np.ndarray:
        """Returns probabilities aligned to input rows.
        Rows before seq_len-1 are NaN (no full sequence available).
        If X has fewer rows than seq_len, returns all-NaN array."""
        all_nan = np.full(len(X), np.nan)
        if len(X) < self.seq_len:
            return all_nan
        X_scaled = self.scaler.transform(X)
        X_seq    = self._make_sequences(X_scaled)
        if X_seq is None:
            return all_nan
        preds = self.model.predict(X_seq, verbose=0).flatten()
        # Pad front with NaN so length matches input
        pad = np.full(self.seq_len - 1, np.nan)
        return np.concatenate([pad, preds])

    # ── Persist ─────────────────────────────────────────────────────────────

    def save(self, path: str):
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(base) + ".keras")
        with open(str(base) + "_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(str(base) + "_params.pkl", "wb") as f:
            pickle.dump({"params": self.params, "seq_len": self.seq_len}, f)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        import tensorflow as tf
        base = Path(path)
        with open(str(base) + "_params.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(seq_len=meta["seq_len"], params=meta["params"])
        obj.model = tf.keras.models.load_model(str(base) + ".keras")
        with open(str(base) + "_scaler.pkl", "rb") as f:
            obj.scaler = pickle.load(f)
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleModel:
    """Weighted average of XGB + LGBM + LSTM probabilities."""

    def __init__(self, weights: Optional[dict] = None):
        from config import ENSEMBLE_WEIGHTS
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.xgb  = XGBModel()
        self.lgbm = LGBMModel()
        self.lstm = LSTMModel()
        self.models = {"xgb": self.xgb, "lgbm": self.lgbm, "lstm": self.lstm}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        for name, m in self.models.items():
            logger.info(f"  Fitting {name}…")
            m.fit(X_train, y_train, X_val, y_val)
        return self

    def predict_proba(self, X) -> np.ndarray:
        preds = {}
        for name, m in self.models.items():
            p = m.predict_proba(X)
            preds[name] = p
        # Weighted average (ignoring NaN from LSTM padding)
        total_w = sum(self.weights.values())
        combined = np.zeros(len(X))
        weight_used = np.zeros(len(X))
        for name, p in preds.items():
            w = self.weights.get(name, 0.33)
            valid = ~np.isnan(p)
            combined[valid] += w * p[valid]
            weight_used[valid] += w
        combined = np.where(weight_used > 0, combined / weight_used, 0.5)
        return combined

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, symbol: str, horizon: int, base_dir: str = "models"):
        self.xgb.save(f"{base_dir}/{symbol}_h{horizon}_xgb.pkl")
        self.lgbm.save(f"{base_dir}/{symbol}_h{horizon}_lgbm.pkl")
        self.lstm.save(f"{base_dir}/{symbol}_h{horizon}_lstm")
        logger.info(f"Saved ensemble for {symbol} h={horizon}")

    @classmethod
    def load(cls, symbol: str, horizon: int, base_dir: str = "models") -> "EnsembleModel":
        obj = cls()
        obj.xgb  = XGBModel.load(f"{base_dir}/{symbol}_h{horizon}_xgb.pkl")
        obj.lgbm = LGBMModel.load(f"{base_dir}/{symbol}_h{horizon}_lgbm.pkl")
        try:
            obj.lstm = LSTMModel.load(f"{base_dir}/{symbol}_h{horizon}_lstm")
        except Exception:
            logger.warning(f"LSTM model not found for {symbol}; using XGB+LGBM only")
            obj.weights = {"xgb": 0.5, "lgbm": 0.5, "lstm": 0.0}
        obj.models = {"xgb": obj.xgb, "lgbm": obj.lgbm, "lstm": obj.lstm}
        return obj
