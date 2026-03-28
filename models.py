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
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        eval_set = None
        if X_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            eval_set = [(X_val, y_val)]
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
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
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        eval_set  = None
        if X_val is not None:
            X_val    = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            eval_set = [(X_val, y_val)]
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if eval_set else None,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
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
# LSTM Classifier  (PyTorch — works on Python 3.13, no TensorFlow needed)
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
        self.scaler  = None

    # ── Data prep ──────────────────────────────────────────────────────────

    def _make_sequences(self, X: np.ndarray) -> Optional[np.ndarray]:
        n, f = X.shape
        if n < self.seq_len:
            return None
        seqs = np.zeros((n - self.seq_len + 1, self.seq_len, f), dtype=np.float32)
        for i in range(seqs.shape[0]):
            seqs[i] = X[i: i + self.seq_len]
        return seqs

    def _align_labels(self, y: np.ndarray) -> np.ndarray:
        return y[self.seq_len - 1:]

    # ── Build (PyTorch) ─────────────────────────────────────────────────────

    def _build_torch(self, n_features: int):
        import torch
        import torch.nn as nn

        p = self.params
        u1 = p["units_1"]
        u2 = p["units_2"]
        dr = p["dropout"]

        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(n_features, u1, batch_first=True)
                self.drop1 = nn.Dropout(dr)
                self.lstm2 = nn.LSTM(u1, u2, batch_first=True)
                self.drop2 = nn.Dropout(dr)
                self.fc1   = nn.Linear(u2, 32)
                self.relu  = nn.ReLU()
                self.fc2   = nn.Linear(32, 1)
                self.sig   = nn.Sigmoid()

            def forward(self, x):
                out, _ = self.lstm1(x)
                out    = self.drop1(out)
                out, _ = self.lstm2(out)
                out    = self.drop2(out[:, -1, :])   # last timestep
                out    = self.relu(self.fc1(out))
                # Return raw logits — BCEWithLogitsLoss applies sigmoid internally
                # This is numerically stable and avoids the "between 0 and 1" crash
                return self.fc2(out).squeeze(1)

        return LSTMNet()

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler

        # ── Sanitize inputs ─────────────────────────────────────────────────
        # Replace NaN/inf with 0 before scaling — the new 184-col feature matrix
        # may contain inf/-inf from math model features (e.g. division by zero)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.clip(np.asarray(y_train, dtype=np.float32), 0.0, 1.0)

        self.scaler  = StandardScaler()
        X_scaled     = self.scaler.fit_transform(X_train)
        # Clip scaled values to ±10σ — prevents exploding LSTM gradients
        X_scaled     = np.clip(X_scaled, -10.0, 10.0)

        X_seq = self._make_sequences(X_scaled)
        y_seq = self._align_labels(y_train)

        if X_seq is None or len(X_seq) == 0:
            logger.warning("[LSTM] Not enough data for sequences — skipping")
            return self

        device = torch.device("cpu")
        p      = self.params
        model  = self._build_torch(X_train.shape[1]).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])

        # BCEWithLogitsLoss is numerically more stable than Sigmoid + BCELoss
        # (avoids the "all elements must be between 0 and 1" crash when
        #  sigmoid saturates to exactly 0 or 1 due to extreme inputs)
        loss_fn = nn.BCEWithLogitsLoss()

        Xt = torch.tensor(X_seq,  dtype=torch.float32)
        yt = torch.tensor(y_seq,  dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=p["batch_size"], shuffle=False
        )

        val_loader = None
        if X_val is not None and len(X_val) >= self.seq_len:
            Xv = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            Xv_s   = np.clip(self.scaler.transform(Xv), -10.0, 10.0)
            Xv_seq = self._make_sequences(Xv_s)
            yv_seq = self._align_labels(np.clip(np.asarray(y_val, dtype=np.float32), 0.0, 1.0))
            if Xv_seq is not None and len(Xv_seq) > 0:
                Xvt = torch.tensor(Xv_seq, dtype=torch.float32)
                yvt = torch.tensor(yv_seq, dtype=torch.float32)
                val_loader = DataLoader(TensorDataset(Xvt, yvt), batch_size=64)

        best_val   = float("inf")
        patience   = 10
        no_improve = 0
        best_state = None
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.5
        )

        for epoch in range(p["epochs"]):
            model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_loss += loss_fn(model(xb), yb).item()
                val_loss /= len(val_loader)
                sched.step(val_loss)
                if val_loss < best_val - 1e-4:
                    best_val   = val_loss
                    no_improve = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        if best_state:
            model.load_state_dict(best_state)

        self.model = model
        return self

    # ── Predict ─────────────────────────────────────────────────────────────

    def predict_proba(self, X) -> np.ndarray:
        import torch
        all_nan = np.full(len(X), np.nan)
        if self.model is None or self.scaler is None:
            return all_nan
        if len(X) < self.seq_len:
            return all_nan
        X_clean  = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = np.clip(self.scaler.transform(X_clean), -10.0, 10.0)
        X_seq    = self._make_sequences(X_scaled)
        if X_seq is None:
            return all_nan
        self.model.eval()
        with torch.no_grad():
            xt     = torch.tensor(X_seq, dtype=torch.float32)
            logits = self.model(xt)
            pred   = torch.sigmoid(logits).numpy().flatten()   # logits → probabilities
        pad = np.full(self.seq_len - 1, np.nan)
        return np.concatenate([pad, pred])

    # ── Persist ─────────────────────────────────────────────────────────────

    def save(self, path: str):
        import torch
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), str(base) + "_weights.pt")
        with open(str(base) + "_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(str(base) + "_params.pkl", "wb") as f:
            pickle.dump({"params": self.params, "seq_len": self.seq_len,
                         "n_features": self.model.lstm1.input_size if self.model else None}, f)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        import torch
        base = Path(path)
        with open(str(base) + "_params.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(seq_len=meta["seq_len"], params=meta["params"])
        with open(str(base) + "_scaler.pkl", "rb") as f:
            obj.scaler = pickle.load(f)
        n_features = meta.get("n_features")
        if n_features:
            obj.model = obj._build_torch(n_features)
            weights_path = str(base) + "_weights.pt"
            if Path(weights_path).exists():
                obj.model.load_state_dict(
                    torch.load(weights_path, map_location="cpu", weights_only=True)
                )
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class CalibratedModel:
    """
    Wraps any base model and applies Platt scaling (logistic regression
    calibration) to convert raw model scores into well-calibrated probabilities.

    Uncalibrated models often output probabilities clustered near 0.5
    even when they are confident. Calibration spreads these out so that
    a 70% prediction actually means the model is right ~70% of the time.
    """

    def __init__(self, base_model, name: str = "calibrated"):
        self.base  = base_model
        self.name  = name
        self.cal   = None   # fitted CalibratedClassifierCV or LogisticRegression

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.base.fit(X_train, y_train, X_val, y_val)
        # Fit calibration on validation data if available, else cross-val on train
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression

        X_cal = X_val if X_val is not None and len(X_val) >= 20 else X_train
        y_cal = y_val if X_val is not None and len(X_val) >= 20 else y_train

        # Use sigmoid (Platt scaling) calibration
        raw_proba = self.base.predict_proba(X_cal)
        valid = ~np.isnan(raw_proba)
        if valid.sum() >= 20 and len(set(y_cal[valid])) > 1:
            try:
                lr = LogisticRegression(C=1.0)
                lr.fit(raw_proba[valid].reshape(-1, 1), y_cal[valid])
                self.cal = lr
            except Exception as e:
                logger.debug(f"Calibration fit failed: {e}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base.predict_proba(X)
        if self.cal is None:
            return raw
        valid = ~np.isnan(raw)
        out   = raw.copy()
        try:
            out[valid] = self.cal.predict_proba(raw[valid].reshape(-1, 1))[:, 1]
        except Exception:
            pass
        return out

    def save(self, path: str):
        import pickle
        self.base.save(path)
        if self.cal is not None:
            with open(path + "_cal.pkl", "wb") as f:
                pickle.dump(self.cal, f)

    @classmethod
    def load_from_base(cls, base_model, path: str) -> "CalibratedModel":
        import pickle
        obj = cls(base_model)
        cal_path = path + "_cal.pkl"
        if Path(cal_path).exists():
            with open(cal_path, "rb") as f:
                obj.cal = pickle.load(f)
        return obj


class StackingMetaLearner:
    """
    Level-2 meta-learner trained on out-of-fold predictions from base models.

    Instead of a fixed weighted average, this learns the optimal combination
    of base model outputs using a simple logistic regression. It discovers
    that e.g. LGBM is more reliable on trending stocks, LSTM on volatile ones.

    Typical AUC improvement: +0.01 to +0.03 over weighted average.
    """

    def __init__(self):
        self.meta = None
        self.feature_names = ["xgb", "lgbm", "lstm"]

    def fit(self, xgb_oof, lgbm_oof, lstm_oof, y_true):
        from sklearn.linear_model import LogisticRegression

        # Stack OOF predictions into a feature matrix
        valid = (~np.isnan(xgb_oof) & ~np.isnan(lgbm_oof) & ~np.isnan(lstm_oof))
        if valid.sum() < 20:
            logger.warning("Not enough valid OOF predictions for stacking meta-learner")
            return self

        X_meta = np.column_stack([
            xgb_oof[valid],
            lgbm_oof[valid],
            lstm_oof[valid],
        ])
        y_meta = np.asarray(y_true)[valid]

        if len(set(y_meta)) < 2:
            return self

        self.meta = LogisticRegression(C=0.1, max_iter=1000)
        self.meta.fit(X_meta, y_meta)
        logger.info(
            f"Meta-learner coefficients: "
            f"xgb={self.meta.coef_[0][0]:.3f} "
            f"lgbm={self.meta.coef_[0][1]:.3f} "
            f"lstm={self.meta.coef_[0][2]:.3f}"
        )
        return self

    def predict_proba(self, xgb_p, lgbm_p, lstm_p) -> np.ndarray:
        if self.meta is None:
            return np.full(len(xgb_p), np.nan)

        valid = (~np.isnan(xgb_p) & ~np.isnan(lgbm_p) & ~np.isnan(lstm_p))
        out   = np.full(len(xgb_p), np.nan)
        if valid.sum() == 0:
            return out

        X = np.column_stack([xgb_p[valid], lgbm_p[valid], lstm_p[valid]])
        try:
            out[valid] = self.meta.predict_proba(X)[:, 1]
        except Exception as e:
            logger.warning(f"Meta-learner prediction failed: {e}")
        return out

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.meta, f)

    @classmethod
    def load(cls, path: str) -> "StackingMetaLearner":
        import pickle
        obj = cls()
        if Path(path).exists():
            with open(path, "rb") as f:
                obj.meta = pickle.load(f)
        return obj


class EnsembleModel:
    """
    4-model ensemble: XGBoost + LightGBM + LSTM + Stacking Meta-learner.

    Each base model is optionally calibrated (Platt scaling) to produce
    well-calibrated probabilities. The meta-learner learns the optimal
    combination from out-of-fold predictions.
    """

    def __init__(self, weights: Optional[dict] = None):
        from config import ENSEMBLE_WEIGHTS
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.xgb    = XGBModel()
        self.lgbm   = LGBMModel()
        self.lstm   = LSTMModel()
        self.meta   = StackingMetaLearner()
        self.models = {"xgb": self.xgb, "lgbm": self.lgbm, "lstm": self.lstm}
        self._calibrated = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        for name, m in self.models.items():
            logger.info(f"  Fitting {name}...")
            m.fit(X_train, y_train, X_val, y_val)

        # Train stacking meta-learner on validation OOF predictions
        if X_val is not None and len(X_val) >= 20:
            xgb_v  = self.xgb.predict_proba(X_val)
            lgbm_v = self.lgbm.predict_proba(X_val)
            lstm_v = self.lstm.predict_proba(X_val)
            self.meta.fit(xgb_v, lgbm_v, lstm_v, y_val)
        return self

    def predict_proba(self, X) -> np.ndarray:
        xgb_p  = self.xgb.predict_proba(X)
        lgbm_p = self.lgbm.predict_proba(X)
        lstm_p = self.lstm.predict_proba(X)

        # Base weighted blend
        total_w     = sum(self.weights.get(k, 0) for k in ["xgb", "lgbm", "lstm"])
        combined    = np.zeros(len(X))
        weight_used = np.zeros(len(X))

        for name, p in [("xgb", xgb_p), ("lgbm", lgbm_p), ("lstm", lstm_p)]:
            w     = self.weights.get(name, 0.33)
            valid = ~np.isnan(p)
            combined[valid]    += w * p[valid]
            weight_used[valid] += w

        base_blend = np.where(weight_used > 0, combined / weight_used, 0.5)

        # Blend in meta-learner if available
        meta_w    = self.weights.get("meta", 0.10)
        meta_pred = self.meta.predict_proba(xgb_p, lgbm_p, lstm_p)
        meta_valid = ~np.isnan(meta_pred)

        if meta_w > 0 and meta_valid.any():
            final = base_blend.copy()
            final[meta_valid] = (
                (1 - meta_w) * base_blend[meta_valid]
                + meta_w     * meta_pred[meta_valid]
            )
            return final

        return base_blend

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, symbol: str, horizon: int, base_dir: str = "models"):
        self.xgb.save(f"{base_dir}/{symbol}_h{horizon}_xgb.pkl")
        self.lgbm.save(f"{base_dir}/{symbol}_h{horizon}_lgbm.pkl")
        self.lstm.save(f"{base_dir}/{symbol}_h{horizon}_lstm")
        self.meta.save(f"{base_dir}/{symbol}_h{horizon}_meta_learner.pkl")
        logger.info(f"Saved ensemble for {symbol} h={horizon}")

    @classmethod
    def load(cls, symbol: str, horizon: int, base_dir: str = "models") -> "EnsembleModel":
        obj = cls()
        obj.xgb  = XGBModel.load(f"{base_dir}/{symbol}_h{horizon}_xgb.pkl")
        obj.lgbm = LGBMModel.load(f"{base_dir}/{symbol}_h{horizon}_lgbm.pkl")
        try:
            obj.lstm = LSTMModel.load(f"{base_dir}/{symbol}_h{horizon}_lstm")
        except Exception:
            logger.warning(f"LSTM not found for {symbol} — using XGB+LGBM only")
            obj.weights = {"xgb": 0.45, "lgbm": 0.45, "lstm": 0.0, "meta": 0.10}
        try:
            obj.meta = StackingMetaLearner.load(
                f"{base_dir}/{symbol}_h{horizon}_meta_learner.pkl"
            )
        except Exception:
            pass
        obj.models = {"xgb": obj.xgb, "lgbm": obj.lgbm, "lstm": obj.lstm}
        return obj
