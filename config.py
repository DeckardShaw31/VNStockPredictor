"""
config.py — Central configuration for the Vietnam Stock AI System
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pytz

# ── Market Config ─────────────────────────────────────────────────────────────
VIETNAM_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

# HOSE market hours: 9:00–11:30, 13:00–14:45 ICT
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MINUTE = 0
MARKET_CLOSE_HOUR  = 14
MARKET_CLOSE_MINUTE = 45

# Retrain triggers: after market close + 15-min buffer
RETRAIN_HOUR   = 15
RETRAIN_MINUTE = 0

# ── Default Stock Universe ─────────────────────────────────────────────────────
DEFAULT_SYMBOLS: List[str] = [
    "VNM",   # Vinamilk         – Consumer
    "VIC",   # Vingroup         – Conglomerate
    "HPG",   # Hoa Phat Group   – Steel
    "VCB",   # Vietcombank      – Banking
    "FPT",   # FPT Corp         – Technology
    "MWG",   # Mobile World     – Retail
    "MSN",   # Masan Group      – Consumer
    "TCB",   # Techcombank      – Banking
    "SSI",   # SSI Securities   – Finance
    "GAS",   # PV Gas           – Energy
    "PLX",   # Petrolimex       – Energy
    "SAB",   # Sabeco           – Beverages
    "VHM",   # Vinhomes         – Real Estate
    "HPA",   # HPA              – New listing (adaptive short-history training)
    "SHS",   # SHS Securities   – Finance
    "VGI",   # Viettel Global   – Telecom

]

# ── Data Config ────────────────────────────────────────────────────────────────
LOOKBACK_DAYS       = 1300    # ~4 years of history for training
FEATURE_WINDOW      = 60     # rolling window for sequence models (LSTM)
CACHE_TTL_HOURS     = 6      # hours before re-fetching price data

# ── Model Config ───────────────────────────────────────────────────────────────
PREDICTION_HORIZON  = 5      # days ahead to predict
TEST_SPLIT_RATIO    = 0.15   # fraction of data held out for test
VAL_SPLIT_RATIO     = 0.15   # fraction of training data used for validation
RANDOM_SEED         = 42

# ── LSTM Hyperparameter Search Space ──────────────────────────────────────────
LSTM_SEARCH_SPACE = {
    "units_1":        (32, 256),
    "units_2":        (16, 128),
    "dropout":        (0.1, 0.5),
    "learning_rate":  (1e-4, 1e-2),
    "batch_size":     [16, 32, 64],
    "epochs":         (20, 80),
}

# ── XGBoost Hyperparameter Search Space ───────────────────────────────────────
XGB_SEARCH_SPACE = {
    "n_estimators":   (100, 800),
    "max_depth":      (3, 10),
    "learning_rate":  (0.01, 0.3),
    "subsample":      (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "gamma":          (0.0, 1.0),
    "reg_alpha":      (0.0, 2.0),
    "reg_lambda":     (0.5, 5.0),
}

# ── LightGBM Hyperparameter Search Space ──────────────────────────────────────
LGBM_SEARCH_SPACE = {
    "n_estimators":     (100, 800),
    "num_leaves":       (20, 150),
    "learning_rate":    (0.01, 0.3),
    "feature_fraction": (0.6, 1.0),
    "bagging_fraction": (0.6, 1.0),
    "bagging_freq":     (1, 7),
    "min_child_samples":(5, 50),
    "reg_alpha":        (0.0, 2.0),
    "reg_lambda":       (0.0, 2.0),
}

# ── Optuna Tuning Config ───────────────────────────────────────────────────────
OPTUNA_TRIALS_XGB  = 60
OPTUNA_TRIALS_LGBM = 60
OPTUNA_TRIALS_LSTM = 30     # LSTM is slow; keep this LOW (30 is plenty)
OPTUNA_TIMEOUT_SEC = 1800   # 30 minutes max per model — hard ceiling

# ── Ensemble Config ────────────────────────────────────────────────────────────
# Final prediction = weighted average of model probabilities
ENSEMBLE_WEIGHTS = {
    "xgb":  0.35,
    "lgbm": 0.35,
    "lstm": 0.30,
}

# ── File Paths ─────────────────────────────────────────────────────────────────
DATA_CACHE_DIR  = "data/cache"
MODEL_DIR       = "models"
RESULTS_DIR     = "results"
LOG_DIR         = "logs"
