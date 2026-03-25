"""
config.py — Central configuration for the Vietnam Stock AI System
"""

from typing import Dict, List
import pytz

# ── Market Config ─────────────────────────────────────────────────────────────
VIETNAM_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

MARKET_OPEN_HOUR    = 9
MARKET_OPEN_MINUTE  = 0
MARKET_CLOSE_HOUR   = 14
MARKET_CLOSE_MINUTE = 45

RETRAIN_HOUR   = 15
RETRAIN_MINUTE = 0

# ── Stock Universe ─────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS: List[str] = [
    "VNM",   # Vinamilk         – Consumer
    "VIC",   # Vingroup         – Conglomerate
    "HPG",   # Hoa Phat Group   – Steel
    "VCB",   # Vietcombank      – Banking
    "FPT",   # FPT Corp         – Technology
    "MWG",   # Mobile World     – Retail
    "MSN",   # Masan Group      – Consumer
    "SSI",   # SSI Securities   – Finance
    "GAS",   # PV Gas           – Energy
    "PLX",   # Petrolimex       – Energy
    "SAB",   # Sabeco           – Beverages
    "VHM",   # Vinhomes         – Real Estate
    "HPA",   # HPA              – New listing (adaptive short-history training)
    "SHS",   # SHS Securities   – Finance
    "VGI",   # Viettel Global   – Telecom
    "GEX",   # Gelex            – Conglomerate  
]

# ── Sector Map ─────────────────────────────────────────────────────────────────
# Used for sector-relative features, portfolio sector exposure, and heatmap
SECTOR_MAP: Dict[str, str] = {
    "VNM": "Consumer Staples",        # Vinamilk
    "VIC": "Conglomerate",            # Vingroup
    "HPG": "Materials",               # Hoa Phat (Steel)
    "VCB": "Banking",                 # Vietcombank
    "FPT": "Technology",              # FPT Corp
    "MWG": "Consumer Discretionary",  # Mobile World (Retail)
    "MSN": "Consumer Staples",        # Masan Group
    "SSI": "Finance",                 # SSI Securities
    "GAS": "Energy",                  # PV Gas
    "PLX": "Energy",                  # Petrolimex
    "SAB": "Consumer Staples",        # Sabeco (Beverages)
    "VHM": "Real Estate",             # Vinhomes
    "HPA": "Healthcare",              # HPA
    "SHS": "Finance",                 # SHS Securities
    "VGI": "Telecom",                 # Viettel Global
    "GEX": "Conglomerate",            # Gelex
}

# ── Benchmark Tickers ─────────────────────────────────────────────────────────
VNINDEX_SYMBOL = "VNINDEX"
VN30_SYMBOL    = "VN30"      # Blue-chip index (fear/greed proxy)

# ── Data Config ────────────────────────────────────────────────────────────────
LOOKBACK_DAYS       = 3650   # ~10 years
FEATURE_WINDOW      = 600
CACHE_TTL_HOURS     = 6

# ── Model Config ───────────────────────────────────────────────────────────────
PREDICTION_HORIZON  = 5
TEST_SPLIT_RATIO    = 0.15
VAL_SPLIT_RATIO     = 0.15
RANDOM_SEED         = 42

# Enable probability calibration (Platt scaling) after training
CALIBRATE_PROBABILITIES = True

# Enable stacking meta-learner (uses OOF predictions from base models)
USE_STACKING = True

# ── LSTM Hyperparameter Search Space ──────────────────────────────────────────
LSTM_SEARCH_SPACE = {
    "units_1":       (32, 256),
    "units_2":       (16, 128),
    "dropout":       (0.1, 0.5),
    "learning_rate": (1e-4, 1e-2),
    "batch_size":    [16, 32, 64],
    "epochs":        (20, 80),
}

# ── XGBoost Hyperparameter Search Space ───────────────────────────────────────
XGB_SEARCH_SPACE = {
    "n_estimators":    (100, 800),
    "max_depth":       (3, 10),
    "learning_rate":   (0.01, 0.3),
    "subsample":       (0.6, 1.0),
    "colsample_bytree":(0.6, 1.0),
    "min_child_weight":(1, 10),
    "gamma":           (0.0, 1.0),
    "reg_alpha":       (0.0, 2.0),
    "reg_lambda":      (0.5, 5.0),
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

# ── Optuna Config ──────────────────────────────────────────────────────────────
OPTUNA_TRIALS_XGB  = 60
OPTUNA_TRIALS_LGBM = 60
OPTUNA_TRIALS_LSTM = 30
OPTUNA_TIMEOUT_SEC = 1800

# ── Ensemble Weights (fallback if Optuna weight optimisation fails) ────────────
ENSEMBLE_WEIGHTS = {
    "xgb":  0.30,
    "lgbm": 0.35,
    "lstm": 0.25,
    "meta": 0.10,   # stacking meta-learner
}

# ── Risk Management ────────────────────────────────────────────────────────────
MAX_SECTOR_EXPOSURE_PCT  = 0.35   # max 35% of portfolio in one sector
MAX_SINGLE_POSITION_PCT  = 0.15   # max 15% in one stock
VAR_CONFIDENCE_LEVEL     = 0.95   # 95% VaR
VAR_LOOKBACK_DAYS        = 252    # 1 year for historical VaR

# ── Signal Thresholds ─────────────────────────────────────────────────────────
MIN_AI_CONFIDENCE    = 0.52   # AI probability must be >= this to fire BUY/SELL
MIN_RR_RATIO         = 1.5    # Minimum risk/reward ratio
ATR_SL_MULTIPLIER    = 1.5    # Stop-loss = entry - 1.5 × ATR
ATR_TP_MULTIPLIER    = 3.0    # Take-profit = entry + 3.0 × ATR
MAX_POSITION_PCT     = 0.15   # Max position size (Kelly-adjusted)
KELLY_FRACTION       = 0.25   # Quarter-Kelly

# ── File Paths ─────────────────────────────────────────────────────────────────
DATA_CACHE_DIR  = "data/cache"
MODEL_DIR       = "models"
RESULTS_DIR     = "results"
LOG_DIR         = "logs"
