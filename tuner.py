"""
tuner.py — Optuna-based hyperparameter optimisation.

Tunes XGBoost, LightGBM and LSTM independently, then optimises
ensemble weights in a final step.
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("tuner")


# ── XGBoost Objective ──────────────────────────────────────────────────────────

def _xgb_objective(trial, X_tr, y_tr, X_val, y_val):
    from xgboost import XGBClassifier

    sp = config.XGB_SEARCH_SPACE
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", *sp["n_estimators"]),
        "max_depth":        trial.suggest_int("max_depth", *sp["max_depth"]),
        "learning_rate":    trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
        "subsample":        trial.suggest_float("subsample", *sp["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *sp["colsample_bytree"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *sp["min_child_weight"]),
        "gamma":            trial.suggest_float("gamma", *sp["gamma"]),
        "reg_alpha":        trial.suggest_float("reg_alpha", *sp["reg_alpha"]),
        "reg_lambda":       trial.suggest_float("reg_lambda", *sp["reg_lambda"]),
        "eval_metric": "logloss",
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,
    }
    mdl = XGBClassifier(**params)
    mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    prob = mdl.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, prob)


# ── LightGBM Objective ─────────────────────────────────────────────────────────

def _lgbm_objective(trial, X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb

    sp = config.LGBM_SEARCH_SPACE
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", *sp["n_estimators"]),
        "num_leaves":       trial.suggest_int("num_leaves", *sp["num_leaves"]),
        "learning_rate":    trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", *sp["feature_fraction"]),
        "bagging_fraction": trial.suggest_float("bagging_fraction", *sp["bagging_fraction"]),
        "bagging_freq":     trial.suggest_int("bagging_freq", *sp["bagging_freq"]),
        "min_child_samples":trial.suggest_int("min_child_samples", *sp["min_child_samples"]),
        "reg_alpha":        trial.suggest_float("reg_alpha", *sp["reg_alpha"]),
        "reg_lambda":       trial.suggest_float("reg_lambda", *sp["reg_lambda"]),
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    mdl = lgb.LGBMClassifier(**params)
    mdl.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
    )
    prob = mdl.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, prob)


# ── LSTM Objective (PyTorch) ───────────────────────────────────────────────────

def _lstm_objective(trial, X_tr, y_tr, X_val, y_val):
    from sklearn.metrics import roc_auc_score

    # Sanitize before any model call — 184-col feature matrix may have NaN/inf
    X_tr_c  = np.nan_to_num(X_tr,  nan=0.0, posinf=0.0, neginf=0.0)
    X_val_c = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    y_tr_c  = np.clip(np.asarray(y_tr,  dtype=np.float32), 0.0, 1.0)
    y_val_c = np.clip(np.asarray(y_val, dtype=np.float32), 0.0, 1.0)

    sp = config.LSTM_SEARCH_SPACE
    params = {
        "units_1":      trial.suggest_int("units_1", *sp["units_1"]),
        "units_2":      trial.suggest_int("units_2", *sp["units_2"]),
        "dropout":      trial.suggest_float("dropout", *sp["dropout"]),
        "learning_rate":trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
        "batch_size":   trial.suggest_categorical("batch_size", sp["batch_size"]),
        "epochs":       trial.suggest_int("epochs", *sp["epochs"]),
    }

    from models import LSTMModel
    mdl = LSTMModel(seq_len=30, params=params)
    mdl.fit(X_tr_c, y_tr_c, X_val_c, y_val_c)
    prob  = mdl.predict_proba(X_val_c)
    valid = ~np.isnan(prob)
    if valid.sum() < 5 or len(set(y_val_c[valid])) < 2:
        return 0.5
    return float(roc_auc_score(y_val_c[valid], prob[valid]))


# ── Public Tuning Functions ────────────────────────────────────────────────────

def tune_xgb(
    X_tr, y_tr, X_val, y_val,
    n_trials: int = config.OPTUNA_TRIALS_XGB,
    timeout: int = config.OPTUNA_TIMEOUT_SEC,
) -> dict:
    logger.info(f"Tuning XGBoost ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _xgb_objective(t, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials, timeout=timeout, show_progress_bar=False,
        catch=(Exception,),
    )
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        logger.warning("XGB tuning: all trials failed — using default params")
        return {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "eval_metric": "logloss", "random_state": config.RANDOM_SEED, "n_jobs": -1,
        }
    best = study.best_params
    best.update({"eval_metric": "logloss",
                 "random_state": config.RANDOM_SEED, "n_jobs": -1})
    logger.info(f"XGB best AUC: {study.best_value:.4f}  params={best}")
    return best


def tune_lgbm(
    X_tr, y_tr, X_val, y_val,
    n_trials: int = config.OPTUNA_TRIALS_LGBM,
    timeout: int = config.OPTUNA_TIMEOUT_SEC,
) -> dict:
    logger.info(f"Tuning LightGBM ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _lgbm_objective(t, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials, timeout=timeout, show_progress_bar=False,
        catch=(Exception,),
    )
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        logger.warning("LGBM tuning: all trials failed — using default params")
        return {
            "n_estimators": 300, "num_leaves": 63, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "random_state": config.RANDOM_SEED, "n_jobs": -1, "verbose": -1,
        }
    best = study.best_params
    best.update({"random_state": config.RANDOM_SEED, "n_jobs": -1, "verbose": -1})
    logger.info(f"LGBM best AUC: {study.best_value:.4f}  params={best}")
    return best


def tune_lstm(
    X_tr, y_tr, X_val, y_val,
    n_trials: int = config.OPTUNA_TRIALS_LSTM,
    timeout: int = config.OPTUNA_TIMEOUT_SEC,
) -> dict:
    logger.info(f"Tuning LSTM ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _lstm_objective(t, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials, timeout=timeout, show_progress_bar=False,
        catch=(Exception,),
    )
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        logger.warning("LSTM tuning: all trials failed — using default params")
        return {
            "units_1": 64, "units_2": 32, "dropout": 0.2,
            "learning_rate": 5e-4, "batch_size": 32, "epochs": 30,
        }
    best = study.best_params
    logger.info(f"LSTM best AUC: {study.best_value:.4f}  params={best}")
    return best


def tune_ensemble_weights(
    xgb_proba, lgbm_proba, lstm_proba, y_true
) -> dict:
    """Optimise ensemble blend weights via Optuna."""
    logger.info("Tuning ensemble weights…")
    lstm_valid = ~np.isnan(lstm_proba)

    def objective(trial):
        w_xgb  = trial.suggest_float("w_xgb",  0.1, 0.8)
        w_lgbm = trial.suggest_float("w_lgbm", 0.1, 0.8)
        w_lstm = trial.suggest_float("w_lstm", 0.0, 0.5)
        total  = w_xgb + w_lgbm + w_lstm
        blend  = (
            w_xgb  * xgb_proba[lstm_valid]  +
            w_lgbm * lgbm_proba[lstm_valid] +
            w_lstm * lstm_proba[lstm_valid]
        ) / total
        return roc_auc_score(y_true[lstm_valid], blend)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    w = study.best_params
    total = w["w_xgb"] + w["w_lgbm"] + w["w_lstm"]
    weights = {
        "xgb":  w["w_xgb"]  / total,
        "lgbm": w["w_lgbm"] / total,
        "lstm": w["w_lstm"] / total,
    }
    logger.info(f"Optimal weights: {weights}  AUC={study.best_value:.4f}")
    return weights


def _tune_four_model_weights(
    xgb_proba, lgbm_proba, lstm_proba, transformer_proba, y_true
) -> dict:
    """Optimise 4-model ensemble blend weights (XGB + LGBM + LSTM + Transformer)."""
    import optuna
    from sklearn.metrics import roc_auc_score
    import numpy as np

    logger.info("Tuning 4-model ensemble weights (XGB+LGBM+LSTM+Transformer)...")

    lstm_valid        = ~np.isnan(lstm_proba)
    transformer_valid = ~np.isnan(transformer_proba)
    valid = lstm_valid & transformer_valid

    def objective(trial):
        w_xgb  = trial.suggest_float("w_xgb",  0.1, 0.6)
        w_lgbm = trial.suggest_float("w_lgbm", 0.1, 0.6)
        w_lstm = trial.suggest_float("w_lstm", 0.0, 0.4)
        w_tr   = trial.suggest_float("w_tr",   0.0, 0.4)
        total  = w_xgb + w_lgbm + w_lstm + w_tr
        blend  = (
            w_xgb  * xgb_proba[valid]         +
            w_lgbm * lgbm_proba[valid]        +
            w_lstm * lstm_proba[valid]         +
            w_tr   * transformer_proba[valid]
        ) / total
        return roc_auc_score(y_true[valid], blend)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    w = study.best_params
    total = w["w_xgb"] + w["w_lgbm"] + w["w_lstm"] + w["w_tr"]
    weights = {
        "xgb":         w["w_xgb"]  / total,
        "lgbm":        w["w_lgbm"] / total,
        "lstm":        w["w_lstm"] / total,
        "transformer": w["w_tr"]   / total,
    }
    logger.info(f"4-model weights: {weights}  AUC={study.best_value:.4f}")
    return weights
