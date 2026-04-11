"""
Module 04 — ML Model (XGBoost Direction Predictor)
====================================================
Trains a gradient-boosted classifier per strategy/hold-period target.

For each ticker:
  - Feature matrix X: the ~190 engineered features
  - Target y: fwd_positive_tN  (1 = price higher at T+N, 0 = lower)

Walk-forward training: train on oldest 80%, validate on newest 20%.

Output:
  data/models/{TICKER}_model.pkl   (trained XGBoost pipeline)
  data/models/{TICKER}_proba.csv   (probability predictions for full history)

Run:
    python modules/04_model.py --ticker HPG
    python modules/04_model.py --all
"""

import json
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT   = Path(__file__).resolve().parent.parent
FEAT   = ROOT / "data" / "features"
MDL    = ROOT / "data" / "models"
MDL.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("model")

# ── Feature groups ─────────────────────────────────────────────────────────────
EXCLUDE_COLS = {
    "date", "ticker", "close_raw",
    # forward targets (leak prevention)
    "fwd_return_t3",  "fwd_return_t5",  "fwd_return_t10",
    "fwd_return_t20", "fwd_return_t60", "fwd_return_t180",
    "fwd_positive_t3",  "fwd_positive_t5",  "fwd_positive_t10",
    "fwd_positive_t20", "fwd_positive_t60", "fwd_positive_t180",
    # raw OHLCV (already encoded as features)
    "open", "high", "low", "close", "volume",
    # misc non-numeric
    "wyckoff_phase",
}

# Target for each primary strategy (optimal hold period)
STRATEGY_TARGETS = {
    "PRICE_DOWN_15_MA20": "fwd_positive_t180",
    "RSI_OVERSOLD":       "fwd_positive_t60",
    "PRICE_DOWN_15_20D":  "fwd_positive_t5",
    "DMI_WAVE":           "fwd_positive_t10",
    "SAR_MACD":           "fwd_positive_t20",
    "BOLLINGER":          "fwd_positive_t5",
    "VOLUME_EXPLOSION":   "fwd_positive_t180",
    "UPTREND":            "fwd_positive_t180",
    "STOCH_RSI":          "fwd_positive_t180",
    "GENERAL":            "fwd_positive_t20",   # fallback model
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """All numeric columns that are not excluded."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in EXCLUDE_COLS]


def _safe_fillna(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
            df[c] = df[c].fillna(df[c].median())
    return df


# ── Walk-forward trainer ───────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    train_frac: float = 0.80,
) -> tuple:
    """
    Trains an XGBoost classifier using walk-forward split.
    Returns (pipeline, val_accuracy, val_auc, feature_importances_dict).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.impute import SimpleImputer

    df_clean = _safe_fillna(df, feature_cols).dropna(subset=[target_col])
    df_clean = df_clean[df_clean[target_col].isin([0, 1])]

    if len(df_clean) < 100:
        return None, 0.0, 0.0, {}

    split = int(len(df_clean) * train_frac)
    train_df = df_clean.iloc[:split]
    val_df   = df_clean.iloc[split:]

    if len(val_df) < 20:
        val_df = train_df.iloc[-50:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.astype(int)
    X_val   = val_df[feature_cols].values
    y_val   = val_df[target_col].values.astype(int)

    # Class imbalance weight
    pos_frac = y_train.mean()
    scale    = (1 - pos_frac) / max(pos_frac, 1e-6)

    try:
        clf = XGBClassifier(
            n_estimators=30000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    except TypeError:
        # fallback without use_label_encoder
        clf = XGBClassifier(
            n_estimators=30000, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    acc  = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba)
    except Exception:
        auc = 0.5

    # Feature importances
    try:
        imps = pipe.named_steps["clf"].feature_importances_
        feat_imp = dict(sorted(
            zip(feature_cols, imps), key=lambda x: x[1], reverse=True
        )[:20])
    except Exception:
        feat_imp = {}

    return pipe, round(acc, 4), round(auc, 4), feat_imp


# ── Predict on full history ────────────────────────────────────────────────────

def predict_proba_series(
    pipe,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Returns probability of positive outcome for every row in df."""
    if pipe is None:
        return np.full(len(df), 0.5)
    df_clean = _safe_fillna(df, feature_cols)
    X = df_clean[feature_cols].values
    try:
        return pipe.predict_proba(X)[:, 1]
    except Exception:
        return np.full(len(df), 0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(ticker: str) -> None:
    feat_path = FEAT / f"{ticker}.csv"
    if not feat_path.exists():
        log.error(f"Feature file not found: {feat_path}")
        return

    df = pd.read_csv(feat_path, low_memory=False)
    feature_cols = _get_feature_cols(df)
    log.info(f"{ticker}: {len(df)} rows, {len(feature_cols)} features")

    model_info = {"ticker": ticker, "models": {}}
    proba_cols = {}

    # Train a model per strategy target (unique targets)
    for strat_key, target_col in STRATEGY_TARGETS.items():
        if target_col not in df.columns:
            continue
        log.info(f"  Training [{strat_key}] → target={target_col} …")
        pipe, acc, auc, feat_imp = train_model(df, target_col, feature_cols)

        if pipe is None:
            log.warning(f"  {ticker}/{strat_key}: insufficient data, skipping")
            continue

        log.info(f"  {strat_key}: acc={acc:.3f}  AUC={auc:.3f}")

        proba_col_name = f"proba_{target_col}"
        proba_cols[proba_col_name] = predict_proba_series(pipe, df, feature_cols)

        model_info["models"][strat_key] = {
            "target":   target_col,
            "accuracy": acc,
            "auc":      auc,
            "top_features": feat_imp,
        }

        # Save individual model
        model_path = MDL / f"{ticker}_{strat_key}.pkl"
        joblib.dump(pipe, model_path)

    # Save combined probability dataframe
    proba_df = df[["date", "close"]].copy()
    for col_name, vals in proba_cols.items():
        proba_df[col_name] = vals

    proba_path = MDL / f"{ticker}_proba.csv"
    proba_df.to_csv(proba_path, index=False)

    # Save model info JSON
    info_path = MDL / f"{ticker}_info.json"
    info_path.write_text(json.dumps(model_info, indent=2, default=str))

    log.info(f"{ticker}: models saved → {MDL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VN100 Model Training")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    tickers = [f.stem for f in FEAT.glob("*.csv")] if args.all else (
        [args.ticker] if args.ticker else [f.stem for f in FEAT.glob("*.csv")]
    )
    log.info(f"=== Model Training: {len(tickers)} tickers ===\n")
    for t in tickers:
        run(t)