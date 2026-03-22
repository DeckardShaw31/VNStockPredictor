"""
target_engineering.py — Better prediction targets and walk-forward CV.

The Problem with Binary Up/Down:
  - A stock moving +0.1% is labeled the same as +5% → massive noise
  - Labels near 0 are effectively random
  - 50% base rate on balanced data means accuracy ceiling ~55-60%

Better Approaches:
  1. Volatility-adjusted threshold: label UP only if return > 0.5 * ATR
     → Ignores noise near zero, only predicts meaningful moves
  2. Magnitude-weighted labels: UP_STRONG / UP / NEUTRAL / DOWN / DOWN_STRONG
     → Train multi-class, evaluate on strong signal only
  3. Purged walk-forward CV: prevents look-ahead leakage from overlapping returns
  4. Risk-adjusted return as regression target (Sharpe-like)
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("target_engineering")


# ── Better Binary Labels ───────────────────────────────────────────────────────

def make_volatility_adjusted_label(
    close: pd.Series,
    horizon: int = 5,
    vol_window: int = 20,
    threshold_multiplier: float = 0.5,
) -> pd.Series:
    """
    Label as UP (1) only if forward return > threshold_multiplier * daily_vol.
    Label as DOWN (0) only if forward return < -threshold_multiplier * daily_vol.
    Rows within the dead zone are dropped (ambiguous near-zero moves).

    This dramatically reduces label noise while preserving real signal.
    """
    fwd_ret  = close.pct_change(horizon).shift(-horizon)
    daily_vol = close.pct_change().rolling(vol_window).std()

    threshold = threshold_multiplier * daily_vol

    label = pd.Series(np.nan, index=close.index)
    label[fwd_ret >  threshold] = 1.0   # clear UP
    label[fwd_ret < -threshold] = 0.0   # clear DOWN
    # Rows within ±threshold remain NaN and will be dropped

    pct_kept = label.notna().mean()
    logger.info(
        f"  Vol-adjusted labels: {pct_kept:.1%} of rows kept "
        f"(threshold={threshold_multiplier}x daily vol, "
        f"dropped {1-pct_kept:.1%} near-zero moves)"
    )
    return label


def make_risk_adjusted_target(
    close: pd.Series,
    horizon: int = 5,
    vol_window: int = 20,
) -> pd.Series:
    """
    Regression target: forward Sharpe-like ratio.
    = forward_return / rolling_volatility

    A model predicting this learns to find high-conviction moves,
    not just direction. Use with a regression model (XGB regressor).
    """
    fwd_ret  = close.pct_change(horizon).shift(-horizon)
    vol      = close.pct_change().rolling(vol_window).std()
    return (fwd_ret / vol.replace(0, np.nan)).clip(-5, 5)


def make_quintile_label(
    close: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """
    Rank stocks into quintiles based on forward return.
    Returns 0-4 (0=worst, 4=best). Only meaningful in cross-sectional context.
    """
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    return pd.qcut(fwd_ret.rank(method="first"), 5, labels=False)


# ── Purged Walk-Forward Cross-Validation ──────────────────────────────────────

class PurgedWalkForwardCV:
    """
    Walk-forward cross-validation with embargo (purging).

    Standard CV has a look-ahead bias problem with overlapping labels:
    if the label for day t uses returns from t to t+5, and the test fold
    starts at t+3, then the training data's last label partially overlaps
    with the test period. Purging removes these contaminated rows.

    Parameters:
        n_splits:     number of walk-forward folds
        embargo_days: days to drop between train and test (= horizon)
        min_train:    minimum training rows before first fold
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 5,
        min_train: int = 100,
    ):
        self.n_splits    = n_splits
        self.embargo_days = embargo_days
        self.min_train   = min_train

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        splits = []

        fold_size = (n - self.min_train) // self.n_splits

        for i in range(self.n_splits):
            train_end = self.min_train + i * fold_size
            test_start = train_end + self.embargo_days   # embargo gap
            test_end   = min(train_end + fold_size + self.embargo_days, n)

            if test_start >= n or test_end <= test_start:
                break

            train_idx = np.arange(0, train_end)
            test_idx  = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        return splits

    def cross_val_auc(
        self,
        model_class,
        model_params: dict,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Run purged CV and return mean AUC across folds."""
        from sklearn.metrics import roc_auc_score

        aucs = []
        for fold_i, (tr_idx, te_idx) in enumerate(self.split(X)):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                continue

            model = model_class(model_params)
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_te)
            valid = ~np.isnan(prob)
            if valid.sum() < 5:
                continue
            auc = roc_auc_score(y_te[valid], prob[valid])
            aucs.append(auc)

        return float(np.mean(aucs)) if aucs else 0.5


# ── Feature Selection ─────────────────────────────────────────────────────────

def select_top_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    top_n: int = 40,
    method: str = "lgbm",
) -> List[str]:
    """
    Select the top_n most predictive features using LightGBM feature importance.

    Removes noisy features that hurt OOB generalisation.
    Typical improvement: +2-4% AUC from reducing noise features.
    """
    if len(feature_names) <= top_n:
        return feature_names

    if method == "lgbm":
        import lightgbm as lgb
        mdl = lgb.LGBMClassifier(
            n_estimators=100, num_leaves=31,
            learning_rate=0.1, random_state=42,
            n_jobs=-1, verbose=-1,
        )
        mdl.fit(X_train, y_train)
        importances = pd.Series(mdl.feature_importances_, index=feature_names)
    else:
        from sklearn.ensemble import RandomForestClassifier
        mdl = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        mdl.fit(X_train, y_train)
        importances = pd.Series(mdl.feature_importances_, index=feature_names)

    top_features = importances.nlargest(top_n).index.tolist()
    dropped = len(feature_names) - top_n
    logger.info(f"  Feature selection: kept {top_n}/{len(feature_names)} features (dropped {dropped} low-importance)")
    return top_features


# ── Market Regime Detection ───────────────────────────────────────────────────

def add_regime_features(df: pd.DataFrame, vnindex: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add market regime features — bull/bear/sideways detection.

    Regime matters because:
    - In bull regimes, momentum signals work better
    - In bear regimes, mean-reversion works better
    - A model that knows the regime can weight signals appropriately
    """
    c = df["close"]

    # Trend regime: price vs 50-day and 200-day MA
    ma50  = c.ewm(span=min(50,  len(c)-1)).mean()
    ma200 = c.ewm(span=min(200, len(c)-1)).mean()

    df["regime_bull"]    = (c > ma50).astype(int)
    df["regime_uptrend"] = (ma50 > ma200).astype(int)
    df["regime_strength"] = (c - ma200) / ma200.replace(0, np.nan)   # distance from 200MA

    # Volatility regime: high/low vol (affects signal reliability)
    vol_20 = c.pct_change().rolling(min(20, len(c)-1)).std() * np.sqrt(252)
    vol_60 = c.pct_change().rolling(min(60, len(c)-1)).std() * np.sqrt(252)
    df["vol_regime"]      = (vol_20 > vol_60).astype(int)   # 1 = rising vol
    df["vol_ratio"]       = vol_20 / vol_60.replace(0, np.nan)

    # Drawdown from 52-week high (regime proxy)
    high_52w = c.rolling(min(252, len(c))).max()
    df["drawdown_52w"] = (c - high_52w) / high_52w.replace(0, np.nan)

    if vnindex is not None:
        vn_c   = vnindex["close"].reindex(df.index)
        vn_ma50 = vn_c.ewm(span=min(50, len(vn_c)-1)).mean()
        df["market_regime_bull"] = (vn_c > vn_ma50).astype(int).reindex(df.index)

        # VN-Index distance from 52W high
        vn_high = vn_c.rolling(min(252, len(vn_c))).max()
        df["market_drawdown"] = ((vn_c - vn_high) / vn_high.replace(0, np.nan)).reindex(df.index)

    return df
