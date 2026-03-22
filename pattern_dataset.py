"""
pattern_dataset.py — Builds the training corpus for the Pattern Transformer.

Concept:
  In NLP, you train a language model on a CORPUS — all books, articles,
  websites combined. The model learns general language patterns.

  Here, we train on a MARKET CORPUS — ALL symbols' price histories combined.
  The transformer learns general Vietnam stock market patterns:
    - Which token sequences tend to precede rallies
    - Which sequences precede sell-offs
    - How volume patterns interact with price patterns
    - How trend context changes what a pattern means

  Cross-symbol training is crucial: a pattern that works for VNM
  is MORE reliable if it also works for VCB, FPT, HPG, etc.
  This is exactly like how an LLM trained on many books generalises
  better than one trained on a single author.

  After pre-training on the full corpus, the model can be fine-tuned
  on a single symbol (transfer learning).
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("pattern_dataset")

CORPUS_CACHE = Path("data/cache/pattern_corpus.pkl")


def build_corpus(
    data: Dict[str, pd.DataFrame],
    seq_len: int = 60,
    horizon: int = 5,
    stride: int = 1,
    balance: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a unified training corpus from all symbols.

    Returns:
        X:       (N, seq_len) token sequences
        y:       (N,) binary direction labels (0=down, 1=up)
        weights: (N,) sample weights (recent data weighted higher)
    """
    from price_tokenizer import PriceTokenizer

    tokenizer = PriceTokenizer()
    all_X, all_y, all_weights = [], [], []

    for sym, ohlcv in data.items():
        if len(ohlcv) < seq_len + horizon + 10:
            logger.warning(f"[corpus] {sym}: too short ({len(ohlcv)} rows), skipping")
            continue

        X_sym, y_sym = tokenizer.encode_for_direction_prediction(
            ohlcv, seq_len=seq_len, horizon=horizon, stride=stride
        )

        if len(X_sym) == 0:
            continue

        # Recency weighting: more recent sequences get higher weight
        # This makes the model prioritise recent market regime
        n = len(X_sym)
        weights = np.linspace(0.5, 1.5, n)   # linear ramp: old=0.5x, new=1.5x

        all_X.append(X_sym)
        all_y.append(y_sym)
        all_weights.append(weights)

        up_pct = y_sym.mean()
        logger.info(f"[corpus] {sym}: {n} sequences  up={up_pct:.1%}  down={(1-up_pct):.1%}")

    if not all_X:
        logger.error("[corpus] No data to build corpus from!")
        return np.array([]), np.array([]), np.array([])

    X       = np.concatenate(all_X,       axis=0)
    y       = np.concatenate(all_y,       axis=0)
    weights = np.concatenate(all_weights, axis=0)

    # Shuffle (maintain temporal integrity within each symbol, but mix symbols)
    idx = np.random.RandomState(42).permutation(len(X))
    X, y, weights = X[idx], y[idx], weights[idx]

    if balance:
        X, y, weights = _balance_classes(X, y, weights)

    logger.info(
        f"[corpus] Total: {len(X)} sequences across {len(data)} symbols  "
        f"up={y.mean():.1%}"
    )
    return X, y, weights


def _balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample majority class to balance training data."""
    n_up   = int(y.sum())
    n_down = int(len(y) - y.sum())
    n_keep = min(n_up, n_down)

    up_idx   = np.where(y == 1)[0]
    down_idx = np.where(y == 0)[0]

    rng = np.random.RandomState(42)
    up_keep   = rng.choice(up_idx,   n_keep, replace=False)
    down_keep = rng.choice(down_idx, n_keep, replace=False)
    keep      = np.sort(np.concatenate([up_keep, down_keep]))

    return X[keep], y[keep], weights[keep]


def time_split_corpus(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
) -> Tuple:
    """
    Time-aware split: test set = last test_ratio of data.
    (No shuffling across the split boundary.)
    """
    n      = len(X)
    n_test = max(int(n * test_ratio), 10)
    n_val  = max(int(n * val_ratio), 10)
    n_tr   = n - n_test - n_val

    if n_tr < 10:
        n_tr, n_val, n_test = max(1, n - 4), 2, 2

    return (
        X[:n_tr],       y[:n_tr],       weights[:n_tr],
        X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val], weights[n_tr:n_tr+n_val],
        X[n_tr+n_val:], y[n_tr+n_val:],
    )


def build_symbol_finetune_data(
    ohlcv: pd.DataFrame,
    seq_len: int = 60,
    horizon: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build fine-tuning data for a SINGLE symbol.
    Used after pre-training on the full corpus.
    """
    from price_tokenizer import PriceTokenizer
    tokenizer = PriceTokenizer()
    return tokenizer.encode_for_direction_prediction(
        ohlcv, seq_len=seq_len, horizon=horizon, stride=1
    )


def save_corpus(X, y, weights, path: Path = CORPUS_CACHE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y, "weights": weights}, f)
    logger.info(f"[corpus] Saved to {path}")


def load_corpus(path: Path = CORPUS_CACHE) -> Optional[Tuple]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["X"], d["y"], d["weights"]
