"""
pattern_transformer.py — A small GPT-style transformer trained locally
on tokenized Vietnam stock price sequences.

Core idea (from the video concept):
  GPT learns: "the quick brown fox" → predicts "jumps"
  This learns: "FLAT|VOL_LOW|DOJI, SMALL_UP|VOL_HIGH|BULL, ..." → predicts direction

The transformer uses self-attention to discover which historical patterns
most reliably precede upward or downward moves. This is pattern recognition
through attention — the same mechanism that makes LLMs understand language,
applied to market microstructure patterns.

Architecture:
  - Embedding layer: token_id → 64-dim vector
  - Positional encoding: learnable position embeddings
  - N transformer blocks: multi-head attention + FFN + LayerNorm
  - Direction head: binary classification (up/down)

This is a LOCAL model — trained on your machine, no API needed.
Training data: tokenized OHLCV sequences from all symbols in your watchlist.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger("pattern_transformer")


# ── Transformer Architecture ───────────────────────────────────────────────────

def build_pattern_transformer(
    vocab_size: int,
    seq_len: int = 60,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_blocks: int = 3,
    ff_dim: int = 128,
    dropout_rate: float = 0.1,
) -> "tf.keras.Model":
    """
    Build a GPT-style transformer for price pattern classification.

    Architecture:
        Input: (batch, seq_len) integer token IDs
        → Token Embedding + Positional Embedding
        → N × TransformerBlock (causal self-attention + FFN)
        → GlobalAveragePooling
        → Dense(64, relu) → Dropout → Dense(1, sigmoid)
        Output: (batch, 1) probability of UP move

    Causal masking ensures the model can only attend to past tokens,
    preserving temporal ordering — exactly like GPT.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input

    # ── Token + Positional Embeddings ──────────────────────────────────────
    token_input = Input(shape=(seq_len,), dtype=tf.int32, name="token_input")

    # Token embedding
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="token_embedding",
    )(token_input)

    # Learnable positional embedding (better than fixed sinusoidal for short sequences)
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(
        input_dim=seq_len,
        output_dim=embed_dim,
        name="position_embedding",
    )(positions)
    x = x + pos_emb   # (batch, seq_len, embed_dim)

    x = layers.Dropout(dropout_rate)(x)

    # ── Transformer Blocks ─────────────────────────────────────────────────
    for block_i in range(num_blocks):
        # Causal multi-head self-attention
        # use_causal_mask ensures each position only attends to previous positions
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name=f"attention_{block_i}",
        )(x, x, use_causal_mask=True)

        # Add & Norm (residual)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"norm1_{block_i}")(x + attn_out)

        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation="gelu", name=f"ffn1_{block_i}")(x)
        ffn = layers.Dropout(dropout_rate)(ffn)
        ffn = layers.Dense(embed_dim, name=f"ffn2_{block_i}")(ffn)

        # Add & Norm (residual)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"norm2_{block_i}")(x + ffn)

    # ── Classification Head ────────────────────────────────────────────────
    # Use the last token's representation (like GPT's [CLS] approach)
    last_token = x[:, -1, :]   # (batch, embed_dim)

    # Also pool across the sequence for global context
    pooled = layers.GlobalAveragePooling1D()(x)   # (batch, embed_dim)

    combined = layers.Concatenate()([last_token, pooled])   # (batch, embed_dim*2)
    combined = layers.Dense(64, activation="gelu", name="head_dense")(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    output   = layers.Dense(1, activation="sigmoid", name="output")(combined)

    model = Model(inputs=token_input, outputs=output, name="PatternTransformer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )

    return model


# ── PatternTransformerModel Wrapper ───────────────────────────────────────────

class PatternTransformerModel:
    """
    Wrapper around the Keras transformer model with the same interface
    as XGBModel / LGBMModel / LSTMModel for drop-in ensemble integration.

    This model operates on TOKEN sequences, not raw feature vectors.
    It requires a PriceTokenizer to convert OHLCV → tokens first.
    """

    name = "transformer"

    def __init__(
        self,
        seq_len: int = 60,
        params: Optional[dict] = None,
        vocab_size: Optional[int] = None,
    ):
        self.seq_len    = seq_len
        self.vocab_size = vocab_size
        self.params     = params or {
            "embed_dim":    64,
            "num_heads":    4,
            "num_blocks":   3,
            "ff_dim":       128,
            "dropout_rate": 0.1,
            "learning_rate":1e-3,
            "batch_size":   32,
            "epochs":       40,
        }
        self.model      = None
        self.tokenizer  = None

    def _get_vocab_size(self):
        from price_tokenizer import TOTAL_VOCAB
        return self.vocab_size or TOTAL_VOCAB

    def fit(
        self,
        X_train: np.ndarray,   # (n, seq_len) token IDs
        y_train: np.ndarray,   # (n,) binary labels
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Train the transformer on tokenized sequences."""
        import tensorflow as tf
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        if len(X_train) == 0:
            logger.warning("[transformer] No training data available")
            return self

        # Truncate/pad sequences to seq_len
        X_train = self._pad_sequences(X_train)
        if X_val is not None:
            X_val = self._pad_sequences(X_val)

        p = self.params
        self.model = build_pattern_transformer(
            vocab_size    = self._get_vocab_size(),
            seq_len       = self.seq_len,
            embed_dim     = p["embed_dim"],
            num_heads     = p["num_heads"],
            num_blocks    = p["num_blocks"],
            ff_dim        = p["ff_dim"],
            dropout_rate  = p["dropout_rate"],
        )

        # Recompile with tuned LR
        self.model.optimizer.learning_rate.assign(p["learning_rate"])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=8, restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5, patience=4, verbose=0,
            ),
        ]

        val_data = (X_val, y_val) if X_val is not None and len(X_val) > 0 else None

        self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        logger.info(f"[transformer] Training complete on {len(X_train)} sequences")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of UP for each sequence."""
        if self.model is None or len(X) == 0:
            return np.full(len(X), np.nan)
        X_padded = self._pad_sequences(X)
        return self.model.predict(X_padded, verbose=0).flatten()

    def _pad_sequences(self, X: np.ndarray) -> np.ndarray:
        """Pad or truncate sequences to self.seq_len."""
        from price_tokenizer import PAD_TOKEN
        n = len(X)
        result = np.full((n, self.seq_len), PAD_TOKEN, dtype=np.int32)
        for i, seq in enumerate(X):
            seq = np.asarray(seq, dtype=np.int32)
            if len(seq) >= self.seq_len:
                result[i] = seq[-self.seq_len:]
            else:
                result[i, -len(seq):] = seq
        return result

    def save(self, path: str):
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save(str(base) + ".keras")
        with open(str(base) + "_config.pkl", "wb") as f:
            pickle.dump({
                "seq_len":    self.seq_len,
                "params":     self.params,
                "vocab_size": self.vocab_size,
            }, f)

    @classmethod
    def load(cls, path: str) -> "PatternTransformerModel":
        import tensorflow as tf
        base = Path(path)
        with open(str(base) + "_config.pkl", "rb") as f:
            cfg = pickle.load(f)
        obj = cls(**cfg)
        keras_path = str(base) + ".keras"
        if Path(keras_path).exists():
            obj.model = tf.keras.models.load_model(keras_path)
        return obj


# ── Optuna Objective for Transformer ──────────────────────────────────────────

def tune_transformer_objective(trial, X_tr, y_tr, X_val, y_val) -> float:
    """Optuna objective — tunes transformer hyperparameters."""
    from sklearn.metrics import roc_auc_score

    params = {
        "embed_dim":    trial.suggest_categorical("embed_dim",    [32, 64, 128]),
        "num_heads":    trial.suggest_categorical("num_heads",    [2, 4, 8]),
        "num_blocks":   trial.suggest_int("num_blocks",           2, 5),
        "ff_dim":       trial.suggest_categorical("ff_dim",       [64, 128, 256]),
        "dropout_rate": trial.suggest_float("dropout_rate",       0.05, 0.4),
        "learning_rate":trial.suggest_float("learning_rate",      1e-4, 5e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size",   [16, 32, 64]),
        "epochs":       trial.suggest_int("epochs",               20, 60),
    }

    # num_heads must divide embed_dim
    if params["embed_dim"] % params["num_heads"] != 0:
        return 0.5

    model = PatternTransformerModel(params=params)
    model.fit(X_tr, y_tr, X_val, y_val)
    if model.model is None:
        return 0.5

    prob = model.predict_proba(X_val)
    valid = ~np.isnan(prob)
    if valid.sum() < 5 or len(set(y_val[valid])) < 2:
        return 0.5
    return float(roc_auc_score(y_val[valid], prob[valid]))
