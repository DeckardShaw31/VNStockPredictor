"""
price_tokenizer.py — Converts raw OHLCV price data into discrete tokens.

Core concept:
  Natural language: "the cat sat" → tokens [the=1, cat=284, sat=442]
  Price sequences:  +2.1%, -0.8%, +0.3% → tokens [BIG_UP=6, SMALL_DOWN=2, FLAT=4]

We build a multi-channel vocabulary where each daily bar becomes a
composite "word" encoding:
  - Return magnitude + direction   (primary token)
  - Volume relative to average     (secondary token)
  - Candlestick body shape         (tertiary token)
  - Trend context (above/below MA) (context token)

Combined, each trading day becomes a sequence of integers that a
transformer can learn from — exactly like GPT learns from word sequences.

Vocabulary design (inspired by FinGPT / chart pattern tokenization):
  Return tokens (21 bins):
    CRASH      < -4%
    BIG_DOWN   -4% to -2%
    MED_DOWN   -2% to -1%
    SMALL_DOWN -1% to -0.3%
    FLAT_DOWN  -0.3% to -0.1%
    FLAT        ±0.1%
    FLAT_UP    +0.1% to +0.3%
    SMALL_UP   +0.3% to +1%
    MED_UP     +1% to +2%
    BIG_UP     +2% to +4%
    SURGE      > +4%

  Volume tokens (5 bins): VERY_LOW, LOW, NORMAL, HIGH, VERY_HIGH
  Body tokens (5 bins):   BEAR_STRONG, BEAR, DOJI, BULL, BULL_STRONG
  Trend tokens (3 bins):  BELOW_MA, AT_MA, ABOVE_MA
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import pickle
from pathlib import Path


# ── Vocabulary Definition ──────────────────────────────────────────────────────

# Return bins (edges in %)
RETURN_BINS   = [-np.inf, -4, -2, -1, -0.5, -0.2, -0.05,
                  0.05, 0.2, 0.5, 1, 2, 4, np.inf]
RETURN_LABELS = list(range(len(RETURN_BINS) - 1))   # 0..12

# Volume bins (relative to 20-day average)
VOLUME_BINS   = [-np.inf, 0.4, 0.7, 1.3, 2.0, np.inf]
VOLUME_LABELS = list(range(len(VOLUME_BINS) - 1))   # 0..4

# Body shape bins (body / range ratio)
BODY_BINS     = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
BODY_LABELS   = list(range(len(BODY_BINS) - 1))      # 0..4

# Trend context bins (price / MA200 - 1)
TREND_BINS    = [-np.inf, -0.03, 0.03, np.inf]
TREND_LABELS  = [0, 1, 2]                             # 0=below, 1=at, 2=above

# ── Vocabulary sizes ───────────────────────────────────────────────────────────
N_RETURN  = len(RETURN_LABELS)    # 13
N_VOLUME  = len(VOLUME_LABELS)    # 5
N_BODY    = len(BODY_LABELS)      # 5
N_TREND   = len(TREND_LABELS)     # 3

# Total composite token space: 13 * 5 * 5 * 3 = 975 tokens
# We use a flat encoding: token_id = r*75 + v*15 + b*3 + t
VOCAB_SIZE = N_RETURN * N_VOLUME * N_BODY * N_TREND   # 975

# Special tokens
PAD_TOKEN = VOCAB_SIZE       # 975
BOS_TOKEN = VOCAB_SIZE + 1   # 976  (beginning of sequence)
EOS_TOKEN = VOCAB_SIZE + 2   # 977  (end of sequence — market close)
TOTAL_VOCAB = VOCAB_SIZE + 3  # 978


def _encode_composite(r: int, v: int, b: int, t: int) -> int:
    """Encode (return, volume, body, trend) bins into a single token ID."""
    return r * (N_VOLUME * N_BODY * N_TREND) + v * (N_BODY * N_TREND) + b * N_TREND + t


def _decode_composite(token_id: int) -> Tuple[int, int, int, int]:
    """Decode token ID back into (return, volume, body, trend) bins."""
    t = token_id % N_TREND
    token_id //= N_TREND
    b = token_id % N_BODY
    token_id //= N_BODY
    v = token_id % N_VOLUME
    r = token_id // N_VOLUME
    return r, v, b, t


class PriceTokenizer:
    """
    Converts OHLCV DataFrames into sequences of integer token IDs.

    Usage:
        tokenizer = PriceTokenizer()
        tokens = tokenizer.encode(ohlcv_df)   # -> List[int]
        decoded = tokenizer.decode(tokens)     # -> human-readable list
    """

    def __init__(self, ma_window: int = 50):
        self.ma_window = ma_window
        self.vocab_size = TOTAL_VOCAB

    def encode(self, df: pd.DataFrame) -> List[int]:
        """
        Encode an OHLCV DataFrame into a sequence of token IDs.
        One token per trading day.
        """
        if len(df) < 3:
            return []

        close  = df["close"].values.astype(float)
        open_  = df["open"].values.astype(float) if "open" in df.columns else close
        high   = df["high"].values.astype(float) if "high" in df.columns else close
        low    = df["low"].values.astype(float)  if "low"  in df.columns else close
        volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(close))

        # Returns (%)
        returns_pct = np.zeros(len(close))
        returns_pct[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9) * 100

        # Volume relative to rolling average
        vol_avg = pd.Series(volume).rolling(min(20, len(volume)-1), min_periods=1).mean().values
        vol_ratio = volume / (vol_avg + 1e-9)

        # Body shape: (close - open) / (high - low)  ∈ [-1, 1]
        range_ = high - low
        body   = np.where(range_ > 0, (close - open_) / range_, 0.0)

        # Trend context: price vs MA(50)
        ma = pd.Series(close).ewm(span=min(self.ma_window, len(close)-1), adjust=False).mean().values
        trend_ratio = (close - ma) / (ma + 1e-9) * 100   # percent above/below MA

        tokens = []
        for i in range(len(close)):
            r = int(np.searchsorted(RETURN_BINS[1:-1], returns_pct[i]))
            r = max(0, min(r, N_RETURN - 1))

            v = int(np.searchsorted(VOLUME_BINS[1:-1], vol_ratio[i]))
            v = max(0, min(v, N_VOLUME - 1))

            b = int(np.searchsorted(BODY_BINS[1:-1], body[i]))
            b = max(0, min(b, N_BODY - 1))

            t = int(np.searchsorted(TREND_BINS[1:-1], trend_ratio[i]))
            t = max(0, min(t, N_TREND - 1))

            tokens.append(_encode_composite(r, v, b, t))

        return tokens

    def decode_token(self, token_id: int) -> str:
        """Convert a token ID to a human-readable description."""
        if token_id == PAD_TOKEN:
            return "[PAD]"
        if token_id == BOS_TOKEN:
            return "[BOS]"
        if token_id == EOS_TOKEN:
            return "[EOS]"

        r, v, b, t = _decode_composite(token_id)

        return_names = [
            "CRASH", "BIG_DOWN", "MED_DOWN", "SMALL_DOWN", "FLAT_DOWN",
            "FLAT_NEG", "FLAT", "FLAT_POS", "SMALL_UP", "MED_UP",
            "BIG_UP", "SURGE_UP", "SURGE"
        ]
        volume_names = ["VOL_VERY_LOW", "VOL_LOW", "VOL_NORMAL", "VOL_HIGH", "VOL_VERY_HIGH"]
        body_names   = ["BEAR_STR", "BEAR", "DOJI", "BULL", "BULL_STR"]
        trend_names  = ["BELOW_MA", "AT_MA", "ABOVE_MA"]

        rn = return_names[r] if r < len(return_names) else f"R{r}"
        vn = volume_names[v] if v < len(volume_names) else f"V{v}"
        bn = body_names[b]   if b < len(body_names)   else f"B{b}"
        tn = trend_names[t]  if t < len(trend_names)  else f"T{t}"

        return f"{rn}|{vn}|{bn}|{tn}"

    def decode(self, tokens: List[int]) -> List[str]:
        return [self.decode_token(t) for t in tokens]

    def encode_sequences(
        self,
        df: pd.DataFrame,
        seq_len: int = 60,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (input_sequences, next_token) pairs for training.
        Like language model next-token prediction.

        Returns:
            X: shape (n_samples, seq_len) — input token sequences
            y: shape (n_samples,)         — next token (classification target)
        """
        tokens = self.encode(df)
        if len(tokens) < seq_len + 1:
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(0, len(tokens) - seq_len, stride):
            X.append(tokens[i: i + seq_len])
            y.append(tokens[i + seq_len])

        return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

    def encode_for_direction_prediction(
        self,
        df: pd.DataFrame,
        seq_len: int = 60,
        horizon: int = 5,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (input_sequences, direction_label) pairs.
        Direction label: 1 if price up after horizon days, 0 if down.
        Filters out ambiguous near-zero moves (volatility-adjusted).
        """
        tokens = self.encode(df)
        close  = df["close"].values.astype(float)

        if len(tokens) < seq_len + horizon + 1:
            return np.array([]), np.array([])

        # Volatility-adjusted labels
        returns   = pd.Series(close).pct_change()
        daily_vol = returns.rolling(min(20, len(returns)-1)).std().values
        fwd_ret   = np.zeros(len(close))
        fwd_ret[:-horizon] = (close[horizon:] - close[:-horizon]) / (close[:-horizon] + 1e-9)

        X, y = [], []
        for i in range(0, len(tokens) - seq_len - horizon, stride):
            seq_end    = i + seq_len
            fwd_return = fwd_ret[seq_end]
            threshold  = 0.3 * (daily_vol[seq_end] if daily_vol[seq_end] > 0 else 0.005)

            if abs(fwd_return) < threshold:
                continue   # skip ambiguous near-zero moves

            X.append(tokens[i: seq_end])
            y.append(1 if fwd_return > 0 else 0)

        return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PriceTokenizer":
        with open(path, "rb") as f:
            return pickle.load(f)
