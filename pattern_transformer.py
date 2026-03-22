"""
pattern_transformer.py — GPT-style transformer for price pattern recognition.
Implemented in PyTorch — works on Python 3.13, no TensorFlow required.

Core idea:
  GPT learns: "the quick brown fox" -> predicts "jumps"
  This learns: "FLAT|DOJI, BIG_UP|BULL_STR|VOL_HIGH, ..." -> predicts UP/DOWN

Architecture:
  Token + Positional Embedding
  N x TransformerBlock (causal multi-head attention + FFN + LayerNorm)
  Last-token + Global-avg pooling -> Dense -> Sigmoid

Runs entirely locally. No API. No internet required during inference.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("pattern_transformer")


# ── PyTorch Model ──────────────────────────────────────────────────────────────

def _build_torch_transformer(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    ff_dim: int,
    dropout: float,
):
    import torch
    import torch.nn as nn
    import math

    class CausalTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                                dropout=dropout, batch_first=True)
            self.ff    = nn.Sequential(
                nn.Linear(embed_dim, ff_dim), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.drop  = nn.Dropout(dropout)

        def forward(self, x, causal_mask):
            attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask,
                                    is_causal=True, need_weights=False)
            x = self.norm1(x + self.drop(attn_out))
            x = self.norm2(x + self.ff(x))
            return x

    class PatternTransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 3)
            self.pos_emb = nn.Embedding(seq_len, embed_dim)
            self.drop    = nn.Dropout(dropout)
            self.blocks  = nn.ModuleList([CausalTransformerBlock() for _ in range(num_blocks)])
            self.head    = nn.Sequential(
                nn.Linear(embed_dim * 2, 64), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1), nn.Sigmoid(),
            )
            self.seq_len = seq_len

        def forward(self, x):
            B, T = x.shape
            pos  = torch.arange(T, device=x.device).unsqueeze(0)
            emb  = self.drop(self.tok_emb(x) + self.pos_emb(pos))

            # Causal mask: upper triangle = -inf
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            for block in self.blocks:
                emb = block(emb, mask)

            last   = emb[:, -1, :]                        # (B, D)
            pooled = emb.mean(dim=1)                       # (B, D)
            out    = self.head(torch.cat([last, pooled], dim=-1))
            return out.squeeze(1)

    return PatternTransformerNet()


# ── Wrapper class ──────────────────────────────────────────────────────────────

class PatternTransformerModel:
    """
    Wrapper around the PyTorch transformer with the same interface as
    XGBModel / LGBMModel / LSTMModel for drop-in ensemble use.
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
        self.model = None

    def _get_vocab_size(self):
        from price_tokenizer import TOTAL_VOCAB
        return self.vocab_size or TOTAL_VOCAB

    def _pad_sequences(self, X: np.ndarray) -> np.ndarray:
        from price_tokenizer import PAD_TOKEN
        n      = len(X)
        result = np.full((n, self.seq_len), PAD_TOKEN, dtype=np.int64)
        for i, seq in enumerate(X):
            seq = np.asarray(seq, dtype=np.int64)
            if len(seq) >= self.seq_len:
                result[i] = seq[-self.seq_len:]
            else:
                result[i, -len(seq):] = seq
        return result

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        if len(X_train) == 0:
            return self

        p   = self.params
        # num_heads must divide embed_dim
        while p["embed_dim"] % p["num_heads"] != 0:
            p["num_heads"] = max(1, p["num_heads"] - 1)

        self.model = _build_torch_transformer(
            vocab_size = self._get_vocab_size(),
            seq_len    = self.seq_len,
            embed_dim  = p["embed_dim"],
            num_heads  = p["num_heads"],
            num_blocks = p["num_blocks"],
            ff_dim     = p["ff_dim"],
            dropout    = p["dropout_rate"],
        )

        Xp = torch.tensor(self._pad_sequences(X_train), dtype=torch.long)
        yp = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xp, yp),
                            batch_size=p["batch_size"], shuffle=False)

        val_loader = None
        if X_val is not None and len(X_val) > 0:
            Xvp = torch.tensor(self._pad_sequences(X_val), dtype=torch.long)
            yvp = torch.tensor(y_val, dtype=torch.float32)
            val_loader = DataLoader(TensorDataset(Xvp, yvp), batch_size=64)

        opt      = torch.optim.Adam(self.model.parameters(), lr=p["learning_rate"])
        sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
        loss_fn  = nn.BCELoss()
        best_val = float("inf")
        patience = 8
        no_impr  = 0
        best_st  = None

        for epoch in range(p["epochs"]):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model(xb), yb).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

            if val_loader:
                self.model.eval()
                vl = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        vl += loss_fn(self.model(xb), yb).item()
                vl /= len(val_loader)
                sched.step(vl)
                if vl < best_val - 1e-4:
                    best_val = vl
                    no_impr  = 0
                    best_st  = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    no_impr += 1
                    if no_impr >= patience:
                        break

        if best_st:
            self.model.load_state_dict(best_st)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self.model is None or len(X) == 0:
            return np.full(len(X), np.nan)
        self.model.eval()
        Xp = torch.tensor(self._pad_sequences(X), dtype=torch.long)
        with torch.no_grad():
            return self.model(Xp).numpy().flatten()

    def save(self, path: str):
        import torch
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), str(base) + "_weights.pt")
        with open(str(base) + "_config.pkl", "wb") as f:
            pickle.dump({
                "seq_len":    self.seq_len,
                "params":     self.params,
                "vocab_size": self.vocab_size,
            }, f)

    @classmethod
    def load(cls, path: str) -> "PatternTransformerModel":
        import torch
        base = Path(path)
        with open(str(base) + "_config.pkl", "rb") as f:
            cfg = pickle.load(f)
        obj = cls(**cfg)
        weights_path = str(base) + "_weights.pt"
        if Path(weights_path).exists():
            p = obj.params
            while p["embed_dim"] % p["num_heads"] != 0:
                p["num_heads"] = max(1, p["num_heads"] - 1)
            obj.model = _build_torch_transformer(
                vocab_size = obj._get_vocab_size(),
                seq_len    = obj.seq_len,
                embed_dim  = p["embed_dim"],
                num_heads  = p["num_heads"],
                num_blocks = p["num_blocks"],
                ff_dim     = p["ff_dim"],
                dropout    = p["dropout_rate"],
            )
            obj.model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
        return obj


# ── Optuna objective ───────────────────────────────────────────────────────────

def tune_transformer_objective(trial, X_tr, y_tr, X_val, y_val) -> float:
    from sklearn.metrics import roc_auc_score

    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    # ensure divisibility
    while embed_dim % num_heads != 0:
        num_heads = max(1, num_heads - 1)

    params = {
        "embed_dim":    embed_dim,
        "num_heads":    num_heads,
        "num_blocks":   trial.suggest_int("num_blocks", 2, 4),
        "ff_dim":       trial.suggest_categorical("ff_dim", [64, 128, 256]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.05, 0.4),
        "learning_rate":trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs":       trial.suggest_int("epochs", 15, 40),
    }

    model = PatternTransformerModel(params=params)
    model.fit(X_tr, y_tr, X_val, y_val)
    if model.model is None:
        return 0.5

    prob  = model.predict_proba(X_val)
    valid = ~np.isnan(prob)
    if valid.sum() < 5 or len(set(y_val[valid])) < 2:
        return 0.5
    return float(roc_auc_score(y_val[valid], prob[valid]))
