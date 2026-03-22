# 🇻🇳 Vietnam Stock AI — Production Trading System

> **Version 3.0** | HOSE / HNX / UPCOM | Live intraday fine-tuning | LLM sentiment | Trade signals

A production-grade AI system for Vietnam stock market prediction and trade signal generation. Runs continuously — learning from daily closes, fine-tuning on live intraday data while the market is open, and emitting actionable **BUY / SELL / HOLD** signals with entry price, stop-loss, and take-profit levels.

---

## What This System Does

```
Pre-market 08:30 ICT       Market open 09:00-14:45 ICT      Post-market 15:00 ICT
         |                           |                                |
   Fresh predictions         Live fine-tuning                 Full retrain +
   for the day using   every 15 min on intraday bars          Optuna tuning
   last night's model  -> update signal confidence            all symbols
```

---

## Feature Overview

### AI / ML Core
| Component | Detail |
|-----------|--------|
| XGBoost | Gradient boosting, Optuna-tuned (60 trials) |
| LightGBM | Gradient boosting, Optuna-tuned (60 trials) |
| LSTM | 2-layer sequence model (30-day window), Optuna-tuned (30 trials) |
| Ensemble | Optuna-optimised weighted blend of all three |
| Labels | Volatility-adjusted binary (drops near-zero noise moves) |
| Feature Selection | Top-40 features by LightGBM importance per symbol |

### Feature Engineering (80+ features)
| Category | Features |
|----------|----------|
| Returns | 1d/3d/5d/10d/20d; log returns |
| Trend | SMA/EMA 5-200; DEMA; MACD; ADX; DI |
| Momentum | RSI(7,14,21); Stochastic; CCI; Williams %R; ROC |
| Volatility | Bollinger Bands; ATR(7,14,21); Historical Vol; Keltner |
| Volume | OBV; MFI; CMF; Force Index; VWAP; Volume ratios |
| Candlestick | Doji; Hammer; Shooting Star; Engulfing; Gaps |
| Calendar | Day-of-week; month-end/quarter-end effects |
| Market | Beta(20d); Relative Strength vs VN-Index; VN-Index RSI |
| Intraday | Realized volatility (5-min); Volume imbalance; VWAP deviation; Open gap |
| Regime | Bull/bear regime; Volatility regime; 52W drawdown; MA200 distance |
| Sentiment | Claude LLM scores on Vietnamese financial news per symbol |
| Math Models | Fibonacci; Pivot Points; Ichimoku; HMA; Parabolic SAR; Donchian |

### Trade Signal Engine
- BUY / SELL / HOLD signal per symbol with confidence score
- Entry price range based on support/pivot alignment
- Stop-loss: ATR-based or below key support level
- Take-profit: Fibonacci extension or resistance target
- Risk/reward ratio (minimum 1:1.5 enforced)
- Position size suggestion based on volatility-adjusted Kelly fraction

---

## Installation

```bash
# 1. Download project
cd vietnam_stock_ai

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set Claude API key for LLM sentiment scoring
set ANTHROPIC_API_KEY=sk-ant-...     # Windows
# export ANTHROPIC_API_KEY=sk-ant-... # Linux/macOS

# 5. Create data directories
mkdir logs models results
mkdir data\cache data\intraday_cache data\sentiment_cache
```

---

## Quick Start

### Step 1 - Initial training
```bash
python main.py --mode train
```
Fetches 4 years of OHLCV data, engineers 80+ features, runs Optuna tuning, saves models.
With Optuna enabled: ~30-60 min per symbol. For a fast first run set OPTUNA_TRIALS_XGB = 10 in config.py.

### Step 2 - Generate pre-market predictions and trade signals
```bash
python main.py --mode predict
```
Outputs per symbol: direction, confidence %, BUY/SELL/HOLD signal, entry, stop-loss, take-profit, R/R ratio.

### Step 3 - Live intraday mode
```bash
python main.py --mode live
```
Runs from 09:00 to 14:45 ICT. Every 15 minutes fetches fresh intraday bars, updates signal confidence, re-emits signals if confidence threshold crossed.

### Step 4 - Launch dashboard
```bash
python main.py --mode dashboard
# Opens http://localhost:8501
```

### Step 5 - Full auto-daemon (recommended)
```bash
python main.py --mode daemon
```

Daily schedule (ICT):

| Time | Action |
|------|--------|
| 08:30 | Pre-market predictions + trade signals |
| 09:05 - 14:30 | Live fine-tuning every 15 min |
| 15:00 | Full retrain + Optuna tuning |
| 15:30 | Post-market analysis report |

---

## Trade Signal Interpretation

```
=== SIGNAL: FPT ===
Direction  : BUY
Confidence : 73.2%
Entry Zone : 118,500 - 119,200
Stop-Loss  : 116,800   (ATR-based, -1.8%)
Take-Profit: 123,500   (+3.6%)
R/R Ratio  : 1 : 2.0
Position   : 8.5% of capital
Basis      : MACD cross + RSI(14)=38 oversold + above MA200 + Fib 61.8% support
```

Signal rules:
- BUY: confidence >= 65% + price above MA200 + at least 2 confirming math models
- SELL: confidence >= 65% + price below MA50 + at least 2 confirming models
- HOLD: confidence 50-65% or conflicting signals
- Stop-loss = entry - (1.5 x ATR14) or below nearest support, whichever is tighter
- Take-profit = entry + (3.0 x ATR14) or nearest Fibonacci extension
- Minimum R/R = 1:1.5 — signals below this are suppressed

---

## Mathematical Models

### Trend
| Model | Usage |
|-------|-------|
| SMA / EMA (5, 10, 20, 50, 100, 200) | Trend direction, dynamic S/R |
| DEMA (Double EMA) | Faster trend, less lag |
| TEMA (Triple EMA) | Ultra-low lag trend |
| Hull Moving Average (HMA) | Smooth responsive trend |
| Ichimoku Cloud | Trend + momentum + S/R combined |
| Parabolic SAR | Trailing stop, trend reversal |
| Donchian Channel | Breakout detection |

### Momentum
| Model | Usage |
|-------|-------|
| RSI (7, 14, 21) | Overbought / oversold |
| Stochastic (14, 3) | Momentum crossover |
| MACD (12, 26, 9) | Trend momentum + crossover |
| CCI (20) | Extreme readings |
| Williams %R (14) | Short-term extremes |

### Support / Resistance
| Model | Usage |
|-------|-------|
| Pivot Points (Classic) | Daily S1/S2/S3, R1/R2/R3 |
| Pivot Points (Camarilla) | Tight intraday levels |
| Pivot Points (Woodie) | Open-weighted levels |
| Fibonacci Retracement | 23.6%, 38.2%, 50%, 61.8%, 78.6% |
| Fibonacci Extension | 127.2%, 161.8%, 200% targets |

### Volatility
| Model | Usage |
|-------|-------|
| Bollinger Bands (20, 2) | Squeeze / expansion |
| Keltner Channel | Volatility-adjusted bands |
| ATR (7, 14, 21) | Stop-loss sizing |

---

## Live Fine-Tuning Architecture

```
Market Opens 09:00 ICT
        |
        v
[live_engine.py - LiveTradingEngine]
        |
        |-- Every 15 min:
        |   1. Fetch latest 5-min bars (vnstock intraday API)
        |   2. Compute intraday features (realized vol, vol imbalance, VWAP dev)
        |   3. Re-score models on fresh features (XGB + LGBM, instant)
        |   4. Recalculate all math models (pivots, ATR, Fibonacci levels)
        |   5. Combine AI confidence + math model agreement score
        |   6. Emit updated BUY/SELL/HOLD signal if confidence changed > 5%
        |   7. Check live price vs stop-loss / take-profit levels
        |   8. Write to results/live_signals_YYYYMMDD.json
        |
Market Closes 14:45 ICT
        |
        v
[Post-market: full retrain + Optuna with today's close included]
```

Why this works without retraining every 15 minutes:
1. XGBoost and LightGBM are fast at inference — re-scoring on new features takes under 1 second per symbol
2. Intraday features (realized vol, volume imbalance) shift the model's confidence without a refit
3. Math models (pivots, ATR, VWAP, Fibonacci) are fully real-time — recalculated on every new bar
4. LSTM is frozen during market hours — it provides sequential context from history; intraday context comes from the tree models updating their feature inputs
5. Full retrain at 15:00 ICT captures the new daily close and resets everything cleanly

Signal confidence update formula:
```
live_confidence = 0.60 x model_confidence + 0.40 x math_model_agreement_score

math_model_agreement_score = (confirming_signals / total_signals)
  where each of: MACD, RSI, MA cross, Pivot level, Fibonacci level, Ichimoku
  votes +1 for bullish or -1 for bearish
```

---

## File Structure

```
vietnam_stock_ai/
|-- main.py                # CLI entry: train/predict/live/daemon/dashboard/backtest
|-- config.py              # All settings
|
|-- data_fetcher.py        # Daily OHLCV (vnstock + yfinance fallback)
|-- intraday_fetcher.py    # 5-min intraday bars + daily aggregated features
|-- sentiment.py           # Claude LLM news sentiment scoring
|
|-- features.py            # 80+ adaptive feature engineering
|-- math_models.py         # Fibonacci, Pivots, Ichimoku, HMA, Parabolic SAR
|-- target_engineering.py  # Vol-adjusted labels, purged CV, feature selection
|
|-- models.py              # XGBoost, LightGBM, LSTM, EnsembleModel
|-- tuner.py               # Optuna hyperparameter search
|-- pipeline.py            # TrainingPipeline + PredictionPipeline
|
|-- trade_signals.py       # BUY/SELL/HOLD signals, entry/SL/TP calculation
|-- live_engine.py         # Intraday fine-tuning loop during market hours
|-- scheduler.py           # APScheduler daemon
|
|-- backtester.py          # Walk-forward backtest with trading simulation
|-- dashboard.py           # Streamlit UI
|-- requirements.txt
|
|-- models/                # Saved model files
|-- data/
|   |-- cache/             # Daily OHLCV cache
|   |-- intraday_cache/    # 5-min bar cache
|   `-- sentiment_cache/   # LLM sentiment cache
|-- results/               # Predictions, signals, backtest outputs
`-- logs/                  # System logs
```

---

## Accuracy Expectations

| Metric | Typical Range | Notes |
|--------|--------------|-------|
| Test AUC | 0.60 - 0.75 | After vol-adjusted labels |
| Directional Accuracy | 55% - 65% | On clear signals (conf >= 65%) |
| Win Rate (backtested) | 52% - 60% | With R/R >= 1:2, mathematically profitable |
| Max Drawdown | -8% to -18% | Depends on position sizing |

A model with 55% win rate and 1:2 R/R ratio is mathematically profitable in the long run. You do not need 70%+ accuracy. You need consistent execution, good position sizing, and strict stop-losses.

---

## Configuration Reference

Key parameters in config.py and trade_signals.py:

```python
LOOKBACK_DAYS       = 1300    # ~4 years training history
PREDICTION_HORIZON  = 5       # Days ahead to predict (1, 3, 5, 10)
OPTUNA_TRIALS_XGB   = 60      # Reduce to 10 for fast testing
OPTUNA_TRIALS_LGBM  = 60
OPTUNA_TRIALS_LSTM  = 30

# Trade signal thresholds (in trade_signals.py)
MIN_CONFIDENCE      = 0.65    # Below this = HOLD
MIN_RR_RATIO        = 1.5     # Below this R/R = suppress signal
ATR_SL_MULTIPLIER   = 1.5     # Stop = entry - 1.5 x ATR14
ATR_TP_MULTIPLIER   = 3.0     # Target = entry + 3.0 x ATR14
MAX_POSITION_PCT    = 0.15    # Max 15% of portfolio per trade
LIVE_INTERVAL_MIN   = 15      # Intraday refresh every 15 minutes
```

---

## Troubleshooting

**vnstock not found:**
```bash
pip install vnstock --upgrade
```

**TensorFlow CPU only:**
```bash
pip install tensorflow-cpu
```

**Sentiment not activating:**
Make sure ANTHROPIC_API_KEY is set in your environment. Sentiment is optional — system works without it.

**Slow training:**
Set OPTUNA_TRIALS_XGB = 10 in config.py, or train a subset:
```bash
python main.py --mode train --symbols VNM VIC HPG FPT
```

**APScheduler version error:**
```bash
pip install "apscheduler>=3.10,<4.0"
```

---

## Disclaimer

This system is for educational and research purposes only. Vietnam stock trading involves significant risk of capital loss. The AI predictions, trade signals, and mathematical models do not constitute financial advice. Past model performance does not guarantee future results. Always consult a licensed securities broker or financial advisor before trading. The authors accept no liability for trading decisions made using this software.

Regulatory note: Ensure compliance with SSC (State Securities Commission of Vietnam) regulations before deploying with real capital.
