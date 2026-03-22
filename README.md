# Vietnam Stock AI — Production Trading System

HOSE / HNX / UPCOM | Real data | Local GPT transformer | Live intraday signals

---

## What this system is

An end-to-end AI trading system for the Vietnam stock market. It fetches real
price data, trains four AI models (including a locally-run GPT-style transformer
that learns chart patterns the same way language models learn grammar), generates
BUY/SELL/HOLD signals with entry price, stop-loss, and take-profit levels, and
runs continuously throughout the trading day, updating signal confidence every
15 minutes as new intraday bars come in.

No external AI API is required. Every model trains and runs on your local machine.

---

## The Pattern Transformer — what it actually does

The core insight comes from treating price history exactly like text.

A language model learns:  "once upon a time" -> predicts "there was"
This transformer learns:  "FLAT|DOJI, BIG_UP|BULL_STR|VOL_HIGH, SMALL_UP..." -> predicts UP or DOWN

Each trading day is converted into a single composite token from a vocabulary of 978
tokens. Each token encodes four things at once: the return magnitude and direction
(13 bins from CRASH to SURGE), volume relative to the 20-day average (5 bins), the
candlestick body shape (5 bins from BEAR_STRONG to BULL_STRONG), and whether the
price is above or below the 50-day moving average (3 bins). So a day where the stock
jumped 2.5% on twice normal volume with a bullish engulfing candle above its MA50
becomes one token: SURGE_UP|VOL_HIGH|BULL_STR|ABOVE_MA.

A 60-day price history becomes a 60-token sequence. The transformer reads these
sequences using causal self-attention (the same mechanism in GPT — it can only look
backwards, never forward) and learns which patterns most reliably precede upward or
downward moves. Head-and-shoulders, breakout consolidations, volume climaxes,
oversold bounces — if they repeat, the attention mechanism finds them.

Training happens in two stages, exactly like how large language models are built:

Stage 1 — Pre-training on the full market corpus. All 16+ symbols are combined into
one dataset of 20,000+ sequences. The model learns general Vietnam stock market
grammar. A pattern that appears before rallies in VNM, VCB, FPT, and HPG is a
stronger signal than one seen only in one stock.

Stage 2 — Per-symbol fine-tuning. The pre-trained weights are transferred to each
individual stock and the model adapts to that stock's personality (a banking stock
behaves differently from a steel stock) without forgetting the general patterns it
learned in pre-training. This is the same process as fine-tuning GPT on a specific
domain after training on the general internet.

The transformer then becomes the fourth member of the ensemble alongside XGBoost,
LightGBM, and LSTM. Optuna finds the optimal blend weight for each symbol.

---

## System overview

```
                        TRAINING (run once, then nightly)
                        ----------------------------------
  Daily OHLCV (4 years)    5-min intraday bars      VN-Index (benchmark)
          |                        |                        |
          v                        v                        v
   Feature Engineering      Intraday Features        Market-Relative
   (80+ indicators)         (realized vol,           (beta, rel strength,
                             vol imbalance,           VN-Index RSI)
                             VWAP deviation)
          |                        |                        |
          +------------------------+------------------------+
                                   |
                                   v
                        Vol-Adjusted Labels
                        (drops near-zero noise)
                                   |
                    +--------------+--------------+
                    |              |              |
                    v              v              v
              XGBoost          LightGBM         LSTM
              Optuna           Optuna           Optuna
              60 trials        60 trials        30 trials
                    |              |              |
                    |              |              |
                    v              v              v
                    +----Optuna 4-model blend-----+
                                   |
                                   v
                    Pattern Transformer (GPT-style)
                    Stage 1: Pre-train on all 16 symbols
                    Stage 2: Fine-tune per symbol
                                   |
                                   v
                          Ensemble Prediction
                          (weighted probability)


                    LIVE (runs 09:00-14:45 ICT daily)
                    -----------------------------------
  Every 15 minutes:
  1. Fetch latest 5-min bars
  2. Recompute intraday features
  3. Re-score XGB + LGBM on fresh features (instant, no retrain)
  4. Recalculate all math models (Pivots, Fibonacci, Ichimoku, etc.)
  5. Combine AI confidence + math model agreement vote
  6. Check live price vs stop-loss / take-profit levels
  7. Emit updated signal if confidence changed >5% or direction flipped


                    DAILY SCHEDULE (ICT timezone)
                    ------------------------------
  08:30  Pre-market predictions + full trade signals for the day
  09:05  Live engine starts — updates every 15 min until market close
  14:45  Market closes — live engine stops, runs final update
  15:00  Full retrain + Optuna tuning on today's new close data
```

---

## Installation

Requires Python 3.10 or 3.11. Python 3.13 works but TensorFlow support is limited.

```
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create data directories (Windows)
mkdir logs
mkdir models
mkdir results
mkdir data\cache
mkdir data\intraday_cache
mkdir data\sentiment_cache

# 3. Create data directories (Linux / macOS)
mkdir -p logs models results data/cache data/intraday_cache data/sentiment_cache
```

If you want to suppress TensorFlow oneDNN messages:

```
# Windows
set TF_ENABLE_ONEDNN_OPTS=0

# Linux / macOS
export TF_ENABLE_ONEDNN_OPTS=0
```

APScheduler version: pin to 3.x to avoid API differences.

```
pip install "apscheduler>=3.10,<4.0"
```

---

## Usage

### Step 1 — Train all models

```
python main.py --mode train
```

This does the following in order:

1. Fetches OHLCV data for all symbols in DEFAULT_SYMBOLS (~4 years by default)
2. Fetches 5-min intraday bars for realized volatility and volume imbalance features
3. Builds the full 80+ feature matrix with adaptive rolling windows
4. Applies volatility-adjusted labels (drops ambiguous near-zero moves)
5. Pre-trains the Pattern Transformer on the full multi-symbol corpus
6. For each symbol: runs Optuna tuning on XGBoost (60 trials), LightGBM (60 trials),
   LSTM (30 trials), then fine-tunes the transformer
7. Optimises 4-model ensemble blend weights per symbol via Optuna
8. Saves all models to models/ and evaluation metrics to results/

Estimated time with full Optuna tuning: 30-60 minutes per symbol.
For a fast first run, set OPTUNA_TRIALS_XGB = 10 in config.py.

```
python main.py --mode train --symbols VNM FPT HPG
```

Train only specific symbols to test the pipeline before running the full list.

---

### Step 2 — Generate pre-market predictions and trade signals

```
python main.py --mode predict
```

Outputs for every symbol:

```
====================================================
  VNM    BUY  [UP]  Conf=71.4%  AUC=0.682
====================================================
  Entry zone :    65,400 -     66,100
  Stop-loss  :    64,200  (-2.1%)
  Take-profit:    68,800  (+4.4%)
  R/R ratio  : 1 : 2.10
  Position   : 9.3% of capital
  Pivot PP   :    65,800  S1=64,400  R1=67,200
  Fib 61.8%  :    64,600  38.2%=65,100
  Basis      : AI conf=71.4% | Models: supertrend, ichimoku, hma20 | ATR14=820
  Math votes : {'ichimoku': 1, 'supertrend': 1, 'psar': 1, 'hma20': 1, ...}
```

The signal is only emitted as BUY or SELL if:
- Combined confidence (AI + math model agreement) is at or above 65%
- The risk/reward ratio is at or above 1:1.5
- If either condition fails, the signal is HOLD

---

### Step 3 — Live intraday mode

```
python main.py --mode live
```

Run this during market hours (09:00-14:45 ICT). The engine fetches fresh 5-min
bars every 15 minutes and re-scores all signals. It does not retrain the models
intraday — it re-scores them on updated feature inputs, which takes under one second
per symbol. All math models (pivots, Fibonacci, ATR levels, Ichimoku) are fully
recalculated on each bar.

If a live price crosses a stop-loss or take-profit level that was set at the open,
an EXIT or STOP signal is emitted immediately without waiting for the next 15-min
tick. All live signals are written to results/live_signals_YYYYMMDD.json.

---

### Step 4 — Full daemon (recommended for production use)

```
python main.py --mode daemon
```

Runs the full automated schedule. Blocks forever. Use Ctrl+C to stop.

To run in the background on Windows:

```
start /B python main.py --mode daemon > logs\daemon.log 2>&1
```

To run in the background on Linux:

```
nohup python main.py --mode daemon > logs/daemon.log 2>&1 &
```

---

### Step 5 — Launch the dashboard

```
python main.py --mode dashboard
```

Opens http://localhost:8501 — Streamlit UI with candlestick charts, signal confidence
heatmap, trade signal table, model performance metrics, and a backtest results viewer.

---

### Step 6 — Backtest

```
python main.py --mode backtest
```

Runs a walk-forward backtest simulating monthly retraining and paper trading. Reports
AUC, accuracy, win rate, strategy return vs buy-and-hold, Sharpe ratio, and max
drawdown per symbol.

---

## How trade signals are calculated

### Signal confidence

```
combined_confidence = 0.60 x AI_ensemble_probability
                    + 0.40 x math_model_agreement_score

math_model_agreement_score = (bullish_votes - bearish_votes) / total_votes

Each of these models votes +1 (bullish) or -1 (bearish):
  Ichimoku Cloud position and TK cross
  Supertrend direction
  Parabolic SAR direction
  Hull Moving Average (20) cross
  Donchian Channel breakout
  Squeeze Momentum direction
  Pivot Point position (above R1 or below S1)
  Fibonacci level proximity (within 1% of 61.8% retracement)
```

### Stop-loss

Computed as the tighter of two levels:
- ATR-based: entry price minus 1.5 x ATR(14)
- Support-based: just below the nearest Pivot S1 or Fibonacci retracement level

Hard cap: stop-loss cannot be more than 4% below entry price.

### Take-profit

Computed as the nearer of two levels:
- ATR-based: entry price plus 3.0 x ATR(14)
- Resistance-based: nearest Pivot R1 or Fibonacci extension level above current price

Hard floor: take-profit is always at least 2% above entry price.

### Position sizing

Uses a quarter-Kelly fraction based on the signal confidence and the expected
risk/reward ratio, capped at 15% of total capital per position:

```
Kelly fraction = (win_probability x reward - loss_probability x risk) / reward
Position size  = Kelly fraction x 0.25   (quarter-Kelly for safety)
Position size  = min(position_size, 15% of capital)
```

If the signal is HOLD, position size is 0.

---

## Mathematical models

### Trend models

| Model | Description |
|-------|-------------|
| SMA 5, 10, 20, 50, 200 | Simple moving averages, trend direction and dynamic S/R |
| EMA 5, 10, 20, 50, 200 | Exponential moving averages, faster reaction than SMA |
| DEMA (20) | Double EMA — reduces lag compared to plain EMA |
| TEMA (20) | Triple EMA — near-zero lag trend line |
| HMA (20, 50) | Hull Moving Average — smooth and responsive, minimal lag |
| MACD (12, 26, 9) | Trend momentum and crossover signals |
| ADX + DI (14) | Trend strength and direction |
| Ichimoku Cloud | Full system: Tenkan/Kijun/Senkou A+B/Chikou with TK cross signals |
| Parabolic SAR | Trailing stop level and trend reversal detection |

### Momentum models

| Model | Description |
|-------|-------------|
| RSI (7, 14, 21) | Overbought / oversold conditions |
| Stochastic (14, 3) | %K and %D momentum crossover |
| CCI (20) | Commodity Channel Index for extreme readings |
| Williams %R (14) | Short-term overbought / oversold |
| Rate of Change (5, 10, 20) | Momentum strength measurement |
| Elder Ray | Bull Power and Bear Power relative to EMA |

### Support and resistance models

| Model | Description |
|-------|-------------|
| Pivot Points (Classic) | Daily PP, S1/S2/S3, R1/R2/R3 from previous session |
| Pivot Points (Camarilla) | Tighter intraday levels using 1.1/12 multiplier |
| Pivot Points (Woodie) | Open-weighted pivot, popular with session traders |
| Fibonacci Retracement | 23.6%, 38.2%, 50%, 61.8%, 78.6% from swing high/low |
| Fibonacci Extension | 127.2%, 161.8%, 200% projection targets |
| Donchian Channel (20) | 20-day highest high and lowest low breakout band |

### Volatility models

| Model | Description |
|-------|-------------|
| Bollinger Bands (20, 2) | Band squeeze and expansion, mean reversion signals |
| Keltner Channel (20, ATR x 2) | ATR-adjusted volatility bands |
| ATR (7, 14, 21) | Average True Range for position sizing and stop placement |
| Historical Volatility (10, 20, 30d) | Annualised close-to-close volatility |
| Supertrend (10, 3.0) | Combines ATR with trend direction, very popular in Vietnam |
| Squeeze Momentum | Detects Bollinger Band squeeze inside Keltner Channel |

---

## Pattern Transformer architecture

```
Input: 60-token sequence (one token per trading day)
Each token: composite of return bin + volume bin + body shape bin + trend bin
Vocabulary: 978 tokens (975 composite + PAD + BOS + EOS)

Token Embedding (978 -> 64 dim)
    +
Position Embedding (60 positions, learnable)
    |
    v
TransformerBlock x 3:
  MultiHeadAttention (4 heads, causal mask — cannot see future tokens)
  LayerNorm + Residual
  FeedForward (64 -> 128 -> 64, GELU activation)
  LayerNorm + Residual
    |
    v
Last token representation + Global Average Pooling
    |
    v
Dense(64, GELU) -> Dropout(0.1) -> Dense(1, Sigmoid)
    |
    v
P(UP) in [0, 1]

Training:
  Stage 1 (Pre-training):  All symbols combined, 50 epochs, batch 64
  Stage 2 (Fine-tuning):   Per symbol, 20 epochs, batch 16, LR 3e-4
  Optuna architecture search: 20 trials on embed_dim, num_heads, num_blocks, ff_dim
```

---

## Feature engineering (80+ features)

| Category | Features |
|----------|----------|
| Returns | 1d, 3d, 5d, 10d, 20d forward returns; log returns |
| Trend | SMA/EMA 5-200; DEMA; MACD + signal + histogram; ADX; +DI/-DI |
| Momentum | RSI(7,14,21); Stochastic %K/%D; CCI(20); Williams %R; ROC(5,10,20) |
| Volatility | Bollinger Bands width and %B; ATR(7,14,21); Historical Vol; Keltner %position |
| Volume | OBV; MFI(14); CMF(20); Force Index(13); VWAP(20); Volume SMA ratios |
| Candlestick | Body size; upper/lower wick; Doji; Hammer; Shooting Star; Engulfing; Gaps |
| Calendar | Day of week; week of year; month; quarter; month-end; quarter-end |
| Market | Beta(20d) vs VN-Index; Relative Strength(20d); VN-Index RSI; VN-Index momentum |
| Intraday | Realized volatility from 5-min bars; Volume imbalance (up vs down tick); VWAP deviation; Open gap; Intraday momentum (morning vs full session); Range ratio |
| Regime | Bull/bear regime flag; volatility regime; distance from MA200; 52-week drawdown; market regime vs VN-Index |
| Math Models | Fibonacci levels; Pivot Points (all variants); Ichimoku components; HMA; TEMA; Parabolic SAR; Donchian; Supertrend; Elder Ray; Squeeze Momentum |

All rolling windows are adaptive — they automatically shrink for newly listed stocks
like HPA that have less history, so every symbol in your watchlist can be trained
regardless of how long it has been on the exchange.

---

## File structure

```
vietnam_stock_ai/
|
|-- main.py                   Entry point for all modes
|-- config.py                 All settings: symbols, horizons, tuning params
|-- requirements.txt          Python dependencies
|
|-- Data
|-- data_fetcher.py           Daily OHLCV via vnstock (primary) + yfinance (fallback)
|-- intraday_fetcher.py       5-min intraday bars + daily aggregated features
|
|-- Feature Engineering
|-- features.py               80+ adaptive features, integrates all modules below
|-- math_models.py            Fibonacci, Pivots, Ichimoku, HMA, TEMA, SAR, Supertrend, etc.
|-- target_engineering.py     Vol-adjusted labels, purged walk-forward CV, feature selection, regime detection
|-- sentiment.py              Optional: Claude LLM news sentiment (requires ANTHROPIC_API_KEY)
|
|-- Pattern Transformer (local GPT — no API needed)
|-- price_tokenizer.py        Converts OHLCV to 978-token vocabulary sequences
|-- pattern_transformer.py    GPT-style transformer architecture (Keras, runs locally)
|-- pattern_dataset.py        Multi-symbol corpus builder for pre-training
|-- transformer_pipeline.py   Stage 1 pre-train + Stage 2 per-symbol fine-tune
|
|-- Classical ML Models
|-- models.py                 XGBoost, LightGBM, LSTM, EnsembleModel wrappers
|-- tuner.py                  Optuna hyperparameter search for all 4 models
|
|-- Pipelines
|-- pipeline.py               TrainingPipeline + PredictionPipeline
|-- live_engine.py            Intraday fine-tuning and signal update loop
|-- scheduler.py              APScheduler daemon (08:30 / 09:05 / 15:00 ICT)
|-- backtester.py             Walk-forward backtest with paper trading simulation
|
|-- Output
|-- trade_signals.py          BUY/SELL/HOLD generation, entry/SL/TP, position sizing
|-- dashboard.py              Streamlit UI
|
|-- models/                   Saved model files (auto-created on first train)
|-- data/
|   |-- cache/                Daily OHLCV cache (6h TTL)
|   |-- intraday_cache/       5-min bar cache
|   `-- sentiment_cache/      LLM sentiment cache
|-- results/                  Predictions, signals, backtest reports (JSON)
`-- logs/                     System logs
```

---

## Configuration reference

All settings are in config.py. The most commonly changed parameters are:

```python
DEFAULT_SYMBOLS     List of stock tickers to track. Edit freely.
                    HPA is included and supported even with very short history.

LOOKBACK_DAYS       How many calendar days of history to fetch for training.
                    Default 730 (2 years). Increase to 1300 for ~4 years.

PREDICTION_HORIZON  How many trading days ahead to predict.
                    Default 5. Valid options: 1, 3, 5, 10.

OPTUNA_TRIALS_XGB   Number of Optuna trials for XGBoost tuning. Default 60.
                    Set to 10 for a fast first run.

OPTUNA_TRIALS_LGBM  Number of Optuna trials for LightGBM tuning. Default 60.

OPTUNA_TRIALS_LSTM  Number of Optuna trials for LSTM tuning. Default 30.

RETRAIN_HOUR        Hour (ICT) at which the nightly retrain triggers. Default 15.
RETRAIN_MINUTE      Minute offset. Default 0. So 15:00 ICT.
```

Trade signal thresholds are in trade_signals.py:

```python
MIN_CONFIDENCE      Minimum combined confidence to emit BUY or SELL. Default 0.65.
MIN_RR_RATIO        Minimum risk/reward ratio. Signals below this become HOLD. Default 1.5.
ATR_SL_MULTIPLIER   Stop-loss distance in ATR multiples. Default 1.5.
ATR_TP_MULTIPLIER   Take-profit distance in ATR multiples. Default 3.0.
MAX_POSITION_PCT    Maximum position size as fraction of capital. Default 0.15 (15%).
KELLY_FRACTION      Safety scaling on Kelly criterion. Default 0.25 (quarter-Kelly).
```

Live engine settings are in live_engine.py:

```python
LIVE_INTERVAL_MIN         Minutes between intraday updates. Default 15.
SIGNAL_CHANGE_THRESHOLD   Minimum confidence change to re-emit a signal. Default 0.05.
```

---

## Accuracy and realistic expectations

A 50% accuracy on 5-day binary prediction is not a failure — it is the baseline.
The market is noisy and genuinely hard to predict. The improvements in this system
work by reducing noise (vol-adjusted labels that drop ambiguous near-zero moves),
finding patterns others miss (the transformer's attention over 60-day sequences),
and applying strict trade filters (R/R ratio enforcement and position sizing) that
make a 55-58% win rate mathematically profitable.

Realistic performance ranges from backtesting:

```
Test AUC (model quality)     0.60 - 0.75   after vol-adjusted labels
Directional accuracy         55% - 65%     on signals where confidence >= 65%
Win rate on executed trades  52% - 60%     after R/R filter applied
Expected Sharpe ratio        0.8 - 1.8     depending on symbol and market regime
```

A 55% win rate with a 1:2 risk/reward ratio produces the following expected value
per trade: 0.55 x 2 - 0.45 x 1 = 0.65 units of profit per unit risked.
You do not need 70%+ accuracy. You need consistent execution, enforced stop-losses,
and position sizing that keeps any single loss from being catastrophic.

---

## Troubleshooting

vnstock not found or import error:
```
pip install vnstock --upgrade
```

TensorFlow CPU warning about AVX/AVX2 instructions — these are harmless.
To suppress them: set TF_CPP_MIN_LOG_LEVEL=2 in your environment.

APScheduler next_run_time attribute error:
```
pip install "apscheduler>=3.10,<4.0"
```

Symbol has too few rows (e.g. HPA with 26 rows):
The system handles this automatically. All rolling windows shrink to fit the
available data. Models will still train and improve daily as more data accumulates.

Training fails with XGBoost base_score error:
This happens when a split has zero rows. The pipeline now catches this and either
falls back to simple binary labels or skips the symbol with a clear error message.

LSTM save error with .keras extension:
Keras 3.x requires the .keras extension for saved models. The current code uses
this correctly. If you have model files from an older version, delete the models/
folder and retrain.

---

## Disclaimer

This system is for educational and research purposes only. Vietnam stock trading
involves significant risk of capital loss. The AI predictions, trade signals, and
mathematical models in this system do not constitute financial advice. Past model
performance does not guarantee future results. Always consult a licensed securities
broker or financial advisor before committing real capital. The authors accept no
liability for any trading decisions made using this software.

Before running with real capital, verify that your use of automated analysis tools
complies with the regulations of the State Securities Commission of Vietnam (SSC)
and the rules of the exchange on which you intend to trade.
