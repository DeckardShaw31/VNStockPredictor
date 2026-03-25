# Vietnam Stock AI — Production Trading System

Version 4.0 | HOSE / HNX / UPCOM | Real data | Local AI | Live signals | Portfolio analytics

---

## What this system is

An end-to-end AI trading system for the Vietnam stock market. It trains four AI models
locally on your machine (no external AI service required), generates BUY/SELL/HOLD
signals with entry, stop-loss, and take-profit levels, manages a live portfolio with
real-time P&L and risk analytics, and runs continuously throughout the trading day —
updating signal confidence every 15 minutes as new price bars come in.

---

## Architecture

```
DATA SOURCES
  vnstock (primary) + yfinance (fallback)
  Daily OHLCV (4 years) + 5-min intraday bars + VN-Index + VN30

FEATURE ENGINEERING  (~120 features per stock per day)
  Price & Returns       SMA/EMA 5-200, DEMA, TEMA, HMA, MACD, ADX
  Momentum              RSI(7/14/21), Stochastic, CCI, Williams %R, ROC
  Volatility            Bollinger Bands, ATR, Keltner, Historical Vol
  Volume                OBV, MFI, CMF, Force Index, VWAP
  Candlestick           Doji, Hammer, Engulfing, Gaps
  Math Models           Fibonacci, Pivots (3 variants), Ichimoku, SAR, Supertrend
                        Elder Ray, Squeeze Momentum, Donchian
  Intraday              Realized vol (5-min), Volume imbalance, VWAP deviation
  Market                Beta vs VN-Index, Relative strength, VN-Index RSI
  Sector                Sector momentum, Stock alpha vs sector peers
  Fear/Greed            VN-Index MA50 distance, RSI, volatility regime, VN30 breadth
  Calendar              Day-of-week, month-end, quarter-end effects
  Regime                Bull/bear/sideways regime, 52W drawdown, MA200 distance

AI MODELS  (all trained locally, PyTorch + scikit-learn + XGBoost + LightGBM)
  XGBoost               Gradient boosting, Optuna 60 trials
  LightGBM              Gradient boosting, Optuna 60 trials
  LSTM (PyTorch)        2-layer sequence model, 30-day window, Optuna 30 trials
  Pattern Transformer   Local GPT-style model on tokenized price sequences
                        Stage 1: Pre-train on all symbols combined (~37,000 sequences)
                        Stage 2: Fine-tune per symbol (transfer learning)
  Stacking Meta-Learner Logistic regression on OOF predictions from base models
  Platt Calibration     Sigmoid calibration applied to each base model

ENSEMBLE
  Weighted blend: XGB 30% + LGBM 35% + LSTM 25% + Meta 10%
  Weights optimised per symbol by Optuna on validation set

SIGNAL GENERATION
  Combined confidence = 0.5 + |AI_strength| + math_model_bonus
  BUY fires when: AI >= 54% bullish + combined confidence >= 52%
  SELL fires when: AI >= 54% bearish + combined confidence >= 52%
  Stop-loss: max(entry - 1.5xATR, nearest support level)
  Take-profit: min(entry + 3.0xATR, nearest resistance level)
  Position size: quarter-Kelly fraction, capped at 15% of capital
  R/R filter: signals with R/R < 1.5 are suppressed

PORTFOLIO ANALYTICS
  Per-position P&L, day change, days held, annualised return
  AI signal alignment (does model agree with your position?)
  Distance to stop-loss (alerts when < 3%, emergency exit when breached)
  Lot-by-lot tracking (multiple buy dates, averaging down support)
  Price unit auto-normalisation (vnstock thousands-VND vs full VND)

RISK MANAGEMENT
  Historical VaR (95%, 1-day and 5-day)
  Conditional VaR / Expected Shortfall
  Portfolio annualised volatility and Sharpe ratio
  Position correlation matrix (warns on >0.80 correlations)
  Sector exposure limits (max 35% per sector)
  Herfindahl concentration index
  Position size auto-reduction when portfolio risk is elevated

LIVE TRADING ENGINE
  Runs 09:00-14:45 ICT Monday-Friday
  Updates every 15 minutes: re-scores AI on fresh intraday features
  Recalculates all math models on every new bar (instant, no refit)
  Monitors live price vs stop-loss and take-profit levels
  Emits EXIT NOW signal immediately when SL/TP crossed
  Saves live signal log to results/live_signals_YYYYMMDD.json

DAILY SCHEDULE  (handled by --mode daemon)
  08:30 ICT  Pre-market predictions + trade signals
  09:05 ICT  Live intraday engine starts (runs until 14:45)
  15:05 ICT  Fast incremental update (~30s per symbol, no Optuna)
  15:30 ICT  Weekly full retrain with Optuna (Fridays only)
```

---

## Installation

Requires Python 3.10, 3.11, or 3.12. Python 3.13 works but TensorFlow is not
supported (the system uses PyTorch instead, which works fine on 3.13).

```
git clone / download the project
cd VNStockPredictor

python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux / macOS

pip install -r requirements.txt

# Create required directories
mkdir logs models results
mkdir data\cache data\intraday_cache data\sentiment_cache   # Windows
mkdir -p data/cache data/intraday_cache data/sentiment_cache  # Linux/macOS
```

Optional: set Claude API key for LLM news sentiment scoring (system works without it):

```
set ANTHROPIC_API_KEY=sk-ant-...      # Windows
export ANTHROPIC_API_KEY=sk-ant-...   # Linux/macOS
```

Suppress TensorFlow startup noise (not needed, but reduces log clutter):

```
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
```

---

## Usage — Command Reference

### Initial training (first time setup)

```
python main.py --mode train
```

Runs the full training pipeline for all symbols in DEFAULT_SYMBOLS:
  1. Fetches 4 years of daily OHLCV data and 30 days of 5-min intraday bars
  2. Builds 120+ features including sector momentum and fear/greed indicators
  3. Pre-trains the Pattern Transformer on all symbols combined (~37,000 sequences)
  4. For each symbol: Optuna tunes XGBoost (60 trials), LightGBM (60 trials),
     LSTM (30 trials), then fine-tunes the transformer
  5. Trains the stacking meta-learner on out-of-fold validation predictions
  6. Saves all models to models/

Time estimate with full Optuna: 30-60 minutes per symbol, 8-16 hours for all 20.
For a fast first run, edit config.py and set OPTUNA_TRIALS_XGB = 10.

To train only specific symbols:

```
python main.py --mode train --symbols VNM HPG FPT
```

### Generate predictions and trade signals

```
python main.py --mode predict
```

Outputs for every symbol:
  - AI direction (UP/DOWN) and confidence %
  - BUY/SELL/HOLD signal
  - Entry zone, stop-loss (with %), take-profit (with %)
  - Risk/reward ratio and suggested position size %
  - Math model votes (each model votes +1/-1/0)
  - Pivot levels and Fibonacci levels

### Fast daily update (run every day after 15:00 ICT)

```
python main.py --mode update
```

Takes ~30 seconds per symbol. Ingests today's new close price, refits
XGBoost and LightGBM with the existing hyperparameters, warm-starts the LSTM
for 5 more epochs. No Optuna. Run this every trading day.

### Live intraday mode (run during market hours)

```
python main.py --mode live
```

Runs from 09:00 to 14:45 ICT. Every 15 minutes:
  - Fetches fresh 5-min intraday bars
  - Recomputes intraday features and all math model levels
  - Re-scores the AI models on updated inputs (no refit — takes <1s per symbol)
  - Emits updated signal if confidence changed more than 5% or direction flipped
  - Monitors live prices against stop-loss and take-profit levels
  - Triggers EXIT NOW immediately if a stop is hit

### Dashboard

```
python main.py --mode dashboard
```

Opens http://localhost:8501 with 7 pages:

  Market Overview    AI predictions table, confidence heatmap, signal counts
  Stock Analysis     Candlestick chart with MA overlays, RSI, volume, math model
                     votes, ensemble weights, model AUC
  Portfolio          Holdings with lot-by-lot tracking, P&L, day change, AI
                     alignment, AI position recommendations, P&L waterfall
  Risk Analytics     VaR gauge, sector exposure chart, correlation matrix,
                     sector performance heatmap, concentration index
  Trade Signals      All signals sortable by direction/confidence, with
                     entry/SL/TP/R/R for every tracked symbol
  Train / Update     Buttons to run update or full train from the UI
  Backtest           Walk-forward backtest results and charts

### Full auto-daemon (recommended for production)

```
python main.py --mode daemon
```

Runs forever. Handles the complete daily schedule automatically:
  08:30  Predictions + signals
  09:05  Live engine (runs until market close)
  15:05  Daily fast update (Mon-Fri)
  15:30  Weekly full retrain with Optuna (Fridays only)

Background on Windows:

```
start /B python main.py --mode daemon > logs\daemon.log 2>&1
```

Background on Linux:

```
nohup python main.py --mode daemon > logs/daemon.log 2>&1 &
```

### Backtest

```
python main.py --mode backtest
```

Walk-forward backtest simulating monthly retraining and paper trading.
Reports AUC, accuracy, win rate, strategy return vs buy-and-hold,
Sharpe ratio, and max drawdown per symbol.

---

## Portfolio Management

### Adding a position

In the dashboard Portfolio tab, expand "Add New Position" and fill in:
  - Symbol (any symbol from DEFAULT_SYMBOLS)
  - Shares
  - Buy price in full VND (e.g. 25,800 for HPG, not 25.8)
  - Buy date
  - Optional note

The system automatically detects if vnstock is returning prices in thousands
of VND (e.g. SHS at 15 = 15,000 VND) and normalises the calculation so P&L
is always shown correctly.

### Adding another lot (averaging down, DCA)

Expand any position in the Portfolio tab and use the "Add another lot" form.
Each lot is tracked separately with its own buy date and note.

### Portfolio data storage

All holdings are saved to data/portfolio.json. Back this file up regularly.
Format:

```json
{
  "VNM": [
    {"shares": 1000, "buy_price": 65400, "buy_date": "2026-03-20", "note": ""},
    {"shares": 500,  "buy_price": 63200, "buy_date": "2026-02-14", "note": "DCA"}
  ]
}
```

---

## AI Recommendation Logic

The Portfolio AI Analysis section shows one of six recommendations per position:

  EXIT NOW            Price is already below your stop-loss level. Exit immediately.
  CUT LOSS            Near stop-loss + AI bearish signal. Exit before stop is hit.
  AVERAGE DOWN?       Big unrealised loss but AI says BUY. Consider adding if conviction high.
  CONSIDER SELLING    AI signals SELL with >58% confidence.
  WATCH CLOSELY       Near stop-loss but AI says BUY — potential support bounce.
  HOLD / ADD          AI agrees with your long, position profitable, trend intact.
  TAKE PARTIAL PROFIT Position up >15%. Lock in gains, move stop to breakeven.
  REDUCE / WATCH      AI disagrees with your long position — reduce exposure.
  HOLD                No strong signal either way. Monitor for changes.

---

## Risk Management Rules

The system enforces the following risk rules:

  Single position limit      Max 15% of total portfolio value in one stock
  Sector exposure limit      Max 35% of portfolio in one sector (auto-blocks signals)
  VaR reduction              If daily VaR > 3%, new position sizes are reduced 30%
  Correlation reduction      If new stock correlates >0.80 with existing, size -30%
  Concentration reduction    If HHI > 0.25 (portfolio concentrated), size -15%
  R/R minimum                Signals with reward/risk < 1.5 are suppressed (HOLD)
  Stop-loss enforcement      EXIT NOW shown immediately when price breaches stop

---

## Mathematical Models

### Trend
  SMA / EMA (5, 10, 20, 50, 100, 200)    Trend direction, dynamic S/R
  DEMA / TEMA                              Low-lag trend lines
  Hull Moving Average (20, 50)             Smooth, near-zero-lag trend
  MACD (12, 26, 9)                        Trend momentum and crossover
  ADX + DI (14)                           Trend strength and direction
  Ichimoku Cloud                           Full Tenkan/Kijun/Senkou/Chikou system

### Momentum
  RSI (7, 14, 21)                         Overbought / oversold
  Stochastic (14, 3)                      %K and %D crossover
  CCI (20), Williams %R (14)              Extreme readings
  Elder Ray                               Bull Power / Bear Power

### Support and Resistance
  Pivot Points Classic                     PP, S1/S2/S3, R1/R2/R3
  Pivot Points Camarilla                   Tight intraday levels
  Pivot Points Woodie                      Open-weighted levels
  Fibonacci Retracement                    23.6, 38.2, 50, 61.8, 78.6%
  Fibonacci Extension                      127.2, 161.8, 200% targets
  Donchian Channel (20)                    Breakout detection

### Volatility
  Bollinger Bands (20, 2)                 Squeeze and expansion
  Keltner Channel                          ATR-adjusted bands
  ATR (7, 14, 21)                         Stop-loss sizing
  Supertrend (10, 3.0)                    Trend direction + trailing stop
  Squeeze Momentum                        BB/KC squeeze breakout detector

---

## Pattern Transformer (Local GPT)

Each trading day is converted to a single composite token from a vocabulary of 978.
Each token encodes: return magnitude (13 bins) + volume vs average (5 bins) +
candlestick body shape (5 bins) + trend context above/below MA50 (3 bins).

A 60-day price history becomes a 60-token sequence. A causal transformer
(cannot look forward — same constraint as GPT) learns which sequences
most reliably precede upward or downward moves.

Training uses two stages:
  Stage 1  Pre-train on all 20 symbols combined (~40,000 sequences).
           Model learns general Vietnam market grammar.
           Corpus is cached and reused on subsequent runs (skips re-tokenization
           if cache is less than 23 hours old and saved model weights exist).
  Stage 2  Fine-tune per symbol for 20 epochs. The pre-trained weights are
           transferred and the model adapts to each stock's personality.

The transformer becomes the 4th ensemble member. Its weight is determined
by Optuna per symbol.

---

## File Structure

```
VNStockPredictor/
|
|-- main.py                  CLI: train/predict/update/live/daemon/dashboard/backtest
|-- config.py                All settings: symbols, sector map, risk limits, Optuna params
|-- requirements.txt
|
|-- Data
|-- data_fetcher.py          Daily OHLCV (vnstock primary, yfinance fallback)
|-- intraday_fetcher.py      5-min bars + daily realized vol / volume imbalance
|
|-- Features
|-- features.py              120+ adaptive features, sector momentum, fear/greed proxy
|-- math_models.py           Fibonacci, Pivots (3 variants), Ichimoku, HMA, TEMA,
|                            Parabolic SAR, Supertrend, Donchian, Elder Ray, Squeeze
|-- target_engineering.py   Vol-adjusted labels, purged CV, feature selection, regimes
|-- sentiment.py             Optional LLM news sentiment (requires ANTHROPIC_API_KEY)
|
|-- Pattern Transformer (local, PyTorch, no API)
|-- price_tokenizer.py       OHLCV -> 978-token vocabulary
|-- pattern_transformer.py   GPT-style causal transformer (PyTorch)
|-- pattern_dataset.py       Multi-symbol corpus builder
|-- transformer_pipeline.py  Stage 1 pre-train + Stage 2 per-symbol fine-tune
|
|-- AI Models
|-- models.py                XGBoost, LightGBM, LSTM (PyTorch), EnsembleModel,
|                            CalibratedModel (Platt scaling), StackingMetaLearner
|-- tuner.py                 Optuna hyperparameter search for all models
|
|-- Pipelines
|-- pipeline.py              TrainingPipeline + PredictionPipeline
|-- updater.py               Fast incremental update (no Optuna)
|-- live_engine.py           Intraday fine-tuning loop (09:00-14:45 ICT)
|-- scheduler.py             APScheduler daemon
|-- backtester.py            Walk-forward backtest with paper trading
|
|-- Signals and Risk
|-- trade_signals.py         BUY/SELL/HOLD generation, entry/SL/TP, position sizing,
|                            price unit normalisation (thousands-VND detection)
|-- risk_manager.py          VaR, CVaR, correlation matrix, sector exposure,
|                            Herfindahl index, position size adjustment
|
|-- Portfolio
|-- portfolio.py             Portfolio storage, lot tracking, P&L, AI recommendations,
|                            price unit normalisation for low-priced stocks
|
|-- UI
|-- dashboard.py             Streamlit: Market Overview, Stock Analysis, Portfolio,
|                            Risk Analytics, Trade Signals, Train/Update, Backtest
|
|-- models/                  Saved model files (auto-created)
|-- data/
|   |-- cache/               Daily OHLCV (6h TTL)
|   |-- intraday_cache/      5-min bars
|   `-- sentiment_cache/     LLM sentiment
|-- results/                 predictions_*.json, signals_*.json, backtest_*.json
`-- logs/                    system.log + daemon output
```

---

## Configuration Reference

Edit config.py before running. Key parameters:

```python
DEFAULT_SYMBOLS         Add or remove stock tickers. HPA supported (adaptive windows).

LOOKBACK_DAYS = 1300    ~4 years of training data. Reduce to 730 for faster runs.

PREDICTION_HORIZON = 5  Days ahead. Options: 1, 3, 5, 10.

OPTUNA_TRIALS_XGB = 60  Reduce to 10 for fast testing.
OPTUNA_TRIALS_LGBM = 60
OPTUNA_TRIALS_LSTM = 30

CALIBRATE_PROBABILITIES = True   Platt scaling on base models. Improves calibration.
USE_STACKING = True              Stacking meta-learner. Improves AUC by 0.01-0.03.

MIN_AI_CONFIDENCE = 0.52   Fires BUY/SELL when AI is 54%+ confident.
MIN_RR_RATIO = 1.5         Signals below this R/R are suppressed to HOLD.
ATR_SL_MULTIPLIER = 1.5    Stop-loss distance in ATR multiples.
ATR_TP_MULTIPLIER = 3.0    Take-profit distance in ATR multiples.

MAX_SECTOR_EXPOSURE_PCT = 0.35   Hard block when sector reaches 35% of portfolio.
MAX_SINGLE_POSITION_PCT = 0.15   Kelly fraction capped at this.
VAR_CONFIDENCE_LEVEL = 0.95      95% VaR calculation.
```

---

## Accuracy Expectations

Stock direction prediction is inherently noisy. Realistic ranges:

  Test AUC (model quality)          0.60 - 0.78
  Directional accuracy on signals   55% - 65%
  Win rate after R/R filter         52% - 60%
  Expected Sharpe ratio             0.8 - 2.0

A model with 55% win rate and 1:2 R/R has expected value of
0.55 * 2 - 0.45 * 1 = 0.65 units per trade. This is profitable.
You do not need 70% accuracy. You need disciplined stop-losses
and position sizing that prevents any single loss from being catastrophic.

What genuinely improves performance:
  1. Vol-adjusted labels (drops ambiguous near-zero moves)
  2. Probability calibration (raw scores are often overconfident)
  3. Stacking meta-learner (learns which model is right on which stocks)
  4. Feature selection (removes noise that hurts OOB generalisation)
  5. Sector features (distinguishes stock-specific vs sector-wide moves)
  6. The Pattern Transformer (catches repeating chart patterns in sequence context)

---

## Troubleshooting

vnstock not found:
```
pip install vnstock --upgrade
```

APScheduler version error:
```
pip install "apscheduler>=3.10,<4.0"
```

TensorFlow DLL error on Python 3.13:
  This is expected. The system uses PyTorch for all neural network components.
  Run: pip uninstall tensorflow tensorflow-intel tf-nightly keras -y
  TensorFlow is not needed.

Signal suppressed (R/R too low):
  This happens when ATR is very small due to vnstock thousands-VND pricing.
  The trade_signals.py auto-detection handles this. If you see persistent
  suppression, check that your stock is in the correct price scale.

Portfolio P&L shows -99%:
  You entered the buy price in full VND (e.g. 21,829) but vnstock returns
  the stock in thousands (e.g. 15 = 15,000 VND). The portfolio module
  auto-detects and normalises this. If it still appears wrong, try
  re-entering the position.

HPA / new listing gives very few features:
  This is handled automatically. All rolling windows shrink to fit available
  data. The model improves daily as more bars accumulate.

Training is slow:
  Set OPTUNA_TRIALS_XGB = 10, OPTUNA_TRIALS_LGBM = 10, OPTUNA_TRIALS_LSTM = 5
  in config.py. Or train a subset:
    python main.py --mode train --symbols VNM HPG FPT VCB

---

## Disclaimer

For educational and research purposes only. Vietnam stock trading involves
significant risk of capital loss. The AI signals, math models, and risk
metrics in this system do not constitute financial advice. Past performance
does not guarantee future results. Always consult a licensed securities broker
before committing real capital. The authors accept no liability for trading
decisions made using this software.

Ensure compliance with SSC (State Securities Commission of Vietnam)
regulations before running with real capital.
