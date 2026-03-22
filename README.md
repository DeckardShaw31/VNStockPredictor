# 🇻🇳 Vietnam Stock AI Prediction System

A production-grade AI system for predicting Vietnam stock market direction (HOSE/HNX/UPCOM), featuring:

- **Real data** from `vnstock` (primary) + `yfinance` (fallback)
- **Ensemble model**: XGBoost + LightGBM + LSTM (Bi-directional)
- **60+ technical features**: price action, trend, momentum, volatility, volume, candlestick patterns, market-relative
- **Optuna hyperparameter tuning** (XGB: 60 trials, LGBM: 60 trials, LSTM: 30 trials)
- **Ensemble weight optimisation** via Optuna
- **Daily auto-retraining** daemon (APScheduler, Vietnam ICT timezone)
- **Walk-forward backtesting** with Sharpe ratio & max drawdown
- **Streamlit dashboard** with candlestick charts, prediction confidence heatmap, model metrics

---

## 🛠️ Installation

```bash
# 1. Clone / download this project
cd vietnam_stock_ai

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create directories
mkdir -p logs models data/cache results
```

---

## 🚀 Quick Start

### Step 1 — Train models (first time)
```bash
python main.py --mode train
```
This will:
- Fetch ~2 years of OHLCV data for 15 Vietnamese stocks
- Engineer 60+ features (RSI, MACD, Bollinger Bands, etc.)
- Run Optuna to tune XGBoost, LightGBM, and LSTM
- Train an ensemble model per symbol
- Evaluate on held-out test set
- Save models to `models/`

> ⏱️ With Optuna tuning enabled: ~30–60 min per symbol.  
> For a fast first run, edit `config.py` and set `OPTUNA_TRIALS_XGB = 10`, etc.

### Step 2 — Generate predictions
```bash
python main.py --mode predict
```
Outputs directional predictions (UP/DOWN), confidence %, and target price for each symbol.

### Step 3 — Launch dashboard
```bash
python main.py --mode dashboard
# → Opens http://localhost:8501
```

### Step 4 — Enable daily auto-retraining
```bash
# Run in foreground (Ctrl+C to stop)
python main.py --mode daemon

# Run in background
nohup python main.py --mode daemon > logs/daemon.log 2>&1 &
```

The daemon:
- **09:05 ICT** (Mon–Fri): Generates fresh predictions before market open
- **15:00 ICT** (Mon–Fri): Retrains & Optuna-tunes all models after market close

---

## 📂 File Structure

```
vietnam_stock_ai/
├── main.py           # Entry point (train / predict / daemon / dashboard)
├── config.py         # All configuration (symbols, hyperparameters, paths)
├── data_fetcher.py   # vnstock + yfinance data fetching with caching
├── features.py       # 60+ feature engineering functions
├── models.py         # XGBoost, LightGBM, LSTM, EnsembleModel wrappers
├── tuner.py          # Optuna-based hyperparameter optimisation
├── pipeline.py       # TrainingPipeline + PredictionPipeline
├── scheduler.py      # APScheduler daily retrain daemon
├── backtester.py     # Walk-forward backtesting engine
├── dashboard.py      # Streamlit UI
├── requirements.txt
│
├── models/           # Saved model files (auto-created)
├── data/cache/       # Cached OHLCV data (auto-created)
├── results/          # Prediction & backtest JSON outputs (auto-created)
└── logs/             # Log files (auto-created)
```

---

## ⚙️ Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_SYMBOLS` | 15 symbols | Stocks to track (edit freely) |
| `LOOKBACK_DAYS` | 730 | Days of history for training |
| `PREDICTION_HORIZON` | 5 | Days ahead to predict |
| `OPTUNA_TRIALS_XGB` | 60 | XGBoost tuning trials |
| `OPTUNA_TRIALS_LGBM` | 60 | LightGBM tuning trials |
| `OPTUNA_TRIALS_LSTM` | 30 | LSTM tuning trials |
| `RETRAIN_HOUR` | 15 | Hour (ICT) for daily retrain |
| `ENSEMBLE_WEIGHTS` | XGB 35%, LGBM 35%, LSTM 30% | Default ensemble blend |

---

## 📊 Model Architecture

```
Input: 60+ features per trading day
    ↓
┌─────────────────────────────────┐
│  XGBoost (Gradient Boosting)    │  → P(up)_xgb
│  Optuna: 60 trials              │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│  LightGBM (Gradient Boosting)   │  → P(up)_lgbm
│  Optuna: 60 trials              │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│  LSTM (sequence 30 days)        │  → P(up)_lstm
│  2 LSTM layers + Dropout        │
│  Optuna: 30 trials              │
└─────────────────────────────────┘
         ↓ Optuna weight blend
    Ensemble P(up) ∈ [0,1]
         ↓ threshold=0.5
    Direction: UP / DOWN
```

---

## 📈 Features (60+)

| Category | Features |
|----------|----------|
| Returns | 1d, 3d, 5d, 10d, 20d returns; log returns |
| Trend | SMA/EMA (5,10,20,50,200); DEMA; MACD; ADX; ±DI |
| Momentum | RSI (7,14,21); Stochastic; CCI; Williams %R; ROC; Momentum |
| Volatility | Bollinger Bands (width, %B); ATR (7,14,21); Historical Vol; Keltner |
| Volume | OBV; MFI; CMF; Force Index; VWAP; Volume SMA ratios |
| Candlestick | Body size, wicks, Doji, Hammer, Shooting Star, Engulfing, Gaps |
| Calendar | Day of week, month, quarter, month-end/start effects |
| Market | Beta (20d), Relative Strength vs VN-Index, VN-Index momentum |

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. Stock market predictions involve inherent uncertainty. Past model performance does not guarantee future results. Do not make financial decisions solely based on AI predictions. Always consult a licensed financial advisor.

---

## 🔧 Troubleshooting

**`vnstock` not found:**
```bash
pip install vnstock --upgrade
```

**TensorFlow not available (no GPU):**
```bash
pip install tensorflow-cpu
```

**Symbol data not available:**
Check that the symbol exists on HOSE/HNX. Some symbols may need `-` prefix for HNX (e.g., `HNX:SHN`). Edit `DEFAULT_SYMBOLS` in `config.py`.

**Slow training:**
Reduce `OPTUNA_TRIALS_*` in `config.py`, or train fewer symbols with `--symbols VNM VIC HPG`.
