"""
dashboard.py — Streamlit dashboard for Vietnam Stock AI.

Run with:  streamlit run dashboard.py
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vietnam Stock AI",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 4px 0;
  }
  .bull { color: #00d09c; font-weight: 700; font-size: 1.2em; }
  .bear { color: #ff4d4d; font-weight: 700; font-size: 1.2em; }
  .conf-bar {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #2d3561 0%, #00d09c var(--pct), #2d3561 var(--pct));
  }
</style>
""", unsafe_allow_html=True)

import config
from data_fetcher import fetch_ohlcv, get_vnindex
from features import build_features, get_feature_cols


# ── Helper: Load predictions ───────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_latest_predictions() -> dict:
    pred_dir = Path(config.RESULTS_DIR)
    files = sorted(pred_dir.glob("predictions_*.json"), reverse=True)
    if not files:
        return {}
    with open(files[0]) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_ohlcv(symbol: str) -> pd.DataFrame:
    try:
        return fetch_ohlcv(symbol)
    except Exception as e:
        st.error(f"Could not fetch data for {symbol}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_vnindex() -> Optional[pd.DataFrame]:
    try:
        return get_vnindex()
    except Exception:
        return None


def load_model_meta(symbol: str, horizon: int) -> dict:
    path = Path(f"{config.MODEL_DIR}/{symbol}_h{horizon}_meta.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_training_summary() -> dict:
    files = sorted(Path(config.RESULTS_DIR).glob("training_summary_*.json"), reverse=True)
    if not files:
        return {}
    with open(files[0]) as f:
        return json.load(f)


# ── Candlestick Chart ──────────────────────────────────────────────────────────
def candlestick_chart(df: pd.DataFrame, symbol: str, pred: Optional[dict] = None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index[-120:],
        open=df["open"].iloc[-120:],
        high=df["high"].iloc[-120:],
        low=df["low"].iloc[-120:],
        close=df["close"].iloc[-120:],
        name=symbol,
        increasing_line_color="#00d09c",
        decreasing_line_color="#ff4d4d",
    ))

    # Add SMA overlays
    for n, color in [(20, "#FFD700"), (50, "#FF8C00")]:
        sma = df["close"].rolling(n).mean()
        fig.add_trace(go.Scatter(
            x=df.index[-120:], y=sma.iloc[-120:],
            mode="lines", name=f"SMA{n}",
            line=dict(color=color, width=1, dash="dot"),
        ))

    # Prediction target line
    if pred and pred.get("target_price"):
        fig.add_hline(
            y=pred["target_price"],
            line_dash="dash",
            line_color="#00d09c" if pred["direction"] == 1 else "#ff4d4d",
            annotation_text=f"Target: {pred['target_price']:.0f}",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"{symbol} — Last 120 days", font=dict(size=14)),
    )
    return fig


def volume_chart(df: pd.DataFrame):
    colors = ["#00d09c" if c >= o else "#ff4d4d"
              for o, c in zip(df["open"].iloc[-60:], df["close"].iloc[-60:])]
    fig = go.Figure(go.Bar(
        x=df.index[-60:], y=df["volume"].iloc[-60:],
        marker_color=colors, name="Volume",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=150, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig


def rsi_chart(df: pd.DataFrame):
    close = df["close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(14).mean()
    loss  = (-delta.clip(upper=0)).ewm(14).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[-60:], y=rsi.iloc[-60:],
        mode="lines", line=dict(color="#7b68ee", width=2), name="RSI(14)"
    ))
    for lvl, col, dash in [(70, "#ff4d4d", "dash"), (30, "#00d09c", "dash"), (50, "#888", "dot")]:
        fig.add_hline(y=lvl, line_color=col, line_dash=dash, line_width=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=150, margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(range=[0, 100]), showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🇻🇳 Vietnam Stock AI")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["📊 Dashboard", "🔍 Stock Analysis", "⚙️ Train / Predict", "📈 Backtest"])

horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    [1, 3, 5, 10],
    index=2,
    format_func=lambda x: f"{x} trading days",
)

symbols_input = st.sidebar.multiselect(
    "Symbols",
    config.DEFAULT_SYMBOLS,
    default=config.DEFAULT_SYMBOLS[:8],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Today:** {datetime.now().strftime('%Y-%m-%d')}")
st.sidebar.markdown(f"**TZ:** Asia/Ho_Chi_Minh")
st.sidebar.markdown("**Market Hours:** 9:00–11:30, 13:00–14:45")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════

if mode == "📊 Dashboard":
    st.title("🇻🇳 Vietnam Stock AI — Prediction Dashboard")

    predictions = load_latest_predictions()

    if not predictions:
        st.warning(
            "No predictions found. Run `python main.py --mode train` first, "
            "then `python main.py --mode predict`."
        )
    else:
        # ── Summary metrics ───────────────────────────────────────────────
        bull_count = sum(1 for p in predictions.values() if p.get("direction") == 1)
        bear_count = len(predictions) - bull_count
        avg_conf   = np.mean([p.get("confidence", 0.5) for p in predictions.values()])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Symbols tracked", len(predictions))
        col2.metric("Bullish signals", f"{bull_count} 🟢")
        col3.metric("Bearish signals", f"{bear_count} 🔴")
        col4.metric("Avg confidence", f"{avg_conf:.1%}")

        st.markdown("---")

        # ── Prediction table ───────────────────────────────────────────────
        rows = []
        for sym, p in predictions.items():
            if sym not in symbols_input:
                continue
            rows.append({
                "Symbol": sym,
                "Direction": "▲ UP" if p.get("direction") == 1 else "▼ DOWN",
                "Confidence": f"{p.get('confidence', 0):.1%}",
                "Exp Return": f"{p.get('return_pct', 0):+.2f}%",
                "Target Price": f"{p.get('target_price', 0):,.0f}",
                "Last Close": f"{p.get('last_close', 0):,.0f}",
                "Horizon": f"{p.get('horizon_days', horizon)}d",
                "Model AUC": f"{p.get('model_auc', 0):.3f}",
            })

        if rows:
            df_pred = pd.DataFrame(rows)
            st.dataframe(
                df_pred.style.applymap(
                    lambda v: "color: #00d09c" if "▲" in str(v) else ("color: #ff4d4d" if "▼" in str(v) else ""),
                    subset=["Direction"]
                ),
                use_container_width=True,
            )

        # ── Confidence heatmap ─────────────────────────────────────────────
        st.markdown("### Confidence Heatmap")
        conf_data = {
            sym: p.get("confidence", 0.5)
            for sym, p in predictions.items()
            if sym in symbols_input
        }
        if conf_data:
            fig = go.Figure(go.Bar(
                x=list(conf_data.keys()),
                y=[v * 100 for v in conf_data.values()],
                marker_color=[
                    "#00d09c" if v >= 0.5 else "#ff4d4d"
                    for v in conf_data.values()
                ],
                text=[f"{v:.1%}" for v in conf_data.values()],
                textposition="outside",
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="#888")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                yaxis=dict(range=[0, 105], title="Confidence (%)"),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Stock Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "🔍 Stock Analysis":
    st.title("🔍 Stock Analysis")

    selected = st.selectbox("Select Symbol", symbols_input or config.DEFAULT_SYMBOLS[:5])

    if selected:
        with st.spinner(f"Loading {selected}…"):
            ohlcv = load_ohlcv(selected)

        if ohlcv.empty:
            st.error(f"No data available for {selected}")
        else:
            predictions = load_latest_predictions()
            pred = predictions.get(selected)
            meta = load_model_meta(selected, horizon)

            # ── Stats row ──────────────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            last   = ohlcv["close"].iloc[-1]
            prev   = ohlcv["close"].iloc[-2]
            chg    = (last - prev) / prev * 100
            high52 = ohlcv["close"].tail(252).max()
            low52  = ohlcv["close"].tail(252).min()

            c1.metric("Last Close", f"{last:,.0f}", f"{chg:+.2f}%")
            c2.metric("52W High",   f"{high52:,.0f}")
            c3.metric("52W Low",    f"{low52:,.0f}")
            c4.metric("Avg Volume (20d)", f"{ohlcv['volume'].tail(20).mean():,.0f}")
            if pred:
                arrow = "▲" if pred["direction"] == 1 else "▼"
                c5.metric("AI Signal",
                          f"{arrow} {pred['return_pct']:+.1f}%",
                          f"conf={pred['confidence']:.1%}")

            st.markdown("---")

            # ── Charts ─────────────────────────────────────────────────────
            st.plotly_chart(candlestick_chart(ohlcv, selected, pred), use_container_width=True)

            col_v, col_r = st.columns(2)
            with col_v:
                st.markdown("**Volume**")
                st.plotly_chart(volume_chart(ohlcv), use_container_width=True)
            with col_r:
                st.markdown("**RSI (14)**")
                st.plotly_chart(rsi_chart(ohlcv), use_container_width=True)

            # ── Model info ─────────────────────────────────────────────────
            if meta:
                st.markdown("### Model Information")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Test AUC",  f"{meta.get('auc', 0):.4f}")
                mc2.metric("Test Acc",  f"{meta.get('accuracy', 0):.3f}")
                mc3.metric("Horizon",   f"{meta.get('horizon', horizon)}d")
                mc4.metric("Trained At", meta.get("trained_at", "N/A")[:10])

                if "weights" in meta:
                    st.markdown("**Ensemble Weights**")
                    w = meta["weights"]
                    wfig = go.Figure(go.Bar(
                        x=list(w.keys()), y=list(w.values()),
                        marker_color=["#7b68ee", "#00d09c", "#FFD700"],
                        text=[f"{v:.1%}" for v in w.values()],
                        textposition="outside",
                    ))
                    wfig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=200, margin=dict(l=0, r=0, t=0, b=0),
                        yaxis=dict(range=[0, 0.7]),
                    )
                    st.plotly_chart(wfig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Train / Predict
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "⚙️ Train / Predict":
    st.title("⚙️ Train & Predict")

    col_tr, col_pr = st.columns(2)

    with col_tr:
        st.markdown("### 🧠 Train Models")
        tune_cb = st.checkbox("Enable Optuna hyperparameter tuning", value=True)
        st.warning("Training with Optuna takes 20–60 min per symbol. Untick for fast run.")
        if st.button("🚀 Start Training", type="primary"):
            from pipeline import TrainingPipeline
            with st.spinner("Training in progress… this may take a while."):
                try:
                    pipeline = TrainingPipeline(
                        symbols=symbols_input or config.DEFAULT_SYMBOLS,
                        horizon=horizon,
                        tune=tune_cb,
                    )
                    summary = pipeline.run()
                    st.success(f"✅ Training complete! {len(summary)} models updated.")
                    st.json(summary)
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with col_pr:
        st.markdown("### 🎯 Generate Predictions")
        if st.button("🔮 Predict Now", type="primary"):
            from pipeline import PredictionPipeline
            with st.spinner("Generating predictions…"):
                try:
                    pipeline = PredictionPipeline(
                        symbols=symbols_input or config.DEFAULT_SYMBOLS,
                        horizon=horizon,
                    )
                    results = pipeline.run()
                    st.success(f"✅ Predictions generated for {len(results)} symbols.")
                    st.json(results)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.markdown("### ⏰ Daemon Scheduler Status")
    st.info(
        "The daily daemon (`python main.py --mode daemon`) will automatically:\n"
        "- **09:05 ICT** Mon–Fri: Generate fresh predictions before market open\n"
        "- **15:00 ICT** Mon–Fri: Retrain + Optuna-tune all models after market close\n\n"
        "Run it in the background: `nohup python main.py --mode daemon &`"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Backtest
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "📈 Backtest":
    st.title("📈 Walk-Forward Backtest")
    st.info("Simulates monthly retraining and paper trading over the full historical dataset.")

    if st.button("▶️ Run Backtest", type="primary"):
        from backtester import Backtester
        with st.spinner("Running walk-forward backtest… (may take a few minutes)"):
            try:
                bt = Backtester(symbols=symbols_input[:5] or config.DEFAULT_SYMBOLS[:5], horizon=horizon)
                report = bt.run()
                st.code(report, language="text")
            except Exception as e:
                st.error(f"Backtest error: {e}")

    # Show last backtest if exists
    bt_files = sorted(Path(config.RESULTS_DIR).glob("backtest_*.json"), reverse=True)
    if bt_files:
        st.markdown("### Last Backtest Results")
        with open(bt_files[0]) as f:
            bt_data = json.load(f)
        rows = []
        for sym, r in bt_data.items():
            if "error" not in r:
                rows.append({
                    "Symbol": sym,
                    "AUC": f"{r.get('auc', 0):.3f}",
                    "Accuracy": f"{r.get('accuracy', 0):.3f}",
                    "Win Rate": f"{r.get('win_rate', 0):.3f}",
                    "Strategy %": f"{r.get('strategy_return_pct', 0):+.1f}%",
                    "Buy & Hold %": f"{r.get('buy_hold_return_pct', 0):+.1f}%",
                    "Sharpe": f"{r.get('sharpe_ratio', 0):.2f}",
                    "Max DD %": f"{r.get('max_drawdown_pct', 0):.1f}%",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
