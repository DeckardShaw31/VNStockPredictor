"""
dashboard.py — Vietnam Stock AI — Full Streamlit Dashboard
Tabs: Market Overview | Stock Analysis | Portfolio | Signals | Train/Predict | Backtest
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Vietnam Stock AI",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  code, .mono { font-family: 'IBM Plex Mono', monospace; }

  div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0f1923, #162032);
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 14px 18px;
  }
  .bull  { color: #00c896; font-weight: 700; }
  .bear  { color: #ff4560; font-weight: 700; }
  .neu   { color: #a0aec0; font-weight: 600; }
  .badge-buy  { background:#00c896; color:#000; padding:2px 10px; border-radius:12px; font-size:0.8em; font-weight:700; }
  .badge-sell { background:#ff4560; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.8em; font-weight:700; }
  .badge-hold { background:#4a5568; color:#e2e8f0; padding:2px 10px; border-radius:12px; font-size:0.8em; font-weight:600; }
  .risk-high  { color:#ff4560; font-weight:700; }
  .risk-med   { color:#f6ad55; font-weight:600; }
  .risk-low   { color:#00c896; font-weight:600; }
</style>
""", unsafe_allow_html=True)

import config
from data_fetcher import fetch_ohlcv, get_vnindex, fetch_multiple
from portfolio import (
    load_portfolio, save_portfolio, add_position, remove_lot,
    build_portfolio_report,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

@st.cache_data(ttl=300)
def _load_predictions() -> dict:
    files = sorted(Path(config.RESULTS_DIR).glob("predictions_*.json"), reverse=True)
    if not files: return {}
    with open(files[0]) as f: return json.load(f)

@st.cache_data(ttl=300)
def _load_signals() -> list:
    files = sorted(Path(config.RESULTS_DIR).glob("signals_*.json"), reverse=True)
    if not files: return []
    with open(files[0], encoding="utf-8") as f: return json.load(f)

@st.cache_data(ttl=1800)
def _fetch(symbol: str) -> pd.DataFrame:
    try: return fetch_ohlcv(symbol)
    except: return pd.DataFrame()

@st.cache_data(ttl=1800)
def _fetch_vnindex() -> Optional[pd.DataFrame]:
    try: return get_vnindex()
    except: return None

def _load_meta(sym: str, horizon: int) -> dict:
    p = Path(f"{config.MODEL_DIR}/{sym}_h{horizon}_meta.json")
    if p.exists():
        with open(p, encoding="utf-8") as f: return json.load(f)
    return {}

def _fmt_vnd(v: float) -> str:
    if abs(v) >= 1e9:  return f"{v/1e9:+.2f}B"
    if abs(v) >= 1e6:  return f"{v/1e6:+.1f}M"
    return f"{v:+,.0f}"

def _signal_badge(sig: str) -> str:
    cls = {"BUY": "buy", "SELL": "sell"}.get(sig, "hold")
    return f'<span class="badge-{cls}">{sig}</span>'

def _candlestick(df: pd.DataFrame, sym: str, signal: Optional[dict] = None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index[-120:], open=df["open"].iloc[-120:],
        high=df["high"].iloc[-120:], low=df["low"].iloc[-120:], close=df["close"].iloc[-120:],
        name=sym, increasing_line_color="#00c896", decreasing_line_color="#ff4560",
    ))
    for n, col in [(20, "#f6ad55"), (50, "#9f7aea")]:
        sma = df["close"].rolling(n).mean()
        fig.add_trace(go.Scatter(x=df.index[-120:], y=sma.iloc[-120:],
            mode="lines", name=f"MA{n}", line=dict(color=col, width=1, dash="dot")))
    if signal:
        if signal.get("stop_loss"):
            fig.add_hline(y=signal["stop_loss"], line_color="#ff4560", line_dash="dash",
                annotation_text=f"SL {signal['stop_loss']:,.0f}", annotation_position="left")
        if signal.get("take_profit"):
            fig.add_hline(y=signal["take_profit"], line_color="#00c896", line_dash="dash",
                annotation_text=f"TP {signal['take_profit']:,.0f}", annotation_position="left")
    fig.update_layout(**DARK, height=380, xaxis_rangeslider_visible=False,
        margin=dict(l=0,r=0,t=30,b=0), title=dict(text=f"{sym} — 120 days", font_size=13))
    return fig

def _rsi_chart(df: pd.DataFrame):
    d = df["close"].diff()
    rsi = 100 - 100/(1 + d.clip(lower=0).ewm(14).mean() / (-d.clip(upper=0)).ewm(14).mean().replace(0,np.nan))
    fig = go.Figure(go.Scatter(x=df.index[-60:], y=rsi.iloc[-60:],
        mode="lines", line=dict(color="#7b68ee", width=2)))
    for lvl, col in [(70,"#ff4560"),(30,"#00c896"),(50,"#555")]:
        fig.add_hline(y=lvl, line_color=col, line_dash="dash", line_width=1)
    fig.update_layout(**DARK, height=140, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
        yaxis=dict(range=[0,100]))
    return fig

def _volume_chart(df: pd.DataFrame):
    colors = ["#00c896" if c>=o else "#ff4560"
              for o,c in zip(df["open"].iloc[-60:], df["close"].iloc[-60:])]
    fig = go.Figure(go.Bar(x=df.index[-60:], y=df["volume"].iloc[-60:], marker_color=colors))
    fig.update_layout(**DARK, height=120, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🇻🇳 Vietnam Stock AI")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Market Overview",
    "🔍 Stock Analysis",
    "💼 Portfolio",
    "🎯 Trade Signals",
    "⚙️ Train / Update",
    "📈 Backtest",
])

horizon = st.sidebar.selectbox("Prediction Horizon",
    [1, 3, 5, 10], index=2, format_func=lambda x: f"{x} trading days")

symbols_input = st.sidebar.multiselect("Watchlist",
    config.DEFAULT_SYMBOLS, default=config.DEFAULT_SYMBOLS[:8])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{datetime.now().strftime('%Y-%m-%d %H:%M')}** ICT")
st.sidebar.markdown("Market: 09:00–11:30, 13:00–14:45")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Market Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Market Overview":
    st.title("🇻🇳 Market Overview")

    predictions = _load_predictions()
    signals_raw = _load_signals()
    sig_map     = {s["symbol"]: s for s in signals_raw}

    if not predictions:
        st.warning("No predictions yet. Run `python main.py --mode predict` first.")
    else:
        bull = sum(1 for p in predictions.values() if p.get("direction")==1)
        bear = len(predictions) - bull
        avg_conf = np.mean([p.get("confidence",0.5) for p in predictions.values()])
        buy_sigs = sum(1 for s in signals_raw if s.get("signal")=="BUY")
        sell_sigs= sum(1 for s in signals_raw if s.get("signal")=="SELL")

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Symbols", len(predictions))
        c2.metric("AI Bullish", f"{bull} 🟢")
        c3.metric("AI Bearish", f"{bear} 🔴")
        c4.metric("BUY Signals", f"{buy_sigs}")
        c5.metric("Avg Confidence", f"{avg_conf:.1%}")

        st.markdown("---")

        # Signal table
        rows = []
        for sym, p in predictions.items():
            if sym not in symbols_input: continue
            s = sig_map.get(sym, {})
            rows.append({
                "Symbol":      sym,
                "AI Direction":  "▲ UP" if p.get("direction")==1 else "▼ DOWN",
                "Confidence":    f"{p.get('confidence',0):.1%}",
                "Signal":        s.get("signal","—"),
                "Entry":         f"{s.get('entry_price',0):,.0f}" if s else "—",
                "Stop-Loss":     f"{s.get('stop_loss',0):,.0f}"   if s else "—",
                "Take-Profit":   f"{s.get('take_profit',0):,.0f}" if s else "—",
                "R/R":           f"1:{s.get('rr_ratio',0):.1f}"   if s else "—",
                "Model AUC":     f"{p.get('model_auc',0):.3f}",
            })

        if rows:
            df_show = pd.DataFrame(rows)
            st.dataframe(df_show.style.applymap(
                lambda v: "color:#00c896;font-weight:700" if "▲" in str(v) or v=="BUY"
                     else ("color:#ff4560;font-weight:700" if "▼" in str(v) or v=="SELL" else ""),
                subset=["AI Direction","Signal"]
            ), use_container_width=True, height=420)

        # Confidence bar chart
        st.markdown("### Signal Confidence")
        conf_data = {sym: p.get("confidence",0.5) for sym,p in predictions.items() if sym in symbols_input}
        if conf_data:
            fig = go.Figure(go.Bar(
                x=list(conf_data.keys()),
                y=[v*100 for v in conf_data.values()],
                marker_color=["#00c896" if v>=0.5 else "#ff4560" for v in conf_data.values()],
                text=[f"{v:.0%}" for v in conf_data.values()], textposition="outside",
            ))
            fig.add_hline(y=65, line_dash="dash", line_color="#f6ad55",
                annotation_text="Signal threshold 65%")
            fig.update_layout(**DARK, height=320, yaxis=dict(range=[0,105],title="Confidence %"),
                margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Stock Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Stock Analysis":
    st.title("🔍 Stock Analysis")

    sym = st.selectbox("Symbol", symbols_input or config.DEFAULT_SYMBOLS[:5])
    if not sym:
        st.info("Select a symbol above.")
        st.stop()

    with st.spinner(f"Loading {sym}..."):
        ohlcv = _fetch(sym)

    if ohlcv.empty:
        st.error(f"No data for {sym}"); st.stop()

    signals_raw = _load_signals()
    sig_map     = {s["symbol"]: s for s in signals_raw}
    sig         = sig_map.get(sym)
    meta        = _load_meta(sym, horizon)

    # Stats row
    last  = float(ohlcv["close"].iloc[-1])
    prev  = float(ohlcv["close"].iloc[-2]) if len(ohlcv)>1 else last
    chg   = (last-prev)/prev*100
    h52   = ohlcv["close"].tail(252).max()
    l52   = ohlcv["close"].tail(252).min()
    avol  = ohlcv["volume"].tail(20).mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Last Close",  f"{last:,.0f}", f"{chg:+.2f}%")
    c2.metric("52W High",    f"{h52:,.0f}")
    c3.metric("52W Low",     f"{l52:,.0f}")
    c4.metric("Avg Vol 20d", f"{avol:,.0f}")
    if sig:
        c5.metric("AI Signal", sig.get("signal","—"),
            f"conf={sig.get('confidence',0):.1%}")

    # Candlestick
    st.plotly_chart(_candlestick(ohlcv, sym, sig), use_container_width=True)

    cv, rv = st.columns(2)
    with cv:
        st.caption("Volume")
        st.plotly_chart(_volume_chart(ohlcv), use_container_width=True)
    with rv:
        st.caption("RSI (14)")
        st.plotly_chart(_rsi_chart(ohlcv), use_container_width=True)

    # Signal detail
    if sig:
        st.markdown("### Trade Signal Detail")
        sa, sb, sc, sd = st.columns(4)
        sa.metric("Signal",      sig.get("signal","—"))
        sb.metric("Entry Zone",  f"{sig.get('entry_low',0):,.0f} – {sig.get('entry_high',0):,.0f}")
        sc.metric("Stop-Loss",   f"{sig.get('stop_loss',0):,.0f}  ({sig.get('stop_loss_pct',0):+.1f}%)")
        sd.metric("Take-Profit", f"{sig.get('take_profit',0):,.0f}  ({sig.get('take_profit_pct',0):+.1f}%)")

        se, sf, sg, sh = st.columns(4)
        se.metric("R/R Ratio",   f"1 : {sig.get('rr_ratio',0):.2f}")
        sf.metric("Position",    f"{sig.get('position_size_pct',0):.1f}% of capital")
        sg.metric("Pivot PP",    f"{sig.get('pivot_pp',0):,.0f}")
        sh.metric("Fib 61.8%",   f"{sig.get('fib_618',0):,.0f}")

        with st.expander("Math model votes"):
            votes = sig.get("math_votes", {})
            vcols = st.columns(4)
            for i,(k,v) in enumerate(votes.items()):
                emoji = "🟢" if v>0 else ("🔴" if v<0 else "⚪")
                vcols[i%4].markdown(f"{emoji} **{k}**: {'+1' if v>0 else ('-1' if v<0 else '0')}")

    # Model info
    if meta:
        st.markdown("### Model Metrics")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Test AUC",  f"{meta.get('auc',0):.4f}")
        m2.metric("Test Acc",  f"{meta.get('accuracy',0):.3f}")
        m3.metric("Trained",   meta.get("trained_at","—")[:10])
        m4.metric("Features",  len(meta.get("features",[])))

        if "weights" in meta:
            st.markdown("**Ensemble weights**")
            w = meta["weights"]
            wfig = go.Figure(go.Bar(
                x=list(w.keys()), y=[v*100 for v in w.values()],
                marker_color=["#7b68ee","#00c896","#f6ad55","#fc8181"][:len(w)],
                text=[f"{v:.0%}" for v in w.values()], textposition="outside",
            ))
            wfig.update_layout(**DARK, height=200, yaxis=dict(range=[0,85]),
                margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(wfig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Portfolio
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💼 Portfolio":
    st.title("💼 My Portfolio")

    portfolio = load_portfolio()
    signals_raw = _load_signals()
    sig_map = {s["symbol"]: s for s in signals_raw}

    # ── Add position form ──────────────────────────────────────────────────
    with st.expander("➕ Add New Position", expanded=not bool(portfolio)):
        with st.form("add_position"):
            fc1, fc2, fc3, fc4 = st.columns(4)
            new_sym   = fc1.selectbox("Symbol", config.DEFAULT_SYMBOLS)
            new_qty   = fc2.number_input("Shares", min_value=1, value=100, step=100)
            new_price = fc3.number_input("Buy Price (VND)", min_value=1.0, value=50000.0, step=100.0)
            new_date  = fc4.date_input("Buy Date", value=datetime.now().date())
            new_note  = st.text_input("Note (optional)", placeholder="e.g. Long-term, DCA lot 2...")
            submitted = st.form_submit_button("Add Position", type="primary")
            if submitted:
                add_position(new_sym, int(new_qty), float(new_price),
                             str(new_date), new_note)
                st.success(f"Added {new_qty:,} shares of {new_sym} @ {new_price:,.0f}")
                st.cache_data.clear()
                st.rerun()

    if not portfolio:
        st.info("Your portfolio is empty. Add your first position above.")
        st.stop()

    # ── Fetch current prices ───────────────────────────────────────────────
    syms_held = list(portfolio.keys())
    with st.spinner("Fetching current prices..."):
        ohlcv_data = {}
        for s in syms_held:
            df = _fetch(s)
            if not df.empty:
                ohlcv_data[s] = df

    positions, summary = build_portfolio_report(ohlcv_data, signals_raw)

    if not positions:
        st.warning("Could not load price data for portfolio symbols.")
        st.stop()

    # ── Portfolio summary metrics ─────────────────────────────────────────
    pnl_color = "normal" if summary["total_pnl"] >= 0 else "inverse"
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Value",    f"{summary['total_value']/1e6:.1f}M",
              f"{_fmt_vnd(summary['day_change_vnd'])} today")
    c2.metric("Total Cost",     f"{summary['total_cost']/1e6:.1f}M")
    c3.metric("Unrealised P&L", f"{_fmt_vnd(summary['total_pnl'])}",
              f"{summary['total_pnl_pct']:+.2f}%",
              delta_color=pnl_color)
    c4.metric("Today's Change", f"{_fmt_vnd(summary['day_change_vnd'])}",
              f"{summary['day_change_pct']:+.2f}%",
              delta_color="normal" if summary["day_change_vnd"]>=0 else "inverse")
    c5.metric("Win / Loss",     f"{summary['n_winners']} / {summary['n_losers']}",
              f"Win rate {summary['win_rate']:.0f}%")
    c6.metric("AI Agrees",      f"{summary['ai_agree_count']} positions",
              f"{summary['ai_disagree_count']} disagree")

    # Risk alert — split below-SL from near-SL
    below_sl = [p for p in positions if p.get("dist_to_sl_pct") is not None and p["dist_to_sl_pct"] < 0]
    near_sl  = [p for p in positions if p.get("dist_to_sl_pct") is not None and 0 <= p["dist_to_sl_pct"] < 3.0]
    if below_sl:
        syms = ", ".join(p["symbol"] for p in below_sl)
        st.error(f"🚨 **STOP-LOSS BREACHED:** {syms} — price is already below stop-loss. Consider exiting immediately.")
    if near_sl:
        syms = ", ".join(p["symbol"] for p in near_sl)
        st.warning(f"⚠️ **Near Stop-Loss:** {syms} — within 3% of stop-loss level.")

    st.markdown("---")

    # ── Allocation pie chart ───────────────────────────────────────────────
    col_pie, col_pnl = st.columns(2)

    with col_pie:
        st.markdown("#### Portfolio Allocation")
        pie_fig = go.Figure(go.Pie(
            labels=[p["symbol"] for p in positions],
            values=[p["current_value"] for p in positions],
            hole=0.45,
            marker_colors=px.colors.qualitative.Set3[:len(positions)],
            textinfo="label+percent",
        ))
        pie_fig.update_layout(**DARK, height=300, margin=dict(l=0,r=0,t=0,b=0),
            showlegend=False)
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_pnl:
        st.markdown("#### P&L by Position")
        pnl_colors = ["#00c896" if p["unrealised_pnl"]>=0 else "#ff4560" for p in positions]
        pnl_fig = go.Figure(go.Bar(
            x=[p["symbol"] for p in positions],
            y=[p["unrealised_pct"] for p in positions],
            marker_color=pnl_colors,
            text=[f"{p['unrealised_pct']:+.1f}%" for p in positions],
            textposition="outside",
        ))
        pnl_fig.add_hline(y=0, line_color="#555", line_width=1)
        pnl_fig.update_layout(**DARK, height=300, yaxis_title="Unrealised %",
            margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(pnl_fig, use_container_width=True)

    # ── Position detail table ─────────────────────────────────────────────
    st.markdown("#### Position Detail")

    for pos in sorted(positions, key=lambda x: x["unrealised_pct"], reverse=True):
        sym      = pos["symbol"]
        sig_str  = pos.get("signal","—") or "—"
        ai_icon  = "🟢" if pos["ai_agrees"] is True else ("🔴" if pos["ai_agrees"] is False else "⚪")
        pnl_icon = "▲" if pos["unrealised_pnl"]>=0 else "▼"
        pnl_cls  = "bull" if pos["unrealised_pnl"]>=0 else "bear"

        with st.expander(
            f"{sym}  |  {pnl_icon} {pos['unrealised_pct']:+.1f}%  "
            f"({_fmt_vnd(pos['unrealised_pnl'])})  |  Signal: {sig_str}  {ai_icon}"
        ):
            d1,d2,d3,d4,d5,d6 = st.columns(6)
            d1.metric("Shares",       f"{pos['total_shares']:,}")
            d2.metric("Avg Cost",     f"{pos['avg_cost']:,.0f}")
            d3.metric("Current",      f"{pos['current_price']:,.0f}",
                      f"{pos['day_change_pct']:+.2f}% today")
            d4.metric("Value",        f"{pos['current_value']/1e6:.2f}M")
            d5.metric("Days Held",    f"{pos['days_held']}d")
            d6.metric("Ann. Return",  f"{pos['ann_return_pct']:+.1f}%")

            if pos.get("stop_loss") or pos.get("take_profit"):
                e1,e2,e3 = st.columns(3)
                e1.metric("AI Signal",  sig_str)
                e2.metric("Stop-Loss",  f"{pos['stop_loss']:,.0f}" if pos.get("stop_loss") else "—",
                          f"{pos.get('dist_to_sl_pct',0):.1f}% away" if pos.get("dist_to_sl_pct") else "")
                e3.metric("Take-Profit",f"{pos['take_profit']:,.0f}" if pos.get("take_profit") else "—")

            # Individual lots
            lots = portfolio.get(sym, [])
            if lots:
                st.markdown("**Lots:**")
                for i, lot in enumerate(lots):
                    lot_pnl = (pos["current_price"] - lot["buy_price"]) * lot["shares"]
                    lot_pct = (pos["current_price"] - lot["buy_price"]) / lot["buy_price"] * 100
                    lc1,lc2,lc3,lc4,lc5 = st.columns([2,2,2,2,1])
                    lc1.write(f"**Lot {i+1}** — {lot['buy_date']}")
                    lc2.write(f"{lot['shares']:,} shares @ {lot['buy_price']:,.0f}")
                    pnl_col = "🟢" if lot_pnl>=0 else "🔴"
                    lc3.write(f"{pnl_col} {lot_pct:+.1f}% ({_fmt_vnd(lot_pnl)})")
                    lc4.write(lot.get("note","") or "—")
                    if lc5.button("🗑", key=f"del_{sym}_{i}", help="Remove this lot"):
                        remove_lot(sym, i)
                        st.cache_data.clear()
                        st.rerun()

            # Quick add another lot to this symbol
            with st.form(f"add_lot_{sym}"):
                st.markdown(f"**Add another lot for {sym}:**")
                al1,al2,al3,al4 = st.columns(4)
                add_qty   = al1.number_input("Shares", min_value=1, value=100, step=100, key=f"aq_{sym}")
                add_price = al2.number_input("Price", min_value=1.0, value=max(1.0, float(pos["current_price"])), step=100.0, key=f"ap_{sym}")
                add_date  = al3.date_input("Date", value=datetime.now().date(), key=f"ad_{sym}")
                add_note  = al4.text_input("Note", key=f"an_{sym}")
                if st.form_submit_button("Add Lot"):
                    add_position(sym, int(add_qty), float(add_price), str(add_date), add_note)
                    st.cache_data.clear()
                    st.rerun()

    # ── Per-position analysis ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### AI Position Analysis")
    st.caption("AI-powered recommendation for each holding based on current price, P&L, and signal.")

    for pos in sorted(positions, key=lambda x: x["unrealised_pct"], reverse=True):
        sym      = pos["symbol"]
        pnl_pct  = pos["unrealised_pct"]
        dist_sl  = pos.get("dist_to_sl_pct")   # can be negative if already below SL
        ai_sig   = pos.get("signal", "HOLD") or "HOLD"
        ai_conf  = pos.get("signal_conf", 0.5)
        ai_agree = pos.get("ai_agrees")

        # Clamp to readable display: negative dist means already below SL
        already_below_sl = dist_sl is not None and dist_sl < 0
        near_sl          = dist_sl is not None and 0 <= dist_sl < 3.0

        # ── Recommendation logic ──────────────────────────────────────────
        # Priority 1: Already blown through stop-loss
        if already_below_sl:
            rec_color  = "#ff4560"
            rec_icon   = "🔴"
            rec_action = "EXIT NOW"
            rec_reason = (f"Price is already {abs(dist_sl):.1f}% BELOW your stop-loss level. "
                          f"The stop was supposed to protect you — exit immediately to stop further damage.")

        # Priority 2: Within 3% of stop-loss AND AI also bearish
        elif near_sl and ai_sig in ("SELL", "HOLD") and ai_conf >= 0.52:
            rec_color  = "#ff4560"
            rec_icon   = "🔴"
            rec_action = "CUT LOSS"
            rec_reason = (f"Price is only {dist_sl:.1f}% above your stop-loss. "
                          f"AI signal is {ai_sig} ({ai_conf:.0%} conf). "
                          f"Consider exiting before stop is hit.")

        # Priority 3: Big loss but AI says BUY — different situation
        elif pnl_pct <= -15 and ai_sig == "BUY" and ai_conf >= 0.58:
            rec_color  = "#f6ad55"
            rec_icon   = "🟡"
            rec_action = "AVERAGE DOWN?"
            rec_reason = (f"Position down {pnl_pct:.1f}% but AI signals BUY ({ai_conf:.0%} conf). "
                          f"Could average down if conviction is high — but set a hard stop first.")

        # Priority 4: Big loss and AI also bearish → cut
        elif pnl_pct <= -12 and ai_sig in ("SELL", "HOLD"):
            rec_color  = "#ff4560"
            rec_icon   = "🔴"
            rec_action = "CUT LOSS"
            rec_reason = (f"Position down {pnl_pct:.1f}% and AI signal is {ai_sig}. "
                          f"No recovery signal detected. Cutting loss limits further damage.")

        # Priority 5: Near SL but AI is bullish — watch carefully
        elif near_sl and ai_sig == "BUY":
            rec_color  = "#f6ad55"
            rec_icon   = "🟡"
            rec_action = "WATCH CLOSELY"
            rec_reason = (f"Only {dist_sl:.1f}% above stop-loss, but AI signals BUY. "
                          f"If price holds support, may recover. Keep stop tight.")

        # Priority 6: AI confirms BUY, position profitable
        elif ai_sig == "BUY" and ai_conf >= 0.58 and pnl_pct >= 0:
            rec_color  = "#00c896"
            rec_icon   = "🟢"
            rec_action = "HOLD / ADD"
            rec_reason = (f"AI signals BUY ({ai_conf:.0%} conf). "
                          f"Position up +{pnl_pct:.1f}%. Trend supports holding.")

        # Priority 7: Strong SELL signal from AI
        elif ai_sig == "SELL" and ai_conf >= 0.58:
            rec_color  = "#ff4560"
            rec_icon   = "🔴"
            rec_action = "CONSIDER SELLING"
            rec_reason = (f"AI signals SELL with {ai_conf:.0%} confidence. "
                          f"Math models confirm bearish trend. Consider exiting.")

        # Priority 8: Taking profit territory
        elif pnl_pct >= 15:
            rec_color  = "#f6ad55"
            rec_icon   = "🟡"
            rec_action = "TAKE PARTIAL PROFIT"
            rec_reason = (f"Position up +{pnl_pct:.1f}%. "
                          f"Consider selling 30–50% to lock in gains and move stop to breakeven.")

        # Priority 9: AI disagrees with hold
        elif ai_agree is False and ai_conf >= 0.55:
            rec_color  = "#f6ad55"
            rec_icon   = "🟡"
            rec_action = "REDUCE / WATCH"
            rec_reason = (f"AI disagrees with your long position (signal={ai_sig}, conf={ai_conf:.0%}). "
                          f"Consider reducing exposure.")

        # Default: hold, no strong signal
        else:
            rec_color  = "#4a90e2"
            rec_icon   = "🔵"
            rec_action = "HOLD"
            rec_reason = (f"No strong signal. AI: {ai_sig} ({ai_conf:.0%}). "
                          f"P&L: {pnl_pct:+.1f}%. Monitor for changes.")

        # Distance display string
        if already_below_sl:
            dist_str = f" | ⚠️ {abs(dist_sl):.1f}% BELOW SL"
        elif near_sl:
            dist_str = f" | ⚠️ {dist_sl:.1f}% above SL"
        else:
            dist_str = ""

        st.markdown(f"""
        <div style="border-left:4px solid {rec_color}; padding:10px 16px;
                    background:#0f1923; border-radius:8px; margin-bottom:8px;
                    display:flex; align-items:flex-start; gap:16px; flex-wrap:wrap;">
          <div style="min-width:60px; font-size:1.1em; font-weight:700;">{sym}</div>
          <div style="min-width:140px; color:{rec_color}; font-weight:700;">
            {rec_icon} {rec_action}
          </div>
          <div style="color:#a0aec0; font-size:0.9em; flex:1;">{rec_reason}</div>
          <div style="min-width:120px; text-align:right; font-size:0.85em; white-space:nowrap;">
            <span style="color:{'#00c896' if pnl_pct>=0 else '#ff4560'}; font-weight:600;">
              {pnl_pct:+.1f}%
            </span>
            <span style="color:#718096;">{dist_str}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Cumulative P&L Waterfall")
    wf_fig = go.Figure(go.Waterfall(
        x=[p["symbol"] for p in positions],
        y=[p["unrealised_pnl"] for p in positions],
        measure=["relative"]*len(positions),
        text=[_fmt_vnd(p["unrealised_pnl"]) for p in positions],
        textposition="outside",
        increasing=dict(marker_color="#00c896"),
        decreasing=dict(marker_color="#ff4560"),
        totals=dict(marker_color="#4a5568"),
    ))
    wf_fig.update_layout(**DARK, height=320, yaxis_title="VND",
        margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(wf_fig, use_container_width=True)

    # ── Best / Worst performers ───────────────────────────────────────────
    if summary.get("best_performer") and summary.get("worst_performer"):
        st.markdown("---")
        bc, wc = st.columns(2)
        best  = summary["best_performer"]
        worst = summary["worst_performer"]
        with bc:
            st.success(f"**Best:** {best['symbol']}  {best['unrealised_pct']:+.1f}%  "
                       f"({_fmt_vnd(best['unrealised_pnl'])})")
        with wc:
            st.error(f"**Worst:** {worst['symbol']}  {worst['unrealised_pct']:+.1f}%  "
                     f"({_fmt_vnd(worst['unrealised_pnl'])})")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Trade Signals
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Trade Signals":
    st.title("🎯 Trade Signals")

    signals_raw = _load_signals()
    if not signals_raw:
        st.warning("No signals yet. Run `python main.py --mode predict`.")
        st.stop()

    # Show ALL symbols that have signals — sidebar watchlist only filters other pages
    all_signal_syms = [s["symbol"] for s in signals_raw]
    st.caption(f"Showing signals for all {len(signals_raw)} tracked symbols. "
               f"Use sidebar watchlist to filter Stock Analysis page only.")

    filt = st.radio("Filter", ["All", "BUY only", "SELL only", "HOLD only",
                                "High confidence (≥60%)", "Watchlist only"],
                    horizontal=True)

    shown = 0
    for sig in sorted(signals_raw, key=lambda x: x.get("confidence", 0), reverse=True):
        s    = sig.get("signal", "HOLD")
        conf = sig.get("confidence", 0)
        sym  = sig.get("symbol", "")

        if filt == "BUY only"  and s != "BUY":  continue
        if filt == "SELL only" and s != "SELL": continue
        if filt == "HOLD only" and s != "HOLD": continue
        if filt == "High confidence (≥60%)" and conf < 0.60: continue
        if filt == "Watchlist only" and sym not in symbols_input: continue

        shown += 1
        col = "#00c896" if s=="BUY" else ("#ff4560" if s=="SELL" else "#4a5568")
        with st.container():
            st.markdown(f"""
            <div style="border-left:4px solid {col}; padding:10px 16px;
                        background:#0f1923; border-radius:8px; margin-bottom:8px;">
              <b style="font-size:1.1em">{sym}</b>
              &nbsp;&nbsp;
              <span style="background:{col}; color:{'#000' if s=='BUY' else '#fff'};
                    padding:2px 10px; border-radius:10px; font-size:0.85em; font-weight:700">
                {s}
              </span>
              &nbsp;&nbsp;
              <span style="color:#a0aec0">Conf: <b style="color:#e2e8f0">{conf:.1%}</b></span>
              &nbsp;&nbsp;
              <span style="color:#a0aec0">Entry: <b style="color:#e2e8f0">{sig.get('entry_low',0):,.0f}–{sig.get('entry_high',0):,.0f}</b></span>
              &nbsp;&nbsp;
              <span style="color:#ff4560">SL: <b>{sig.get('stop_loss',0):,.0f}</b> ({sig.get('stop_loss_pct',0):+.1f}%)</span>
              &nbsp;&nbsp;
              <span style="color:#00c896">TP: <b>{sig.get('take_profit',0):,.0f}</b> ({sig.get('take_profit_pct',0):+.1f}%)</span>
              &nbsp;&nbsp;
              <span style="color:#a0aec0">R/R: <b style="color:#e2e8f0">1:{sig.get('rr_ratio',0):.1f}</b></span>
              &nbsp;&nbsp;
              <span style="color:#a0aec0">Pos: <b style="color:#e2e8f0">{sig.get('position_size_pct',0):.1f}%</b></span>
              &nbsp;&nbsp;
              <span style="color:#718096; font-size:0.8em">AUC:{sig.get('model_auc',0):.3f}</span>
            </div>
            """, unsafe_allow_html=True)

    if shown == 0:
        st.info("No signals match the current filter.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Train / Update
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Train / Update":
    st.title("⚙️ Train & Update Models")

    tab_update, tab_train, tab_predict = st.tabs(
        ["⚡ Quick Update", "🧠 Full Train", "🔮 Predict"])

    with tab_update:
        st.markdown("### Fast Incremental Update")
        st.info(
            "**~30 seconds per symbol.** Ingests today's new close, refits models "
            "using saved hyperparameters. Run this every day after 15:00 ICT."
        )
        if st.button("⚡ Run Update Now", type="primary"):
            from updater import IncrementalUpdater
            with st.spinner("Updating models with today's data..."):
                try:
                    u = IncrementalUpdater(
                        symbols=symbols_input or config.DEFAULT_SYMBOLS,
                        horizon=horizon)
                    summary = u.run()
                    st.success(f"Updated {sum(1 for v in summary.values() if 'error' not in v)} symbols.")
                    st.json(summary)
                except Exception as e:
                    st.error(f"Update failed: {e}")

    with tab_train:
        st.markdown("### Full Retrain + Optuna Tuning")
        st.warning("**30–60 min per symbol.** Runs Optuna hyperparameter search. Use weekly.")
        tune_cb = st.checkbox("Enable Optuna tuning", value=True)
        if st.button("🚀 Start Full Training", type="primary"):
            from pipeline import TrainingPipeline
            with st.spinner("Training in progress..."):
                try:
                    p = TrainingPipeline(
                        symbols=symbols_input or config.DEFAULT_SYMBOLS,
                        horizon=horizon, tune=tune_cb)
                    s = p.run()
                    st.success(f"Training complete. {len(s)} models updated.")
                    st.json(s)
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with tab_predict:
        st.markdown("### Generate Predictions + Signals")
        if st.button("🔮 Predict Now", type="primary"):
            from pipeline import PredictionPipeline
            with st.spinner("Generating predictions..."):
                try:
                    p = PredictionPipeline(
                        symbols=symbols_input or config.DEFAULT_SYMBOLS,
                        horizon=horizon)
                    r = p.run()
                    st.success(f"Predictions generated for {len(r)} symbols.")
                    st.json(r)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Backtest
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Backtest":
    st.title("📈 Walk-Forward Backtest")
    st.info("Simulates monthly retraining and paper trading over historical data.")

    if st.button("▶️ Run Backtest", type="primary"):
        from backtester import Backtester
        with st.spinner("Running walk-forward backtest..."):
            try:
                bt = Backtester(symbols=symbols_input[:5] or config.DEFAULT_SYMBOLS[:5],
                                horizon=horizon)
                report = bt.run()
                st.code(report, language="text")
            except Exception as e:
                st.error(f"Backtest error: {e}")

    bt_files = sorted(Path(config.RESULTS_DIR).glob("backtest_*.json"), reverse=True)
    if bt_files:
        st.markdown("### Last Backtest Results")
        with open(bt_files[0]) as f:
            bt_data = json.load(f)
        rows = []
        for sym, r in bt_data.items():
            if "error" not in r:
                rows.append({
                    "Symbol":      sym,
                    "AUC":         f"{r.get('auc',0):.3f}",
                    "Accuracy":    f"{r.get('accuracy',0):.3f}",
                    "Win Rate":    f"{r.get('win_rate',0):.3f}",
                    "Strategy %":  f"{r.get('strategy_return_pct',0):+.1f}%",
                    "B&H %":       f"{r.get('buy_hold_return_pct',0):+.1f}%",
                    "Sharpe":      f"{r.get('sharpe_ratio',0):.2f}",
                    "Max DD %":    f"{r.get('max_drawdown_pct',0):.1f}%",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
