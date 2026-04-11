"""
Module 05 — Walk-Forward Backtester  (v2 — fixed)
====================================================
Key fixes vs v1:
  1. ONE position at a time per ticker — new signals are skipped while a
     position is open (position_open_until guard).  Eliminates the
     exponential compounding from 500 overlapping 180-day positions.
  2. Position value anchored to STARTING_CAP, not live equity.
     Capital P&L still accrues; only the bet size is stabilised.
  3. TIER_SIZE now realistic (10 / 7 / 4 % of starting capital).
  4. Commission fixed to one-way rate applied on both legs.
"""

import json
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
FEAT    = ROOT / "data" / "features"
SIG_DIR = ROOT / "data" / "signals"
REP     = ROOT / "data" / "reports"
REP.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")

# ── Constants ──────────────────────────────────────────────────────────────────
SLIPPAGE      = 0.0015        # 0.15% market impact per side
COMMISSION    = 0.0015        # 0.15% brokerage, one-way (applied on buy AND sell)
PRICE_LIMIT   = 0.07          # VN ±7% daily circuit breaker
STARTING_CAP  = 100_000_000   # 100 M VND

# Bet size as % of STARTING_CAP (anchored — does NOT grow with capital)
TIER_SIZE = {1: 0.10, 2: 0.07, 3: 0.04}


def _execute_price(price: float, direction: str = "buy") -> float:
    slip = SLIPPAGE if direction == "buy" else -SLIPPAGE
    return price * (1 + slip)


def _apply_price_limit(prev_close: float, raw_price: float) -> float:
    return float(np.clip(raw_price,
                         prev_close * (1 - PRICE_LIMIT),
                         prev_close * (1 + PRICE_LIMIT)))


def _count_bdays(start_str: str, end_str: str, bday_set: set) -> int:
    s = pd.Timestamp(start_str)
    e = pd.Timestamp(end_str)
    return sum(1 for d in bday_set if s < d <= e)


def backtest_ticker(
    ticker: str,
    df_feat: pd.DataFrame,
    sig_df: pd.DataFrame,
) -> tuple[dict, list, list]:
    df = df_feat.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dates_list = df["date"].tolist()
    bday_set   = set(dates_list)

    empty = ({"ticker": ticker, "n_trades": 0}, [], [])
    if sig_df is None or sig_df.empty:
        return empty

    sig_df = sig_df.copy()
    sig_df["date"] = pd.to_datetime(sig_df["date"])
    sig_df = sig_df.sort_values("date").reset_index(drop=True)

    capital      = float(STARTING_CAP)
    equity_curve = [{"date": str(dates_list[0].date()), "equity": capital}]
    trades       = []

    # Guard: skip signals while this position is still open
    position_open_until: pd.Timestamp | None = None

    for _, sig in sig_df.iterrows():
        sig_date  = sig["date"]
        tier      = int(sig.get("tier", 3))
        hold_days = int(sig.get("hold_days", 10))
        stop_loss = float(sig.get("stop_loss", sig["current_price"] * 0.942))

        # T+3 execution delay
        entry_idx = next((i for i, d in enumerate(dates_list) if d > sig_date), None)
        if entry_idx is None:
            continue
        exec_idx = min(entry_idx + 3, len(dates_list) - 1)
        if exec_idx >= len(dates_list) - 1:
            continue
        exec_date = dates_list[exec_idx]

        # ── CORE FIX: skip if position still open ─────────────────────────────
        if position_open_until is not None and exec_date <= position_open_until:
            continue

        entry_price = _execute_price(float(df.iloc[exec_idx]["close"]), "buy")

        # ── Fixed bet size anchored to STARTING_CAP ───────────────────────────
        pos_size_pct = TIER_SIZE.get(tier, 0.04)
        pos_value    = STARTING_CAP * pos_size_pct
        pos_value    = min(pos_value, capital)   # never bet more than we have
        if pos_value <= 0:
            continue
        shares = pos_value / entry_price

        # ── Find exit ─────────────────────────────────────────────────────────
        exit_price  = None
        exit_date   = None
        exit_reason = "hold_period"

        for day_offset in range(1, hold_days + 2):
            check_idx = exec_idx + day_offset
            if check_idx >= len(dates_list):
                break

            check_date  = dates_list[check_idx]
            check_price = _apply_price_limit(
                float(df.iloc[check_idx - 1]["close"]),
                float(df.iloc[check_idx]["close"]),
            )

            if check_price <= stop_loss:
                exit_price  = _execute_price(
                    max(check_price, float(df.iloc[check_idx - 1]["close"]) * (1 - PRICE_LIMIT)),
                    "sell",
                )
                exit_date   = check_date
                exit_reason = "stop_loss"
                break

            if day_offset >= hold_days:
                exit_price  = _execute_price(check_price, "sell")
                exit_date   = check_date
                exit_reason = "hold_period"
                break

        if exit_price is None or exit_date is None:
            continue

        # ── P&L ───────────────────────────────────────────────────────────────
        gross_pnl = (exit_price - entry_price) * shares
        buy_comm  = entry_price * shares * COMMISSION
        sell_comm = exit_price  * shares * COMMISSION
        net_pnl   = gross_pnl - buy_comm - sell_comm
        pnl_pct   = net_pnl / pos_value

        capital            += net_pnl
        position_open_until = exit_date

        if capital <= 0:
            log.warning(f"{ticker}: capital exhausted — stopping")
            break

        trades.append({
            "ticker":           ticker,
            "entry_date":       str(exec_date.date()),
            "exit_date":        str(exit_date.date()),
            "entry_price":      round(entry_price, 0),
            "exit_price":       round(exit_price, 0),
            "shares":           round(shares, 4),
            "pos_value_vnd":    round(pos_value, 0),
            "tier":             tier,
            "strategy":         str(sig.get("primary_strategy", "")),
            "pnl_vnd":          round(net_pnl, 0),
            "pnl_pct":          round(pnl_pct, 4),
            "reason":           exit_reason,
            "hold_days_actual": _count_bdays(
                str(exec_date.date()), str(exit_date.date()), bday_set
            ),
        })
        equity_curve.append({"date": str(exit_date.date()), "equity": round(capital, 0)})

    # ── Metrics ───────────────────────────────────────────────────────────────
    if not trades:
        return {"ticker": ticker, "n_trades": 0}, [], equity_curve

    trade_df = pd.DataFrame(trades)
    wins     = trade_df[trade_df["pnl_pct"] > 0]
    losses   = trade_df[trade_df["pnl_pct"] <= 0]

    win_rate      = len(wins) / len(trade_df)
    avg_win       = float(wins["pnl_pct"].mean())   if len(wins)   > 0 else 0.0
    avg_loss      = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
    profit_factor = wins["pnl_vnd"].sum() / max(losses["pnl_vnd"].abs().sum(), 1.0)

    eq_vals   = [e["equity"] for e in equity_curve]
    eq_series = pd.Series(eq_vals, dtype=float)
    # Use actual calendar span of the feature data, NOT len(equity_curve)/252.
    # equity_curve has one row per trade exit (e.g. 3 rows ≠ 3 years).
    first_date = dates_list[0]
    last_date  = dates_list[-1]
    n_years    = max((last_date - first_date).days / 365.25, 1 / 365.25)
    ann_ret    = (eq_vals[-1] / eq_vals[0]) ** (1 / n_years) - 1

    roll_max = eq_series.cummax()
    max_dd   = float(((eq_series - roll_max) / roll_max).min())

    daily_ret = eq_series.pct_change().dropna()
    sharpe    = (float(daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
                 if daily_ret.std() > 0 else 0.0)
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    stats = {
        "ticker":            ticker,
        "n_trades":          len(trade_df),
        "win_rate":          round(win_rate, 4),
        "avg_win":           round(avg_win, 4),
        "avg_loss":          round(avg_loss, 4),
        "profit_factor":     round(profit_factor, 3),
        "ann_return":        round(ann_ret, 4),
        "max_drawdown":      round(max_dd, 4),
        "sharpe":            round(sharpe, 3),
        "calmar":            round(calmar, 3),
        "final_capital":     round(capital, 0),
        "total_return_pct":  round((capital / STARTING_CAP - 1) * 100, 2),
    }
    return stats, trades, equity_curve


def run(ticker: str) -> tuple[dict, list, list]:
    feat_path = FEAT / f"{ticker}.csv"
    sig_path  = SIG_DIR / f"{ticker}_signals.csv"

    if not feat_path.exists():
        log.error(f"{ticker}: feature file missing")
        return {}, [], []

    df_feat = pd.read_csv(feat_path, low_memory=False)
    sig_df  = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()

    log.info(f"{ticker}: backtesting {len(sig_df)} signals …")
    stats, trades, equity = backtest_ticker(ticker, df_feat, sig_df)

    if equity:
        pd.DataFrame(equity).to_csv(REP / f"{ticker}_equity.csv", index=False)

    log.info(
        f"  {ticker}: {stats.get('n_trades',0):3d} trades | "
        f"WR={stats.get('win_rate',0):.1%} | "
        f"Sharpe={stats.get('sharpe',0):6.2f} | "
        f"CAGR={stats.get('ann_return',0)*100:6.1f}% | "
        f"DD={stats.get('max_drawdown',0)*100:.1f}%"
    )
    return stats, trades, equity


def run_all(tickers: list[str]) -> None:
    all_stats, all_trades, all_equity = [], {}, {}

    for t in tickers:
        stats, trades, equity = run(t)
        all_stats.append(stats)
        all_trades[t] = trades
        all_equity[t] = equity

    valid = [s for s in all_stats if s.get("n_trades", 0) > 0]

    def _mean(key):
        vals = [s[key] for s in valid if key in s]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    summary = {
        "tickers_tested":      len(all_stats),
        "tickers_with_trades": len(valid),
        "avg_win_rate":        _mean("win_rate"),
        "avg_sharpe":          _mean("sharpe"),
        "avg_ann_return":      _mean("ann_return"),
        "avg_max_dd":          _mean("max_drawdown"),
    }

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "per_ticker":   all_stats,
        "trades":       all_trades,
        "equity":       all_equity,
        "summary":      summary,
    }
    out = REP / "backtest_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    log.info(f"\n=== Backtest complete → {out} ===")
    log.info(f"  Tickers tested   : {summary['tickers_tested']}")
    log.info(f"  Avg win rate     : {summary['avg_win_rate']:.1%}")
    log.info(f"  Avg Sharpe       : {summary['avg_sharpe']:.2f}")
    log.info(f"  Avg CAGR         : {summary['avg_ann_return']*100:.1f}%")
    log.info(f"  Avg max drawdown : {summary['avg_max_dd']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VN100 Backtester v2")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--all",    action="store_true")
    args = parser.parse_args()

    tickers = ([f.stem for f in FEAT.glob("*.csv")]
               if (args.all or args.ticker is None)
               else [args.ticker])

    if len(tickers) > 1:
        run_all(tickers)
    else:
        for t in tickers:
            run(t)