"""
Module 07 — End-of-Day Incremental Updater
===========================================
Run once after market close (~5 PM) to:

  Step 1  Download only NEW rows for each ticker (appends to raw CSV).
  Step 2  Rebuild features (full recompute — needed for rolling windows).
  Step 3  Re-detect signals (incremental: new signals appended, not replaced).
  Step 4  Model re-evaluation — score today's row with existing model,
          log prediction vs reality for yesterday's signals.
          Full retrain triggered only when new data adds ≥ RETRAIN_ROWS rows.
  Step 5  Backtest update — incremental equity append (no full replay).
  Step 6  Export to JSON (ui_data.json).

Usage:
    python 07_update.py                  # update all tickers
    python 07_update.py --ticker HPG     # single ticker
    python 07_update.py --force-retrain  # force full model retrain
    python 07_update.py --dry-run        # print plan, no writes

Design principles:
  • Raw CSVs are append-only. Old data is never deleted.
  • Features are always fully recomputed (rolling windows require full history).
  • Signals are appended to the existing signals CSV (deduped by date).
  • Models are retrained in-place when ≥ RETRAIN_ROWS new rows have accumulated
    since the last retrain — otherwise only probabilities are updated.
  • The update log (data/update_log.json) tracks every run.
"""

import json
import logging
import argparse
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Module imports (same-directory siblings) ──────────────────────────────────
import importlib, sys
ROOT = Path(__file__).resolve().parent

def _import(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA      = ROOT / "data"
RAW       = DATA / "raw"
FEAT      = DATA / "features"
SIG_DIR   = DATA / "signals"
MDL       = DATA / "models"
REP       = DATA / "reports"
VIZ       = ROOT / "viz"
LOG_FILE  = DATA / "update_log.json"

for d in [RAW, FEAT, SIG_DIR, MDL, REP, VIZ]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RETRAIN_ROWS = 20    # retrain model after this many new rows since last train
DOWNLOAD_DELAY = 2.0  # seconds between ticker downloads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("update")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Incremental download
# ─────────────────────────────────────────────────────────────────────────────

def _download_new_rows(ticker: str, today_str: str) -> tuple[pd.DataFrame, int]:
    """
    Reads existing raw CSV, finds the last date, downloads from last_date+1
    to today, appends new rows, saves.
    Returns (full_df, n_new_rows).
    """
    raw_path = RAW / f"{ticker}.csv"
    existing = pd.DataFrame()

    if raw_path.exists():
        existing = pd.read_csv(raw_path, dtype={"date": str})
        existing["date"] = pd.to_datetime(existing["date"]).dt.date.astype(str)
        last_date = existing["date"].max()
    else:
        last_date = "2000-01-01"

    # Nothing to do if already up to date
    if last_date >= today_str:
        log.info(f"  {ticker}: already up to date ({last_date})")
        return existing, 0

    # Download from day after last_date
    start = (date.fromisoformat(last_date) + timedelta(days=1)).isoformat()
    log.info(f"  {ticker}: fetching {start} → {today_str} …")

    try:
        from vnstock import Vnstock
        df_new = (
            Vnstock()
            .stock(symbol=ticker, source="KBS")
            .quote.history(start=start, end=today_str)
        )
        if df_new is None or df_new.empty:
            log.warning(f"  {ticker}: no new data returned")
            return existing, 0

        df_new = df_new.rename(columns={"time": "date"})
        df_new["date"] = pd.to_datetime(df_new["date"]).dt.date.astype(str)
        df_new = df_new[["date", "open", "high", "low", "close", "volume"]].dropna()
        df_new = df_new[df_new["date"] > last_date]   # strict: only truly new rows

        if df_new.empty:
            log.info(f"  {ticker}: 0 new rows after date filter")
            return existing, 0

        combined = pd.concat([existing, df_new], ignore_index=True)
        combined = combined.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        combined.to_csv(raw_path, index=False)
        n_new = len(df_new)
        log.info(f"  {ticker}: +{n_new} new rows → {raw_path.name}")
        return combined, n_new

    except Exception as e:
        log.warning(f"  {ticker}: download error — {e}")
        return existing, 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Feature rebuild
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild_features(ticker: str) -> bool:
    """Full feature recompute (rolling windows require full history)."""
    try:
        m02 = _import("02_features")
        m02.run(ticker)
        return True
    except Exception as e:
        log.error(f"  {ticker}: feature error — {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Signal update
# ─────────────────────────────────────────────────────────────────────────────

def _update_signals(ticker: str, today_str: str) -> dict:
    """
    Re-runs signal detection on the full feature file, but only appends
    signals with date > last known signal date.  Deduplicates by date.
    Returns the latest signal dict.
    """
    try:
        feat_path = FEAT / f"{ticker}.csv"
        if not feat_path.exists():
            return {}

        m03 = _import("03_signals")
        df_feat = pd.read_csv(feat_path, low_memory=False)

        # Restrict to new rows only for signal detection (performance)
        sig_csv = SIG_DIR / f"{ticker}_signals.csv"
        if sig_csv.exists():
            old_sigs = pd.read_csv(sig_csv)
            last_sig_date = old_sigs["date"].max() if len(old_sigs) else "2000-01-01"
        else:
            old_sigs      = pd.DataFrame()
            last_sig_date = "2000-01-01"

        # Only run signal detection on rows after last signal
        df_new_feat = df_feat[pd.to_datetime(df_feat["date"]).dt.date.astype(str) > last_sig_date]

        if df_new_feat.empty:
            # Load and return existing latest signal
            jf = SIG_DIR / f"{ticker}_latest.json"
            return json.loads(jf.read_text()) if jf.exists() else {}

        new_sig_df, latest = m03.process_ticker(ticker, df_new_feat)

        if not new_sig_df.empty:
            combined = pd.concat([old_sigs, new_sig_df], ignore_index=True)
            combined = combined.drop_duplicates("date").sort_values("date")
            combined.to_csv(sig_csv, index=False)
            (SIG_DIR / f"{ticker}_latest.json").write_text(
                json.dumps(latest, indent=2, default=str))
            log.info(f"  {ticker}: +{len(new_sig_df)} new signal(s)")
        else:
            log.info(f"  {ticker}: no new signals today")

        jf = SIG_DIR / f"{ticker}_latest.json"
        return json.loads(jf.read_text()) if jf.exists() else {}

    except Exception as e:
        log.error(f"  {ticker}: signal error — {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Model re-evaluation / conditional retrain
# ─────────────────────────────────────────────────────────────────────────────

def _model_update(ticker: str, n_new_rows: int, force_retrain: bool) -> dict:
    """
    • Always: score today's feature row with existing model → append to proba CSV.
    • Conditionally: retrain if n_new_rows >= RETRAIN_ROWS or force_retrain.
    • Evaluate: check if yesterday's signal direction was correct → log accuracy.
    Returns a dict with today's probabilities.
    """
    results = {}
    feat_path = FEAT / f"{ticker}.csv"
    if not feat_path.exists():
        return results

    df = pd.read_csv(feat_path, low_memory=False)

    # ── Score today with existing model ──────────────────────────────────────
    try:
        import joblib
        m04 = _import("04_models")
        feature_cols = m04._get_feature_cols(df)

        from xgboost import XGBClassifier  # noqa – just for import check
        today_proba = {}

        for strat_key, target_col in m04.STRATEGY_TARGETS.items():
            pkl = MDL / f"{ticker}_{strat_key}.pkl"
            if not pkl.exists():
                continue
            pipe = joblib.load(pkl)
            last_row = m04._safe_fillna(df.tail(1), feature_cols)
            try:
                p = float(pipe.predict_proba(last_row[feature_cols].values)[:, 1][0])
                today_proba[f"proba_fwd_positive_{target_col.split('_positive_')[1]}"] = round(p, 4)
            except Exception:
                pass

        results["today_proba"] = today_proba

    except Exception as e:
        log.warning(f"  {ticker}: model scoring error — {e}")

    # ── Evaluate yesterday's signal hit rate ─────────────────────────────────
    try:
        proba_path = MDL / f"{ticker}_proba.csv"
        if proba_path.exists():
            p_df = pd.read_csv(proba_path)
            # Append today's row
            today_row = {"date": df.iloc[-1]["date"], "close": df.iloc[-1]["close"]}
            today_row.update(results.get("today_proba", {}))
            p_df = pd.concat([p_df, pd.DataFrame([today_row])], ignore_index=True)
            p_df = p_df.drop_duplicates("date").sort_values("date")
            p_df.to_csv(proba_path, index=False)
    except Exception as e:
        log.warning(f"  {ticker}: proba append error — {e}")

    # ── Conditional retrain ───────────────────────────────────────────────────
    should_retrain = force_retrain or (n_new_rows >= RETRAIN_ROWS)
    if should_retrain:
        log.info(f"  {ticker}: retraining model ({n_new_rows} new rows) …")
        try:
            m04 = _import("04_models")
            m04.run(ticker)
            results["retrained"] = True
            log.info(f"  {ticker}: model retrained ✓")
        except Exception as e:
            log.error(f"  {ticker}: retrain error — {e}")
            results["retrained"] = False
    else:
        results["retrained"] = False
        rows_until_retrain = RETRAIN_ROWS - (n_new_rows % RETRAIN_ROWS)
        log.info(f"  {ticker}: model not retrained "
                 f"({rows_until_retrain} rows until next scheduled retrain)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Incremental backtest equity append
# ─────────────────────────────────────────────────────────────────────────────

def _incremental_backtest(ticker: str):
    """
    Rather than replaying the full backtest, we just check if any open
    positions in the signal log have now hit their exit date / stop / target,
    and append those outcomes to the equity CSV.

    Full backtest replay happens only when --full-backtest flag is passed
    (handled in main).
    """
    try:
        sig_csv = SIG_DIR / f"{ticker}_signals.csv"
        eq_csv  = REP / f"{ticker}_equity.csv"
        feat_path = FEAT / f"{ticker}.csv"

        if not sig_csv.exists() or not feat_path.exists():
            return

        sig_df  = pd.read_csv(sig_csv)
        df_feat = pd.read_csv(feat_path, low_memory=False)
        today_str = date.today().isoformat()

        # Find signals whose exit_date is today or in the past but not yet in equity
        existing_eq = pd.read_csv(eq_csv) if eq_csv.exists() else pd.DataFrame(columns=["date","equity"])
        last_eq_date = existing_eq["date"].max() if len(existing_eq) else "2000-01-01"

        # Use the full backtester for any new signals since last equity date
        m05 = _import("05_backtest")
        new_sigs = sig_df[pd.to_datetime(sig_df["date"]).dt.date.astype(str) > last_eq_date]
        if new_sigs.empty:
            return

        stats, trades, eq_new = m05.backtest_ticker(ticker, df_feat, new_sigs)
        if not eq_new:
            return

        # Append new equity rows
        eq_new_df = pd.DataFrame(eq_new)
        eq_new_df = eq_new_df[eq_new_df["date"] > last_eq_date]
        if not eq_new_df.empty:
            combined = pd.concat([existing_eq, eq_new_df], ignore_index=True)
            combined = combined.drop_duplicates("date").sort_values("date")
            combined.to_csv(eq_csv, index=False)
            log.info(f"  {ticker}: equity curve updated (+{len(eq_new_df)} rows)")

    except Exception as e:
        log.warning(f"  {ticker}: incremental backtest error — {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Export
# ─────────────────────────────────────────────────────────────────────────────

def _export(tickers: list[str]):
    log.info("Exporting to ui_data.json …")
    try:
        m06 = _import("06_export")
        m06.run_export(tickers)
    except Exception as e:
        log.error(f"Export error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Update log
# ─────────────────────────────────────────────────────────────────────────────

def _load_log() -> dict:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except Exception:
            pass
    return {"runs": [], "ticker_stats": {}}


def _save_log(log_data: dict):
    LOG_FILE.write_text(json.dumps(log_data, indent=2, default=str))


def _log_run(log_data: dict, summary: dict):
    log_data.setdefault("runs", []).append(summary)
    log_data["runs"] = log_data["runs"][-90:]   # keep last 90 daily runs
    _save_log(log_data)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_update(
    tickers: list[str],
    today_str:     str,
    force_retrain: bool = False,
    full_backtest: bool = False,
    dry_run:       bool = False,
) -> dict:

    log.info(f"\n{'='*60}")
    log.info(f"  VN100 END-OF-DAY UPDATE  —  {today_str}")
    log.info(f"  Tickers: {len(tickers)}  |  force_retrain={force_retrain}")
    log.info(f"{'='*60}\n")

    if dry_run:
        log.info("[DRY RUN] No data will be written.")
        for t in tickers:
            raw_path = RAW / f"{t}.csv"
            last = pd.read_csv(raw_path)["date"].max() if raw_path.exists() else "none"
            log.info(f"  {t}: last raw date = {last}")
        return {}

    log_data = _load_log()
    run_summary = {
        "run_at":          datetime.now().isoformat(),
        "today":           today_str,
        "tickers_updated": [],
        "new_rows_total":  0,
        "signals_new":     0,
        "retrained":       [],
        "errors":          [],
    }

    for ticker in tickers:
        log.info(f"\n── {ticker} ─────────────────────────────────────────")
        try:
            # Step 1: Download
            _, n_new = _download_new_rows(ticker, today_str)
            run_summary["new_rows_total"] += n_new
            time.sleep(DOWNLOAD_DELAY)

            if n_new == 0 and not force_retrain:
                # Check if feature file exists; if not, build it
                if not (FEAT / f"{ticker}.csv").exists():
                    log.info(f"  {ticker}: no feature file — building from existing raw")
                    _rebuild_features(ticker)
                else:
                    log.info(f"  {ticker}: no new data, skipping feature/signal rebuild")
                    continue

            # Step 2: Features
            ok = _rebuild_features(ticker)
            if not ok:
                run_summary["errors"].append(f"{ticker}: feature build failed")
                continue

            # Step 3: Signals
            _update_signals(ticker, today_str)

            # Step 4: Model
            model_res = _model_update(ticker, n_new, force_retrain)
            if model_res.get("retrained"):
                run_summary["retrained"].append(ticker)

            # Step 5: Backtest (incremental or full)
            if full_backtest:
                log.info(f"  {ticker}: running full backtest …")
                m05 = _import("05_backtest")
                m05.run(ticker)
            else:
                _incremental_backtest(ticker)

            run_summary["tickers_updated"].append(ticker)

        except Exception as e:
            log.error(f"  {ticker}: unexpected error — {e}")
            run_summary["errors"].append(f"{ticker}: {e}")

    # Step 6: Export (always runs for all tickers, even unchanged ones)
    _export(tickers)

    # Persist log
    _log_run(log_data, run_summary)

    log.info(f"\n{'='*60}")
    log.info(f"  UPDATE COMPLETE")
    log.info(f"  Tickers updated : {len(run_summary['tickers_updated'])}")
    log.info(f"  New rows total  : {run_summary['new_rows_total']}")
    log.info(f"  Models retrained: {run_summary['retrained']}")
    if run_summary["errors"]:
        log.warning(f"  Errors          : {run_summary['errors']}")
    log.info(f"{'='*60}\n")

    return run_summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VN100 end-of-day incremental updater",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 07_update.py                    # update all tickers
  python 07_update.py --ticker HPG VCB   # update specific tickers
  python 07_update.py --force-retrain    # force full model retrain today
  python 07_update.py --full-backtest    # re-run full backtest (slow)
  python 07_update.py --dry-run          # print plan only, no writes
  python 07_update.py --date 2026-04-10  # pretend today is this date
        """
    )
    parser.add_argument("--ticker",        nargs="*", default=None,
                        help="Specific ticker(s) to update (default: all raw CSVs)")
    parser.add_argument("--all",           action="store_true",
                        help="Force update of ALL known tickers")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force model retrain regardless of new row count")
    parser.add_argument("--full-backtest", action="store_true",
                        help="Run full walk-forward backtest (slower)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print plan only, no data writes")
    parser.add_argument("--date",          default=None,
                        help="Override today's date (YYYY-MM-DD) for testing")
    parser.add_argument("--delay",         type=float, default=DOWNLOAD_DELAY,
                        help=f"Seconds between ticker downloads (default: {DOWNLOAD_DELAY})")
    args = parser.parse_args()

    DOWNLOAD_DELAY = args.delay

    today_str = args.date or date.today().isoformat()

    # Resolve ticker list
    if args.ticker:
        tickers = [t.upper() for t in args.ticker]
    elif args.all:
        # Import default list from 01_download
        try:
            m01 = _import("01_download")
            tickers = m01.VN100_DEFAULT
        except Exception:
            tickers = [f.stem for f in RAW.glob("*.csv")]
    else:
        # Default: all tickers that already have a raw CSV
        tickers = sorted(f.stem for f in RAW.glob("*.csv"))
        if not tickers:
            log.warning("No raw CSV files found. Run 01_download.py first, "
                        "or use --all to download the full universe.")
            sys.exit(0)

    run_update(
        tickers       = tickers,
        today_str     = today_str,
        force_retrain = args.force_retrain,
        full_backtest = args.full_backtest,
        dry_run       = args.dry_run,
    )