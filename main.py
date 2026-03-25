"""
Vietnam Stock AI Prediction System
===================================
Entry point. Run:
  python main.py --mode train   # one-time train + tune
  python main.py --mode predict # predict today
  python main.py --mode daemon  # auto-tune daily (background scheduler)
  python main.py --mode dashboard # launch Streamlit UI
"""

import argparse
import sys
import logging
import io
from pathlib import Path

import config

# ── Force UTF-8 output on Windows (fixes cp1252 UnicodeEncodeError) ───────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/system.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

Path("logs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("data/cache").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Vietnam Stock AI Prediction System")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "live", "update", "daemon", "dashboard", "backtest"],
        default="dashboard",
        help="Execution mode",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Stock symbols (e.g. VNM VIC HPG). Defaults to config.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,   # None = read from config.PREDICTION_HORIZON
        help="Prediction horizon in trading days. Defaults to config.PREDICTION_HORIZON.",
    )
    parser.add_argument(
        "--retrain-transformer",
        action="store_true",
        default=False,
        help="Force re-train the Pattern Transformer from scratch (ignores saved weights)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        default=False,
        help="Skip Optuna tuning — use default hyperparameters (much faster)",
    )
    args = parser.parse_args()

    # Resolve horizon: CLI arg takes priority, then config.PREDICTION_HORIZON
    effective_horizon = args.horizon if args.horizon is not None else config.PREDICTION_HORIZON
    if args.horizon is not None and args.horizon != config.PREDICTION_HORIZON:
        logger.info(f"Horizon override: {effective_horizon}d (config default: {config.PREDICTION_HORIZON}d)")
    else:
        logger.info(f"Horizon: {effective_horizon}d (from config.PREDICTION_HORIZON)")

    if args.mode == "train":
        from pipeline import TrainingPipeline

        # ── Interactive: ask about transformer retraining ──────────────────
        force_retrain_transformer = args.retrain_transformer

        if not force_retrain_transformer:
            from pathlib import Path
            pretrained_exists = (
                Path(f"{__import__('config').MODEL_DIR}/pretrained_transformer_weights.pt").exists()
            )
            if pretrained_exists:
                print("\n" + "="*60)
                print("  Pattern Transformer — saved weights found")
                print("="*60)
                answer = input("  Retrain transformer from scratch? [y/N]: ").strip().lower()
                force_retrain_transformer = answer in ("y", "yes")
                if force_retrain_transformer:
                    print("  Will rebuild transformer from scratch.")
                else:
                    print("  Loading existing transformer weights (faster).")
                print("="*60 + "\n")

        pipeline = TrainingPipeline(
            symbols=args.symbols,
            horizon=effective_horizon,
            tune=not args.no_tune,
            force_retrain_transformer=force_retrain_transformer,
        )
        pipeline.run()

    elif args.mode == "predict":
        from pipeline import PredictionPipeline
        from data_fetcher import fetch_multiple, get_vnindex
        from math_models import build_math_model_features, get_math_signal_votes
        from trade_signals import generate_signal, format_signal, save_signals

        pipeline = PredictionPipeline(symbols=args.symbols, horizon=effective_horizon)
        results  = pipeline.run()

        syms   = list(results.keys())
        ohlcvs = fetch_multiple(syms)
        try:
            vnidx = get_vnindex()
        except Exception:
            vnidx = None

        signals = []
        for sym, pred in results.items():
            direction = "[UP]  " if pred["direction"] == 1 else "[DOWN]"
            logger.info(
                f"{sym}: {direction} {pred['return_pct']:+.2f}% | "
                f"confidence={pred['confidence']:.1%} | "
                f"target_price={pred['target_price']:.2f}"
            )
            if sym in ohlcvs:
                try:
                    math_df    = build_math_model_features(
                        ohlcvs[sym]["high"], ohlcvs[sym]["low"],
                        ohlcvs[sym]["close"], ohlcvs[sym]["open"]
                    )
                    math_votes = get_math_signal_votes(pred["last_close"], math_df)
                    sig = generate_signal(
                        symbol=sym, ohlcv=ohlcvs[sym],
                        ai_confidence=pred["confidence"],
                        math_df=math_df, math_votes=math_votes,
                        horizon=effective_horizon, model_auc=pred.get("model_auc", 0),
                    )
                    signals.append(sig)
                    logger.info(f"\n{format_signal(sig)}")
                except Exception as e:
                    logger.warning(f"Signal generation failed for {sym}: {e}")

        if signals:
            save_signals(signals)

    elif args.mode == "update":
        from updater import IncrementalUpdater
        updater = IncrementalUpdater(symbols=args.symbols, horizon=effective_horizon)
        summary = updater.run()
        ok  = sum(1 for v in summary.values() if "error" not in v)
        err = len(summary) - ok
        logger.info(f"Update complete: {ok} symbols updated, {err} failed")

    elif args.mode == "live":
        from live_engine import LiveTradingEngine
        engine = LiveTradingEngine(symbols=args.symbols, horizon=effective_horizon)
        engine.run_until_close()

    elif args.mode == "daemon":
        from scheduler import DailyScheduler
        sched = DailyScheduler(symbols=args.symbols, horizon=effective_horizon)
        sched.start()  # blocks forever

    elif args.mode == "backtest":
        from backtester import Backtester
        bt = Backtester(symbols=args.symbols, horizon=effective_horizon)
        report = bt.run()
        logger.info(f"\n{report}")

    elif args.mode == "dashboard":
        import subprocess
        subprocess.run(
            ["streamlit", "run", "dashboard.py", "--server.port=8501"],
            check=True,
        )


if __name__ == "__main__":
    main()