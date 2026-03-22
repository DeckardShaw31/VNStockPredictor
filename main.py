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
        choices=["train", "predict", "daemon", "dashboard", "backtest"],
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
        default=5,
        help="Prediction horizon in trading days (default: 5)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        from pipeline import TrainingPipeline
        pipeline = TrainingPipeline(symbols=args.symbols, horizon=args.horizon)
        pipeline.run()

    elif args.mode == "predict":
        from pipeline import PredictionPipeline
        pipeline = PredictionPipeline(symbols=args.symbols, horizon=args.horizon)
        results = pipeline.run()
        for sym, pred in results.items():
            direction = "[UP]  " if pred["direction"] == 1 else "[DOWN]"
            logger.info(
                f"{sym}: {direction} {pred['return_pct']:+.2f}% | "
                f"confidence={pred['confidence']:.1%} | "
                f"target_price={pred['target_price']:.2f}"
            )

    elif args.mode == "daemon":
        from scheduler import DailyScheduler
        sched = DailyScheduler(symbols=args.symbols, horizon=args.horizon)
        sched.start()  # blocks forever

    elif args.mode == "backtest":
        from backtester import Backtester
        bt = Backtester(symbols=args.symbols, horizon=args.horizon)
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
