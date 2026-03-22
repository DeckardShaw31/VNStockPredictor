"""
scheduler.py — Daily auto-tune daemon.

Triggers a full re-train + tune after Vietnam market close each trading day.
Compatible with APScheduler 3.x and 4.x.

Schedule:
  - 15:00 ICT Mon-Fri: re-fetch data, retrain & tune all symbols
  - 09:05 ICT Mon-Fri: generate fresh predictions before market open
"""

import logging
import signal
import sys
from datetime import datetime

import pytz

# ── APScheduler version compatibility ─────────────────────────────────────────
try:
    # APScheduler 3.x
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_V4 = False
except ImportError:
    # APScheduler 4.x
    from apscheduler import Scheduler as BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_V4 = True

import config
from pipeline import TrainingPipeline, PredictionPipeline

logger = logging.getLogger("scheduler")

VN_TZ = config.VIETNAM_TZ


# ── Job Functions ──────────────────────────────────────────────────────────────

def retrain_job(symbols=None, horizon=config.PREDICTION_HORIZON):
    """Full retrain + Optuna tune. Called after market close."""
    logger.info("=" * 60)
    logger.info(f"[RETRAIN JOB] Starting | {datetime.now(VN_TZ).isoformat()}")
    try:
        pipeline = TrainingPipeline(symbols=symbols, horizon=horizon, tune=True)
        summary  = pipeline.run()
        n_ok = sum(1 for v in summary.values() if "auc" in v)
        logger.info(f"[RETRAIN JOB] Complete | {n_ok}/{len(summary)} models updated")
    except Exception as e:
        logger.error(f"[RETRAIN JOB] FAILED: {e}", exc_info=True)


def predict_job(symbols=None, horizon=config.PREDICTION_HORIZON):
    """Generate predictions for the coming trading session."""
    logger.info(f"[PREDICT JOB] Generating predictions | {datetime.now(VN_TZ).isoformat()}")
    try:
        pipeline = PredictionPipeline(symbols=symbols, horizon=horizon)
        results  = pipeline.run()
        for sym, pred in results.items():
            arrow = "UP  " if pred["direction"] == 1 else "DOWN"
            logger.info(
                f"  {sym:6s} {arrow} {pred['return_pct']:+.2f}%  "
                f"conf={pred['confidence']:.1%}  "
                f"target={pred['target_price']:.2f}"
            )
    except Exception as e:
        logger.error(f"[PREDICT JOB] FAILED: {e}", exc_info=True)


def health_check_job():
    """Simple heartbeat log."""
    logger.info(f"[HEALTH] System alive | {datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M %Z')}")


# ── Scheduler Class ────────────────────────────────────────────────────────────

class DailyScheduler:
    def __init__(
        self,
        symbols=None,
        horizon: int = config.PREDICTION_HORIZON,
    ):
        self.symbols  = symbols or config.DEFAULT_SYMBOLS
        self.horizon  = horizon
        # APScheduler 4.x removed the timezone constructor arg
        if APSCHEDULER_V4:
            self.scheduler = BlockingScheduler()
        else:
            self.scheduler = BlockingScheduler(timezone=VN_TZ)
        self._register_jobs()

    def _add_job(self, func, cron_kwargs, job_id, job_name, job_kwargs, misfire_grace_time=300):
        """Version-safe add_job wrapper."""
        trigger = CronTrigger(**cron_kwargs, timezone=VN_TZ)
        if APSCHEDULER_V4:
            # APScheduler 4.x API
            self.scheduler.add_schedule(
                func,
                trigger=trigger,
                id=job_id,
                args=(),
                kwargs=job_kwargs,
            )
        else:
            # APScheduler 3.x API
            self.scheduler.add_job(
                func,
                trigger=trigger,
                kwargs=job_kwargs,
                id=job_id,
                name=job_name,
                replace_existing=True,
                misfire_grace_time=misfire_grace_time,
            )

    def _register_jobs(self):
        # 1. Pre-market predictions at 09:05 ICT Mon-Fri
        self._add_job(
            predict_job,
            cron_kwargs={"day_of_week": "mon-fri", "hour": 9, "minute": 5},
            job_id="predict_morning",
            job_name="Morning prediction",
            job_kwargs={"symbols": self.symbols, "horizon": self.horizon},
            misfire_grace_time=300,
        )

        # 2. Post-market retrain at 15:00 ICT Mon-Fri
        self._add_job(
            retrain_job,
            cron_kwargs={
                "day_of_week": "mon-fri",
                "hour": config.RETRAIN_HOUR,
                "minute": config.RETRAIN_MINUTE,
            },
            job_id="retrain_daily",
            job_name="Daily retrain + tune",
            job_kwargs={"symbols": self.symbols, "horizon": self.horizon},
            misfire_grace_time=1800,
        )

        # 3. Hourly health check
        self._add_job(
            health_check_job,
            cron_kwargs={"minute": 0},
            job_id="health_check",
            job_name="Health check",
            job_kwargs={},
        )

        logger.info("Registered jobs:")
        jobs = self.scheduler.get_jobs() if not APSCHEDULER_V4 else self.scheduler.get_schedules()
        for job in jobs:
            job_id   = getattr(job, "id", "?")
            job_name = getattr(job, "name", job_id)
            next_run = (
                getattr(job, "next_fire_time", None)
                or getattr(job, "next_run_time", None)
                or "calculated at start"
            )
            logger.info(f"  [{job_id}] {job_name}  next_run={next_run}")

    def start(self):
        """Block forever, running jobs on schedule."""
        logger.info("=" * 60)
        logger.info("Vietnam Stock AI -- Daily Scheduler STARTED")
        logger.info(f"Symbols  : {', '.join(self.symbols)}")
        logger.info(f"Horizon  : {self.horizon} days")
        logger.info(f"Timezone : {VN_TZ}")
        logger.info("Press Ctrl+C to stop.")
        logger.info("=" * 60)

        # Graceful shutdown on SIGINT / SIGTERM
        def _shutdown(signum, frame):
            logger.info("Shutting down scheduler...")
            try:
                if APSCHEDULER_V4:
                    self.scheduler.stop()
                else:
                    self.scheduler.shutdown(wait=False)
            except Exception:
                pass
            sys.exit(0)

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Run predict immediately on startup so you get today's predictions
        logger.info("Running initial prediction now...")
        predict_job(self.symbols, self.horizon)

        # Start the blocking scheduler
        if APSCHEDULER_V4:
            with self.scheduler:
                self.scheduler.run_until_stopped()
        else:
            self.scheduler.start()  # blocks
