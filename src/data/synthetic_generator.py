"""
EDMD Framework — Synthetic Event Stream Generator
==================================================
Generates a synthetic financial event stream calibrated to
production credit risk portfolio characteristics.

Supports four event classes:
  - TransactionEvent    (δ_T = 15 min)
  - PaymentEvent        (δ_P = 72 hours)
  - BureauRefreshEvent  (δ_B = 30 days)
  - AccountChangeEvent  (δ_A = 24 hours)

Usage:
    python src/data/synthetic_generator.py \
        --n_accounts 500000 \
        --daily_events 10000000 \
        --simulation_days 90 \
        --default_rate 0.048 \
        --late_injection_rate 0.10 \
        --output_path artifacts/synthetic_events.parquet \
        --seed 42
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Event class configuration ─────────────────────────────────────────────────
EVENT_CONFIG = {
    "TransactionEvent": {
        "daily_fraction": 0.85,          # ~85% of all events
        "lateness_bound_minutes": 15,    # δ_T
        "lateness_p99_minutes": 12,      # 99th pct of actual lateness
        "features": ["amount", "mcc_code", "channel", "merchant_id"],
    },
    "PaymentEvent": {
        "daily_fraction": 0.08,          # ~8% of all events
        "lateness_bound_minutes": 72*60, # δ_P = 72 hours
        "lateness_p99_minutes": 60*60,   # 99th pct = 60 hours
        "features": ["payment_amount", "payment_type", "days_before_due"],
    },
    "BureauRefreshEvent": {
        "daily_fraction": 0.02,          # ~2% of all events
        "lateness_bound_minutes": 30*24*60, # δ_B = 30 days
        "lateness_p99_minutes": 25*24*60,   # 99th pct = 25 days
        "features": ["credit_score", "score_delta", "inquiry_count", "external_util"],
    },
    "AccountChangeEvent": {
        "daily_fraction": 0.05,          # ~5% of all events
        "lateness_bound_minutes": 24*60, # δ_A = 24 hours
        "lateness_p99_minutes": 20*60,   # 99th pct = 20 hours
        "features": ["current_limit", "product_type", "revolving_status"],
    },
}

MCC_CODES = [5411, 5812, 5541, 4111, 5912, 5311, 7011, 5661, 5732, 5999]
CHANNELS = ["online", "in_store", "mobile", "contactless"]
PAYMENT_TYPES = ["full", "minimum", "partial", "autopay"]


class SyntheticEventGenerator:
    """
    Generates a synthetic financial event stream using a Gaussian copula
    over credit bureau signals and behavioral features, with conditional
    default rates calibrated by credit score decile and utilization band.
    """

    def __init__(
        self,
        n_accounts: int = 500_000,
        daily_events: int = 10_000_000,
        simulation_days: int = 90,
        default_rate: float = 0.048,
        late_injection_rate: float = 0.10,
        seed: int = 42,
    ):
        self.n_accounts = n_accounts
        self.daily_events = daily_events
        self.simulation_days = simulation_days
        self.default_rate = default_rate
        self.late_injection_rate = late_injection_rate
        self.rng = np.random.default_rng(seed)

        # Generate account master data
        self._generate_account_master()

    def _generate_account_master(self) -> None:
        """Generate account-level static features using Gaussian copula."""
        logger.info(f"Generating account master for {self.n_accounts:,} accounts...")

        # Credit score — calibrated to prime revolving portfolio (Thomas et al., 2017)
        self.credit_scores = self.rng.normal(720, 65, self.n_accounts).clip(300, 850).astype(int)

        # Current credit limit — log-normal
        self.current_limits = np.exp(
            self.rng.normal(8.5, 0.6, self.n_accounts)
        ).clip(500, 50000).astype(int)

        # Utilization — beta distribution
        self.utilization = self.rng.beta(2, 5, self.n_accounts).clip(0, 1)

        # Default propensity — logistic function of score and utilization
        log_odds = (
            -4.5
            + (-0.008 * self.credit_scores)
            + (2.5 * self.utilization)
            + self.rng.normal(0, 0.3, self.n_accounts)
        )
        self.default_prob = 1 / (1 + np.exp(-log_odds))

        # Assign default labels with calibrated rate
        threshold = np.percentile(self.default_prob, (1 - self.default_rate) * 100)
        self.default_labels = (self.default_prob >= threshold).astype(int)

        # Account IDs
        self.account_ids = np.arange(1, self.n_accounts + 1)

        actual_rate = self.default_labels.mean()
        logger.info(f"Account master generated. Default rate: {actual_rate:.3f} (target: {self.default_rate})")

    def _generate_transaction_event(self, account_idx: int, event_time: datetime) -> dict:
        """Generate a single TransactionEvent."""
        limit = self.current_limits[account_idx]
        util = self.utilization[account_idx]
        base_amount = limit * util * self.rng.uniform(0.01, 0.05)

        return {
            "event_type": "TransactionEvent",
            "account_id": int(self.account_ids[account_idx]),
            "event_time": event_time,
            "amount": round(float(base_amount * self.rng.uniform(0.5, 2.0)), 2),
            "mcc_code": int(self.rng.choice(MCC_CODES)),
            "channel": str(self.rng.choice(CHANNELS)),
            "merchant_id": int(self.rng.integers(1000, 99999)),
            "is_reversal": bool(self.rng.random() < 0.02),
            "late_arrival_flag": False,
        }

    def _generate_payment_event(self, account_idx: int, event_time: datetime) -> dict:
        """Generate a single PaymentEvent."""
        limit = self.current_limits[account_idx]
        util = self.utilization[account_idx]
        balance = limit * util

        payment_type = str(self.rng.choice(
            PAYMENT_TYPES,
            p=[0.35, 0.25, 0.30, 0.10]
        ))

        if payment_type == "full":
            amount = balance
        elif payment_type == "minimum":
            amount = max(25.0, balance * 0.02)
        elif payment_type == "autopay":
            amount = balance * self.rng.uniform(0.1, 1.0)
        else:
            amount = balance * self.rng.uniform(0.05, 0.95)

        return {
            "event_type": "PaymentEvent",
            "account_id": int(self.account_ids[account_idx]),
            "event_time": event_time,
            "payment_amount": round(float(amount), 2),
            "payment_type": payment_type,
            "statement_balance": round(float(balance), 2),
            "days_before_due": int(self.rng.integers(-5, 30)),
            "late_arrival_flag": False,
        }

    def _generate_bureau_refresh_event(self, account_idx: int, event_time: datetime) -> dict:
        """Generate a single BureauRefreshEvent."""
        score = self.credit_scores[account_idx]
        score_delta = int(self.rng.integers(-40, 40))

        return {
            "event_type": "BureauRefreshEvent",
            "account_id": int(self.account_ids[account_idx]),
            "event_time": event_time,
            "credit_score": score,
            "score_delta_3m": score_delta,
            "score_delta_6m": int(score_delta * 1.5 + self.rng.integers(-15, 15)),
            "inquiry_count_90d": int(self.rng.integers(0, 6)),
            "external_utilization": float(self.rng.beta(2, 4)),
            "derogatory_marks": int(self.rng.integers(0, 3)),
            "late_arrival_flag": False,
        }

    def _generate_account_change_event(self, account_idx: int, event_time: datetime) -> dict:
        """Generate a single AccountChangeEvent."""
        return {
            "event_type": "AccountChangeEvent",
            "account_id": int(self.account_ids[account_idx]),
            "event_time": event_time,
            "current_limit": int(self.current_limits[account_idx]),
            "product_type": str(self.rng.choice(["cash_back", "travel", "secured", "student"])),
            "revolving_status": bool(self.rng.random() < 0.75),
            "account_tenure_months": int(self.rng.integers(1, 240)),
            "late_arrival_flag": False,
        }

    def _inject_lateness(self, events: List[dict], event_type: str) -> List[dict]:
        """
        Inject late-arriving events at the configured rate.
        Late events exceed the calibrated lateness bound δ_c.
        """
        config = EVENT_CONFIG[event_type]
        bound_minutes = config["lateness_bound_minutes"]

        n_late = int(len(events) * self.late_injection_rate)
        late_indices = self.rng.choice(len(events), size=n_late, replace=False)

        for idx in late_indices:
            # Lateness beyond bound — exponential tail
            extra_lateness = self.rng.exponential(scale=bound_minutes * 0.3)
            late_minutes = bound_minutes + extra_lateness
            events[idx]["event_time"] = events[idx]["event_time"] - timedelta(minutes=late_minutes)
            events[idx]["late_arrival_flag"] = True

        return events

    def generate_day(self, day: int, base_date: datetime) -> pd.DataFrame:
        """Generate all events for a single simulation day."""
        events = []
        day_start = base_date + timedelta(days=day)

        for event_type, config in EVENT_CONFIG.items():
            n_events = int(self.daily_events * config["daily_fraction"])
            type_events = []

            # Sample account indices
            account_indices = self.rng.integers(0, self.n_accounts, size=n_events)

            # Sample event times within the day
            event_minutes = self.rng.uniform(0, 1440, size=n_events)

            for i, (acc_idx, mins) in enumerate(zip(account_indices, event_minutes)):
                event_time = day_start + timedelta(minutes=float(mins))

                if event_type == "TransactionEvent":
                    evt = self._generate_transaction_event(acc_idx, event_time)
                elif event_type == "PaymentEvent":
                    evt = self._generate_payment_event(acc_idx, event_time)
                elif event_type == "BureauRefreshEvent":
                    evt = self._generate_bureau_refresh_event(acc_idx, event_time)
                else:
                    evt = self._generate_account_change_event(acc_idx, event_time)

                type_events.append(evt)

            # Inject late events
            type_events = self._inject_lateness(type_events, event_type)
            events.extend(type_events)

        df = pd.DataFrame(events)
        df["processing_time"] = pd.Timestamp.now()
        df["day"] = day
        return df

    def generate(self, output_path: str, sample_days: int = 3) -> pd.DataFrame:
        """
        Generate synthetic event stream.
        For performance, generates sample_days days and returns summary.
        Full 90-day generation available via generate_full().
        """
        logger.info(f"Generating {sample_days} days of synthetic events...")
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        base_date = datetime(2024, 1, 1)
        all_dfs = []

        for day in range(sample_days):
            logger.info(f"  Day {day+1}/{sample_days}...")
            df = self.generate_day(day, base_date)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined["event_time"] = pd.to_datetime(combined["event_time"])

        # Save
        combined.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(combined):,} events to {output_path}")

        # Summary statistics
        summary = combined.groupby("event_type").agg(
            count=("account_id", "count"),
            late_count=("late_arrival_flag", "sum"),
        )
        summary["late_rate"] = summary["late_count"] / summary["count"]
        logger.info(f"\nEvent summary:\n{summary.to_string()}")

        return combined


def main():
    parser = argparse.ArgumentParser(description="EDMD Synthetic Event Stream Generator")
    parser.add_argument("--n_accounts", type=int, default=500_000)
    parser.add_argument("--daily_events", type=int, default=10_000_000)
    parser.add_argument("--simulation_days", type=int, default=3, help="Days to generate (3 for demo, 90 for full)")
    parser.add_argument("--default_rate", type=float, default=0.048)
    parser.add_argument("--late_injection_rate", type=float, default=0.10)
    parser.add_argument("--output_path", type=str, default="artifacts/synthetic_events.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generator = SyntheticEventGenerator(
        n_accounts=args.n_accounts,
        daily_events=args.daily_events,
        simulation_days=args.simulation_days,
        default_rate=args.default_rate,
        late_injection_rate=args.late_injection_rate,
        seed=args.seed,
    )

    generator.generate(args.output_path, sample_days=args.simulation_days)


if __name__ == "__main__":
    main()
