"""
EDMD Framework — Population Stability Index (PSI) Drift Monitor
===============================================================
Monitors feature and score distribution drift using PSI.

PSI = Σ_i (p_i − q_i) · ln(p_i / q_i)

Thresholds (industry standard):
  PSI < 0.10 : Stable — no action required
  PSI 0.10–0.25: Moderate drift — investigation triggered
  PSI > 0.25 : Significant drift — retraining initiated

Usage:
    python src/monitoring/drift_monitor.py \
        --reference_path artifacts/reference_features.parquet \
        --current_path artifacts/current_features.parquet \
        --output_path artifacts/psi_report.json
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PSI_STABLE = 0.10
PSI_MODERATE = 0.25

STATUS_STABLE = "STABLE"
STATUS_MODERATE = "MODERATE_DRIFT"
STATUS_SIGNIFICANT = "SIGNIFICANT_DRIFT"


def compute_psi_single(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute PSI between reference and current distributions.

    PSI = Σ_i (p_i − q_i) · ln(p_i / q_i)

    Parameters
    ----------
    reference : reference distribution (training window)
    current   : current distribution (scoring window)
    n_bins    : number of equal-frequency bins
    epsilon   : smoothing constant to prevent log(0)
    """
    # Use reference distribution to define bin edges
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Count observations in each bin
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    # Convert to proportions with smoothing
    p = ref_counts / len(reference) + epsilon
    q = cur_counts / len(current) + epsilon

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    psi = np.sum((p - q) * np.log(p / q))
    return float(psi)


def interpret_psi(psi: float) -> Tuple[str, str]:
    """Return status and recommended action for a PSI value."""
    if psi < PSI_STABLE:
        return STATUS_STABLE, "No action required."
    elif psi < PSI_MODERATE:
        return STATUS_MODERATE, "Investigate feature distribution. Consider retraining if trend continues."
    else:
        return STATUS_SIGNIFICANT, "Significant drift detected. Initiate model retraining."


class EDMDDriftMonitor:
    """
    PSI-based drift monitor for EDMD Platinum tier features.
    Monitors both feature distributions and model score distribution.
    """

    FEATURE_FAMILIES = {
        "utilization_dynamics": [
            "util_mean_3m", "util_mean_6m", "util_mean_12m",
            "util_slope_6m", "util_volatility", "mcc_entropy_6m",
        ],
        "payment_stability": [
            "ptr_mean_6m", "ptr_mean_3m", "full_to_min_ratio",
            "consecutive_ontime_payments", "payment_timing_dev_days",
        ],
        "risk_drift_indicators": [
            "credit_score_latest", "score_delta_3m", "score_delta_6m",
            "inquiry_count_90d", "external_utilization",
        ],
        "behavioral_stability": [
            "spend_cv_6m", "channel_entropy_6m", "monthly_spend_cv",
            "transaction_count_6m", "revolver_consistency",
        ],
    }

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def monitor(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        score_col: str = "pd_score",
    ) -> Dict:
        """
        Compute PSI for all features and model score.

        Returns
        -------
        dict with PSI values, status, and recommended actions per feature
        """
        report = {
            "monitoring_timestamp": str(pd.Timestamp.now()),
            "reference_n": len(reference),
            "current_n": len(current),
            "feature_psi": {},
            "family_psi": {},
            "score_psi": None,
            "overall_status": STATUS_STABLE,
            "alerts": [],
        }

        # Monitor each feature family
        for family, features in self.FEATURE_FAMILIES.items():
            family_psi_values = []

            for feat in features:
                if feat not in reference.columns or feat not in current.columns:
                    continue

                ref_vals = reference[feat].dropna().values
                cur_vals = current[feat].dropna().values

                if len(ref_vals) < 10 or len(cur_vals) < 10:
                    continue

                psi = compute_psi_single(ref_vals, cur_vals, self.n_bins)
                status, action = interpret_psi(psi)

                report["feature_psi"][feat] = {
                    "psi": round(psi, 4),
                    "status": status,
                    "action": action,
                }

                family_psi_values.append(psi)

                if psi >= PSI_MODERATE:
                    report["alerts"].append({
                        "feature": feat,
                        "family": family,
                        "psi": round(psi, 4),
                        "status": status,
                    })

            if family_psi_values:
                family_psi = float(np.mean(family_psi_values))
                family_status, _ = interpret_psi(family_psi)
                report["family_psi"][family] = {
                    "mean_psi": round(family_psi, 4),
                    "max_psi": round(max(family_psi_values), 4),
                    "status": family_status,
                }

        # Monitor model score distribution
        if score_col in reference.columns and score_col in current.columns:
            score_psi = compute_psi_single(
                reference[score_col].dropna().values,
                current[score_col].dropna().values,
                self.n_bins,
            )
            score_status, score_action = interpret_psi(score_psi)
            report["score_psi"] = {
                "psi": round(score_psi, 4),
                "status": score_status,
                "action": score_action,
            }

            if score_psi >= PSI_MODERATE:
                report["alerts"].append({
                    "feature": "model_score",
                    "family": "scoring",
                    "psi": round(score_psi, 4),
                    "status": score_status,
                })

        # Overall status = worst status observed
        all_psi = [v["psi"] for v in report["feature_psi"].values()]
        if report["score_psi"]:
            all_psi.append(report["score_psi"]["psi"])

        if all_psi:
            max_psi = max(all_psi)
            if max_psi >= PSI_MODERATE:
                report["overall_status"] = STATUS_SIGNIFICANT
            elif max_psi >= PSI_STABLE:
                report["overall_status"] = STATUS_MODERATE
            else:
                report["overall_status"] = STATUS_STABLE

        # Summary
        n_alerts = len(report["alerts"])
        logger.info(f"PSI monitoring complete.")
        logger.info(f"  Overall status: {report['overall_status']}")
        logger.info(f"  Features monitored: {len(report['feature_psi'])}")
        logger.info(f"  Alerts triggered: {n_alerts}")

        if n_alerts > 0:
            logger.warning(f"  ALERTS:")
            for alert in report["alerts"]:
                logger.warning(f"    [{alert['status']}] {alert['feature']}: PSI={alert['psi']:.4f}")

        return report


def main():
    parser = argparse.ArgumentParser(description="EDMD PSI Drift Monitor")
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--current_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="artifacts/psi_report.json")
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    reference = pd.read_parquet(args.reference_path)
    current = pd.read_parquet(args.current_path)

    monitor = EDMDDriftMonitor(n_bins=args.n_bins)
    report = monitor.monitor(reference, current)

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"PSI report saved to {args.output_path}")


if __name__ == "__main__":
    main()
