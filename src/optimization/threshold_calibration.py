"""
EDMD Framework — Algorithm 1: Risk-Constrained Threshold Calibration
=====================================================================
Selects the optimal PD threshold τ* that simultaneously:
  1. Maximises portfolio profit Σ Π_i
  2. Satisfies portfolio EL constraint: EL(τ) ≤ EL_target
  3. Satisfies fairness constraint:    EOD(τ) ≤ ε

Complexity: O(N log N) from initial sort; O(N) for cumulative EL scan.

Usage:
    python src/optimization/threshold_calibration.py \
        --scores_path artifacts/val_scores.parquet \
        --el_target 16000000 \
        --fairness_tolerance 0.05 \
        --lgd 0.70 \
        --net_margin 0.025 \
        --mean_cli 2000 \
        --mean_utilization 0.35
"""

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Result of Algorithm 1 threshold calibration."""
    tau_star: float
    approved_count: int
    approval_rate: float
    expected_loss: float
    expected_profit: float
    revenue_lift: float
    eod: float
    dpd: float
    dir_ratio: float
    el_constraint_met: bool
    fairness_constraint_met: bool
    all_constraints_met: bool

    def to_dict(self) -> dict:
        return {
            "tau_star": round(self.tau_star, 6),
            "approved_count": self.approved_count,
            "approval_rate": round(self.approval_rate, 4),
            "expected_loss": round(self.expected_loss, 2),
            "expected_profit": round(self.expected_profit, 2),
            "revenue_lift": round(self.revenue_lift, 4),
            "eod": round(self.eod, 4),
            "dpd": round(self.dpd, 4),
            "dir_ratio": round(self.dir_ratio, 4),
            "el_constraint_met": self.el_constraint_met,
            "fairness_constraint_met": self.fairness_constraint_met,
            "all_constraints_met": self.all_constraints_met,
        }


class RiskConstrainedThresholdCalibration:
    """
    Algorithm 1 from the EDMD paper (Section 3.4).

    Formal specification:
      Input:  {PD_i, LGD_i, EAD_i, Δ_i} for i=1..N, EL_target, ε
      Output: Optimal threshold τ*

      1. Sort accounts ascending by PD_i
      2. For each candidate τ in {sorted PD values}:
         a. S(τ) ← {i : PD_i < τ}
         b. EL(τ) ← Σ_{i∈S(τ)} PD_i · LGD_i · (EAD_i + Δ_i)
         c. EOD(τ) ← |P(Approve|Y=1,A=0) − P(Approve|Y=1,A=1)|
         d. If EL(τ) ≤ EL_target AND EOD(τ) ≤ ε:
               Profit(τ) ← Σ_{i∈S(τ)} Π_i
      3. Return τ* ← argmax Profit(τ)
    """

    def __init__(
        self,
        el_target: float = 16_000_000,
        fairness_tolerance: float = 0.05,
        lgd: float = 0.70,
        net_margin: float = 0.025,
        mean_cli_amount: float = 2_000,
        mean_utilization: float = 0.35,
    ):
        self.el_target = el_target
        self.fairness_tolerance = fairness_tolerance
        self.lgd = lgd
        self.net_margin = net_margin
        self.mean_cli_amount = mean_cli_amount
        self.mean_utilization = mean_utilization

    def _compute_incremental_profit(
        self,
        pd: np.ndarray,
        ead: np.ndarray,
        cli_amount: np.ndarray,
        utilization: np.ndarray,
    ) -> np.ndarray:
        """
        Π_i = γ · Δ_i · U_i − PD_i · LGD · (EAD_i + Δ_i)
        """
        revenue = self.net_margin * cli_amount * utilization
        loss = pd * self.lgd * (ead + cli_amount)
        return revenue - loss

    def _compute_eod(
        self,
        approved_mask: np.ndarray,
        labels: np.ndarray,
        protected: np.ndarray,
    ) -> float:
        """
        EOD = |P(Approve|Y=1, A=0) − P(Approve|Y=1, A=1)|
        """
        group_0_creditworthy = (labels == 0) & (protected == 0)
        group_1_creditworthy = (labels == 0) & (protected == 1)

        tpr_0 = approved_mask[group_0_creditworthy].mean() if group_0_creditworthy.sum() > 0 else 0.0
        tpr_1 = approved_mask[group_1_creditworthy].mean() if group_1_creditworthy.sum() > 0 else 0.0

        return abs(tpr_0 - tpr_1)

    def _compute_dpd(
        self,
        approved_mask: np.ndarray,
        protected: np.ndarray,
    ) -> float:
        """DPD = |P(Approve|A=0) − P(Approve|A=1)|"""
        apr_0 = approved_mask[protected == 0].mean() if (protected == 0).sum() > 0 else 0.0
        apr_1 = approved_mask[protected == 1].mean() if (protected == 1).sum() > 0 else 0.0
        return abs(apr_0 - apr_1)

    def _compute_dir(
        self,
        approved_mask: np.ndarray,
        protected: np.ndarray,
    ) -> float:
        """DIR = P(Approve|A=0) / P(Approve|A=1)"""
        apr_0 = approved_mask[protected == 0].mean() if (protected == 0).sum() > 0 else 0.0
        apr_1 = approved_mask[protected == 1].mean() if (protected == 1).sum() > 0 else 1.0
        return apr_0 / max(apr_1, 1e-6)

    def calibrate(
        self,
        pd_scores: np.ndarray,
        ead: np.ndarray,
        labels: np.ndarray,
        protected: np.ndarray,
        cli_amount: Optional[np.ndarray] = None,
        utilization: Optional[np.ndarray] = None,
    ) -> ThresholdResult:
        """
        Execute Algorithm 1: Risk-Constrained Threshold Calibration.

        Parameters
        ----------
        pd_scores : np.ndarray
            Calibrated probability of default estimates.
        ead : np.ndarray
            Exposure at default for each account.
        labels : np.ndarray
            True default labels (0=non-default, 1=default).
        protected : np.ndarray
            Protected attribute proxy (0/1).
        cli_amount : np.ndarray, optional
            Proposed CLI amount per account. Defaults to mean_cli_amount.
        utilization : np.ndarray, optional
            Expected utilization fraction. Defaults to mean_utilization.

        Returns
        -------
        ThresholdResult
            Optimal threshold and associated metrics.
        """
        N = len(pd_scores)

        if cli_amount is None:
            cli_amount = np.full(N, self.mean_cli_amount)
        if utilization is None:
            utilization = np.full(N, self.mean_utilization)

        # Precompute per-account profit and loss contribution
        profit = self._compute_incremental_profit(pd_scores, ead, cli_amount, utilization)
        loss_contrib = pd_scores * self.lgd * (ead + cli_amount)

        # Step 1: Sort ascending by PD
        sort_idx = np.argsort(pd_scores)
        sorted_pd = pd_scores[sort_idx]
        sorted_profit = profit[sort_idx]
        sorted_loss = loss_contrib[sort_idx]
        sorted_labels = labels[sort_idx]
        sorted_protected = protected[sort_idx]

        # Step 2: Cumulative scan — O(N)
        cum_profit = np.cumsum(sorted_profit)
        cum_loss = np.cumsum(sorted_loss)

        best_profit = -np.inf
        best_tau = 0.0
        best_k = 0

        logger.info(f"Scanning {N:,} candidate thresholds...")
        logger.info(f"EL target: ${self.el_target:,.0f} | Fairness tolerance: {self.fairness_tolerance}")

        # Use unique PD values as candidate thresholds
        unique_pds = np.unique(sorted_pd)

        for tau in unique_pds:
            k = np.searchsorted(sorted_pd, tau, side="right")
            if k == 0:
                continue

            el = float(cum_loss[k - 1])

            # EL constraint check — early exit if violated
            if el > self.el_target:
                continue

            # Fairness constraint check
            approved_mask = np.zeros(N, dtype=bool)
            approved_mask[sort_idx[:k]] = True
            eod = self._compute_eod(approved_mask, labels, protected)

            if eod > self.fairness_tolerance:
                continue

            # Both constraints satisfied — check profit
            p = float(cum_profit[k - 1])
            if p > best_profit:
                best_profit = p
                best_tau = float(tau)
                best_k = k

        # Compute final metrics at τ*
        if best_k == 0:
            logger.warning("No feasible threshold found — returning τ=0")
            best_tau = 0.0
            best_k = 0

        final_approved = np.zeros(N, dtype=bool)
        final_approved[sort_idx[:best_k]] = True

        el_final = float(cum_loss[best_k - 1]) if best_k > 0 else 0.0
        profit_final = float(cum_profit[best_k - 1]) if best_k > 0 else 0.0
        revenue = float((self.net_margin * cli_amount[sort_idx[:best_k]] * utilization[sort_idx[:best_k]]).sum()) if best_k > 0 else 0.0
        eod_final = self._compute_eod(final_approved, labels, protected)
        dpd_final = self._compute_dpd(final_approved, protected)
        dir_final = self._compute_dir(final_approved, protected)

        result = ThresholdResult(
            tau_star=best_tau,
            approved_count=int(best_k),
            approval_rate=float(best_k / N),
            expected_loss=el_final,
            expected_profit=profit_final,
            revenue_lift=revenue,
            eod=eod_final,
            dpd=dpd_final,
            dir_ratio=dir_final,
            el_constraint_met=el_final <= self.el_target,
            fairness_constraint_met=eod_final <= self.fairness_tolerance,
            all_constraints_met=(el_final <= self.el_target) and (eod_final <= self.fairness_tolerance),
        )

        logger.info(f"\nAlgorithm 1 Result:")
        logger.info(f"  τ*            = {result.tau_star:.4f}")
        logger.info(f"  Approval rate = {result.approval_rate:.1%}")
        logger.info(f"  Expected loss = ${result.expected_loss:,.0f}")
        logger.info(f"  Net profit    = ${result.expected_profit:,.0f}")
        logger.info(f"  EOD           = {result.eod:.4f} (limit: {self.fairness_tolerance})")
        logger.info(f"  DIR           = {result.dir_ratio:.4f}")
        logger.info(f"  All constraints met: {result.all_constraints_met}")

        return result


def run_from_scores(
    scores_path: str,
    el_target: float,
    fairness_tolerance: float,
    lgd: float,
    net_margin: float,
    mean_cli: float,
    mean_util: float,
    output_path: str = "artifacts/threshold_result.json",
) -> ThresholdResult:
    """Load pre-computed scores and run Algorithm 1."""
    logger.info(f"Loading scores from {scores_path}...")
    df = pd.read_parquet(scores_path)

    required = ["pd_score", "ead", "label", "protected_proxy"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    calibrator = RiskConstrainedThresholdCalibration(
        el_target=el_target,
        fairness_tolerance=fairness_tolerance,
        lgd=lgd,
        net_margin=net_margin,
        mean_cli_amount=mean_cli,
        mean_utilization=mean_util,
    )

    result = calibrator.calibrate(
        pd_scores=df["pd_score"].values,
        ead=df["ead"].values,
        labels=df["label"].values,
        protected=df["protected_proxy"].values,
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Result saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Algorithm 1: Risk-Constrained Threshold Calibration")
    parser.add_argument("--scores_path", type=str, default="artifacts/val_scores.parquet")
    parser.add_argument("--el_target", type=float, default=16_000_000)
    parser.add_argument("--fairness_tolerance", type=float, default=0.05)
    parser.add_argument("--lgd", type=float, default=0.70)
    parser.add_argument("--net_margin", type=float, default=0.025)
    parser.add_argument("--mean_cli", type=float, default=2_000)
    parser.add_argument("--mean_util", type=float, default=0.35)
    parser.add_argument("--output_path", type=str, default="artifacts/threshold_result.json")
    args = parser.parse_args()

    run_from_scores(
        scores_path=args.scores_path,
        el_target=args.el_target,
        fairness_tolerance=args.fairness_tolerance,
        lgd=args.lgd,
        net_margin=args.net_margin,
        mean_cli=args.mean_cli,
        mean_util=args.mean_util,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
