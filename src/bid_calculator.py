"""
Bid Calculator V3: Economic bidding with empirical win rate signal.

V3 Changes:
- Reintroduce win rate adjustment using EMPIRICAL segment rates (not model predictions)
- Formula: bid = EV_cpm x (1 - margin) x win_rate_adjustment

The key insight: V2 removed win rate entirely because the MODEL was bad.
But the SIGNAL is still useful - we just need to observe it directly.

Win Rate Adjustment:
- If win rate HIGH (>target) → we're overpaying → bid LOWER
- If win rate LOW (<target) → we're losing auctions → bid HIGHER

This aligns with optimal bid shading literature:
    optimal_bid = argmax_b [(value - b) × P(win|b)]
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .config import OptimizerConfig
from .models.ctr_model import CTRModel
from .models.empirical_win_rate_model import EmpiricalWinRateModel


@dataclass
class BidResult:
    """Result of bid calculation for a segment."""
    segment_key: Dict[str, any]
    ctr: float
    win_rate: float                       # V3: Reintroduced (empirical)
    expected_cpc: float
    expected_value_per_impression: float
    expected_value_cpm: float
    win_rate_adjustment: float            # V3: The multiplier applied
    raw_bid: float
    final_bid: float
    observation_count: int
    is_profitable: bool
    exclusion_reason: Optional[str] = None


class BidCalculator:
    """
    V3: Economic bidding with empirical win rate signal.

    Formula: bid = EV_cpm × (1 - margin) × win_rate_adjustment

    Win rate adjustment uses OBSERVED empirical rates (not model predictions).
    This gives us market signal without model calibration issues.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        ctr_model: CTRModel,
        win_rate_model: EmpiricalWinRateModel  # V3: Use empirical model
    ):
        self.config = config
        self.ctr_model = ctr_model
        self.win_rate_model = win_rate_model
        self.avg_cpc: float = 0.0

    def set_average_cpc(self, df_clicks: pd.DataFrame) -> None:
        """Calculate average CPC from click data."""
        self.avg_cpc = df_clicks['cpc_value'].mean()
        print(f"    Average CPC: ${self.avg_cpc:.2f}")

    def calculate_bids_for_segments(
        self,
        df_segments: pd.DataFrame,
        features: List[str]
    ) -> List[BidResult]:
        """Calculate optimal bid for each segment."""
        results = []

        for _, row in tqdm(df_segments.iterrows(), total=len(df_segments), desc="    Calculating bids"):
            segment_key = {f: row[f] for f in features}
            observation_count = row.get('count', 0)

            result = self._calculate_single_bid(
                segment_key=segment_key,
                observation_count=observation_count
            )
            results.append(result)

        return results

    def _calculate_single_bid(
        self,
        segment_key: Dict[str, any],
        observation_count: int
    ) -> BidResult:
        """
        Calculate optimal bid for a single segment.

        V3 FORMULA:
            raw_bid = EV_cpm × (1 - margin) × win_rate_adjustment

        Where win_rate_adjustment = 1 + (target_wr - empirical_wr) × sensitivity
            - If empirical > target: adjustment < 1 (bid lower)
            - If empirical < target: adjustment > 1 (bid higher)
        """
        # Get predictions
        ctr = self.ctr_model.get_ctr_for_segment(segment_key)
        win_rate = self.win_rate_model.get_win_rate_for_segment(segment_key)

        # Expected value calculation
        expected_value_per_impression = ctr * self.avg_cpc
        expected_value_cpm = expected_value_per_impression * 1000

        # V3: Win rate adjustment (using empirical rates)
        target_win_rate = self.config.business.target_win_rate
        sensitivity = self.config.business.win_rate_sensitivity

        # Calculate adjustment
        # If win_rate=0.70, target=0.50: adjustment = 1 + (0.50-0.70)*0.5 = 0.90
        # If win_rate=0.30, target=0.50: adjustment = 1 + (0.50-0.30)*0.5 = 1.10
        win_rate_adjustment = 1.0 + (target_win_rate - win_rate) * sensitivity

        # Clip to safety bounds [0.8, 1.2]
        win_rate_adjustment = np.clip(
            win_rate_adjustment,
            self.config.technical.min_win_rate_adjustment,
            self.config.technical.max_win_rate_adjustment
        )

        # Base shading factor
        shading_factor = 1 - self.config.business.target_margin

        # Final formula: EV × (1 - margin) × win_rate_adjustment
        raw_bid = expected_value_cpm * shading_factor * win_rate_adjustment

        # Check economic profitability
        min_bid = self.config.technical.min_bid_cpm
        is_profitable = expected_value_cpm >= min_bid

        exclusion_reason = None
        if not is_profitable:
            exclusion_reason = f"EV_CPM=${expected_value_cpm:.2f} < floor=${min_bid}"

        # Apply floor and ceiling
        final_bid = np.clip(
            raw_bid,
            self.config.technical.min_bid_cpm,
            self.config.technical.max_bid_cpm
        )
        final_bid = round(final_bid, 2)

        return BidResult(
            segment_key=segment_key,
            ctr=round(ctr, 8),
            win_rate=round(win_rate, 4),
            expected_cpc=round(self.avg_cpc, 2),
            expected_value_per_impression=round(expected_value_per_impression, 8),
            expected_value_cpm=round(expected_value_cpm, 4),
            win_rate_adjustment=round(win_rate_adjustment, 4),
            raw_bid=round(raw_bid, 4),
            final_bid=final_bid,
            observation_count=observation_count,
            is_profitable=is_profitable,
            exclusion_reason=exclusion_reason
        )
