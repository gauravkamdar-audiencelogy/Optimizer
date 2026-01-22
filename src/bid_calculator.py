"""
Bid Calculator V4: Hybrid optimal bidding with bid landscape model.

V4 Changes:
- Add BidLandscapeModel that includes bid_amount as a feature
- Use optimal bid formula: argmax_b [(EV - b) × P(win|b, segment)]
- Fall back to V3 empirical formula when landscape model unavailable/invalid

V3 Formula (fallback):
    bid = EV_cpm × (1 - margin) × win_rate_adjustment

V4 Formula (when landscape model available):
    optimal_bid = argmax_b [(EV_cpm - b) × P(win | b, segment)]

Literature:
- "Bid Shading by Win-Rate Estimation and Surplus Maximization" (arXiv:2009.09259)
- "Optimal Real-Time Bidding for Display Advertising" (Zhang et al., KDD 2014)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .config import OptimizerConfig
from .models.ctr_model import CTRModel
from .models.empirical_win_rate_model import EmpiricalWinRateModel
from .models.bid_landscape_model import BidLandscapeModel


@dataclass
class BidResult:
    """Result of bid calculation for a segment."""
    segment_key: Dict[str, any]
    ctr: float
    win_rate: float
    expected_cpc: float
    expected_value_per_impression: float
    expected_value_cpm: float
    win_rate_adjustment: float  # V3 adjustment (1.0 for V4)
    raw_bid: float
    final_bid: float
    observation_count: int
    is_profitable: bool
    bid_method: str = 'empirical'  # V4: 'landscape' or 'empirical'
    expected_surplus: Optional[float] = None  # V4: For landscape method
    exclusion_reason: Optional[str] = None


class BidCalculator:
    """
    V4: Hybrid bidding with bid landscape model.

    Uses BidLandscapeModel for segments when:
    - Model is available and valid (positive bid coefficient)
    - Segment has sufficient observations

    Falls back to V3 empirical formula otherwise.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        ctr_model: CTRModel,
        empirical_win_rate_model: EmpiricalWinRateModel,
        bid_landscape_model: Optional[BidLandscapeModel] = None,
        use_landscape: bool = True
    ):
        self.config = config
        self.ctr_model = ctr_model
        self.empirical_model = empirical_win_rate_model
        self.landscape_model = bid_landscape_model
        self.avg_cpc: float = 0.0

        # Determine if landscape model should be used
        self.use_landscape = (
            use_landscape and
            bid_landscape_model is not None and
            bid_landscape_model.is_valid()
        )

        # Track method usage for reporting
        self.landscape_count = 0
        self.empirical_count = 0

        if self.use_landscape:
            print(f"    V4: BidLandscapeModel enabled (coefficient: {bid_landscape_model.bid_coefficient:.4f})")
        else:
            if bid_landscape_model is None:
                print(f"    V3: Using empirical model only (no landscape model)")
            elif not bid_landscape_model.is_valid():
                print(f"    V3: Landscape model disabled (invalid coefficient: {bid_landscape_model.bid_coefficient:.4f})")

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

        V4: Uses landscape model if available and valid, otherwise V3 formula.
        """
        # Get CTR prediction
        ctr = self.ctr_model.get_ctr_for_segment(segment_key)

        # Expected value calculation
        expected_value_per_impression = ctr * self.avg_cpc
        expected_value_cpm = expected_value_per_impression * 1000

        # Decide which method to use for this segment
        use_landscape_for_segment = (
            self.use_landscape and
            observation_count >= self.config.technical.min_observations
        )

        if use_landscape_for_segment:
            # V4: Optimal bid via landscape model
            result = self._calculate_landscape_bid(
                segment_key, observation_count, ctr,
                expected_value_per_impression, expected_value_cpm
            )
        else:
            # V3: Empirical formula fallback
            result = self._calculate_empirical_bid(
                segment_key, observation_count, ctr,
                expected_value_per_impression, expected_value_cpm
            )

        return result

    def _calculate_landscape_bid(
        self,
        segment_key: Dict[str, any],
        observation_count: int,
        ctr: float,
        expected_value_per_impression: float,
        expected_value_cpm: float
    ) -> BidResult:
        """
        V4: Calculate optimal bid using bid landscape model.

        Formula: optimal_bid = argmax_b [(EV_cpm - b) × P(win | b, segment)]
        """
        self.landscape_count += 1

        # Find optimal bid using golden section search
        optimal_bid, expected_surplus = self.landscape_model.find_optimal_bid(
            expected_value_cpm=expected_value_cpm,
            segment_values=segment_key
        )

        # Get win rate at optimal bid
        win_rate = self.landscape_model.predict_win_rate(optimal_bid, segment_key)

        raw_bid = optimal_bid

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
            win_rate_adjustment=1.0,  # Not used in V4
            raw_bid=round(raw_bid, 4),
            final_bid=final_bid,
            observation_count=observation_count,
            is_profitable=is_profitable,
            bid_method='landscape',
            expected_surplus=expected_surplus,
            exclusion_reason=exclusion_reason
        )

    def _calculate_empirical_bid(
        self,
        segment_key: Dict[str, any],
        observation_count: int,
        ctr: float,
        expected_value_per_impression: float,
        expected_value_cpm: float
    ) -> BidResult:
        """
        V3: Calculate bid using empirical win rate formula.

        Formula: bid = EV_cpm × (1 - margin) × win_rate_adjustment
        """
        self.empirical_count += 1

        # Get empirical win rate
        win_rate = self.empirical_model.get_win_rate_for_segment(segment_key)

        # Calculate win rate adjustment
        target_win_rate = self.config.business.target_win_rate
        sensitivity = self.config.business.win_rate_sensitivity

        # If win_rate=0.70, target=0.50: adjustment = 1 + (0.50-0.70)*0.5 = 0.90
        # If win_rate=0.30, target=0.50: adjustment = 1 + (0.50-0.30)*0.5 = 1.10
        win_rate_adjustment = 1.0 + (target_win_rate - win_rate) * sensitivity

        # Clip to safety bounds
        win_rate_adjustment = np.clip(
            win_rate_adjustment,
            self.config.technical.min_win_rate_adjustment,
            self.config.technical.max_win_rate_adjustment
        )

        # Base shading factor
        shading_factor = 1 - self.config.business.target_margin

        # Final formula
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
            bid_method='empirical',
            expected_surplus=None,
            exclusion_reason=exclusion_reason
        )

    def get_method_stats(self) -> Dict:
        """Return statistics on which method was used."""
        total = self.landscape_count + self.empirical_count
        return {
            'landscape_count': self.landscape_count,
            'empirical_count': self.empirical_count,
            'total': total,
            'landscape_pct': round(100 * self.landscape_count / total, 1) if total > 0 else 0,
            'empirical_pct': round(100 * self.empirical_count / total, 1) if total > 0 else 0
        }
