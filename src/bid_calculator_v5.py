"""
Bid Calculator V5: Volume-First with Asymmetric Exploration.

V5 Philosophy:
- Priority is DATA COLLECTION, not margin optimization
- Losing segments → Bid HIGHER to learn ceiling
- Winning segments → Bid LOWER to find floor (don't overpay)
- Target: 65% win rate to learn the full bid landscape
- Include ALL segments, even with zero observations

Why V4 Was Wrong:
- V4 formula `argmax_b [(EV - b) × P(win)]` maximizes margin
- This outputs $2-3 bids which would CUT volume in half
- During learning phase, we need volume, not margin

V5 Formula:
    wr_gap = target_wr - current_wr
    if wr_gap > 0:  # UNDER-WINNING
        adjustment = 1.0 + wr_gap * 1.3  # Bid UP
    else:  # OVER-WINNING
        adjustment = 1.0 + wr_gap * 0.7  # Bid DOWN (less aggressive)

    bid = base_bid * adjustment * npi_multiplier

Literature:
- Multi-Armed Bandits (Thompson Sampling) - exploration/exploitation
- UCB (Upper Confidence Bound) for uncertainty-based exploration
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
class BidResultV5:
    """Result of V5 bid calculation for a segment."""
    segment_key: Dict[str, any]
    ctr: float
    win_rate: float
    expected_cpc: float
    expected_value_per_impression: float
    expected_value_cpm: float
    base_bid: float
    exploration_adjustment: float
    npi_multiplier: float
    raw_bid: float
    final_bid: float
    observation_count: int
    is_profitable: bool
    allows_negative_margin: bool
    exploration_direction: str  # 'up', 'down', or 'neutral'
    bid_method: str  # 'v5_explore_zero', 'v5_explore_low', 'v5_explore_medium', 'v5_empirical'
    exclusion_reason: Optional[str] = None


class VolumeFirstBidCalculator:
    """
    V5: Asymmetric exploration to learn bid landscape.

    - Losing segments → bid UP to learn ceiling
    - Winning segments → bid DOWN to find floor
    - Include ALL segments (no filtering)
    - NPI integration for high-value prescribers
    """

    def __init__(
        self,
        config: OptimizerConfig,
        ctr_model: CTRModel,
        empirical_win_rate_model: EmpiricalWinRateModel,
        npi_model: Optional['NPIValueModel'] = None,
        global_stats: Optional[Dict] = None
    ):
        self.config = config
        self.ctr_model = ctr_model
        self.empirical_model = empirical_win_rate_model
        self.npi_model = npi_model
        self.avg_cpc: float = 0.0

        # Global stats for exploration baseline
        self.global_stats = global_stats or {}
        self.global_median_bid = self.global_stats.get('median_winning_bid', config.technical.default_bid_cpm)

        # Track method usage for reporting
        self.method_counts = {
            'v5_explore_zero': 0,
            'v5_explore_low': 0,
            'v5_explore_medium': 0,
            'v5_empirical': 0
        }
        self.exploration_directions = {'up': 0, 'down': 0, 'neutral': 0}

        print(f"    V5: VolumeFirstBidCalculator initialized")
        print(f"    Target win rate: {config.business.target_win_rate:.0%}")
        print(f"    Exploration mode: {config.business.exploration_mode}")
        print(f"    Global median bid baseline: ${self.global_median_bid:.2f}")
        print(f"    NPI model: {'enabled' if npi_model else 'disabled'}")

    def set_average_cpc(self, df_clicks: pd.DataFrame) -> None:
        """Calculate average CPC from click data."""
        self.avg_cpc = df_clicks['cpc_value'].mean()
        print(f"    Average CPC: ${self.avg_cpc:.2f}")

    def calculate_bids_for_segments(
        self,
        df_segments: pd.DataFrame,
        features: List[str]
    ) -> List[BidResultV5]:
        """Calculate optimal bid for each segment using asymmetric exploration."""
        results = []

        for _, row in tqdm(df_segments.iterrows(), total=len(df_segments), desc="    Calculating V5 bids"):
            segment_key = {f: row[f] for f in features}
            observation_count = row.get('count', 0)

            # Extract NPI if available (from external_userid in original data)
            npi = row.get('external_userid', None)

            result = self._calculate_single_bid(
                segment_key=segment_key,
                observation_count=observation_count,
                npi=npi
            )
            results.append(result)

        return results

    def _calculate_single_bid(
        self,
        segment_key: Dict[str, any],
        observation_count: int,
        npi: Optional[str] = None
    ) -> BidResultV5:
        """
        Calculate bid for a single segment with asymmetric exploration.

        V5: Bid based on win rate gap from target:
        - Under-winning → bid UP (aggressive, multiplier 1.3)
        - Over-winning → bid DOWN (cautious, multiplier 0.7)
        """
        # Get CTR prediction
        ctr = self.ctr_model.get_ctr_for_segment(segment_key)

        # Expected value calculation
        expected_value_per_impression = ctr * self.avg_cpc
        expected_value_cpm = expected_value_per_impression * 1000

        # Get current win rate for segment
        current_wr = self.empirical_model.get_win_rate_for_segment(segment_key)

        # Determine base bid and method based on observation count
        base_bid, bid_method = self._get_base_bid_and_method(
            segment_key, observation_count
        )

        # Calculate asymmetric exploration adjustment
        exploration_adjustment, direction = self._calculate_exploration_adjustment(
            current_wr, observation_count
        )

        # Apply NPI multiplier
        npi_multiplier = 1.0
        if self.npi_model and npi:
            npi_multiplier = self.npi_model.get_value_multiplier(npi)

        # Calculate raw bid
        raw_bid = base_bid * exploration_adjustment * npi_multiplier

        # Check profitability (V5: allow negative margin during learning)
        min_bid = self.config.technical.min_bid_cpm
        is_profitable = expected_value_cpm >= min_bid
        allows_negative_margin = self.config.business.accept_negative_margin

        # Apply guardrails
        final_bid = self._apply_guardrails(
            raw_bid, expected_value_cpm, allows_negative_margin
        )

        # Track method usage
        self.method_counts[bid_method] += 1
        self.exploration_directions[direction] += 1

        exclusion_reason = None
        if not is_profitable and not allows_negative_margin:
            exclusion_reason = f"EV_CPM=${expected_value_cpm:.2f} < floor=${min_bid}"

        return BidResultV5(
            segment_key=segment_key,
            ctr=round(ctr, 8),
            win_rate=round(current_wr, 4),
            expected_cpc=round(self.avg_cpc, 2),
            expected_value_per_impression=round(expected_value_per_impression, 8),
            expected_value_cpm=round(expected_value_cpm, 4),
            base_bid=round(base_bid, 4),
            exploration_adjustment=round(exploration_adjustment, 4),
            npi_multiplier=round(npi_multiplier, 4),
            raw_bid=round(raw_bid, 4),
            final_bid=final_bid,
            observation_count=observation_count,
            is_profitable=is_profitable,
            allows_negative_margin=allows_negative_margin,
            exploration_direction=direction,
            bid_method=bid_method,
            exclusion_reason=exclusion_reason
        )

    def _get_base_bid_and_method(
        self,
        segment_key: Dict[str, any],
        observation_count: int
    ) -> tuple:
        """
        Determine base bid and method based on observation count.

        Tiered approach:
        - 0 obs: Pure exploration (global median * 1.5)
        - 1-9 obs: Exploration with bonus (global median * 1.35)
        - 10-49 obs: Empirical + small exploration bonus
        - 50+ obs: Full empirical
        """
        min_for_empirical = self.config.technical.min_observations_for_empirical
        min_for_landscape = self.config.technical.min_observations_for_landscape

        if observation_count == 0:
            # Pure exploration for unknown segments
            bonus = 1.0 + self.config.technical.exploration_bonus_zero_obs
            base_bid = self.global_median_bid * bonus
            method = 'v5_explore_zero'

        elif observation_count < min_for_empirical:
            # Low observation: use global with exploration bonus
            bonus = 1.0 + self.config.technical.exploration_bonus_low_obs
            base_bid = self.global_median_bid * bonus
            method = 'v5_explore_low'

        elif observation_count < min_for_landscape:
            # Medium observation: use empirical with small bonus
            segment_stats = self.empirical_model.get_segment_stats(segment_key)
            empirical_wr = segment_stats.get('shrunk_win_rate', self.empirical_model.global_win_rate)

            # Use global median scaled by exploration bonus
            bonus = 1.0 + self.config.technical.exploration_bonus_medium_obs
            base_bid = self.global_median_bid * bonus
            method = 'v5_explore_medium'

        else:
            # High observation: use empirical directly
            # Base bid is the global median (we'll adjust via exploration)
            base_bid = self.global_median_bid
            method = 'v5_empirical'

        return base_bid, method

    def _calculate_exploration_adjustment(
        self,
        current_wr: float,
        observation_count: int
    ) -> tuple:
        """
        Calculate asymmetric exploration adjustment.

        LOSING (WR < target): Bid UP to learn ceiling
        WINNING (WR > target): Bid DOWN to find floor

        Asymmetry:
        - Up multiplier: 1.3 (aggressive on losers)
        - Down multiplier: 0.7 (cautious on winners)
        """
        target_wr = self.config.business.target_win_rate
        wr_gap = target_wr - current_wr

        # Determine direction
        if abs(wr_gap) < 0.05:  # Within 5% of target
            direction = 'neutral'
        elif wr_gap > 0:
            direction = 'up'
        else:
            direction = 'down'

        if not self.config.business.exploration_mode:
            # Exploration disabled: use simple linear adjustment
            adjustment = 1.0 + wr_gap * self.config.business.win_rate_sensitivity
        else:
            # V5: Asymmetric exploration
            if wr_gap > 0:
                # UNDER-WINNING: Bid higher (aggressive)
                # wr_gap of 0.35 (WR=30% vs target=65%) → +45% increase
                up_mult = self.config.business.exploration_up_multiplier
                adjustment = 1.0 + wr_gap * up_mult
            else:
                # OVER-WINNING: Bid lower (cautious)
                # wr_gap of -0.15 (WR=80% vs target=65%) → -10% decrease
                down_mult = self.config.business.exploration_down_multiplier
                adjustment = 1.0 + wr_gap * down_mult

        # Scale by uncertainty (more exploration when less data)
        # uncertainty is higher when obs_count is lower
        if observation_count > 0:
            uncertainty = 1.0 / (1.0 + np.log1p(observation_count))
            # Blend: more uncertain segments get more extreme adjustments
            adjustment = 1.0 + (adjustment - 1.0) * (0.5 + 0.5 * uncertainty)

        # Clip to safety bounds
        adjustment = np.clip(
            adjustment,
            self.config.technical.min_win_rate_adjustment,
            self.config.technical.max_win_rate_adjustment
        )

        return adjustment, direction

    def _apply_guardrails(
        self,
        raw_bid: float,
        expected_value_cpm: float,
        allows_negative_margin: bool
    ) -> float:
        """
        Apply bid guardrails.

        V5: Allow negative margins during learning, but cap at max_negative_margin_pct.
        Floor is ALWAYS respected regardless of EV.
        """
        min_bid = self.config.technical.min_bid_cpm
        max_bid = self.config.technical.max_bid_cpm

        # V5: Cap at max overpay (e.g., 1.5x EV for 50% negative margin)
        # But NEVER go below floor
        if allows_negative_margin and expected_value_cpm > 0:
            max_overpay = expected_value_cpm * (1.0 + self.config.business.max_negative_margin_pct)
            # Only apply max_overpay cap if it's above the floor
            if max_overpay >= min_bid:
                max_bid = min(max_bid, max_overpay)
            # If max_overpay < floor, we'll still bid at floor (exploration)

        # Apply floor and ceiling - floor is ALWAYS enforced
        final_bid = np.clip(raw_bid, min_bid, max_bid)

        return round(final_bid, 2)

    def get_method_stats(self) -> Dict:
        """Return statistics on which method was used."""
        total = sum(self.method_counts.values())

        stats = {
            'total': total,
        }

        # Method breakdown
        for method, count in self.method_counts.items():
            stats[f'{method}_count'] = count
            stats[f'{method}_pct'] = round(100 * count / total, 1) if total > 0 else 0

        # Direction breakdown
        for direction, count in self.exploration_directions.items():
            stats[f'exploration_{direction}_count'] = count
            stats[f'exploration_{direction}_pct'] = round(100 * count / total, 1) if total > 0 else 0

        return stats

    def get_exploration_summary(self) -> Dict:
        """Return summary of exploration statistics."""
        total_up = self.exploration_directions['up']
        total_down = self.exploration_directions['down']
        total_neutral = self.exploration_directions['neutral']
        total = total_up + total_down + total_neutral

        return {
            'segments_bid_up': total_up,
            'segments_bid_down': total_down,
            'segments_neutral': total_neutral,
            'pct_bid_up': round(100 * total_up / total, 1) if total > 0 else 0,
            'pct_bid_down': round(100 * total_down / total, 1) if total > 0 else 0,
            'asymmetry_ratio': round(total_up / total_down, 2) if total_down > 0 else float('inf'),
            'target_win_rate': self.config.business.target_win_rate,
            'up_multiplier': self.config.business.exploration_up_multiplier,
            'down_multiplier': self.config.business.exploration_down_multiplier
        }
