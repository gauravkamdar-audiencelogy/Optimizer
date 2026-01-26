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
from .models.bid_landscape_model import BidLandscapeModel


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
    ev_used_for_bidding: bool = False  # V7: True only in margin_optimize mode


class VolumeFirstBidCalculator:
    """
    V5/V6: Asymmetric exploration to learn bid landscape.

    V6 Enhancements:
    - Config-driven strategy selection (volume_first, margin_optimize, adaptive)
    - Optional bid landscape model for volume targeting
    - Per-segment adaptive strategy when configured

    Core behavior:
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
        bid_landscape_model: Optional[BidLandscapeModel] = None,
        global_stats: Optional[Dict] = None
    ):
        self.config = config
        self.ctr_model = ctr_model
        self.empirical_model = empirical_win_rate_model
        self.npi_model = npi_model
        self.bid_landscape_model = bid_landscape_model
        self.avg_cpc: float = 0.0

        # Check if bid landscape can be used for volume targeting
        self.use_landscape_for_volume = (
            self.config.bidding.use_bid_landscape_for_volume and
            bid_landscape_model is not None and
            bid_landscape_model.is_valid()
        )

        # Global stats for exploration baseline
        self.global_stats = global_stats or {}
        self.global_median_bid = self.global_stats.get('median_winning_bid', config.technical.default_bid_cpm)

        # Track method usage for reporting
        self.method_counts = {
            'v5_explore_zero': 0,
            'v5_explore_low': 0,
            'v5_explore_medium': 0,
            'v5_empirical': 0,
            'v6_landscape_volume': 0,
            'v6_landscape_margin': 0
        }
        self.exploration_directions = {'up': 0, 'down': 0, 'neutral': 0}
        self.strategy_counts = {'volume_first': 0, 'margin_optimize': 0}

        # Log configuration
        strategy = config.bidding.strategy
        print(f"    V6: VolumeFirstBidCalculator initialized")
        print(f"    Strategy: {strategy}")
        print(f"    Target win rate: {config.business.target_win_rate:.0%}")
        print(f"    Exploration mode: {config.business.exploration_mode}")
        print(f"    Global median bid baseline: ${self.global_median_bid:.2f}")
        print(f"    Bid landscape for volume: {'enabled' if self.use_landscape_for_volume else 'disabled'}")
        print(f"    NPI model: {'enabled' if npi_model else 'disabled'}")

        if strategy == 'adaptive':
            print(f"    Adaptive thresholds: WR>={config.bidding.min_win_rate_for_margin:.0%}, obs>={config.bidding.min_observations_for_margin}")

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
        Calculate bid for a single segment.

        V6: Strategy-driven bid calculation:
        - volume_first: Use asymmetric exploration or bid landscape to achieve target WR
        - margin_optimize: Use bid landscape to maximize surplus
        - adaptive: Per-segment strategy based on maturity

        Falls back to heuristic (V5) if bid landscape unavailable.
        """
        # Get CTR prediction
        ctr = self.ctr_model.get_ctr_for_segment(segment_key)

        # V7: Expected value calculation - FOR REPORTING/METRICS ONLY
        # In volume_first mode, actual bid is driven by win rate gap, NOT by EV.
        # EV uses avg_cpc which is demand-side confounded (depends on campaign payouts).
        # We keep EV for profitability monitoring, but it doesn't affect volume_first bids.
        expected_value_per_impression = ctr * self.avg_cpc
        expected_value_cpm = expected_value_per_impression * 1000

        # Get current win rate for segment
        current_wr = self.empirical_model.get_win_rate_for_segment(segment_key)

        # Determine which strategy to use for this segment
        strategy = self._select_strategy_for_segment(current_wr, observation_count)
        self.strategy_counts[strategy] += 1

        # V7: Track whether EV actually affects bidding (only in margin_optimize)
        ev_used_for_bidding = False

        # Calculate bid based on selected strategy
        if strategy == 'margin_optimize' and self.bid_landscape_model and self.bid_landscape_model.is_valid():
            # Use bid landscape for margin optimization - EV IS used here
            base_bid, bid_method = self._calculate_margin_optimized_bid(
                segment_key, expected_value_cpm, observation_count
            )
            exploration_adjustment = 1.0
            direction = 'neutral'
            ev_used_for_bidding = True  # V7: EV affects bid in margin mode

        elif strategy == 'volume_first' and self.use_landscape_for_volume and observation_count >= 50:
            # Use bid landscape for volume targeting (find bid for target WR)
            # V7: EV NOT used - we find bid for target WR, not for margin
            base_bid, bid_method, achieved_wr = self._calculate_volume_landscape_bid(
                segment_key, observation_count
            )
            exploration_adjustment = 1.0
            direction = 'up' if achieved_wr < self.config.business.target_win_rate else 'neutral'

        else:
            # V5 heuristic: Asymmetric exploration
            # V7: EV NOT used - bid driven by win rate gap
            base_bid, bid_method = self._get_base_bid_and_method(
                segment_key, observation_count
            )
            exploration_adjustment, direction = self._calculate_exploration_adjustment(
                current_wr, observation_count
            )

        # Apply NPI multiplier
        npi_multiplier = 1.0
        if self.npi_model and npi:
            npi_multiplier = self.npi_model.get_value_multiplier(npi)

        # Calculate raw bid
        raw_bid = base_bid * exploration_adjustment * npi_multiplier

        # Check profitability (allow negative margin during learning in volume mode)
        min_bid = self.config.technical.min_bid_cpm
        is_profitable = expected_value_cpm >= min_bid
        allows_negative_margin = (
            self.config.business.accept_negative_margin and
            strategy == 'volume_first'
        )

        # Apply guardrails
        final_bid = self._apply_guardrails(
            raw_bid, expected_value_cpm, allows_negative_margin
        )

        # Track method usage
        if bid_method in self.method_counts:
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
            exclusion_reason=exclusion_reason,
            ev_used_for_bidding=ev_used_for_bidding
        )

    def _select_strategy_for_segment(
        self,
        current_wr: float,
        observation_count: int
    ) -> str:
        """
        Select bidding strategy for this segment.

        V6: Config-driven with per-segment adaptation.
        """
        strategy = self.config.bidding.strategy

        if strategy in ['volume_first', 'margin_optimize']:
            return strategy

        elif strategy == 'adaptive':
            # Per-segment automatic switching based on maturity
            thresholds = self.config.bidding
            if (current_wr >= thresholds.min_win_rate_for_margin and
                observation_count >= thresholds.min_observations_for_margin):
                return 'margin_optimize'  # Mature segment
            else:
                return 'volume_first'  # Still learning

        else:
            return 'volume_first'  # Default fallback

    def _calculate_volume_landscape_bid(
        self,
        segment_key: Dict[str, any],
        observation_count: int
    ) -> tuple:
        """
        Use bid landscape to find bid that achieves target win rate.

        V6: This is the "proper" volume-first approach - use P(win|bid)
        to determine exact bid needed for target WR.
        """
        try:
            target_wr = self.config.business.target_win_rate

            # Use bid landscape with uncertainty buffer
            buffered_bid, base_bid, buffer = self.bid_landscape_model.find_bid_for_win_rate_with_buffer(
                target_win_rate=target_wr,
                segment_values=segment_key,
                observation_count=observation_count
            )

            achieved_wr = self.bid_landscape_model.predict_win_rate(buffered_bid, segment_key)
            return buffered_bid, 'v6_landscape_volume', achieved_wr

        except Exception as e:
            # Fall back to heuristic if landscape fails
            base_bid, method = self._get_base_bid_and_method(segment_key, observation_count)
            return base_bid, method, 0.0

    def _calculate_margin_optimized_bid(
        self,
        segment_key: Dict[str, any],
        expected_value_cpm: float,
        observation_count: int
    ) -> tuple:
        """
        Use bid landscape for margin optimization.

        Classic surplus maximization: argmax_b [(EV - b) × P(win|b)]
        """
        try:
            optimal_bid, expected_surplus = self.bid_landscape_model.find_optimal_bid(
                expected_value_cpm=expected_value_cpm,
                segment_values=segment_key
            )
            return optimal_bid, 'v6_landscape_margin'

        except Exception as e:
            # Fall back to heuristic
            base_bid, method = self._get_base_bid_and_method(segment_key, observation_count)
            return base_bid, method

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

        V6 FIX: Removed EV cap in volume_first mode.
        The EV cap (max_negative_margin_pct) caused a cliff effect where
        segments with EV just below threshold got full bids while those
        just above got heavily capped. In learning phase, just use floor/ceiling.

        The EV cap should only apply in margin_optimize mode.
        """
        min_bid = self.config.technical.min_bid_cpm
        max_bid = self.config.technical.max_bid_cpm

        # V6: Only apply EV cap in margin_optimize mode
        # In volume_first mode, we want to explore - just use floor/ceiling
        strategy = self.config.bidding.strategy
        if strategy == 'margin_optimize' and expected_value_cpm > 0:
            # Only cap in margin mode to ensure profitability
            max_overpay = expected_value_cpm * (1.0 + self.config.business.max_negative_margin_pct)
            if max_overpay >= min_bid:
                max_bid = min(max_bid, max_overpay)
        # In volume_first or adaptive (during learning), no EV cap

        # Apply floor and ceiling
        final_bid = np.clip(raw_bid, min_bid, max_bid)

        return round(final_bid, 2)

    def get_method_stats(self) -> Dict:
        """Return statistics on which method was used."""
        total = sum(self.method_counts.values())

        stats = {
            'total': total,
            'config_strategy': self.config.bidding.strategy,
        }

        # Method breakdown
        for method, count in self.method_counts.items():
            if count > 0:  # Only include used methods
                stats[f'{method}_count'] = count
                stats[f'{method}_pct'] = round(100 * count / total, 1) if total > 0 else 0

        # Direction breakdown
        for direction, count in self.exploration_directions.items():
            stats[f'exploration_{direction}_count'] = count
            stats[f'exploration_{direction}_pct'] = round(100 * count / total, 1) if total > 0 else 0

        # V6: Strategy breakdown (for adaptive mode)
        total_strategy = sum(self.strategy_counts.values())
        for strategy, count in self.strategy_counts.items():
            if count > 0:
                stats[f'strategy_{strategy}_count'] = count
                stats[f'strategy_{strategy}_pct'] = round(100 * count / total_strategy, 1) if total_strategy > 0 else 0

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
