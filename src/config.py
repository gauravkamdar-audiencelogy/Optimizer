"""
V5 Configuration: Volume-first optimizer with asymmetric exploration.

V5 Changes:
- New exploration_mode for asymmetric bid adjustment
- Accept negative margins during data collection
- NPI value integration
- Include ALL segments (remove min_observations filter effect)
- Wider adjustment bounds [0.6, 1.8]

V3/V4 Features Retained:
- Empirical win rate model
- Bayesian shrinkage
- Feature auto-exclusion

Philosophy: During data collection phase, prioritize VOLUME over MARGIN.
Learn the bid landscape through asymmetric exploration.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class BusinessControls:
    """
    Controls exposed to business/product team.

    V5: Volume-first with asymmetric exploration.
    """
    target_margin: float = 0.30           # Bid shading factor (bid at 70% of EV)
    target_win_rate: float = 0.65         # V5: Higher target for volume (was 0.50)
    win_rate_sensitivity: float = 0.5     # How aggressively to adjust (0.3-0.7)

    # V5: Exploration mode
    exploration_mode: bool = True         # Enable asymmetric exploration
    exploration_up_multiplier: float = 1.3   # Multiplier for under-winning segments
    exploration_down_multiplier: float = 0.7 # Multiplier for over-winning segments

    # V5: Accept negative margin during learning
    accept_negative_margin: bool = True   # Allow bids > EV during data collection
    max_negative_margin_pct: float = 0.50 # Max 50% overpay (bid up to 1.5x EV)

    # V5: NPI integration
    use_npi_value: bool = True            # Enable NPI-based bid multipliers
    npi_data_path: Optional[str] = None   # Path to NPI value CSV


@dataclass
class TechnicalControls:
    """
    Controls for technical team only.

    V5: Volume-first with exploration.
    """
    # Bid bounds
    min_bid_cpm: float = 2.00              # Floor
    max_bid_cpm: float = 30.00             # Ceiling
    default_bid_cpm: float = 7.50          # V5: Raised for exploration (was 5.00)

    # Floor price handling (Phase 1: disabled)
    floor_available: bool = False          # If True, parse floor from bid_amount column

    # V5: Tiered observation thresholds (instead of binary filter)
    min_observations: int = 1              # V5: Include ALL segments (was 100!)
    min_observations_for_empirical: int = 10   # V5: Use empirical WR with 10+ obs
    min_observations_for_landscape: int = 50   # V5: Use landscape model with 50+ obs

    # Feature selection
    max_features: int = 3                  # Maximum features in memcache key

    # Model shrinkage
    ctr_shrinkage_k: int = 30              # CTR Bayesian shrinkage strength
    win_rate_shrinkage_k: int = 30         # Win rate shrinkage strength

    # Soft exclusion thresholds (algorithm uses these to auto-exclude)
    min_signal_score: float = 50.0         # Below this = auto-exclude feature
    min_coverage_at_threshold: float = 0.5 # Must cover 50% of data
    max_null_pct: float = 30.0             # Above this = auto-exclude feature

    # Effective cardinality filter (prevents dominated features)
    min_effective_cardinality: int = 2     # Must have >=2 values with significant share
    effective_cardinality_min_share: float = 0.05  # 5% threshold for "significant"

    # V5: WIDER adjustment bounds for exploration
    min_win_rate_adjustment: float = 0.6   # V5: Lowered (was 0.8)
    max_win_rate_adjustment: float = 1.8   # V5: Raised (was 1.2)

    # V5: Exploration bonuses for sparse segments
    exploration_bonus_zero_obs: float = 0.50   # +50% for unknown segments
    exploration_bonus_low_obs: float = 0.35    # +35% for 1-9 observations
    exploration_bonus_medium_obs: float = 0.15 # +15% for 10-49 observations


@dataclass
class FeatureConfig:
    """
    Feature selection configuration.

    V3: Split into hard exclusions (never use) vs candidates (algorithm decides).
    Removed data-driven exclusions from config - algorithm will auto-detect.
    """
    candidate_features: List[str] = field(default_factory=list)
    anchor_features: List[str] = field(default_factory=list)
    hard_exclude_features: List[str] = field(default_factory=list)  # V3: Renamed

    def __post_init__(self):
        if not self.candidate_features:
            # V3: Include ALL potentially useful features
            # Let algorithm decide which have low signal
            self.candidate_features = [
                'internal_adspace_id',
                'geo_region_name',
                'geo_country_code2',  # V3: Let algorithm decide (was hardcoded excluded)
                'os_code',
                'pageurl_truncated',  # Full URL truncated at ? (matches bidder format)
                'domain',             # For future multi-domain SSPs
                'browser_code',
                'hour_of_day',        # V3: Let algorithm decide (was hardcoded excluded)
                'day_of_week',         # V3: Let algorithm decide (was hardcoded excluded)
                'media_type',
                'domain',
                'make_id',
                'model_id',
                'carrier_code',
                
            ]
        if not self.anchor_features:
            self.anchor_features = ['internal_adspace_id']
        if not self.hard_exclude_features:
            # V5: Only truly technical/structural exclusions
            # NOT data-driven decisions
            # NOTE: external_userid removed - now used for NPI lookup
            self.hard_exclude_features = [
                'geo_postal_code',    # Too sparse (7,027 unique)
                'geo_city_name',      # Too sparse (3,333 unique)
                'log_txnid',          # Technical: join key
                'internal_txn_id',    # Technical: join key
                'log_dt',             # Technical: timestamp
            ]


@dataclass
class OptimizerConfig:
    """Master configuration."""
    business: BusinessControls = field(default_factory=BusinessControls)
    technical: TechnicalControls = field(default_factory=TechnicalControls)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'OptimizerConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        business_data = data.get('business', {})
        technical_data = data.get('technical', {})
        features_data = data.get('features', {})

        return cls(
            business=BusinessControls(**business_data) if business_data else BusinessControls(),
            technical=TechnicalControls(**technical_data) if technical_data else TechnicalControls(),
            features=FeatureConfig(**features_data) if features_data else FeatureConfig()
        )

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        data = {
            'business': {
                'target_margin': self.business.target_margin,
                'target_win_rate': self.business.target_win_rate,
                'win_rate_sensitivity': self.business.win_rate_sensitivity
            },
            'technical': {
                'min_bid_cpm': self.technical.min_bid_cpm,
                'max_bid_cpm': self.technical.max_bid_cpm,
                'default_bid_cpm': self.technical.default_bid_cpm,
                'min_observations': self.technical.min_observations,
                'max_features': self.technical.max_features,
                'ctr_shrinkage_k': self.technical.ctr_shrinkage_k,
                'win_rate_shrinkage_k': self.technical.win_rate_shrinkage_k,
                'min_signal_score': self.technical.min_signal_score,
                'min_coverage_at_threshold': self.technical.min_coverage_at_threshold,
                'max_null_pct': self.technical.max_null_pct,
                'min_win_rate_adjustment': self.technical.min_win_rate_adjustment,
                'max_win_rate_adjustment': self.technical.max_win_rate_adjustment
            },
            'features': {
                'candidate_features': self.features.candidate_features,
                'anchor_features': self.features.anchor_features,
                'hard_exclude_features': self.features.hard_exclude_features
            }
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
