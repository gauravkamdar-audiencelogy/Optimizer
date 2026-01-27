"""
V8 Configuration: Volume-first optimizer with exploration toggle.

V8 Changes:
- Added aggressive_exploration toggle (default: false = gradual)
- Exploration settings moved into aggressive/gradual presets
- Toggle allows easy experimentation without code changes

V5-V7 Features Retained:
- Asymmetric bid adjustment
- Accept negative margins during data collection
- NPI value integration (click-based tiering)
- Adaptive strategy (auto-switch mature segments to margin_optimize)
- Calibration gate pattern

Philosophy: During data collection phase, prioritize VOLUME over MARGIN.
Control exploration aggressiveness via config toggle.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class BiddingStrategy:
    """
    Bidding strategy configuration.

    V6: Config-driven strategy selection.
    Models are built optimally, config selects WHICH strategy to use.
    """
    # Strategy selection: volume_first, margin_optimize, adaptive
    strategy: str = "volume_first"

    # Adaptive thresholds (for per-segment switching when strategy="adaptive")
    min_win_rate_for_margin: float = 0.55  # Switch to margin when WR >= 55%
    min_observations_for_margin: int = 100  # Need 100+ obs to trust margin optimization

    # Volume-first settings (used when strategy="volume_first" or adaptive segment is learning)
    use_bid_landscape_for_volume: bool = True  # Use bid landscape to find target WR bid

    # Margin optimization settings (used when strategy="margin_optimize" or adaptive segment is mature)
    min_margin_pct: float = 0.20  # Target 20% margin


@dataclass
class CalibrationGate:
    """
    Runtime calibration gate settings.

    Models are built optimally, then gated at runtime based on calibration quality.
    If a model's ECE exceeds threshold, fall back to simpler model.

    This makes the pipeline data-agnostic - no hard-coding "don't use model X".
    """
    max_ece_threshold: float = 0.10  # ECE must be < 10% to use model
    min_observations_for_eval: int = 1000  # Need enough data to evaluate calibration
    log_gate_decisions: bool = True  # Log why model was/wasn't used


@dataclass
class ExplorationPreset:
    """
    V8: Exploration settings preset (aggressive or gradual).

    Controls how aggressively the optimizer explores unexplored bid ranges.
    """
    exploration_bonus_zero_obs: float = 0.50   # Bonus for unknown segments (0 obs)
    exploration_bonus_low_obs: float = 0.35    # Bonus for sparse segments (1-9 obs)
    exploration_bonus_medium_obs: float = 0.15 # Bonus for medium segments (10-49 obs)
    max_win_rate_adjustment: float = 1.8       # Max multiplier on base bid
    max_bid_cpm: float = 30.00                 # Hard bid ceiling


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

    # V5: NPI integration (updated for click data)
    use_npi_value: bool = True            # Enable NPI-based bid multipliers
    npi_exists: bool = True               # True for HCP targeting (drugs.com), False for non-HCP
    npi_data_path: Optional[str] = None   # Legacy: single NPI file (deprecated)
    npi_1year_path: Optional[str] = None  # Path to 1-year NPI click data
    npi_20day_path: Optional[str] = None  # Path to 20-day NPI click data (recency)
    npi_max_multiplier: float = 3.0       # Maximum NPI bid multiplier
    npi_recency_boost: float = 1.2        # Boost for recent clickers (+20%)


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

    # V5: WIDER adjustment bounds for exploration (min bound - max comes from preset)
    min_win_rate_adjustment: float = 0.6   # V5: Lowered (was 0.8)

    # V8: Exploration aggressiveness toggle
    # true = aggressive (larger bid increases, faster learning, higher risk)
    # false = gradual (smaller bid increases, slower learning, lower risk)
    aggressive_exploration: bool = False  # Default to gradual

    # V8: Exploration presets (populated from YAML or defaults)
    aggressive: Optional[ExplorationPreset] = None
    gradual: Optional[ExplorationPreset] = None

    # Legacy fields (kept for backward compatibility, overridden by presets)
    max_win_rate_adjustment: float = 1.8   # Fallback if no preset
    exploration_bonus_zero_obs: float = 0.50
    exploration_bonus_low_obs: float = 0.35
    exploration_bonus_medium_obs: float = 0.15

    def __post_init__(self):
        """Initialize default presets if not provided."""
        if self.aggressive is None:
            self.aggressive = ExplorationPreset(
                exploration_bonus_zero_obs=0.50,
                exploration_bonus_low_obs=0.35,
                exploration_bonus_medium_obs=0.15,
                max_win_rate_adjustment=1.8,
                max_bid_cpm=30.00
            )
        if self.gradual is None:
            self.gradual = ExplorationPreset(
                exploration_bonus_zero_obs=0.25,
                exploration_bonus_low_obs=0.15,
                exploration_bonus_medium_obs=0.08,
                max_win_rate_adjustment=1.4,
                max_bid_cpm=20.00
            )

    def get_active_exploration_settings(self) -> ExplorationPreset:
        """Return exploration settings based on toggle."""
        if self.aggressive_exploration:
            return self.aggressive
        else:
            return self.gradual


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
    calibration_gate: CalibrationGate = field(default_factory=CalibrationGate)
    bidding: BiddingStrategy = field(default_factory=BiddingStrategy)

    @classmethod
    def from_yaml(cls, path: str) -> 'OptimizerConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        business_data = data.get('business', {})
        technical_data = data.get('technical', {})
        features_data = data.get('features', {})
        calibration_gate_data = data.get('calibration_gate', {})
        bidding_data = data.get('bidding', {})

        # V8: Handle nested exploration presets in technical_data
        if technical_data:
            # Extract and convert nested presets to ExplorationPreset objects
            aggressive_data = technical_data.pop('aggressive', None)
            gradual_data = technical_data.pop('gradual', None)

            if aggressive_data:
                technical_data['aggressive'] = ExplorationPreset(**aggressive_data)
            if gradual_data:
                technical_data['gradual'] = ExplorationPreset(**gradual_data)

        return cls(
            business=BusinessControls(**business_data) if business_data else BusinessControls(),
            technical=TechnicalControls(**technical_data) if technical_data else TechnicalControls(),
            features=FeatureConfig(**features_data) if features_data else FeatureConfig(),
            calibration_gate=CalibrationGate(**calibration_gate_data) if calibration_gate_data else CalibrationGate(),
            bidding=BiddingStrategy(**bidding_data) if bidding_data else BiddingStrategy()
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
