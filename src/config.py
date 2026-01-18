"""
V3 Configuration: Complete optimizer with empirical win rate signal.

V3 Changes:
- Reintroduce target_win_rate (now usable with empirical rates)
- Add win_rate_sensitivity control
- Split feature exclusions: hard (config) vs soft (algorithm-determined)

Philosophy: Controls should be meaningful and data-backed.
Algorithm makes data-driven decisions; config handles business/technical constraints.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class BusinessControls:
    """
    Controls exposed to business/product team.

    V3: Reintroduce target_win_rate now that we have empirical (not model) win rates.
    """
    target_margin: float = 0.30           # Bid shading factor (bid at 70% of EV)
    target_win_rate: float = 0.50         # V3: REINTRODUCE - target auction win rate
    win_rate_sensitivity: float = 0.5     # V3: How aggressively to adjust (0.3-0.7)


@dataclass
class TechnicalControls:
    """
    Controls for technical team only.

    V3: Added soft exclusion thresholds for algorithm-determined feature filtering.
    """
    # Bid bounds
    min_bid_cpm: float = 2.00              # Floor (lowered from $5 in V2)
    max_bid_cpm: float = 50.00             # Ceiling
    default_bid_cpm: float = 5.00          # For unmatched segments

    # Segment filtering (binary)
    min_observations: int = 100            # Minimum bids per segment

    # Feature selection
    max_features: int = 3                  # Maximum features in memcache key

    # Model shrinkage
    ctr_shrinkage_k: int = 30              # CTR Bayesian shrinkage strength
    win_rate_shrinkage_k: int = 30         # V3: Win rate shrinkage strength

    # V3: Soft exclusion thresholds (algorithm uses these to auto-exclude)
    min_signal_score: float = 50.0         # Below this = auto-exclude feature
    min_coverage_at_threshold: float = 0.5 # Must cover 50% of data
    max_null_pct: float = 30.0             # Above this = auto-exclude feature

    # V3.1: Effective cardinality filter (prevents dominated features)
    min_effective_cardinality: int = 2     # Must have >=2 values with significant share
    effective_cardinality_min_share: float = 0.05  # 5% threshold for "significant"

    # Win rate adjustment clipping (safety bounds)
    min_win_rate_adjustment: float = 0.8   # Floor for win rate multiplier
    max_win_rate_adjustment: float = 1.2   # Ceiling for win rate multiplier


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
                'page_category',
                'browser_code',
                'hour_of_day',        # V3: Let algorithm decide (was hardcoded excluded)
                'day_of_week'         # V3: Let algorithm decide (was hardcoded excluded)
            ]
        if not self.anchor_features:
            self.anchor_features = ['internal_adspace_id']
        if not self.hard_exclude_features:
            # V3: Only truly technical/structural exclusions
            # NOT data-driven decisions
            self.hard_exclude_features = [
                'geo_postal_code',    # Too sparse (7,027 unique)
                'geo_city_name',      # Too sparse (3,333 unique)
                'log_txnid',          # Technical: join key
                'internal_txn_id',    # Technical: join key
                'log_dt',             # Technical: timestamp
                'external_userid'     # PII / too sparse
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
