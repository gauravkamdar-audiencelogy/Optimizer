"""
Configuration for RTB Bid Optimizer.

Loads config from YAML files with dataset-specific settings.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class DatasetConfig:
    """Dataset identification and auto-derived paths."""
    name: str = "drugs"

    def get_data_dir(self) -> str:
        return f"data/{self.name}/"

    def get_output_dir(self) -> str:
        return f"output/{self.name}/"

    def get_data_filename(self) -> str:
        return f"data_{self.name}.csv"


@dataclass
class ExplorationPreset:
    """Exploration aggressiveness settings."""
    bonus_zero_obs: float = 0.50
    bonus_low_obs: float = 0.35
    bonus_medium_obs: float = 0.15
    max_adjustment: float = 1.8
    max_bid_cpm: float = 30.00

    # Aliases for backward compatibility
    @property
    def exploration_bonus_zero_obs(self): return self.bonus_zero_obs
    @property
    def exploration_bonus_low_obs(self): return self.bonus_low_obs
    @property
    def exploration_bonus_medium_obs(self): return self.bonus_medium_obs
    @property
    def max_win_rate_adjustment(self): return self.max_adjustment


@dataclass
class NPIConfig:
    """NPI (healthcare provider) value settings."""
    enabled: bool = False
    data_1year: Optional[str] = None
    data_20day: Optional[str] = None
    max_multiplier: float = 3.0
    recency_boost: float = 1.2

    # Tiering method: 'iqr' (hierarchical IQR) or 'percentile' (legacy)
    tiering_method: str = 'percentile'

    # IQR parameters (only used when tiering_method='iqr')
    iqr_multiplier_stars: float = 1.5    # IQR multiplier for stars threshold
    iqr_multiplier_extreme: float = 3.0  # IQR multiplier for extreme stars
    poor_rate_factor: float = 0.3        # Below this percentile = poor

    # Multipliers for 5-tier IQR system
    multiplier_extreme_elite: float = 3.00  # > Q3 + 3.0×IQR (capped at max_multiplier)
    multiplier_elite: float = 2.50          # > Q3 + 1.5×IQR
    multiplier_cream: float = 1.80          # > Middle_Q3 + 1.5×Middle_IQR
    multiplier_baseline: float = 1.00       # > poor_threshold
    multiplier_poor: float = 0.70           # ≤ poor_threshold

    # Observation threshold for IQR tiering
    min_clicks_for_tiering: int = 5


@dataclass
class DomainConfig:
    """Domain-level optimization settings (tiered multipliers like NPI)."""
    enabled: bool = False

    # Tiering method: 'iqr' (hierarchical IQR) or 'percentile' (legacy)
    tiering_method: str = 'percentile'

    # --- IQR parameters (only used when tiering_method='iqr') ---
    iqr_multiplier_stars: float = 1.5    # IQR multiplier for stars threshold
    iqr_multiplier_extreme: float = 3.0  # IQR multiplier for extreme stars
    poor_rate_factor: float = 0.3        # Below global_rate × factor = poor

    # Multipliers for 5-tier IQR system
    multiplier_extreme_stars: float = 1.50  # > Q3 + 3.0×IQR
    multiplier_stars: float = 1.30          # > Q3 + 1.5×IQR
    multiplier_cream: float = 1.15          # > Middle_Q3 + 1.5×Middle_IQR
    multiplier_baseline: float = 1.00       # > poor_threshold
    multiplier_poor_iqr: float = 0.60       # ≤ poor_threshold (IQR method)

    # Observation threshold for IQR tiering
    min_bids_for_tiering: int = 30

    # --- Legacy percentile thresholds (used when tiering_method='percentile') ---
    premium_percentile: float = 95.0    # Top 5% → premium tier
    standard_percentile: float = 50.0   # Top 50% → standard tier
    below_avg_percentile: float = 10.0  # Top 90% → below_avg tier

    # Multipliers for legacy percentile system
    multiplier_premium: float = 1.3
    multiplier_standard: float = 1.0
    multiplier_below_avg: float = 0.8
    multiplier_poor: float = 0.5
    multiplier_blocklist: float = 0.0

    # --- Common settings ---
    # Blocklist criteria
    blocklist_rate_factor: float = 0.1   # Block if rate < global_rate × factor
    blocklist_ctr_factor: float = 0.1    # Alias for backwards compatibility
    min_bids_for_blocklist: int = 100    # Need enough data to judge

    # Shrinkage
    shrinkage_k: int = 30                # Same as segment model


@dataclass
class BusinessControls:
    """High-level business settings."""
    target_win_rate: float = 0.65
    exploration_mode: bool = True
    target_margin: float = 0.30

    # Defaults for backward compatibility
    win_rate_sensitivity: float = 0.5
    exploration_up_multiplier: float = 1.3
    exploration_down_multiplier: float = 0.7
    accept_negative_margin: bool = True
    max_negative_margin_pct: float = 0.50


@dataclass
class TechnicalControls:
    """Technical bidding parameters."""
    aggressive_exploration: bool = False
    min_bid_cpm: float = 3.00
    max_bid_cpm: float = 20.00
    default_bid_cpm: float = 12.50
    floor_available: bool = False

    # Exploration presets
    aggressive: ExplorationPreset = field(default_factory=lambda: ExplorationPreset(
        bonus_zero_obs=0.50, bonus_low_obs=0.35, bonus_medium_obs=0.15,
        max_adjustment=1.8, max_bid_cpm=30.00
    ))
    gradual: ExplorationPreset = field(default_factory=lambda: ExplorationPreset(
        bonus_zero_obs=0.25, bonus_low_obs=0.15, bonus_medium_obs=0.08,
        max_adjustment=1.4, max_bid_cpm=20.00
    ))

    # Backward compatibility defaults (these get overridden by active preset)
    min_win_rate_adjustment: float = 0.6

    def get_active_exploration_settings(self) -> ExplorationPreset:
        """Return exploration settings based on toggle."""
        return self.aggressive if self.aggressive_exploration else self.gradual

    # Backward compatibility properties
    @property
    def max_win_rate_adjustment(self):
        return self.get_active_exploration_settings().max_adjustment

    @property
    def exploration_bonus_zero_obs(self):
        return self.get_active_exploration_settings().bonus_zero_obs

    @property
    def exploration_bonus_low_obs(self):
        return self.get_active_exploration_settings().bonus_low_obs

    @property
    def exploration_bonus_medium_obs(self):
        return self.get_active_exploration_settings().bonus_medium_obs


@dataclass
class BiddingStrategy:
    """Bidding strategy configuration."""
    strategy: str = "adaptive"
    use_bid_landscape_for_volume: bool = True


@dataclass
class DataConfig:
    """Data filtering settings."""
    min_bid_date: Optional[str] = None
    min_view_date: Optional[str] = None
    combined_file: Optional[str] = None


@dataclass
class FeatureConfig:
    """Feature selection settings."""
    anchor: List[str] = field(default_factory=lambda: ['internal_adspace_id'])
    candidates: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.candidates:
            self.candidates = [
                'internal_adspace_id', 'geo_region_name', 'geo_country_code2',
                'os_code', 'domain', 'browser_code', 'hour_of_day', 'day_of_week',
                'media_type'
            ]
        if not self.exclude:
            self.exclude = [
                'geo_postal_code', 'geo_city_name', 'log_txnid',
                'internal_txn_id', 'log_dt'
            ]

    # Backward compatibility aliases
    @property
    def anchor_features(self): return self.anchor
    @property
    def candidate_features(self): return self.candidates
    @property
    def hard_exclude_features(self): return self.exclude


@dataclass
class AdvancedConfig:
    """Advanced settings (rarely changed)."""
    # Observation thresholds
    min_observations: int = 1
    min_observations_for_empirical: int = 10
    min_observations_for_landscape: int = 50

    # Adaptive strategy thresholds
    min_win_rate_for_margin: float = 0.55
    min_observations_for_margin: int = 100

    # Model settings
    shrinkage_k: int = 30
    calibration_ece_threshold: float = 0.10

    # Feature selection
    min_signal_score: float = 50.0
    max_features: int = 3
    min_coverage_at_threshold: float = 0.5
    max_null_pct: float = 30.0
    min_effective_cardinality: int = 2
    effective_cardinality_min_share: float = 0.05

    # Technical settings that can be in advanced section of YAML
    default_bid_cpm: float = 12.50
    floor_available: bool = False
    target_margin: float = 0.30

    # Backward compatibility aliases
    @property
    def ctr_shrinkage_k(self): return self.shrinkage_k
    @property
    def win_rate_shrinkage_k(self): return self.shrinkage_k


@dataclass
class ValidationConfig:
    """Validation rules for optimizer output."""
    enabled: bool = False

    # Hard guardrails (block deployment if violated)
    hard_rules: dict = field(default_factory=lambda: {
        'coverage_min_pct': 80.0,           # Min % segments vs previous run
        'calibration_ece_max': 0.15,        # Max ECE for used models
        'bid_floor_respected': True,        # All bids >= min_bid
        'bid_ceiling_respected': True,      # All bids <= max_bid
    })

    # Soft guardrails (warn but allow override)
    soft_rules: dict = field(default_factory=lambda: {
        'bid_median_change_max_pct': 50.0,  # Max % change vs previous
        'pct_at_floor_max': 30.0,           # Max % bids at floor
        'pct_at_ceiling_max': 30.0,         # Max % bids at ceiling
        'pct_profitable_min': 40.0,         # Min % profitable segments
    })

    # Previous run for comparison (optional)
    previous_run_path: Optional[str] = None


@dataclass
class OptimizerConfig:
    """Master configuration container."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    business: BusinessControls = field(default_factory=BusinessControls)
    technical: TechnicalControls = field(default_factory=TechnicalControls)
    bidding: BiddingStrategy = field(default_factory=BiddingStrategy)
    data: DataConfig = field(default_factory=DataConfig)
    npi: NPIConfig = field(default_factory=NPIConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self):
        """Wire up backward compatibility references."""
        # Make technical settings point to advanced for backward compat
        self._wire_backward_compat()

    def _wire_backward_compat(self):
        """Set up backward compatibility between config sections."""
        # Copy advanced settings to technical for backward compat
        pass  # Properties handle this now

    @classmethod
    def from_yaml(cls, path: str) -> 'OptimizerConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse each section
        dataset = DatasetConfig(**data.get('dataset', {}))

        # Business controls
        business_data = data.get('business', {})
        business = BusinessControls(**business_data)

        # Bidding strategy
        bidding_data = data.get('bidding', {})
        bidding = BiddingStrategy(**bidding_data)

        # Data config
        data_config = DataConfig(**data.get('data', {}))

        # NPI config
        npi_data = data.get('npi', {})
        npi = NPIConfig(**npi_data)

        # Domain config
        domain_data = data.get('domain', {})
        domain = DomainConfig(**domain_data)

        # Advanced config
        advanced_data = data.get('advanced', {})
        advanced = AdvancedConfig(**advanced_data)

        # Validation config
        validation_data = data.get('validation', {})
        validation = ValidationConfig(
            enabled=validation_data.get('enabled', False),
            hard_rules=validation_data.get('hard_rules', ValidationConfig().hard_rules),
            soft_rules=validation_data.get('soft_rules', ValidationConfig().soft_rules),
            previous_run_path=validation_data.get('previous_run_path', None)
        )

        # Features config
        features_data = data.get('features', {})
        features = FeatureConfig(
            anchor=features_data.get('anchor', ['internal_adspace_id']),
            candidates=features_data.get('candidates', []),
            exclude=features_data.get('exclude', [])
        )

        # Technical config with exploration presets
        tech_data = data.get('technical', {})
        presets_data = data.get('exploration_presets', {})

        aggressive_preset = ExplorationPreset(**presets_data.get('aggressive', {})) if presets_data.get('aggressive') else None
        gradual_preset = ExplorationPreset(**presets_data.get('gradual', {})) if presets_data.get('gradual') else None

        technical = TechnicalControls(
            aggressive_exploration=tech_data.get('aggressive_exploration', False),
            min_bid_cpm=tech_data.get('min_bid_cpm', 3.00),
            max_bid_cpm=tech_data.get('max_bid_cpm', 20.00),
            default_bid_cpm=advanced_data.get('default_bid_cpm', tech_data.get('default_bid_cpm', 12.50)),
            floor_available=advanced_data.get('floor_available', tech_data.get('floor_available', False)),
        )

        # Override presets if provided
        if aggressive_preset:
            technical.aggressive = aggressive_preset
        if gradual_preset:
            technical.gradual = gradual_preset

        config = cls(
            dataset=dataset,
            business=business,
            technical=technical,
            bidding=bidding,
            data=data_config,
            npi=npi,
            domain=domain,
            features=features,
            advanced=advanced,
            validation=validation
        )

        # Wire backward compatibility after creation
        config._setup_backward_compat()

        return config

    def _setup_backward_compat(self):
        """Set up cross-section backward compatibility after load."""
        # Make business.npi_exists point to npi.enabled
        self.business.npi_exists = self.npi.enabled
        self.business.use_npi_value = self.npi.enabled
        self.business.npi_1year_path = self.npi.data_1year
        self.business.npi_20day_path = self.npi.data_20day
        self.business.npi_max_multiplier = self.npi.max_multiplier
        self.business.npi_recency_boost = self.npi.recency_boost
        self.business.npi_data_path = None  # Legacy field

        # Make bidding point to advanced thresholds
        self.bidding.min_win_rate_for_margin = self.advanced.min_win_rate_for_margin
        self.bidding.min_observations_for_margin = self.advanced.min_observations_for_margin

        # Make technical point to advanced settings (backward compat)
        self.technical.min_observations = self.advanced.min_observations
        self.technical.min_observations_for_empirical = self.advanced.min_observations_for_empirical
        self.technical.min_observations_for_landscape = self.advanced.min_observations_for_landscape
        self.technical.max_features = self.advanced.max_features
        self.technical.ctr_shrinkage_k = self.advanced.shrinkage_k
        self.technical.win_rate_shrinkage_k = self.advanced.shrinkage_k
        self.technical.min_signal_score = self.advanced.min_signal_score
        self.technical.min_coverage_at_threshold = self.advanced.min_coverage_at_threshold
        self.technical.max_null_pct = self.advanced.max_null_pct
        self.technical.min_effective_cardinality = self.advanced.min_effective_cardinality
        self.technical.effective_cardinality_min_share = self.advanced.effective_cardinality_min_share

    # Legacy property for calibration_gate
    @property
    def calibration_gate(self):
        """Backward compatibility: calibration_gate.*"""
        class _Gate:
            def __init__(self, threshold):
                self.max_ece_threshold = threshold
                self.min_observations_for_eval = 1000
                self.log_gate_decisions = True
        return _Gate(self.advanced.calibration_ece_threshold)
