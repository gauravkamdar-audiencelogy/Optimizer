"""
Configuration for RTB Bid Optimizer.

Supports two loading modes:
1. NEW: from_entity() - loads system.yaml + entities/{name}.yaml (recommended)
2. LEGACY: from_yaml() - loads single config file (backward compatible)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import os


# =============================================================================
# NEW CONFIG CLASSES (entity-based structure)
# =============================================================================

@dataclass
class EntityConfig:
    """Entity identity - defines what this SSP-Type combo is."""
    name: str = "unknown"
    floor_available: bool = False
    targeting_type: str = "consumer"  # 'hcp' or 'consumer'
    ssp_exclusions: List[str] = field(default_factory=list)

    @property
    def is_hcp(self) -> bool:
        return self.targeting_type == "hcp"

    @property
    def npi_enabled(self) -> bool:
        """NPI model is enabled for HCP targeting."""
        return self.is_hcp

    @property
    def domain_enabled(self) -> bool:
        """Domain model is enabled for consumer targeting."""
        return not self.is_hcp


@dataclass
class RunConfig:
    """Per-run config - user-facing settings (will come from MySQL later)."""
    target_win_rate: float = 0.65
    max_bid_cpm: float = 20.00
    fast_learning: bool = False
    training_start_date: Optional[str] = None
    training_end_date: Optional[str] = None
    user_disabled_features: List[str] = field(default_factory=list)


# =============================================================================
# EXISTING CONFIG CLASSES (kept for backward compatibility)
# =============================================================================

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


# =============================================================================
# MASTER CONFIG CLASS
# =============================================================================

@dataclass
class OptimizerConfig:
    """Master configuration container."""
    # New structure
    entity: EntityConfig = field(default_factory=EntityConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # Legacy structure (populated for backward compatibility)
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
        self._wire_backward_compat()

    def _wire_backward_compat(self):
        """Set up backward compatibility between config sections."""
        pass  # Properties handle this now

    @classmethod
    def from_entity(cls, entity_name: str, config_dir: str = "config") -> 'OptimizerConfig':
        """
        NEW: Load config from system.yaml + entities/{entity_name}.yaml.

        This is the recommended way to load configs.
        """
        config_path = Path(config_dir)
        system_path = config_path / "system.yaml"
        entity_path = config_path / "entities" / f"{entity_name}.yaml"

        # Check files exist
        if not system_path.exists():
            raise FileNotFoundError(f"System config not found: {system_path}")
        if not entity_path.exists():
            raise FileNotFoundError(f"Entity config not found: {entity_path}")

        # Load both files
        with open(system_path, 'r') as f:
            system_data = yaml.safe_load(f)
        with open(entity_path, 'r') as f:
            entity_data = yaml.safe_load(f)

        # Apply overrides from entity to system
        overrides = entity_data.get('overrides', {})
        merged_system = _deep_merge(system_data, overrides)

        # Build config from merged data
        return cls._build_from_new_structure(entity_data, merged_system)

    @classmethod
    def _build_from_new_structure(cls, entity_data: dict, system_data: dict) -> 'OptimizerConfig':
        """Build OptimizerConfig from new entity + system structure."""

        # Parse entity config
        entity_section = entity_data.get('entity', {})
        entity = EntityConfig(
            name=entity_section.get('name', 'unknown'),
            floor_available=entity_section.get('floor_available', False),
            targeting_type=entity_section.get('targeting_type', 'consumer'),
            ssp_exclusions=entity_section.get('ssp_exclusions', [])
        )

        # Parse run config
        run_section = entity_data.get('run', {})
        run = RunConfig(
            target_win_rate=run_section.get('target_win_rate', 0.65),
            max_bid_cpm=run_section.get('max_bid_cpm', 20.00),
            fast_learning=run_section.get('fast_learning', False),
            training_start_date=run_section.get('training_start_date'),
            training_end_date=run_section.get('training_end_date'),
            user_disabled_features=run_section.get('user_disabled_features', [])
        )

        # Get system sections
        model_data = system_data.get('model', {})
        feature_sel_data = system_data.get('feature_selection', {})
        presets_data = system_data.get('exploration_presets', {})
        npi_tier_data = system_data.get('npi_tiering', {})
        domain_tier_data = system_data.get('domain_tiering', {})
        validation_data = system_data.get('validation', {})
        bidding_data = system_data.get('bidding', {})
        overrides = entity_data.get('overrides', {})

        # Build legacy structures for backward compatibility
        dataset = DatasetConfig(name=entity.name)

        business = BusinessControls(
            target_win_rate=run.target_win_rate,
            exploration_mode=True  # Always on in new structure
        )

        # Parse exploration presets
        aggressive_preset = ExplorationPreset(**presets_data.get('aggressive', {})) if presets_data.get('aggressive') else ExplorationPreset()
        gradual_preset = ExplorationPreset(**presets_data.get('gradual', {})) if presets_data.get('gradual') else ExplorationPreset(
            bonus_zero_obs=0.25, bonus_low_obs=0.15, bonus_medium_obs=0.08,
            max_adjustment=1.4, max_bid_cpm=20.00
        )

        technical = TechnicalControls(
            aggressive_exploration=run.fast_learning,
            min_bid_cpm=overrides.get('min_bid_cpm', 3.00),
            max_bid_cpm=run.max_bid_cpm,
            default_bid_cpm=overrides.get('default_bid_cpm', 12.50),
            floor_available=entity.floor_available,
            aggressive=aggressive_preset,
            gradual=gradual_preset
        )

        bidding = BiddingStrategy(
            strategy=bidding_data.get('strategy', 'adaptive')
        )

        data_config = DataConfig(
            min_bid_date=run.training_start_date,
            min_view_date=overrides.get('min_view_date', run.training_start_date)
        )

        # NPI config - enabled based on targeting_type
        npi = NPIConfig(
            enabled=entity.npi_enabled,
            data_1year=npi_tier_data.get('data_1year'),
            data_20day=npi_tier_data.get('data_20day'),
            max_multiplier=npi_tier_data.get('max_multiplier', 3.0),
            recency_boost=npi_tier_data.get('recency_boost', 1.2),
            tiering_method=npi_tier_data.get('tiering_method', 'iqr'),
            iqr_multiplier_stars=npi_tier_data.get('iqr_multiplier_stars', 1.5),
            iqr_multiplier_extreme=npi_tier_data.get('iqr_multiplier_extreme', 3.0),
            poor_rate_factor=npi_tier_data.get('poor_rate_factor', 0.3),
            multiplier_extreme_elite=npi_tier_data.get('multiplier_extreme_elite', 3.00),
            multiplier_elite=npi_tier_data.get('multiplier_elite', 2.50),
            multiplier_cream=npi_tier_data.get('multiplier_cream', 1.80),
            multiplier_baseline=npi_tier_data.get('multiplier_baseline', 1.00),
            multiplier_poor=npi_tier_data.get('multiplier_poor', 0.70),
            min_clicks_for_tiering=npi_tier_data.get('min_clicks_for_tiering', 5)
        )

        # Domain config - enabled based on targeting_type
        domain = DomainConfig(
            enabled=entity.domain_enabled,
            tiering_method=domain_tier_data.get('tiering_method', 'iqr'),
            iqr_multiplier_stars=domain_tier_data.get('iqr_multiplier_stars', 1.5),
            iqr_multiplier_extreme=domain_tier_data.get('iqr_multiplier_extreme', 3.0),
            poor_rate_factor=domain_tier_data.get('poor_rate_factor', 0.3),
            multiplier_extreme_stars=domain_tier_data.get('multiplier_extreme_stars', 1.50),
            multiplier_stars=domain_tier_data.get('multiplier_stars', 1.30),
            multiplier_cream=domain_tier_data.get('multiplier_cream', 1.15),
            multiplier_baseline=domain_tier_data.get('multiplier_baseline', 1.00),
            multiplier_poor_iqr=domain_tier_data.get('multiplier_poor_iqr', 0.60),
            multiplier_blocklist=domain_tier_data.get('multiplier_blocklist', 0.0),
            blocklist_rate_factor=domain_tier_data.get('blocklist_rate_factor', 0.1),
            min_bids_for_blocklist=domain_tier_data.get('min_bids_for_blocklist', 100),
            shrinkage_k=domain_tier_data.get('shrinkage_k', 30),
            min_bids_for_tiering=domain_tier_data.get('min_bids_for_tiering', 30),
            # Legacy percentile settings
            premium_percentile=domain_tier_data.get('premium_percentile', 95.0),
            standard_percentile=domain_tier_data.get('standard_percentile', 50.0),
            below_avg_percentile=domain_tier_data.get('below_avg_percentile', 10.0),
            multiplier_premium=domain_tier_data.get('multiplier_premium', 1.3),
            multiplier_standard=domain_tier_data.get('multiplier_standard', 1.0),
            multiplier_below_avg=domain_tier_data.get('multiplier_below_avg', 0.8),
            multiplier_poor=domain_tier_data.get('multiplier_poor', 0.5)
        )

        # Features - combine system exclusions + entity ssp_exclusions
        system_exclusions = system_data.get('system_exclusions', ['log_txnid', 'internal_txn_id', 'log_dt'])
        all_exclusions = list(set(system_exclusions + entity.ssp_exclusions + run.user_disabled_features))
        features = FeatureConfig(
            anchor=[],  # Auto-select in new structure
            candidates=[],  # Auto-select in new structure
            exclude=all_exclusions
        )

        # Advanced config
        advanced = AdvancedConfig(
            min_observations=model_data.get('min_observations', 1),
            min_observations_for_empirical=model_data.get('min_observations_for_empirical', 10),
            min_observations_for_landscape=model_data.get('min_observations_for_landscape', 50),
            min_win_rate_for_margin=model_data.get('min_win_rate_for_margin', 0.55),
            min_observations_for_margin=model_data.get('min_observations_for_margin', 100),
            shrinkage_k=model_data.get('shrinkage_k', 30),
            calibration_ece_threshold=model_data.get('calibration_ece_threshold', 0.10),
            min_signal_score=feature_sel_data.get('min_signal_score', 50.0),
            max_features=feature_sel_data.get('max_features', 3),
            min_coverage_at_threshold=feature_sel_data.get('min_coverage_at_threshold', 0.5),
            max_null_pct=feature_sel_data.get('max_null_pct', 30.0),
            min_effective_cardinality=feature_sel_data.get('min_effective_cardinality', 2),
            effective_cardinality_min_share=feature_sel_data.get('effective_cardinality_min_share', 0.05)
        )

        # Validation config - always enabled in new structure
        validation = ValidationConfig(
            enabled=True,  # Always on
            hard_rules=validation_data.get('hard_rules', ValidationConfig().hard_rules),
            soft_rules=validation_data.get('soft_rules', ValidationConfig().soft_rules),
            previous_run_path=None
        )

        config = cls(
            entity=entity,
            run=run,
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

    @classmethod
    def from_yaml(cls, path: str) -> 'OptimizerConfig':
        """LEGACY: Load config from single YAML file (backward compatible)."""
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
            default_bid_cpm=tech_data.get('default_bid_cpm', 12.50),
            floor_available=tech_data.get('floor_available', False),
        )

        # Override presets if provided
        if aggressive_preset:
            technical.aggressive = aggressive_preset
        if gradual_preset:
            technical.gradual = gradual_preset

        # Build entity and run from legacy structure for compatibility
        entity = EntityConfig(
            name=dataset.name,
            floor_available=technical.floor_available,
            targeting_type='hcp' if npi.enabled else 'consumer',
            ssp_exclusions=features.exclude
        )

        run = RunConfig(
            target_win_rate=business.target_win_rate,
            max_bid_cpm=technical.max_bid_cpm,
            fast_learning=technical.aggressive_exploration,
            training_start_date=data_config.min_bid_date,
            training_end_date=None,
            user_disabled_features=[]
        )

        config = cls(
            entity=entity,
            run=run,
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict. Override wins on conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
