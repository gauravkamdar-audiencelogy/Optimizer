# UPDATE_2: Radical Simplification - Aligning Model with Economic Reality

**Purpose**: Remove complexity that adds noise, not signal. Current controls are fighting the data.  
**Priority**: CRITICAL - 92.67% of bids hitting floor means controls are overriding model entirely  
**Philosophy**: With 214 clicks and miscalibrated win rate model, simpler is better

---

## Executive Summary: Why Simplify?

### Evidence from Run 20260114_093840

| Finding | Value | Implication |
|---------|-------|-------------|
| Win rate ECE | 0.176 | Model overestimates by ~1.7x across all buckets |
| CTR AUC-ROC | 0.516 | Barely better than random - can't discriminate segments |
| Bids at floor | 92.67% | All complexity thrown away by floor clipping |
| Raw bid median | $2.20 | Model wants to bid $2, we force $5 |
| EV median | $3.65 | Expected value < floor for most segments |
| Profitable segments | 22.52% | 77.5% of segments are UNPROFITABLE |

### The Core Problem

Current bid formula:
```python
raw_bid = EV × (1 - margin) × (1 + (target_wr - pred_wr) × 0.5) × (0.5 + 0.5 × confidence)
```

This formula uses a **miscalibrated win rate model** (ECE=0.176) to adjust bids. When the model predicts 50% win rate but actual is 30%, the adjustment factor is **based on wrong information**.

**Result**: Noise amplification, not optimization.

---

## CHANGE 1: Remove Win Rate Adjustment from Bid Formula

### File: `src/bid_calculator.py`

### Reasoning

The win rate model has ECE = 0.176 with consistent overestimation:

| Predicted WR | Actual WR | Error |
|--------------|-----------|-------|
| 35.7% | 18.0% | +17.7pp (1.98x over) |
| 43.9% | 26.4% | +17.5pp (1.66x over) |
| 54.0% | 36.4% | +17.6pp (1.48x over) |
| 65.1% | 45.1% | +20.0pp (1.44x over) |
| 74.0% | 57.2% | +16.8pp (1.29x over) |
| 91.7% | 30.4% | +61.4pp (3.02x over) |

Using this model to ADJUST bids means we're adjusting based on wrong predictions.

The `win_rate_adjustment = 1 + (target_wr - pred_wr) × 0.5` formula:
- When pred_wr=0.48, target_wr=0.80: adjustment = 1.16 (bid 16% higher)
- But pred_wr is WRONG (actual is ~0.28)
- So we're bidding higher based on false information

**Literature Note**: The VerizonMedia paper (arXiv:2009.09259) that inspired this formula had billions of observations and well-calibrated models. With 200K bids and ECE=0.18, we don't have the foundation to use win rate for bid adjustment.

### Implementation

```python
# File: src/bid_calculator.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class BidResult:
    """Result of bid calculation for a segment."""
    segment_key: Dict[str, any]
    ctr: float
    expected_cpc: float
    expected_value_per_impression: float
    expected_value_cpm: float
    raw_bid: float
    final_bid: float
    observation_count: int
    is_profitable: bool
    exclusion_reason: Optional[str] = None
    # REMOVED: win_rate (not using for bid calculation)
    # REMOVED: confidence (using binary filter instead)


class BidCalculator:
    """
    V2: Simplified economic bidding.
    
    Formula: bid = EV_cpm × (1 - target_margin)
    
    That's it. No win rate adjustment. No confidence scaling.
    Complex adjustments require accurate models. We don't have them.
    """
    
    def __init__(
        self,
        config: 'OptimizerConfig',
        ctr_model: 'CTRModel'
        # REMOVED: win_rate_model parameter - not using it for bidding
    ):
        self.config = config
        self.ctr_model = ctr_model
        self.avg_cpc: float = 0.0
    
    def set_average_cpc(self, df_clicks: pd.DataFrame) -> None:
        """Calculate average CPC from click data."""
        self.avg_cpc = df_clicks['cpc_value'].mean()
    
    def calculate_bids_for_segments(
        self,
        df_segments: pd.DataFrame,
        features: List[str]
    ) -> List[BidResult]:
        """Calculate optimal bid for each segment using simplified formula."""
        results = []
        
        for _, row in df_segments.iterrows():
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
        
        V2 SIMPLIFIED FORMULA:
            raw_bid = EV_cpm × (1 - target_margin)
        
        REMOVED:
            - win_rate_adjustment: Based on miscalibrated model (ECE=0.18)
            - confidence_factor: Binary filter via min_observations instead
            - target_win_rate: Can't optimize with current data quality
        """
        # Get CTR prediction (well-calibrated, ECE=0.00003)
        ctr = self.ctr_model.get_ctr_for_segment(segment_key)
        
        # Expected value calculation
        # EV per impression = P(click) × E[CPC]
        expected_value_per_impression = ctr * self.avg_cpc
        expected_value_cpm = expected_value_per_impression * 1000
        
        # SIMPLIFIED: Single shading factor based on target margin
        # No win rate adjustment, no confidence scaling
        shading_factor = 1 - self.config.business.target_margin
        
        raw_bid = expected_value_cpm * shading_factor
        
        # Check economic profitability
        # Profitable if EV > floor (with margin for error)
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
            expected_cpc=round(self.avg_cpc, 2),
            expected_value_per_impression=round(expected_value_per_impression, 8),
            expected_value_cpm=round(expected_value_cpm, 4),
            raw_bid=round(raw_bid, 4),
            final_bid=final_bid,
            observation_count=observation_count,
            is_profitable=is_profitable,
            exclusion_reason=exclusion_reason
        )
```

---

## CHANGE 2: Simplify Config - Remove Unused Controls

### File: `src/config.py`

### Reasoning

Current config has controls we can't meaningfully use:

| Control | Problem | Action |
|---------|---------|--------|
| `target_win_rate` | Model can't optimize for this (ECE=0.18) | REMOVE |
| `exploration_budget` | Not implemented in bid formula | REMOVE |
| `volume_boost_mode` | Not implemented | REMOVE |
| `confidence_threshold` | Using binary filter instead | REMOVE |
| `decay_factor` | Not implemented | REMOVE |

**Principle**: Every parameter that isn't data-driven adds arbitrary complexity.

### Implementation

```python
# File: src/config.py

"""
V2 Simplified Configuration.

Philosophy: Only include controls that we can actually optimize for.
With 214 clicks and miscalibrated win rate model, that's not much.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class BusinessControls:
    """
    Controls exposed to business/product team.
    
    V2: Simplified to controls we can actually use.
    """
    target_margin: float = 0.30  # Target profit margin (bid shading)
    # REMOVED: target_win_rate - can't optimize with miscalibrated model
    # REMOVED: exploration_budget - not implemented, do via A/B test
    # REMOVED: volume_boost_mode - solve with floor price instead


@dataclass  
class TechnicalControls:
    """
    Controls for technical team only.
    
    V2: Simplified, aligned with economic reality.
    """
    # Bid bounds
    min_bid_cpm: float = 2.00              # LOWERED: $5→$2 (median raw bid was $2.20)
    max_bid_cpm: float = 50.00             # Keep as sanity ceiling
    default_bid_cpm: float = 5.00          # For unmatched segments
    
    # Segment filtering (binary: include or exclude)
    min_observations: int = 100            # INCREASED: 50→100 for stability
    
    # Feature selection
    max_features: int = 3                  # REDUCED: 4→3 (more samples per segment)
    
    # CTR model shrinkage
    ctr_shrinkage_k: int = 30              # Bayesian shrinkage strength
    
    # REMOVED: confidence_threshold - using binary filter
    # REMOVED: training_window_days - not implemented
    # REMOVED: decay_factor - not implemented


@dataclass
class FeatureConfig:
    """Feature selection configuration."""
    candidate_features: List[str] = field(default_factory=list)
    anchor_features: List[str] = field(default_factory=list)
    exclude_features: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.candidate_features:
            self.candidate_features = [
                'internal_adspace_id',
                'geo_region_name',
                'os_code',
                'page_category',
                'browser_code',
                # REMOVED: hour_of_day, day_of_week (low signal scores)
                # REMOVED: geo_country_code2 (98% US, adds segments without signal)
            ]
        if not self.anchor_features:
            self.anchor_features = ['internal_adspace_id']
        if not self.exclude_features:
            self.exclude_features = [
                'geo_postal_code', 
                'geo_city_name',
                'geo_country_code2',  # ADDED: 98% US, not useful
                'hour_of_day',        # ADDED: Low signal (score=22.9)
                'day_of_week'         # ADDED: Low signal (score=12.1)
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
                'target_margin': self.business.target_margin
            },
            'technical': {
                'min_bid_cpm': self.technical.min_bid_cpm,
                'max_bid_cpm': self.technical.max_bid_cpm,
                'default_bid_cpm': self.technical.default_bid_cpm,
                'min_observations': self.technical.min_observations,
                'max_features': self.technical.max_features,
                'ctr_shrinkage_k': self.technical.ctr_shrinkage_k
            },
            'features': {
                'candidate_features': self.features.candidate_features,
                'anchor_features': self.features.anchor_features,
                'exclude_features': self.features.exclude_features
            }
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
```

---

## CHANGE 3: Update Memcache Builder for Binary Filtering

### File: `src/memcache_builder.py`

### Reasoning

Current confidence_factor creates half-hearted bids that get clipped to floor anyway. Replace with binary filter: either a segment has enough observations and is profitable, or it's excluded.

### Implementation

```python
# File: src/memcache_builder.py

"""
V2: Simplified memcache generation with binary filtering.
"""
import pandas as pd
from typing import List
from pathlib import Path
from datetime import datetime


class MemcacheBuilder:
    def __init__(self, config: 'OptimizerConfig'):
        self.config = config
        self.filter_stats = {
            'total_segments': 0,
            'excluded_low_observations': 0,
            'excluded_unprofitable': 0,
            'included': 0
        }
    
    def build_memcache(
        self,
        bid_results: List['BidResult'],
        features: List[str]
    ) -> pd.DataFrame:
        """
        Build memcache DataFrame with BINARY filtering.
        
        V2 SIMPLIFIED:
        - Include if: observations >= min_observations AND is_profitable
        - Exclude otherwise
        
        No half-measures. No confidence scaling.
        """
        rows = []
        min_obs = self.config.technical.min_observations
        
        self.filter_stats['total_segments'] = len(bid_results)
        
        for result in bid_results:
            # Binary filter 1: Enough observations?
            if result.observation_count < min_obs:
                self.filter_stats['excluded_low_observations'] += 1
                continue
            
            # Binary filter 2: Economically profitable?
            if not result.is_profitable:
                self.filter_stats['excluded_unprofitable'] += 1
                continue
            
            # Include this segment
            row = {}
            for feat in features:
                row[feat] = result.segment_key.get(feat, '')
            
            row['suggested_bid_cpm'] = result.final_bid
            rows.append(row)
            self.filter_stats['included'] += 1
        
        df = pd.DataFrame(rows)
        
        # Ensure column order: features first, then bid
        if len(df) > 0:
            column_order = features + ['suggested_bid_cpm']
            df = df[column_order]
        
        return df
    
    def write_memcache(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        timestamp: str = None
    ) -> Path:
        """Write memcache to TSV file."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'memcache_{timestamp}.csv'
        filepath = output_dir / filename
        
        # Write as tab-separated
        df.to_csv(filepath, sep='\t', index=False)
        
        return filepath
    
    def get_filter_stats(self) -> dict:
        """Return filtering statistics for metrics."""
        return self.filter_stats
```

---

## CHANGE 4: Simplify Metrics Reporter

### File: `src/metrics_reporter.py`

### Reasoning

Remove metrics for controls we've eliminated. Add clarity on what we're actually measuring.

### Implementation (Key Changes Only)

```python
# In compile_metrics method, update the config section:

'config': {
    'business': {
        'target_margin': self.config.business.target_margin
        # REMOVED: target_win_rate, exploration_budget, volume_boost_mode
    },
    'technical': {
        'min_bid_cpm': self.config.technical.min_bid_cpm,
        'max_bid_cpm': self.config.technical.max_bid_cpm,
        'default_bid_cpm': self.config.technical.default_bid_cpm,
        'min_observations': self.config.technical.min_observations,
        'max_features': self.config.technical.max_features,
        'ctr_shrinkage_k': self.config.technical.ctr_shrinkage_k
        # REMOVED: confidence_threshold, training_window_days, decay_factor
    }
},

# Add new section for formula transparency:
'bid_formula': {
    'version': 'v2_simplified',
    'formula': 'bid = EV_cpm × (1 - target_margin)',
    'shading_factor': 1 - self.config.business.target_margin,
    'rationale': 'Removed win_rate_adjustment due to model miscalibration (ECE=0.18)'
},

# Update segment_distribution to use new filter_stats:
'segment_distribution': {
    'total_segments': memcache_builder.filter_stats['total_segments'],
    'excluded_low_observations': memcache_builder.filter_stats['excluded_low_observations'],
    'excluded_unprofitable': memcache_builder.filter_stats['excluded_unprofitable'],
    'segments_in_memcache': memcache_builder.filter_stats['included'],
    'inclusion_rate': round(
        memcache_builder.filter_stats['included'] / 
        max(1, memcache_builder.filter_stats['total_segments']) * 100, 2
    )
}
```

---

## CHANGE 5: Update Main Entry Point

### File: `run_optimizer.py`

### Reasoning

Remove win rate model from bid calculation pipeline. Keep it only for diagnostics.

### Implementation

```python
#!/usr/bin/env python3
"""
RTB Optimizer Pipeline - V2 Simplified

Key changes from V1:
- Bid formula: bid = EV_cpm × (1 - target_margin)
- No win rate adjustment (model miscalibrated)
- Binary segment filtering (no confidence scaling)
- Fewer features (more samples per segment)
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

from src.config import OptimizerConfig
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.feature_selector import FeatureSelector
from src.models.win_rate_model import WinRateModel  # Keep for diagnostics only
from src.models.ctr_model import CTRModel
from src.bid_calculator import BidCalculator
from src.memcache_builder import MemcacheBuilder
from src.metrics_reporter import MetricsReporter


def main():
    parser = argparse.ArgumentParser(description='RTB Optimizer Pipeline V2')
    parser.add_argument('--config', type=str, default='config/optimizer_config.yaml')
    parser.add_argument('--data-dir', type=str, default='data/')
    parser.add_argument('--output-dir', type=str, default='output/')
    args = parser.parse_args()
    
    # Generate run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Starting optimizer run: {run_id}")
    print(f"Version: V2 Simplified (no win rate adjustment)")
    
    # Create output directory
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = OptimizerConfig.from_yaml(str(config_path))
    else:
        print(f"Config not found, using defaults")
        config = OptimizerConfig()
    
    print(f"\nConfiguration:")
    print(f"  target_margin: {config.business.target_margin}")
    print(f"  min_bid_cpm: ${config.technical.min_bid_cpm}")
    print(f"  max_features: {config.technical.max_features}")
    print(f"  min_observations: {config.technical.min_observations}")
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    data_loader = DataLoader(args.data_dir, config)
    df_bids, df_views, df_clicks = data_loader.load_all()
    print(f"  Loaded: {len(df_bids):,} bids, {len(df_views):,} views, {len(df_clicks):,} clicks")
    
    # Step 2: Feature engineering
    print("\n[2/6] Engineering features...")
    feature_engineer = FeatureEngineer(config)
    df_bids = feature_engineer.create_features(df_bids)
    df_views = feature_engineer.create_features(df_views)
    
    # Create training datasets
    df_train_wr = feature_engineer.create_training_data(df_bids, df_views, df_clicks)
    df_train_ctr = feature_engineer.create_ctr_data(df_views, df_clicks)
    print(f"  Win rate training: {len(df_train_wr):,} samples")
    print(f"  CTR training: {len(df_train_ctr):,} samples, {df_train_ctr['clicked'].sum()} clicks")
    
    # Step 3: Feature selection
    print("\n[3/6] Selecting features...")
    feature_selector = FeatureSelector(config)
    selected_features = feature_selector.select_features(df_train_wr, target_col='won')
    print(f"  Selected: {selected_features}")
    
    # Step 4: Train models
    print("\n[4/6] Training models...")
    
    # Win rate model (for DIAGNOSTICS ONLY, not used in bidding)
    win_rate_model = WinRateModel(config)
    win_rate_model.train(df_train_wr, selected_features, target='won')
    print(f"  Win rate model: base_rate={win_rate_model.training_stats['base_rate']:.2%}")
    print(f"    ⚠️  Note: Win rate model NOT used for bid calculation (ECE too high)")
    
    # CTR model (used for bidding)
    ctr_model = CTRModel(config)
    ctr_model.train(df_train_ctr, selected_features, target='clicked')
    print(f"  CTR model: global_ctr={ctr_model.training_stats['global_ctr']:.4%}")
    print(f"    ✓ CTR model used for bid calculation (shrinkage-based)")
    
    # Step 5: Calculate bids
    print("\n[5/6] Calculating bids...")
    
    # V2: BidCalculator only needs CTR model
    bid_calculator = BidCalculator(config, ctr_model)
    bid_calculator.set_average_cpc(df_clicks)
    print(f"  Average CPC: ${bid_calculator.avg_cpc:.2f}")
    
    # Get unique segments from CTR training data (has more samples)
    df_segments = df_train_ctr.groupby(selected_features).size().reset_index(name='count')
    print(f"  Unique segments: {len(df_segments):,}")
    
    bid_results = bid_calculator.calculate_bids_for_segments(df_segments, selected_features)
    
    # Summarize bid distribution
    raw_bids = [r.raw_bid for r in bid_results]
    evs = [r.expected_value_cpm for r in bid_results]
    profitable_count = sum(1 for r in bid_results if r.is_profitable)
    
    print(f"\n  Bid Summary:")
    print(f"    Raw bid range: ${min(raw_bids):.2f} - ${max(raw_bids):.2f}")
    print(f"    Raw bid median: ${sorted(raw_bids)[len(raw_bids)//2]:.2f}")
    print(f"    EV median: ${sorted(evs)[len(evs)//2]:.2f}")
    print(f"    Profitable segments: {profitable_count:,} / {len(bid_results):,} ({profitable_count/len(bid_results)*100:.1f}%)")
    
    # Step 6: Build outputs
    print("\n[6/6] Building outputs...")
    
    # Memcache with binary filtering
    memcache_builder = MemcacheBuilder(config)
    df_memcache = memcache_builder.build_memcache(bid_results, selected_features)
    memcache_path = memcache_builder.write_memcache(df_memcache, output_dir, run_id)
    
    filter_stats = memcache_builder.get_filter_stats()
    print(f"\n  Filtering Results:")
    print(f"    Total segments: {filter_stats['total_segments']:,}")
    print(f"    Excluded (low obs): {filter_stats['excluded_low_observations']:,}")
    print(f"    Excluded (unprofitable): {filter_stats['excluded_unprofitable']:,}")
    print(f"    Included in memcache: {filter_stats['included']:,}")
    
    print(f"\n  Output: {memcache_path}")
    
    # Metrics
    metrics_reporter = MetricsReporter(config)
    metrics = metrics_reporter.compile_metrics(
        run_id=run_id,
        data_loader=data_loader,
        feature_selector=feature_selector,
        win_rate_model=win_rate_model,  # For diagnostics
        ctr_model=ctr_model,
        bid_results=bid_results,
        memcache_path=memcache_path,
        memcache_builder=memcache_builder,
        df_train_wr=df_train_wr,
        df_train_ctr=df_train_ctr
    )
    metrics_path = metrics_reporter.write_metrics(output_dir, run_id)
    print(f"  Metrics: {metrics_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Optimizer V2 run complete: {run_id}")
    print(f"{'='*60}")
    print(f"  Segments in memcache: {filter_stats['included']:,}")
    print(f"  Bid formula: bid = EV × {1 - config.business.target_margin} (fixed shading)")
    print(f"  Floor: ${config.technical.min_bid_cpm}, Ceiling: ${config.technical.max_bid_cpm}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

---

## CHANGE 6: Remove Win Rate Model from Bid Calculation

### File: `src/models/win_rate_model.py`

### Reasoning

Keep the win rate model for DIAGNOSTICS (we want to see calibration metrics) but it should NOT be used in bid calculation.

### Implementation

Add clear documentation:

```python
# File: src/models/win_rate_model.py

"""
Win Rate Model: Predicts P(win | features).

⚠️  V2 NOTE: This model is for DIAGNOSTICS ONLY.
    It is NOT used in bid calculation because:
    - ECE = 0.176 (consistently overestimates by ~1.7x)
    - Using miscalibrated predictions to adjust bids adds noise
    
    We keep it to:
    1. Track calibration over time
    2. Identify when model quality improves enough to use
    3. Understand market dynamics
"""

class WinRateModel:
    """
    Win rate prediction model.
    
    DIAGNOSTIC USE ONLY - not used for bid calculation in V2.
    """
    # ... rest of class unchanged
```

---

## CHANGE 7: Create Default YAML Config

### File: `config/optimizer_config.yaml`

### Reasoning

Provide a clean config file reflecting V2 simplified settings.

### Implementation

```yaml
# V2 Simplified Configuration
# 
# Philosophy: Only include controls we can actually optimize.
# With 214 clicks and miscalibrated win rate model, that's not much.

business:
  target_margin: 0.30  # 30% margin = bid at 70% of expected value

technical:
  min_bid_cpm: 2.00         # Lowered from $5 to capture more segments
  max_bid_cpm: 50.00        # Sanity ceiling
  default_bid_cpm: 5.00     # For unmatched segments
  min_observations: 100     # Binary filter threshold
  max_features: 3           # Reduced for more samples per segment
  ctr_shrinkage_k: 30       # Bayesian shrinkage strength

features:
  anchor_features:
    - internal_adspace_id
  
  candidate_features:
    - internal_adspace_id
    - geo_region_name
    - os_code
    - page_category
    - browser_code
  
  exclude_features:
    - geo_postal_code
    - geo_city_name
    - geo_country_code2    # 98% US, not useful
    - hour_of_day          # Low signal
    - day_of_week          # Low signal
```

---

## Summary of Changes

| Change | File | What Changed | Why |
|--------|------|--------------|-----|
| 1 | bid_calculator.py | Remove win_rate_adjustment, simplify formula | Win rate model ECE=0.18, adds noise |
| 2 | config.py | Remove target_win_rate, exploration_budget, etc. | Controls not backed by data |
| 3 | memcache_builder.py | Binary filtering instead of confidence scaling | Half-measures hit floor anyway |
| 4 | metrics_reporter.py | Update for simplified config | Align with removed controls |
| 5 | run_optimizer.py | Remove win rate from bid calculation | Only CTR model is usable |
| 6 | win_rate_model.py | Mark as diagnostics-only | Don't use for bidding |
| 7 | optimizer_config.yaml | New V2 defaults | Clean config |

---

## Expected Results After V2

| Metric | Before (V1) | Expected (V2) |
|--------|-------------|---------------|
| Segments in memcache | 77 | 300-500 |
| Bids at floor | 92.67% | 30-40% |
| Profitable segments | 22.5% | 60-70% |
| Run-to-run variance | High | Low |
| Bid formula | Complex (5 terms) | Simple (2 terms) |
| Controls count | 10+ | 6 |

---

## Economic Reality Check

With V2 settings:
- Floor: $2.00 CPM
- Margin: 30%
- Global CTR: 0.037%
- Avg CPC: $13.36

Expected value for global average segment:
```
EV_cpm = 0.00037 × $13.36 × 1000 = $4.94

Raw bid = $4.94 × 0.70 = $3.46

Final bid = max($3.46, $2.00) = $3.46  ✓ Not clipped!
```

For segments with shrunk CTR (zero clicks, low prior):
```
Shrunk CTR ≈ 0.00025 (shrunk toward global)
EV_cpm = 0.00025 × $13.36 × 1000 = $3.34

Raw bid = $3.34 × 0.70 = $2.34

Final bid = max($2.34, $2.00) = $2.34  ✓ Not clipped!
```

**This is economically sensible bidding.** Low-value segments get low bids. High-value segments get high bids. No artificial floors distorting the model's output.

---

## Philosophical Note

> "Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away." — Antoine de Saint-Exupéry

The original system tried to optimize win rate and CTR and confidence and exploration all at once. With 214 clicks, we can't optimize for all of that. 

V2 optimizes for ONE thing: **economic value**. 

Bid = Expected Value × Shading.

That's the economic truth. Everything else was noise.
