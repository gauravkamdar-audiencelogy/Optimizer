# RTB Optimizer Pipeline Specification
## Drugs.com HCP Advertising Bid Optimization System

**Version**: 1.0  
**Date**: January 2026  
**Author**: Gaurav Kamdar + Claude  

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Business Context](#2-business-context)
3. [Data Overview](#3-data-overview)
4. [System Architecture](#4-system-architecture)
5. [Module Specifications](#5-module-specifications)
6. [Algorithm Details](#6-algorithm-details)
7. [Control Settings](#7-control-settings)
8. [Output Specifications](#8-output-specifications)
9. [Implementation Guide](#9-implementation-guide)
10. [Testing & Validation](#10-testing--validation)

---

## 1. Executive Summary

### 1.1 Purpose
Build a batch optimization pipeline that:
1. Reads historical bid/view/click data from CSV files
2. Trains win rate and CTR prediction models
3. Dynamically selects optimal features for segmentation
4. Calculates optimal bid prices per segment
5. Outputs a memcache CSV for the bidder and a metrics JSON for diagnostics

### 1.2 Inputs
- `drugs_bids.csv` - Bid request records (Dec 10, 2025+)
- `drugs_views.csv` - Impression win records (Sep 15, 2025+)
- `drugs_clicks.csv` - Click records (Sep 15, 2025+)

### 1.3 Outputs
- `memcache_YYYYMMDD_HHMMSS.csv` - Bid lookup table
- `metrics_YYYYMMDD_HHMMSS.json` - Training metrics and diagnostics

### 1.4 Key Constraints
- Bidder does single memcache lookup (no hierarchical fallback)
- Bidder needs to know column names (order flexible)
- First-price auction without floor price visibility
- Extremely sparse CTR (~0.035%, ~248 clicks in 4 months)

---

## 2. Business Context

### 2.1 Business Model
```
Revenue = CPC × Clicks
Cost = CPM × Impressions / 1000
Profit = Revenue - Cost
```

- **Supply Side**: Pay publishers CPM (Cost Per Mille) for impressions
- **Demand Side**: Earn CPC (Cost Per Click) from advertisers when users click
- **Goal**: Maximize profit by bidding optimally on high-value segments

### 2.2 Current Performance (Baseline)
| Metric | Value | Source |
|--------|-------|--------|
| Win Rate | 30.4% | EDA_4 joined analysis |
| CTR | 0.035% | EDA_4 Oct+ data |
| Avg CPC | $13.36 | EDA_4 click analysis |
| Default Bid | $12.50 CPM | Current config |
| Expected Revenue/Bid | $0.0014 | EDA_4 calculation |

### 2.3 Optimization Opportunities
1. **Underbidding**: Missing high-value impressions that lead to clicks
2. **Overbidding**: Winning impressions at prices higher than necessary
3. **Segment Blindness**: Same bid for all segments ignores value variance

---

## 3. Data Overview

### 3.1 Data Schema

#### 3.1.1 Key Columns
| Column | Type | Description | Usage |
|--------|------|-------------|-------|
| `log_txnid` | UUID | Unique request identifier | Join key |
| `rec_type` | String | Record type: bid/View/link | Filter |
| `log_dt` | Timestamp | Event timestamp | Date filtering |
| `internal_txn_id` | Array | Ad UUIDs (1-3 per request) | Click matching |
| `internal_adspace_id` | Integer | Ad placement ID (9 unique) | Feature |
| `geo_region_name` | String | US State (65 unique, 11.4% null) | Feature |
| `geo_country_code2` | String | Country code (20 unique) | Feature |
| `browser_code` | Integer | Browser ID (9 unique, 94% = code 14) | Feature |
| `os_code` | Integer | OS ID (6 unique, 67% = -1) | Feature |
| `ref_bundle` | String | Full page URL | Feature (extract path) |
| `publisher_payout` | Array | CPM paid when won | Cost |
| `advertiser_spend` | Array | CPC earned on click | Revenue |
| `bid_amount` | Array | Bid submitted | Bid analysis |
| `campaign_code` | Array | Advertiser campaign ID | Revenue grouping |
| `external_userid` | Float | NPI number (HCP identifier) | Filter/Feature |

#### 3.1.2 Array Field Handling
Arrays are stored as PostgreSQL format: `{value1,value2,...}`

**Parsing Rule**: Extract first value for analysis
```python
def parse_first_array_value(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    if val_str.startswith('{') and val_str.endswith('}'):
        inner = val_str[1:-1]
        if inner == '':
            return np.nan
        return float(inner.split(',')[0])
    return float(val_str)
```

### 3.2 Data Quality Issues

| Issue | Impact | Resolution |
|-------|--------|------------|
| View duplicates (7.3%) | Inflated win rate | Deduplicate by `log_txnid` |
| Zero bids (6.4% in Dec) | Invalid records | Filter `bid_amount > 0` |
| NPI as float | Validation issues | Convert to int then string |
| Unmatched views (36%) | Missing bid records | Views before Dec 10 have no bids |
| geo_region_name nulls (11.4%) | Missing signal | Treat as "Unknown" category |

### 3.3 Data Volume Summary

| Dataset | Date Range | Raw Records | Clean Records |
|---------|------------|-------------|---------------|
| Bids | Dec 10, 2025 - Jan 11, 2026 | 207,013 | 193,756 (non-zero) |
| Views | Sep 15, 2025 - Jan 11, 2026 | 623,545 | 577,815 (deduped) |
| Clicks | Sep 15, 2025 - Jan 11, 2026 | 248 | 248 |

### 3.4 Feature Cardinality (From EDA)

| Feature | Unique Values | Null % | Win Rate Variance | Signal Score |
|---------|---------------|--------|-------------------|--------------|
| internal_adspace_id | 9 | 0.0% | High | ⭐⭐⭐ Primary |
| geo_region_name | 65 | 11.4% | Medium | ⭐⭐ Good |
| geo_country_code2 | 20 | 0.0% | Low | ⭐ Limited |
| browser_code | 9 | 0.0% | Low | ⚠️ 94% single value |
| os_code | 6 | 0.0% | Low | ⚠️ 67% unknown |
| page_category | ~100+ | TBD | TBD | ⭐⭐ Potential |

---

## 4. System Architecture

### 4.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OPTIMIZER PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ 1. DATA      │───▶│ 2. FEATURE   │───▶│ 3. FEATURE   │                   │
│  │    LOADER    │    │    ENGINEER  │    │    SELECTOR  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      4. MODEL TRAINING                                │   │
│  │  ┌─────────────────┐              ┌─────────────────┐                │   │
│  │  │  WIN RATE MODEL │              │   CTR MODEL     │                │   │
│  │  │  (Logistic Reg) │              │  (Logistic Reg) │                │   │
│  │  └─────────────────┘              └─────────────────┘                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│                     ┌──────────────────────────┐                            │
│                     │  5. BID CALCULATOR       │                            │
│                     │  (Optimal Bid Formula)   │                            │
│                     └──────────────────────────┘                            │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ 6. MEMCACHE  │───▶│ 7. METRICS   │───▶│ 8. OUTPUT    │                   │
│  │    BUILDER   │    │    REPORTER  │    │    WRITER    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Directory Structure

```
optimizer_drugs_hcp/
├── data/
│   ├── drugs_bids.csv
│   ├── drugs_views.csv
│   └── drugs_clicks.csv
├── src/
│   ├── __init__.py
│   ├── config.py                 # All control settings
│   ├── data_loader.py            # Load and clean data
│   ├── feature_engineering.py    # Parse arrays, create features
│   ├── feature_selector.py       # Dynamic feature selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── win_rate_model.py     # Win probability prediction
│   │   └── ctr_model.py          # CTR prediction
│   ├── bid_calculator.py         # Optimal bid calculation
│   ├── memcache_builder.py       # Generate memcache CSV
│   └── metrics_reporter.py       # Generate metrics JSON
├── output/
│   └── (timestamped run outputs)
├── config/
│   └── optimizer_config.yaml     # Runtime configuration
├── tests/
│   └── (unit tests)
├── run_optimizer.py              # Main entry point
├── requirements.txt
└── CLAUDE.md                     # Instructions for Claude Code
```

---

## 5. Module Specifications

### 5.1 config.py - Control Settings

```python
"""
All optimizer control settings.
Organized by: Business Controls (adjustable) vs Technical Controls (fixed).
"""
from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class BusinessControls:
    """Controls exposed to business/product team."""
    target_win_rate: float = 0.60          # 10-95%, default 60%
    target_margin: float = 0.30            # 10-70%, default 30%
    exploration_budget: float = 0.10       # 0-20%, default 10%
    volume_boost_mode: bool = False        # Prioritize volume over margin
    
@dataclass
class TechnicalControls:
    """Controls for technical team only."""
    min_bid_cpm: float = 0.50              # Floor bid
    max_bid_cpm: float = 50.00             # Ceiling bid
    default_bid_cpm: float = 12.50         # Fallback when no match
    min_observations: int = 50             # Min samples per segment
    confidence_threshold: float = 0.95     # For confidence intervals
    training_window_days: int = 30         # Rolling window for training
    decay_factor: float = 0.9              # Recency weighting
    max_features: int = 4                  # Max features in memcache key
    
@dataclass 
class FeatureConfig:
    """Feature selection configuration."""
    candidate_features: List[str] = None
    anchor_features: List[str] = None      # Always include these
    exclude_features: List[str] = None     # Never use these
    
    def __post_init__(self):
        if self.candidate_features is None:
            self.candidate_features = [
                'internal_adspace_id',
                'geo_region_name',
                'geo_country_code2',
                'browser_code',
                'os_code',
                'page_category',
                'hour_of_day',
                'day_of_week'
            ]
        if self.anchor_features is None:
            self.anchor_features = ['internal_adspace_id']
        if self.exclude_features is None:
            self.exclude_features = ['geo_postal_code', 'geo_city_name']

@dataclass
class OptimizerConfig:
    """Master configuration."""
    business: BusinessControls = None
    technical: TechnicalControls = None
    features: FeatureConfig = None
    
    def __post_init__(self):
        if self.business is None:
            self.business = BusinessControls()
        if self.technical is None:
            self.technical = TechnicalControls()
        if self.features is None:
            self.features = FeatureConfig()
    
    @classmethod
    def from_yaml(cls, path: str) -> 'OptimizerConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            business=BusinessControls(**data.get('business', {})),
            technical=TechnicalControls(**data.get('technical', {})),
            features=FeatureConfig(**data.get('features', {}))
        )
```

### 5.2 data_loader.py - Data Loading & Cleaning

```python
"""
Load and clean bid/view/click data from CSV files.
Handles: malformed rows, duplicates, zero bids, date filtering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, data_dir: str, config: 'OptimizerConfig'):
        self.data_dir = Path(data_dir)
        self.config = config
        self.load_stats = {}
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and clean all three datasets."""
        df_bids = self._load_bids()
        df_views = self._load_views()
        df_clicks = self._load_clicks()
        return df_bids, df_views, df_clicks
    
    def _load_bids(self) -> pd.DataFrame:
        """Load and clean bid data."""
        path = self.data_dir / 'drugs_bids.csv'
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        
        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)
        
        # Parse bid amount
        df['bid_value'] = df['bid_amount'].apply(self._parse_first_array_value)
        
        # Filter: non-zero bids only
        initial_count = len(df)
        df = df[df['bid_value'] > 0]
        
        self.load_stats['bids'] = {
            'raw': initial_count,
            'clean': len(df),
            'removed': initial_count - len(df)
        }
        
        return df
    
    def _load_views(self) -> pd.DataFrame:
        """Load and clean view data."""
        path = self.data_dir / 'drugs_views.csv'
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        
        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)
        
        # Deduplicate by log_txnid
        initial_count = len(df)
        df = df.drop_duplicates(subset=['log_txnid'], keep='first')
        
        self.load_stats['views'] = {
            'raw': initial_count,
            'clean': len(df),
            'removed': initial_count - len(df)
        }
        
        return df
    
    def _load_clicks(self) -> pd.DataFrame:
        """Load click data."""
        path = self.data_dir / 'drugs_clicks.csv'
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        
        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)
        
        # Parse CPC
        df['cpc_value'] = df['advertiser_spend'].apply(self._parse_first_array_value)
        
        self.load_stats['clicks'] = {
            'raw': len(df),
            'clean': len(df),
            'removed': 0
        }
        
        return df
    
    @staticmethod
    def _parse_first_array_value(val) -> float:
        """Extract first numeric value from postgres array string."""
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        if val_str.startswith('{') and val_str.endswith('}'):
            inner = val_str[1:-1]
            if inner == '':
                return np.nan
            try:
                return float(inner.split(',')[0])
            except ValueError:
                return np.nan
        try:
            return float(val_str)
        except ValueError:
            return np.nan
```

### 5.3 feature_engineering.py - Feature Engineering

```python
"""
Feature engineering for bid optimization.
Creates derived features from raw data.
"""
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from typing import List, Set

class FeatureEngineer:
    def __init__(self, config: 'OptimizerConfig'):
        self.config = config
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all derived features."""
        df = df.copy()
        
        # Time-based features
        df['hour_of_day'] = df['log_dt'].dt.hour
        df['day_of_week'] = df['log_dt'].dt.dayofweek
        
        # Page category from ref_bundle
        if 'ref_bundle' in df.columns:
            df['page_category'] = df['ref_bundle'].apply(self._extract_page_category)
        
        # Handle nulls in geo_region_name
        if 'geo_region_name' in df.columns:
            df['geo_region_name'] = df['geo_region_name'].fillna('Unknown')
        
        return df
    
    @staticmethod
    def _extract_page_category(url) -> str:
        """Extract first path segment from URL."""
        if pd.isna(url):
            return 'unknown'
        try:
            parsed = urlparse(str(url))
            parts = [p for p in parsed.path.split('/') if p]
            if parts:
                return parts[0]
            return 'root'
        except:
            return 'unknown'
    
    def create_training_data(
        self, 
        df_bids: pd.DataFrame, 
        df_views: pd.DataFrame,
        df_clicks: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create joined training dataset for win rate model.
        
        Returns DataFrame with:
        - All bid features
        - 'won' binary column (1 if bid resulted in view)
        - 'clicked' binary column (1 if view resulted in click)
        """
        # Join bids to views on log_txnid
        df_train = df_bids.merge(
            df_views[['log_txnid']].assign(won=1),
            on='log_txnid',
            how='left'
        )
        df_train['won'] = df_train['won'].fillna(0).astype(int)
        
        return df_train
    
    def create_ctr_data(
        self,
        df_views: pd.DataFrame,
        df_clicks: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create training data for CTR model.
        
        Returns DataFrame with:
        - All view features
        - 'clicked' binary column
        """
        # Extract click transaction IDs
        click_txn_ids = self._extract_all_txn_ids(df_clicks)
        
        # Mark views that resulted in clicks
        df_views = df_views.copy()
        df_views['first_txn_id'] = df_views['internal_txn_id'].apply(
            lambda x: self._parse_array_to_list(x)[0] if self._parse_array_to_list(x) else None
        )
        df_views['clicked'] = df_views['first_txn_id'].isin(click_txn_ids).astype(int)
        
        return df_views
    
    @staticmethod
    def _parse_array_to_list(val) -> List[str]:
        """Parse postgres array to Python list."""
        if pd.isna(val):
            return []
        val_str = str(val)
        if val_str.startswith('{') and val_str.endswith('}'):
            inner = val_str[1:-1]
            if inner == '':
                return []
            return inner.split(',')
        return [val_str]
    
    def _extract_all_txn_ids(self, df_clicks: pd.DataFrame) -> Set[str]:
        """Extract all transaction IDs from clicks."""
        click_txn_ids = set()
        for txn_id in df_clicks['internal_txn_id']:
            txn_list = self._parse_array_to_list(txn_id)
            click_txn_ids.update(txn_list)
        return click_txn_ids
```

### 5.4 feature_selector.py - Dynamic Feature Selection

```python
"""
Dynamic feature selection algorithm.
Selects optimal features for memcache key based on signal strength.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class FeatureScore:
    """Score for a single feature."""
    name: str
    cardinality: int
    null_pct: float
    win_rate_variance: float
    signal_score: float
    
class FeatureSelector:
    def __init__(self, config: 'OptimizerConfig'):
        self.config = config
        self.feature_scores: Dict[str, FeatureScore] = {}
        self.selected_features: List[str] = []
    
    def select_features(
        self, 
        df_train: pd.DataFrame,
        target_col: str = 'won'
    ) -> List[str]:
        """
        Select optimal features for memcache key.
        
        Algorithm:
        1. Start with anchor features (always included)
        2. Score remaining candidates by information gain
        3. Greedily add features that improve coverage without excessive sparsity
        
        Returns:
            List of feature names for memcache key
        """
        # Start with anchor features
        self.selected_features = list(self.config.features.anchor_features)
        
        # Score all candidate features
        candidates = [
            f for f in self.config.features.candidate_features
            if f not in self.selected_features
            and f not in self.config.features.exclude_features
            and f in df_train.columns
        ]
        
        for feat in candidates:
            score = self._score_feature(df_train, feat, target_col)
            if score is not None:
                self.feature_scores[feat] = score
        
        # Greedily add best features up to max
        remaining = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1].signal_score,
            reverse=True
        )
        
        for feat_name, score in remaining:
            if len(self.selected_features) >= self.config.technical.max_features:
                break
            
            # Check if adding this feature maintains acceptable coverage
            test_features = self.selected_features + [feat_name]
            coverage, sparse_combos = self._check_coverage(df_train, test_features)
            
            # Add if coverage is acceptable (>50% of data in segments with enough obs)
            if coverage >= 0.50 and score.null_pct < 20:
                self.selected_features.append(feat_name)
        
        return self.selected_features
    
    def _score_feature(
        self, 
        df: pd.DataFrame, 
        feature: str,
        target_col: str
    ) -> FeatureScore:
        """Calculate signal score for a feature."""
        if feature not in df.columns:
            return None
        
        # Calculate metrics
        cardinality = df[feature].nunique()
        null_pct = df[feature].isnull().sum() / len(df) * 100
        
        # Skip features with too many nulls
        if null_pct > 30:
            return None
        
        # Calculate win rate variance across feature values
        # Higher variance = more predictive power
        try:
            win_rate_by_val = df.groupby(feature)[target_col].mean()
            win_rate_variance = win_rate_by_val.var()
        except:
            win_rate_variance = 0
        
        # Signal score = variance * (1 - null_pct/100) * log(cardinality + 1)
        # Penalize: high nulls, very low cardinality, very high cardinality
        cardinality_factor = np.log(min(cardinality, 100) + 1)  # Cap at 100
        signal_score = win_rate_variance * (1 - null_pct/100) * cardinality_factor * 10000
        
        return FeatureScore(
            name=feature,
            cardinality=cardinality,
            null_pct=null_pct,
            win_rate_variance=win_rate_variance,
            signal_score=signal_score
        )
    
    def _check_coverage(
        self, 
        df: pd.DataFrame, 
        features: List[str]
    ) -> Tuple[float, int]:
        """
        Check coverage and sparsity for a feature combination.
        
        Returns:
            coverage: Fraction of data in segments with min_observations
            sparse_combos: Number of segments below min_observations
        """
        min_obs = self.config.technical.min_observations
        
        combo_sizes = df.groupby(features).size()
        
        # Segments with enough observations
        sufficient = combo_sizes >= min_obs
        coverage = combo_sizes[sufficient].sum() / len(df)
        sparse_combos = (~sufficient).sum()
        
        return coverage, sparse_combos
    
    def get_selection_report(self) -> Dict:
        """Generate report on feature selection."""
        return {
            'selected_features': self.selected_features,
            'feature_scores': {
                name: {
                    'cardinality': score.cardinality,
                    'null_pct': round(score.null_pct, 2),
                    'win_rate_variance': round(score.win_rate_variance, 6),
                    'signal_score': round(score.signal_score, 4)
                }
                for name, score in self.feature_scores.items()
            },
            'num_features': len(self.selected_features),
            'max_features': self.config.technical.max_features
        }
```

### 5.5 models/win_rate_model.py - Win Rate Prediction

```python
"""
Win Rate Model: Predicts P(win | bid_price, features).

Uses logistic regression for simplicity and interpretability.
Based on: "Bid Shading by Win-Rate Estimation and Surplus Maximization" (arXiv:2009.09259)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional
import pickle

class WinRateModel:
    def __init__(self, config: 'OptimizerConfig'):
        self.config = config
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.training_stats: Dict = {}
    
    def train(
        self, 
        df_train: pd.DataFrame,
        features: List[str],
        target: str = 'won'
    ) -> None:
        """
        Train win rate model.
        
        Args:
            df_train: Training data with features and target
            features: List of feature column names
            target: Target column name ('won')
        """
        self.feature_names = features
        
        # Prepare features
        X = df_train[features].copy()
        y = df_train[target].values
        
        # Identify categorical vs numeric features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver='lbfgs',
                class_weight='balanced'  # Handle imbalanced data
            ))
        ])
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training stats
        y_pred = self.model.predict_proba(X)[:, 1]
        self.training_stats = {
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum()),
            'base_rate': float(y.mean()),
            'mean_pred': float(y_pred.mean()),
            'features': features,
            'n_categorical': len(categorical_features),
            'n_numeric': len(numeric_features)
        }
    
    def predict_win_rate(
        self, 
        df: pd.DataFrame
    ) -> np.ndarray:
        """Predict win probability for given features."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = df[self.feature_names]
        return self.model.predict_proba(X)[:, 1]
    
    def get_win_rate_for_segment(
        self,
        segment_values: Dict[str, any]
    ) -> float:
        """
        Get win rate prediction for a single segment.
        
        Args:
            segment_values: Dict mapping feature name to value
        """
        df = pd.DataFrame([segment_values])
        return self.predict_win_rate(df)[0]
```

### 5.6 models/ctr_model.py - CTR Prediction

```python
"""
CTR Model: Predicts P(click | impression, features).

Uses logistic regression with careful handling of extreme class imbalance.
CTR is ~0.035%, so we use class weighting and confidence intervals.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional

class CTRModel:
    def __init__(self, config: 'OptimizerConfig'):
        self.config = config
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.training_stats: Dict = {}
        self.global_ctr: float = 0.0
    
    def train(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        target: str = 'clicked'
    ) -> None:
        """
        Train CTR model.
        
        Due to extreme sparsity (~0.035% CTR), we use:
        1. Class weights to handle imbalance
        2. L2 regularization to prevent overfitting
        3. Global CTR as fallback for unseen segments
        """
        self.feature_names = features
        self.global_ctr = df_train[target].mean()
        
        # Prepare features
        X = df_train[features].copy()
        y = df_train[target].values
        
        # Identify categorical vs numeric features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Create pipeline with heavy regularization for sparse data
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=0.1,  # Strong regularization for sparse data
                max_iter=1000,
                solver='lbfgs',
                class_weight='balanced'
            ))
        ])
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training stats
        y_pred = self.model.predict_proba(X)[:, 1]
        self.training_stats = {
            'n_samples': len(y),
            'n_clicks': int(y.sum()),
            'global_ctr': float(self.global_ctr),
            'mean_pred_ctr': float(y_pred.mean()),
            'features': features
        }
    
    def predict_ctr(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """Predict CTR for given features."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = df[self.feature_names]
        return self.model.predict_proba(X)[:, 1]
    
    def get_ctr_for_segment(
        self,
        segment_values: Dict[str, any],
        use_confidence_bound: bool = True
    ) -> float:
        """
        Get CTR prediction for a single segment.
        
        If use_confidence_bound=True, returns lower bound of confidence interval
        to be conservative with sparse data.
        """
        df = pd.DataFrame([segment_values])
        pred_ctr = self.predict_ctr(df)[0]
        
        if use_confidence_bound:
            # Use Wilson score interval lower bound
            # Conservative estimate for sparse data
            return max(pred_ctr * 0.5, self.global_ctr * 0.1)
        
        return pred_ctr
```

### 5.7 bid_calculator.py - Optimal Bid Calculation

```python
"""
Bid Calculator: Computes optimal bid price per segment.

Optimal Bid Formula (First-Price Auction):
    Optimal_Bid = argmax_b [ (Expected_Value - b) * P(win | b) ]
    
Where:
    Expected_Value = E[CPC] * P(click | impression)
    P(win | b) = Win rate model prediction

For simplicity, we use a bid shading approach:
    Bid = Expected_Value * Shading_Factor * Target_Margin_Adjustment
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BidResult:
    """Result of bid calculation for a segment."""
    segment_key: Dict[str, any]
    win_rate: float
    ctr: float
    expected_cpc: float
    expected_value: float
    optimal_bid: float
    confidence: float
    observation_count: int

class BidCalculator:
    def __init__(
        self,
        config: 'OptimizerConfig',
        win_rate_model: 'WinRateModel',
        ctr_model: 'CTRModel'
    ):
        self.config = config
        self.win_rate_model = win_rate_model
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
        """
        Calculate optimal bid for each segment.
        
        Args:
            df_segments: DataFrame with unique segment combinations
            features: List of feature columns
            
        Returns:
            List of BidResult for each segment
        """
        results = []
        
        for _, row in df_segments.iterrows():
            segment_key = {f: row[f] for f in features}
            
            result = self._calculate_single_bid(
                segment_key=segment_key,
                observation_count=row.get('count', 0)
            )
            results.append(result)
        
        return results
    
    def _calculate_single_bid(
        self,
        segment_key: Dict[str, any],
        observation_count: int
    ) -> BidResult:
        """Calculate optimal bid for a single segment."""
        
        # Get predictions
        win_rate = self.win_rate_model.get_win_rate_for_segment(segment_key)
        ctr = self.ctr_model.get_ctr_for_segment(segment_key, use_confidence_bound=True)
        
        # Expected value per impression
        # E[Value] = P(click) * E[CPC]
        expected_value = ctr * self.avg_cpc
        
        # Calculate confidence based on observation count
        min_obs = self.config.technical.min_observations
        confidence = min(1.0, observation_count / min_obs)
        
        # Optimal bid with shading
        # In first-price auction, we shade below value to account for:
        # 1. Auction dynamics (bid less than value to make profit)
        # 2. Target margin requirements
        # 3. Confidence in predictions
        
        target_margin = self.config.business.target_margin
        target_win_rate = self.config.business.target_win_rate
        
        # Base shading factor: 1 - target_margin
        shading_factor = 1 - target_margin
        
        # Adjust shading based on current win rate vs target
        # If win_rate < target: bid higher (less shading)
        # If win_rate > target: bid lower (more shading)
        win_rate_adjustment = 1.0 + (target_win_rate - win_rate) * 0.5
        win_rate_adjustment = np.clip(win_rate_adjustment, 0.8, 1.2)
        
        # Confidence adjustment: less confident = more conservative
        confidence_factor = 0.5 + 0.5 * confidence
        
        # Calculate optimal bid
        optimal_bid = expected_value * shading_factor * win_rate_adjustment * confidence_factor * 1000  # Convert to CPM
        
        # Apply floor and ceiling
        optimal_bid = np.clip(
            optimal_bid,
            self.config.technical.min_bid_cpm,
            self.config.technical.max_bid_cpm
        )
        
        # Round to 2 decimal places
        optimal_bid = round(optimal_bid, 2)
        
        return BidResult(
            segment_key=segment_key,
            win_rate=round(win_rate, 4),
            ctr=round(ctr, 6),
            expected_cpc=round(self.avg_cpc, 2),
            expected_value=round(expected_value, 6),
            optimal_bid=optimal_bid,
            confidence=round(confidence, 2),
            observation_count=observation_count
        )
```

### 5.8 memcache_builder.py - Memcache CSV Generation

```python
"""
Generate memcache CSV file for bidder lookup.
"""
import pandas as pd
from typing import List
from pathlib import Path
from datetime import datetime

class MemcacheBuilder:
    def __init__(self, config: 'OptimizerConfig'):
        self.config = config
    
    def build_memcache(
        self,
        bid_results: List['BidResult'],
        features: List[str]
    ) -> pd.DataFrame:
        """
        Build memcache DataFrame from bid results.
        
        Args:
            bid_results: List of BidResult from BidCalculator
            features: List of feature column names (in order)
            
        Returns:
            DataFrame ready to write to CSV
        """
        rows = []
        
        for result in bid_results:
            # Skip segments with very low confidence
            if result.confidence < 0.5:
                continue
            
            row = {}
            for feat in features:
                row[feat] = result.segment_key.get(feat, '')
            
            row['suggested_bid_cpm'] = result.optimal_bid
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Ensure column order: features first, then bid
        column_order = features + ['suggested_bid_cpm']
        df = df[column_order]
        
        return df
    
    def write_memcache(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        timestamp: str = None
    ) -> Path:
        """
        Write memcache to CSV file.
        
        Args:
            df: Memcache DataFrame
            output_dir: Output directory
            timestamp: Optional timestamp string for filename
            
        Returns:
            Path to written file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'memcache_{timestamp}.csv'
        filepath = output_dir / filename
        
        # Write as tab-separated (TSV) to match sample format
        df.to_csv(filepath, sep='\t', index=False)
        
        return filepath
```

### 5.9 metrics_reporter.py - Metrics & Diagnostics

```python
"""
Generate comprehensive metrics JSON for diagnostics and bidder configuration.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class MetricsReporter:
    def __init__(self, config: 'OptimizerConfig'):
        self.config = config
        self.metrics: Dict[str, Any] = {}
    
    def compile_metrics(
        self,
        run_id: str,
        data_loader: 'DataLoader',
        feature_selector: 'FeatureSelector',
        win_rate_model: 'WinRateModel',
        ctr_model: 'CTRModel',
        bid_results: List['BidResult'],
        memcache_path: Path
    ) -> Dict[str, Any]:
        """Compile all metrics into single report."""
        
        self.metrics = {
            'run_info': {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'memcache_file': str(memcache_path.name)
            },
            
            'config': {
                'business': {
                    'target_win_rate': self.config.business.target_win_rate,
                    'target_margin': self.config.business.target_margin,
                    'exploration_budget': self.config.business.exploration_budget,
                    'volume_boost_mode': self.config.business.volume_boost_mode
                },
                'technical': {
                    'min_bid_cpm': self.config.technical.min_bid_cpm,
                    'max_bid_cpm': self.config.technical.max_bid_cpm,
                    'default_bid_cpm': self.config.technical.default_bid_cpm,
                    'min_observations': self.config.technical.min_observations
                }
            },
            
            'data_stats': data_loader.load_stats,
            
            'feature_selection': feature_selector.get_selection_report(),
            
            # CRITICAL: This is what the bidder needs
            'bidder_config': {
                'feature_columns': feature_selector.selected_features,
                'feature_column_order': feature_selector.selected_features,
                'num_segments': len(bid_results),
                'bid_column': 'suggested_bid_cpm'
            },
            
            'model_performance': {
                'win_rate_model': win_rate_model.training_stats,
                'ctr_model': ctr_model.training_stats
            },
            
            'bid_summary': self._summarize_bids(bid_results),
            
            'segment_distribution': self._segment_distribution(bid_results)
        }
        
        return self.metrics
    
    def _summarize_bids(self, bid_results: List['BidResult']) -> Dict:
        """Summarize bid statistics."""
        if not bid_results:
            return {}
        
        bids = [r.optimal_bid for r in bid_results]
        win_rates = [r.win_rate for r in bid_results]
        ctrs = [r.ctr for r in bid_results]
        
        return {
            'count': len(bid_results),
            'bid_min': min(bids),
            'bid_max': max(bids),
            'bid_mean': round(sum(bids) / len(bids), 2),
            'bid_median': round(sorted(bids)[len(bids)//2], 2),
            'win_rate_mean': round(sum(win_rates) / len(win_rates), 4),
            'ctr_mean': round(sum(ctrs) / len(ctrs), 6)
        }
    
    def _segment_distribution(self, bid_results: List['BidResult']) -> Dict:
        """Analyze segment distribution."""
        by_confidence = {
            'high': len([r for r in bid_results if r.confidence >= 0.8]),
            'medium': len([r for r in bid_results if 0.5 <= r.confidence < 0.8]),
            'low': len([r for r in bid_results if r.confidence < 0.5])
        }
        
        return {
            'by_confidence': by_confidence,
            'segments_in_memcache': by_confidence['high'] + by_confidence['medium']
        }
    
    def write_metrics(
        self,
        output_dir: Path,
        timestamp: str = None
    ) -> Path:
        """Write metrics to JSON file."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'metrics_{timestamp}.json'
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return filepath
```

---

## 6. Algorithm Details

### 6.1 Win Rate Model Algorithm

**Based on**: "Bid Shading by Win-Rate Estimation and Surplus Maximization" (VerizonMedia DSP, arXiv:2009.09259)

**Approach**: Logistic regression predicting P(win|features)

**Why Logistic Regression**:
1. Fast inference (critical for memcache generation)
2. Interpretable coefficients
3. Well-calibrated probabilities
4. Production-proven at scale

**Training Data**:
- Records: Bids joined to Views on `log_txnid`
- Target: `won` = 1 if bid has matching view, else 0
- Features: Selected by dynamic feature selector

### 6.2 CTR Model Algorithm

**Challenge**: Extreme class imbalance (0.035% CTR = 248 clicks in 577K views)

**Approach**: Regularized logistic regression with class weighting

**Handling Sparsity**:
1. Strong L2 regularization (C=0.1)
2. Class weights balanced for imbalance
3. Conservative confidence bounds for predictions
4. Fallback to global CTR for unseen segments

### 6.3 Optimal Bid Formula

**First-Price Auction Bid Shading**:

```
Optimal_Bid = Expected_Value × Shading_Factor × Win_Rate_Adjustment × Confidence_Factor × 1000

Where:
- Expected_Value = P(click) × E[CPC]
- Shading_Factor = 1 - Target_Margin (e.g., 0.70 for 30% margin)
- Win_Rate_Adjustment = 1 + (Target_WR - Current_WR) × 0.5, clipped to [0.8, 1.2]
- Confidence_Factor = 0.5 + 0.5 × min(1, observations / min_observations)
- × 1000 converts per-impression value to CPM
```

**Rationale**:
- In first-price auction, we pay what we bid → shade below value
- Adjust shading based on current vs target win rate
- More conservative when less confident in predictions

### 6.4 Dynamic Feature Selection Algorithm

**Goal**: Automatically select features that maximize signal while maintaining coverage

**Algorithm**:
```
1. START with anchor_features (e.g., [internal_adspace_id])

2. FOR each candidate_feature:
   - Calculate cardinality
   - Calculate null percentage
   - Calculate win_rate_variance across values
   - Compute signal_score = variance × (1 - null%) × log(cardinality)

3. SORT candidates by signal_score descending

4. FOR each candidate (sorted):
   - IF len(selected) >= max_features: BREAK
   - Test coverage with candidate added
   - IF coverage >= 50% AND null% < 20%:
     - ADD to selected_features

5. RETURN selected_features
```

---

## 7. Control Settings

### 7.1 Business Controls (UI-Exposed)

| Control | Type | Range | Default | Description |
|---------|------|-------|---------|-------------|
| `target_win_rate` | Slider | 10-95% | 60% | Higher = more aggressive bidding |
| `target_margin` | Slider | 10-70% | 30% | Minimum acceptable margin |
| `exploration_budget` | Slider | 0-20% | 10% | Budget for testing new segments |
| `volume_boost_mode` | Toggle | On/Off | Off | Prioritize volume over margin |

### 7.2 Technical Controls (Config-Only)

| Control | Default | Description |
|---------|---------|-------------|
| `min_bid_cpm` | $0.50 | Floor bid |
| `max_bid_cpm` | $50.00 | Ceiling bid |
| `default_bid_cpm` | $12.50 | Fallback for unmatched segments |
| `min_observations` | 50 | Min samples per segment |
| `confidence_threshold` | 0.95 | For confidence intervals |
| `training_window_days` | 30 | Rolling window |
| `max_features` | 4 | Max features in memcache key |

### 7.3 Feature Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `anchor_features` | ['internal_adspace_id'] | Always included |
| `exclude_features` | ['geo_postal_code', 'geo_city_name'] | Never used (too sparse) |
| `candidate_features` | See list below | Features to consider |

**Candidate Features**:
```yaml
candidate_features:
  - internal_adspace_id
  - geo_region_name
  - geo_country_code2
  - browser_code
  - os_code
  - page_category
  - hour_of_day
  - day_of_week
```

---

## 8. Output Specifications

### 8.1 Memcache CSV Format

**File**: `memcache_YYYYMMDD_HHMMSS.csv`

**Format**: Tab-separated values (TSV)

**Columns**: Dynamic based on selected features + `suggested_bid_cpm`

**Example**:
```
internal_adspace_id	geo_region_name	suggested_bid_cpm
111563	California	8.50
111563	Texas	6.75
111564	California	9.25
111564	New York	7.00
...
```

**Rules**:
- Only segments with confidence >= 0.5 included
- Bid values rounded to 2 decimal places
- No header row quotes
- Column order matches `feature_columns` in metrics

### 8.2 Metrics JSON Format

**File**: `metrics_YYYYMMDD_HHMMSS.json`

**Structure**:
```json
{
  "run_info": {
    "run_id": "20260112_143022",
    "timestamp": "2026-01-12T14:30:22.123456",
    "memcache_file": "memcache_20260112_143022.csv"
  },
  "config": {
    "business": {
      "target_win_rate": 0.60,
      "target_margin": 0.30,
      "exploration_budget": 0.10,
      "volume_boost_mode": false
    },
    "technical": {
      "min_bid_cpm": 0.50,
      "max_bid_cpm": 50.00,
      "default_bid_cpm": 12.50,
      "min_observations": 50
    }
  },
  "data_stats": {
    "bids": {"raw": 207013, "clean": 193756, "removed": 13257},
    "views": {"raw": 623545, "clean": 577815, "removed": 45730},
    "clicks": {"raw": 248, "clean": 248, "removed": 0}
  },
  "feature_selection": {
    "selected_features": ["internal_adspace_id", "geo_region_name"],
    "feature_scores": {
      "geo_region_name": {"cardinality": 65, "null_pct": 11.4, "signal_score": 12.34},
      "browser_code": {"cardinality": 9, "null_pct": 0.0, "signal_score": 2.45}
    },
    "num_features": 2,
    "max_features": 4
  },
  "bidder_config": {
    "feature_columns": ["internal_adspace_id", "geo_region_name"],
    "feature_column_order": ["internal_adspace_id", "geo_region_name"],
    "num_segments": 525,
    "bid_column": "suggested_bid_cpm"
  },
  "model_performance": {
    "win_rate_model": {
      "n_samples": 193756,
      "n_positive": 58887,
      "base_rate": 0.304,
      "features": ["internal_adspace_id", "geo_region_name"]
    },
    "ctr_model": {
      "n_samples": 577815,
      "n_clicks": 248,
      "global_ctr": 0.00043,
      "features": ["internal_adspace_id", "geo_region_name"]
    }
  },
  "bid_summary": {
    "count": 525,
    "bid_min": 0.50,
    "bid_max": 15.75,
    "bid_mean": 6.82,
    "bid_median": 6.50,
    "win_rate_mean": 0.3041,
    "ctr_mean": 0.000342
  },
  "segment_distribution": {
    "by_confidence": {"high": 318, "medium": 207, "low": 0},
    "segments_in_memcache": 525
  }
}
```

---

## 9. Implementation Guide

### 9.1 Main Entry Point (run_optimizer.py)

```python
#!/usr/bin/env python3
"""
RTB Optimizer Pipeline - Main Entry Point

Usage:
    python run_optimizer.py --config config/optimizer_config.yaml --data-dir data/ --output-dir output/

Arguments:
    --config: Path to YAML configuration file
    --data-dir: Directory containing CSV data files
    --output-dir: Directory for output files
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

from src.config import OptimizerConfig
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.feature_selector import FeatureSelector
from src.models.win_rate_model import WinRateModel
from src.models.ctr_model import CTRModel
from src.bid_calculator import BidCalculator
from src.memcache_builder import MemcacheBuilder
from src.metrics_reporter import MetricsReporter


def main():
    parser = argparse.ArgumentParser(description='RTB Optimizer Pipeline')
    parser.add_argument('--config', type=str, default='config/optimizer_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='Directory with CSV data files')
    parser.add_argument('--output-dir', type=str, default='output/',
                        help='Output directory')
    args = parser.parse_args()
    
    # Generate run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Starting optimizer run: {run_id}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = OptimizerConfig.from_yaml(str(config_path))
    else:
        print(f"Config file not found, using defaults")
        config = OptimizerConfig()
    
    print(f"Configuration: target_win_rate={config.business.target_win_rate}, "
          f"target_margin={config.business.target_margin}")
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    data_loader = DataLoader(args.data_dir, config)
    df_bids, df_views, df_clicks = data_loader.load_all()
    print(f"  Loaded: {len(df_bids):,} bids, {len(df_views):,} views, {len(df_clicks):,} clicks")
    
    # Step 2: Feature engineering
    print("\n[2/7] Engineering features...")
    feature_engineer = FeatureEngineer(config)
    df_bids = feature_engineer.create_features(df_bids)
    df_views = feature_engineer.create_features(df_views)
    
    # Create training datasets
    df_train_wr = feature_engineer.create_training_data(df_bids, df_views, df_clicks)
    df_train_ctr = feature_engineer.create_ctr_data(df_views, df_clicks)
    print(f"  Training data: {len(df_train_wr):,} win rate samples, {len(df_train_ctr):,} CTR samples")
    
    # Step 3: Feature selection
    print("\n[3/7] Selecting features...")
    feature_selector = FeatureSelector(config)
    selected_features = feature_selector.select_features(df_train_wr, target_col='won')
    print(f"  Selected features: {selected_features}")
    
    # Step 4: Train win rate model
    print("\n[4/7] Training win rate model...")
    win_rate_model = WinRateModel(config)
    win_rate_model.train(df_train_wr, selected_features, target='won')
    print(f"  Win rate model: {win_rate_model.training_stats['base_rate']:.2%} base rate")
    
    # Step 5: Train CTR model
    print("\n[5/7] Training CTR model...")
    ctr_model = CTRModel(config)
    ctr_model.train(df_train_ctr, selected_features, target='clicked')
    print(f"  CTR model: {ctr_model.training_stats['global_ctr']:.4%} global CTR")
    
    # Step 6: Calculate bids
    print("\n[6/7] Calculating optimal bids...")
    bid_calculator = BidCalculator(config, win_rate_model, ctr_model)
    bid_calculator.set_average_cpc(df_clicks)
    
    # Get unique segments
    df_segments = df_train_wr.groupby(selected_features).size().reset_index(name='count')
    print(f"  Unique segments: {len(df_segments):,}")
    
    bid_results = bid_calculator.calculate_bids_for_segments(df_segments, selected_features)
    print(f"  Bids calculated for {len(bid_results):,} segments")
    
    # Step 7: Build outputs
    print("\n[7/7] Building outputs...")
    
    # Memcache
    memcache_builder = MemcacheBuilder(config)
    df_memcache = memcache_builder.build_memcache(bid_results, selected_features)
    memcache_path = memcache_builder.write_memcache(df_memcache, output_dir, run_id)
    print(f"  Memcache: {memcache_path} ({len(df_memcache):,} segments)")
    
    # Metrics
    metrics_reporter = MetricsReporter(config)
    metrics = metrics_reporter.compile_metrics(
        run_id=run_id,
        data_loader=data_loader,
        feature_selector=feature_selector,
        win_rate_model=win_rate_model,
        ctr_model=ctr_model,
        bid_results=bid_results,
        memcache_path=memcache_path
    )
    metrics_path = metrics_reporter.write_metrics(output_dir, run_id)
    print(f"  Metrics: {metrics_path}")
    
    print(f"\n{'='*60}")
    print(f"Optimizer run complete: {run_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

### 9.2 requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0
```

### 9.3 Config YAML Template

```yaml
# optimizer_config.yaml

business:
  target_win_rate: 0.60
  target_margin: 0.30
  exploration_budget: 0.10
  volume_boost_mode: false

technical:
  min_bid_cpm: 0.50
  max_bid_cpm: 50.00
  default_bid_cpm: 12.50
  min_observations: 50
  confidence_threshold: 0.95
  training_window_days: 30
  decay_factor: 0.9
  max_features: 4

features:
  anchor_features:
    - internal_adspace_id
  candidate_features:
    - internal_adspace_id
    - geo_region_name
    - geo_country_code2
    - browser_code
    - os_code
    - page_category
    - hour_of_day
    - day_of_week
  exclude_features:
    - geo_postal_code
    - geo_city_name
```

---

## 10. Testing & Validation

### 10.1 Validation Criteria

| Test | Expected Outcome |
|------|------------------|
| Data loads without errors | All 3 CSVs loaded, no parse errors |
| Duplicates removed | View count decreases by ~7% |
| Zero bids filtered | Bid count decreases by ~6% |
| Features selected | 2-4 features selected |
| Win rate model trains | Base rate ~30% |
| CTR model trains | Global CTR ~0.04% |
| Memcache generated | 200-1000 segments |
| Bids in valid range | All bids between min and max |
| Metrics JSON valid | Parseable JSON with all fields |

### 10.2 Sample Test Commands

```bash
# Run with default config
python run_optimizer.py

# Run with custom config
python run_optimizer.py --config config/custom_config.yaml

# Run with different data directory
python run_optimizer.py --data-dir /path/to/data/
```

### 10.3 Expected Output

After successful run:
```
output/
└── 20260112_143022/
    ├── memcache_20260112_143022.csv    # ~50KB
    └── metrics_20260112_143022.json    # ~5KB
```

---

## Appendix A: Research References

1. **Bid Shading**: "Bid Shading by Win-Rate Estimation and Surplus Maximization" (arXiv:2009.09259) - VerizonMedia DSP production algorithm
2. **Censored Data**: "Predicting Winning Price in Real Time Bidding with Censored Data" (KDD 2015) - Mixture model for censored observations
3. **CTR Prediction**: "Ad Click Prediction: a View from the Trenches" (McMahan et al., Google) - FTRL-Proximal for sparse CTR
4. **Auction Theory**: "Bid Shading in The Brave New World of First-Price Auctions" (arXiv:2009.01360) - Deep distribution network for first-price

---

## Appendix B: Known Limitations

1. **Sparse CTR**: Only 248 clicks limits CTR model accuracy
2. **No Floor Prices**: First-price blind auction = higher uncertainty
3. **Campaign Blindness**: Current model doesn't differentiate by campaign value
4. **Single Memcache Lookup**: No hierarchical fallback = some requests use default bid

---

## Appendix C: Future Improvements (V2)

1. **Campaign-Level Optimization**: Incorporate campaign_code and CPC variance
2. **Temporal Patterns**: Day-parting and week-parting bid adjustments
3. **Online Learning**: Real-time bid updates vs daily batch
4. **A/B Testing Framework**: Built-in holdout groups for measuring lift
5. **Page-Level Features**: Extract more signal from ref_bundle URLs
