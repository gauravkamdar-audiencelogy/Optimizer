"""
NPI Value Model: Maps National Provider Identifiers to bid multipliers.

V5 Addition: Use NPI data to adjust bids for high-value prescribers.

The external_userid field contains NPI numbers. This model:
1. Loads NPI click/revenue data from CSV files
2. Assigns tiers based on revenue percentiles
3. Applies recency boost for recent clickers
4. Returns bid multipliers (up to 3.0x for top prescribers)

Data sources:
- 1-year click data: Historical RPU (revenue per user)
- 20-day click data: Recent activity signal

Tier definitions (percentile-based on 1-year RPU):
- Tier 1 (Elite): Top 1% (RPU > $72) → 2.5x base
- Tier 2 (High): Top 5% (RPU > $32) → 1.8x base
- Tier 3 (Medium): Top 20% (RPU > $12) → 1.3x base
- Tier 4 (Standard): Rest → 1.0x base

Recency boost: +20% if NPI clicked in last 20 days
Final multiplier capped at max_multiplier (default 3.0x)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class NPIValueModel:
    """
    NPI-based bid multiplier model.

    Maps NPI → bid multiplier based on prescriber value and recency.
    """

    # Tier multipliers (percentile-based)
    TIER_MULTIPLIERS = {
        1: 2.50,  # Elite (Top 1%): 2.5x base
        2: 1.80,  # High (Top 5%): 1.8x base
        3: 1.30,  # Medium (Top 20%): 1.3x base
        4: 1.00,  # Standard: baseline
    }

    # Default percentile thresholds (can be overridden by config)
    DEFAULT_PERCENTILES = {
        1: 99,   # Top 1%
        2: 95,   # Top 5%
        3: 80,   # Top 20%
    }

    def __init__(self, max_multiplier: float = 3.0, recency_boost: float = 1.2):
        self.profiles: Dict[str, Dict] = {}
        self.is_loaded: bool = False
        self.load_stats: Dict = {}
        self.max_multiplier = max_multiplier
        self.recency_boost = recency_boost
        self.percentile_thresholds: Dict[int, float] = {}  # tier -> RPU threshold

    @classmethod
    def from_csv(cls, path: str) -> 'NPIValueModel':
        """
        Load NPI profiles from CSV file.

        Args:
            path: Path to CSV file with NPI data

        Returns:
            Initialized NPIValueModel
        """
        model = cls()
        csv_path = Path(path)

        if not csv_path.exists():
            print(f"    WARNING: NPI data file not found at {path}")
            print(f"    NPI model will return default multipliers (1.0)")
            model.load_stats = {
                'loaded': False,
                'reason': f'File not found: {path}',
                'profiles_count': 0
            }
            return model

        try:
            df = pd.read_csv(csv_path)
            model._load_from_dataframe(df)
            print(f"    Loaded {len(model.profiles):,} NPI profiles from {path}")
            model.load_stats = {
                'loaded': True,
                'path': str(path),
                'profiles_count': len(model.profiles)
            }
        except Exception as e:
            print(f"    WARNING: Failed to load NPI data: {e}")
            print(f"    NPI model will return default multipliers (1.0)")
            model.load_stats = {
                'loaded': False,
                'reason': str(e),
                'profiles_count': 0
            }

        return model

    @classmethod
    def from_click_data(
        cls,
        path_1year: str,
        path_20day: str = None,
        max_multiplier: float = 3.0,
        recency_boost: float = 1.2
    ) -> 'NPIValueModel':
        """
        Load NPI profiles from click/revenue data files.

        Args:
            path_1year: Path to 1-year click data CSV (external_userid, rpu, count)
            path_20day: Path to 20-day click data CSV (optional, for recency)
            max_multiplier: Maximum allowed multiplier (default 3.0)
            recency_boost: Boost factor for recent clickers (default 1.2 = +20%)

        Returns:
            Initialized NPIValueModel with tier assignments
        """
        model = cls(max_multiplier=max_multiplier, recency_boost=recency_boost)

        # Load 1-year data
        path_1y = Path(path_1year)
        if not path_1y.exists():
            print(f"    WARNING: 1-year NPI data not found at {path_1year}")
            model.load_stats = {'loaded': False, 'reason': f'File not found: {path_1year}'}
            return model

        try:
            df_1y = pd.read_csv(path_1y)
            print(f"    Loading 1-year NPI data from {path_1year}...")

            # Filter to valid 10-digit NPIs
            df_1y['external_userid'] = df_1y['external_userid'].astype(str)
            df_1y = df_1y[df_1y['external_userid'].str.len() == 10].copy()
            print(f"    Valid 10-digit NPIs: {len(df_1y):,}")

            # Load 20-day data for recency
            recent_npis = set()
            recent_rpu = {}
            if path_20day:
                path_20d = Path(path_20day)
                if path_20d.exists():
                    df_20d = pd.read_csv(path_20d)
                    df_20d['external_userid'] = df_20d['external_userid'].astype(str)
                    recent_npis = set(df_20d['external_userid'])
                    recent_rpu = dict(zip(df_20d['external_userid'], df_20d['rpu']))
                    print(f"    Recent clickers (20-day): {len(recent_npis):,}")

            # Calculate percentile thresholds from 1-year RPU
            rpu_values = df_1y['rpu'].values
            model.percentile_thresholds = {
                1: np.percentile(rpu_values, 99),   # Top 1%
                2: np.percentile(rpu_values, 95),   # Top 5%
                3: np.percentile(rpu_values, 80),   # Top 20%
            }
            print(f"    RPU thresholds: Tier1>${model.percentile_thresholds[1]:.2f}, "
                  f"Tier2>${model.percentile_thresholds[2]:.2f}, "
                  f"Tier3>${model.percentile_thresholds[3]:.2f}")

            # Assign tiers and build profiles
            tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            recency_count = 0

            for _, row in df_1y.iterrows():
                npi = row['external_userid']
                rpu_1y = float(row['rpu'])

                # Assign tier based on RPU percentile
                if rpu_1y >= model.percentile_thresholds[1]:
                    tier = 1
                elif rpu_1y >= model.percentile_thresholds[2]:
                    tier = 2
                elif rpu_1y >= model.percentile_thresholds[3]:
                    tier = 3
                else:
                    tier = 4

                tier_counts[tier] += 1

                # Check recency
                is_recent = npi in recent_npis
                if is_recent:
                    recency_count += 1

                # Calculate multiplier
                base_mult = cls.TIER_MULTIPLIERS.get(tier, 1.0)
                if is_recent:
                    final_mult = min(base_mult * recency_boost, max_multiplier)
                else:
                    final_mult = base_mult

                model.profiles[npi] = {
                    'tier': tier,
                    'rpu_1year': rpu_1y,
                    'rpu_20day': recent_rpu.get(npi),
                    'is_recent': is_recent,
                    'multiplier': round(final_mult, 2),
                }

            model.is_loaded = True
            model.load_stats = {
                'loaded': True,
                'path_1year': str(path_1year),
                'path_20day': str(path_20day) if path_20day else None,
                'total_npis': len(model.profiles),
                'tier_counts': tier_counts,
                'recent_clickers': recency_count,
                'percentile_thresholds': {k: round(v, 2) for k, v in model.percentile_thresholds.items()},
                'max_multiplier': max_multiplier,
                'recency_boost': recency_boost,
            }

            print(f"    Tier distribution: {tier_counts}")
            print(f"    Recent clickers with boost: {recency_count:,}")

        except Exception as e:
            print(f"    WARNING: Failed to load NPI click data: {e}")
            import traceback
            traceback.print_exc()
            model.load_stats = {'loaded': False, 'reason': str(e)}

        return model

    def _load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load profiles from a DataFrame."""
        required_cols = ['npi']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for _, row in df.iterrows():
            npi = str(row['npi'])
            self.profiles[npi] = {
                'tier': int(row.get('tier', 3)),
                'revenue_likelihood': float(row.get('revenue_likelihood', 0.5)),
                'specialty': row.get('specialty', 'Unknown')
            }

        self.is_loaded = True

        # Compute stats
        if self.profiles:
            tier_counts = {}
            for profile in self.profiles.values():
                tier = profile['tier']
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            print(f"    NPI tier distribution: {tier_counts}")

    def get_value_multiplier(self, npi: Optional[str]) -> float:
        """
        Get bid multiplier for an NPI.

        Args:
            npi: National Provider Identifier (string or None)

        Returns:
            Bid multiplier (default 1.0 if unknown)
        """
        if npi is None or not self.is_loaded:
            return 1.0

        npi_str = str(npi).strip()

        if npi_str not in self.profiles:
            return 1.0

        profile = self.profiles[npi_str]

        # Return pre-computed multiplier (includes tier + recency)
        return profile.get('multiplier', 1.0)

    def get_profile(self, npi: str) -> Optional[Dict]:
        """Get full profile for an NPI (for debugging)."""
        npi_str = str(npi).strip()
        return self.profiles.get(npi_str)

    def get_tier_stats(self) -> Dict:
        """Get statistics about loaded NPI tiers."""
        if not self.is_loaded:
            return {'loaded': False}

        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        recent_count = 0
        multiplier_sum = 0.0

        for profile in self.profiles.values():
            tier = profile.get('tier', 4)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if profile.get('is_recent', False):
                recent_count += 1
            multiplier_sum += profile.get('multiplier', 1.0)

        avg_multiplier = multiplier_sum / len(self.profiles) if self.profiles else 1.0

        return {
            'loaded': True,
            'total_profiles': len(self.profiles),
            'tier_counts': tier_counts,
            'recent_clickers': recent_count,
            'avg_multiplier': round(avg_multiplier, 3),
            'percentile_thresholds': self.percentile_thresholds,
        }

    def get_all_profiles_df(self) -> pd.DataFrame:
        """
        Get all NPI profiles as a DataFrame for export.

        Returns:
            DataFrame with columns: external_userid, multiplier, tier, is_recent, rpu_1year, rpu_20day
        """
        if not self.is_loaded or not self.profiles:
            return pd.DataFrame()

        rows = []
        for npi, profile in self.profiles.items():
            rows.append({
                'external_userid': npi,
                'multiplier': profile.get('multiplier', 1.0),
                'tier': profile.get('tier', 4),
                'is_recent': profile.get('is_recent', False),
                'rpu_1year': profile.get('rpu_1year', 0),
                'rpu_20day': profile.get('rpu_20day'),
            })

        df = pd.DataFrame(rows)
        # Sort by multiplier descending (highest value first)
        df = df.sort_values('multiplier', ascending=False)
        return df


def create_sample_npi_file(path: str, n_profiles: int = 1000) -> None:
    """
    Create a sample NPI data file for testing.

    Args:
        path: Output file path
        n_profiles: Number of sample profiles to generate
    """
    import random

    specialties = [
        'Oncology', 'Cardiology', 'Neurology', 'Pulmonology',
        'Rheumatology', 'Endocrinology', 'Dermatology', 'General Practice'
    ]

    rows = []
    for i in range(n_profiles):
        # Generate fake NPI (10-digit)
        npi = f"{1000000000 + i}"

        # Assign tier (weighted: fewer high-value)
        tier = random.choices([1, 2, 3], weights=[0.1, 0.3, 0.6])[0]

        # Revenue likelihood correlates with tier
        if tier == 1:
            likelihood = random.uniform(0.6, 0.95)
        elif tier == 2:
            likelihood = random.uniform(0.4, 0.7)
        else:
            likelihood = random.uniform(0.2, 0.5)

        specialty = random.choice(specialties)

        rows.append({
            'npi': npi,
            'tier': tier,
            'revenue_likelihood': round(likelihood, 2),
            'specialty': specialty
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Created sample NPI file with {n_profiles} profiles at {path}")
