"""
NPI Value Model: Maps National Provider Identifiers to bid multipliers.

V5 Addition: Use NPI data to adjust bids for high-value prescribers.

The external_userid field contains NPI numbers. This model:
1. Loads NPI value data from CSV (if available)
2. Returns bid multipliers based on prescriber tier/value
3. Defaults to 1.0 for unknown NPIs

Expected CSV format:
    npi,tier,revenue_likelihood,specialty
    1234567890,1,0.85,Oncology
    0987654321,2,0.65,Cardiology

Tier definitions:
- Tier 1: High-volume prescribers (1.5x multiplier)
- Tier 2: Medium-volume prescribers (1.2x multiplier)
- Tier 3: Low-volume prescribers (1.0x multiplier)

Additional adjustment for revenue likelihood (0-1 scale):
- likelihood > 0.7 → +20% boost
- likelihood 0.5-0.7 → no change
- likelihood < 0.5 → -10% reduction
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class NPIValueModel:
    """
    NPI-based bid multiplier model.

    Maps NPI → bid multiplier based on prescriber value.
    """

    # Tier multipliers
    TIER_MULTIPLIERS = {
        1: 1.50,  # High-value: 50% premium
        2: 1.20,  # Medium-value: 20% premium
        3: 1.00,  # Low-value: baseline
    }

    def __init__(self):
        self.profiles: Dict[str, Dict] = {}
        self.is_loaded: bool = False
        self.load_stats: Dict = {}

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

        # Base multiplier from tier
        tier = profile.get('tier', 3)
        tier_mult = self.TIER_MULTIPLIERS.get(tier, 1.0)

        # Adjust based on revenue likelihood
        likelihood = profile.get('revenue_likelihood', 0.5)
        if likelihood > 0.7:
            likelihood_mult = 1.20  # High likelihood: +20%
        elif likelihood < 0.5:
            likelihood_mult = 0.90  # Low likelihood: -10%
        else:
            likelihood_mult = 1.00  # Medium: no change

        # Combined multiplier
        multiplier = tier_mult * likelihood_mult

        # Cap at reasonable bounds
        multiplier = np.clip(multiplier, 0.5, 2.0)

        return float(round(multiplier, 2))

    def get_profile(self, npi: str) -> Optional[Dict]:
        """Get full profile for an NPI (for debugging)."""
        npi_str = str(npi).strip()
        return self.profiles.get(npi_str)

    def get_tier_stats(self) -> Dict:
        """Get statistics about loaded NPI tiers."""
        if not self.is_loaded:
            return {'loaded': False}

        tier_counts = {}
        likelihood_sum = 0.0

        for profile in self.profiles.values():
            tier = profile.get('tier', 3)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            likelihood_sum += profile.get('revenue_likelihood', 0.5)

        avg_likelihood = likelihood_sum / len(self.profiles) if self.profiles else 0.5

        return {
            'loaded': True,
            'total_profiles': len(self.profiles),
            'tier_counts': tier_counts,
            'avg_revenue_likelihood': round(avg_likelihood, 3)
        }


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
