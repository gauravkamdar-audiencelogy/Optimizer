"""
Domain Value Model: Maps domains to bid multipliers.

Similar to NPI model, this assigns domains to performance tiers
based on historical CTR (click-through rate) with Bayesian shrinkage.

Architecture:
- Memcache: segment → base_bid
- Domain:   domain → multiplier (NEW)
- Bidder:   final_bid = base_bid × domain_multiplier × npi_multiplier

Tiering Methods:

1. IQR (Hierarchical Interquartile Range) - Recommended
   Uses data-derived thresholds based on statistical outlier detection.
   43x more stable than percentiles (0.05% vs 2.13% flip-flop rate).

   Tier definitions (5 tiers):
   - Extreme Stars: > Q3 + 3.0×IQR  → 1.50x base (~3%)
   - Stars:         > Q3 + 1.5×IQR  → 1.30x base (~4%)
   - Cream:         > Middle_Q3 + 1.5×Middle_IQR → 1.15x base (~2%)
   - Baseline:      > poor_threshold → 1.00x base (~70%)
   - Poor:          ≤ poor_threshold → 0.60x base (~21%)
   - Blocklist:     < global × 0.1 (severe) → 0.00x (varies)

2. Percentile (Legacy) - Original approach
   Uses fixed percentile cutoffs.

   Tier definitions (percentile-based on shrunk CTR):
   - Premium:      Top 5%  → 1.3x base
   - Standard:     Top 50% → 1.0x base
   - Below Avg:    Top 90% → 0.8x base
   - Poor:         Bottom 10% → 0.5x base
   - Blocklist:    CTR < threshold → 0.0x (don't bid)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from collections import defaultdict


class DomainValueModel:
    """
    Domain-based bid multiplier model.

    Maps domain → bid multiplier based on historical CTR performance.
    Uses Bayesian shrinkage to handle sparse domains.
    """

    def __init__(self, config):
        """
        Initialize domain model.

        Args:
            config: OptimizerConfig with domain settings
        """
        self.config = config
        self.domain_config = config.domain
        self.profiles: Dict[str, Dict] = {}
        self.is_loaded: bool = False
        self.training_stats: Dict = {}
        self.percentile_thresholds: Dict[str, float] = {}
        self.iqr_thresholds: Dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> None:
        """
        Train domain model from bid/view/click data.

        Expects df to have:
        - domain: Domain name
        - won: 1 if bid won (view occurred), 0 otherwise
        - clicked: 1 if click occurred, 0 otherwise (optional)

        Args:
            df: DataFrame with domain, won, clicked columns
        """
        if 'domain' not in df.columns:
            print("    WARNING: 'domain' column not found, domain model will be empty")
            self.training_stats = {'loaded': False, 'reason': 'domain column missing'}
            return

        # Use clicked if available, otherwise fall back to won
        if 'clicked' in df.columns:
            signal_col = 'clicked'
            signal_name = 'CTR'
        else:
            signal_col = 'won'
            signal_name = 'win_rate'

        print(f"    Training domain model using {signal_name}...")

        # Aggregate domain stats
        domain_stats = df.groupby('domain').agg({
            'won': ['sum', 'count'],  # views (wins) and bids
        }).reset_index()
        domain_stats.columns = ['domain', 'views', 'bids']

        # Add clicks if available
        if 'clicked' in df.columns:
            click_stats = df[df['won'] == 1].groupby('domain')['clicked'].sum().reset_index()
            click_stats.columns = ['domain', 'clicks']
            domain_stats = domain_stats.merge(click_stats, on='domain', how='left')
            domain_stats['clicks'] = domain_stats['clicks'].fillna(0).astype(int)
        else:
            domain_stats['clicks'] = domain_stats['views']  # Use views as proxy

        print(f"    Unique domains: {len(domain_stats):,}")

        # Calculate global rate
        if signal_col == 'clicked':
            total_views = domain_stats['views'].sum()
            total_clicks = domain_stats['clicks'].sum()
            global_rate = total_clicks / total_views if total_views > 0 else 0
        else:
            total_bids = domain_stats['bids'].sum()
            total_views = domain_stats['views'].sum()
            global_rate = total_views / total_bids if total_bids > 0 else 0

        print(f"    Global {signal_name}: {global_rate:.4%}")

        # Apply Bayesian shrinkage
        k = self.domain_config.shrinkage_k
        if signal_col == 'clicked':
            domain_stats['shrunk_rate'] = (
                (domain_stats['clicks'] + k * global_rate) /
                (domain_stats['views'] + k)
            )
        else:
            domain_stats['shrunk_rate'] = (
                (domain_stats['views'] + k * global_rate) /
                (domain_stats['bids'] + k)
            )

        # Branch on tiering method
        tiering_method = getattr(self.domain_config, 'tiering_method', 'percentile')
        if tiering_method == 'iqr':
            self._train_iqr(domain_stats, global_rate, signal_name, k)
        else:
            self._train_percentile(domain_stats, global_rate, signal_name, k)

    def _compute_iqr_thresholds(self, rates: np.ndarray) -> dict:
        """
        Compute IQR-based outlier thresholds.

        Args:
            rates: Array of shrunk rates for domains with adequate data

        Returns:
            Dict with Q1, Q3, IQR, and computed thresholds
        """
        Q1 = np.percentile(rates, 25)
        Q3 = np.percentile(rates, 75)
        IQR = Q3 - Q1

        return {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'stars_threshold': Q3 + self.domain_config.iqr_multiplier_stars * IQR,
            'extreme_threshold': Q3 + self.domain_config.iqr_multiplier_extreme * IQR,
        }

    def _compute_cream_threshold(self, middle_rates: np.ndarray, stars_threshold: float) -> float:
        """
        Compute recursive IQR threshold for cream tier.

        The cream tier captures the "best of the middle" - domains that aren't
        statistical outliers but are still performing above average.

        Args:
            middle_rates: Rates for domains not in stars/extreme_stars tiers
            stars_threshold: The stars tier threshold (cream must be below this)

        Returns:
            Cream threshold (or infinity if insufficient data)
        """
        if len(middle_rates) < 10:
            return float('inf')  # Not enough data for cream tier

        M_Q1 = np.percentile(middle_rates, 25)
        M_Q3 = np.percentile(middle_rates, 75)
        M_IQR = M_Q3 - M_Q1

        raw_cream = M_Q3 + self.domain_config.iqr_multiplier_stars * M_IQR

        # CRITICAL: Ensure cream_threshold < stars_threshold
        # This can happen with discrete data where IQR is small
        # Use 95% of stars_threshold as upper bound to guarantee separation
        max_cream = stars_threshold * 0.95
        if raw_cream >= stars_threshold:
            return max_cream

        return raw_cream

    def _assign_tier_iqr(self, rate: float, thresholds: dict) -> tuple:
        """
        Assign tier using IQR method.

        Args:
            rate: Domain's shrunk rate
            thresholds: Dict with all computed thresholds

        Returns:
            Tuple of (tier_name, multiplier)
        """
        if rate > thresholds['extreme_threshold']:
            return 'extreme_stars', self.domain_config.multiplier_extreme_stars
        elif rate > thresholds['stars_threshold']:
            return 'stars', self.domain_config.multiplier_stars
        elif rate > thresholds['cream_threshold']:
            return 'cream', self.domain_config.multiplier_cream
        elif rate > thresholds['poor_threshold']:
            return 'baseline', self.domain_config.multiplier_baseline
        else:
            return 'poor', self.domain_config.multiplier_poor_iqr

    def _train_iqr(self, domain_stats: pd.DataFrame, global_rate: float,
                   signal_name: str, shrinkage_k: int) -> None:
        """
        Train domain model using Hierarchical IQR tiering.

        This method is more stable than percentile-based tiering because:
        - Thresholds are derived from data distribution (Q1, Q3, IQR)
        - Small data changes don't shift tier boundaries
        - Captures statistical outliers vs arbitrary percentile cuts

        Args:
            domain_stats: DataFrame with domain, bids, views, shrunk_rate
            global_rate: Global win/click rate across all domains
            signal_name: Name of signal being used (for logging)
            shrinkage_k: Shrinkage parameter used
        """
        # Filter to domains with enough observations
        min_bids = self.domain_config.min_bids_for_tiering
        adequate_data = domain_stats[domain_stats['bids'] >= min_bids].copy()
        sparse_data = domain_stats[domain_stats['bids'] < min_bids].copy()

        print(f"    Domains with >= {min_bids} bids: {len(adequate_data):,}")
        print(f"    Domains with < {min_bids} bids (sparse): {len(sparse_data):,}")

        if len(adequate_data) < 10:
            print("    WARNING: Not enough domains with adequate data for IQR tiering")
            print("    Falling back to percentile method")
            self._train_percentile(domain_stats, global_rate, signal_name, shrinkage_k)
            return

        rates = adequate_data['shrunk_rate'].values

        # Step 1: Compute IQR thresholds
        iqr_thresholds = self._compute_iqr_thresholds(rates)

        # Step 2: Identify middle segment and compute cream threshold
        middle_mask = rates <= iqr_thresholds['stars_threshold']
        middle_rates = rates[middle_mask]
        cream_threshold = self._compute_cream_threshold(middle_rates, iqr_thresholds['stars_threshold'])

        # Step 3: Poor threshold (fraction of global rate)
        poor_threshold = global_rate * self.domain_config.poor_rate_factor

        # Step 4: Blocklist threshold (severe underperformance)
        blocklist_rate_factor = getattr(self.domain_config, 'blocklist_rate_factor',
                                        self.domain_config.blocklist_ctr_factor)
        blocklist_threshold = global_rate * blocklist_rate_factor

        # Combine all thresholds
        all_thresholds = {
            **iqr_thresholds,
            'cream_threshold': cream_threshold,
            'poor_threshold': poor_threshold,
            'blocklist_threshold': blocklist_threshold,
        }

        self.iqr_thresholds = all_thresholds

        print(f"    IQR Thresholds: "
              f"Extreme>{all_thresholds['extreme_threshold']:.4%}, "
              f"Stars>{all_thresholds['stars_threshold']:.4%}, "
              f"Cream>{cream_threshold:.4%}, "
              f"Poor<{poor_threshold:.4%}")

        # Step 5: Assign tiers
        tier_counts = defaultdict(int)
        min_bids_for_blocklist = self.domain_config.min_bids_for_blocklist

        # Domains with adequate data
        for _, row in adequate_data.iterrows():
            domain = row['domain']
            shrunk_rate = row['shrunk_rate']
            bids = int(row['bids'])
            views = int(row['views'])
            clicks = int(row.get('clicks', views))

            tier, multiplier = self._assign_tier_iqr(shrunk_rate, all_thresholds)

            # Check blocklist override (severe underperformance with enough data)
            if (bids >= min_bids_for_blocklist and
                shrunk_rate < blocklist_threshold):
                tier = 'blocklist'
                multiplier = self.domain_config.multiplier_blocklist

            tier_counts[tier] += 1
            self.profiles[domain] = {
                'tier': tier,
                'multiplier': round(multiplier, 2),
                'bids': bids,
                'views': views,
                'clicks': clicks,
                'shrunk_rate': round(shrunk_rate, 6),
            }

        # Domains with sparse data → baseline (insufficient data to tier)
        for _, row in sparse_data.iterrows():
            domain = row['domain']
            tier_counts['insufficient_data'] += 1
            self.profiles[domain] = {
                'tier': 'insufficient_data',
                'multiplier': 1.0,  # Default to baseline
                'bids': int(row['bids']),
                'views': int(row['views']),
                'clicks': int(row.get('clicks', row['views'])),
                'shrunk_rate': round(row['shrunk_rate'], 6),
            }

        self.is_loaded = True
        self.training_stats = {
            'loaded': True,
            'method': 'iqr',
            'total_domains': len(self.profiles),
            'tier_counts': dict(tier_counts),
            'global_rate': global_rate,
            'signal_used': signal_name,
            'shrinkage_k': shrinkage_k,
            'iqr_thresholds': {k: float(v) for k, v in all_thresholds.items()},
            'domains_tiered': len(adequate_data),
            'domains_sparse': len(sparse_data),
        }

        print(f"    Tier distribution: {dict(tier_counts)}")

    def _train_percentile(self, domain_stats: pd.DataFrame, global_rate: float,
                          signal_name: str, shrinkage_k: int) -> None:
        """
        Train domain model using legacy percentile-based tiering.

        Args:
            domain_stats: DataFrame with domain, bids, views, shrunk_rate
            global_rate: Global win/click rate across all domains
            signal_name: Name of signal being used (for logging)
            shrinkage_k: Shrinkage parameter used
        """
        # Calculate percentile thresholds
        rates = domain_stats['shrunk_rate'].values
        self.percentile_thresholds = {
            'premium': np.percentile(rates, self.domain_config.premium_percentile),
            'standard': np.percentile(rates, self.domain_config.standard_percentile),
            'below_avg': np.percentile(rates, self.domain_config.below_avg_percentile),
        }

        print(f"    {signal_name} thresholds: "
              f"Premium>={self.percentile_thresholds['premium']:.4%}, "
              f"Standard>={self.percentile_thresholds['standard']:.4%}, "
              f"BelowAvg>={self.percentile_thresholds['below_avg']:.4%}")

        # Blocklist threshold
        blocklist_threshold = global_rate * self.domain_config.blocklist_ctr_factor
        min_bids_for_blocklist = self.domain_config.min_bids_for_blocklist

        # Assign tiers and multipliers
        tier_counts = {'premium': 0, 'standard': 0, 'below_avg': 0, 'poor': 0, 'blocklist': 0}

        for _, row in domain_stats.iterrows():
            domain = row['domain']
            shrunk_rate = row['shrunk_rate']
            bids = row['bids']
            views = row['views']
            clicks = row.get('clicks', views)

            # Determine tier
            if shrunk_rate >= self.percentile_thresholds['premium']:
                tier = 'premium'
                multiplier = self.domain_config.multiplier_premium
            elif shrunk_rate >= self.percentile_thresholds['standard']:
                tier = 'standard'
                multiplier = self.domain_config.multiplier_standard
            elif shrunk_rate >= self.percentile_thresholds['below_avg']:
                tier = 'below_avg'
                multiplier = self.domain_config.multiplier_below_avg
            else:
                tier = 'poor'
                multiplier = self.domain_config.multiplier_poor

            # Check for blocklist (only if enough data to judge)
            if bids >= min_bids_for_blocklist and shrunk_rate < blocklist_threshold:
                tier = 'blocklist'
                multiplier = self.domain_config.multiplier_blocklist

            tier_counts[tier] += 1

            self.profiles[domain] = {
                'tier': tier,
                'multiplier': round(multiplier, 2),
                'bids': int(bids),
                'views': int(views),
                'clicks': int(clicks),
                'shrunk_rate': round(shrunk_rate, 6),
            }

        self.is_loaded = True
        self.training_stats = {
            'loaded': True,
            'method': 'percentile',
            'total_domains': len(self.profiles),
            'tier_counts': tier_counts,
            'global_rate': global_rate,
            'signal_used': signal_name,
            'shrinkage_k': shrinkage_k,
            'percentile_thresholds': self.percentile_thresholds,
            'blocklist_threshold': blocklist_threshold,
        }

        print(f"    Tier distribution: {tier_counts}")

    def get_multiplier(self, domain: Optional[str]) -> float:
        """
        Get bid multiplier for a domain.

        Args:
            domain: Domain name (string or None)

        Returns:
            Bid multiplier (default 1.0 if unknown)
        """
        if domain is None or not self.is_loaded:
            return 1.0

        domain_str = str(domain).strip().lower()

        if domain_str not in self.profiles:
            return 1.0  # Unknown domain → baseline

        return self.profiles[domain_str].get('multiplier', 1.0)

    def get_profile(self, domain: str) -> Optional[Dict]:
        """Get full profile for a domain (for debugging)."""
        domain_str = str(domain).strip().lower()
        return self.profiles.get(domain_str)

    def get_tier_stats(self) -> Dict:
        """Get statistics about loaded domain tiers."""
        if not self.is_loaded:
            return {'loaded': False}

        tier_counts = defaultdict(int)
        multiplier_sum = 0.0

        for profile in self.profiles.values():
            tier = profile.get('tier', 'baseline')
            tier_counts[tier] += 1
            multiplier_sum += profile.get('multiplier', 1.0)

        avg_multiplier = multiplier_sum / len(self.profiles) if self.profiles else 1.0

        result = {
            'loaded': True,
            'method': self.training_stats.get('method', 'percentile'),
            'total_domains': len(self.profiles),
            'tier_counts': dict(tier_counts),
            'avg_multiplier': round(avg_multiplier, 3),
        }

        # Add method-specific thresholds
        if self.iqr_thresholds:
            result['iqr_thresholds'] = {k: round(v, 6) if isinstance(v, float) else v
                                        for k, v in self.iqr_thresholds.items()}
        if self.percentile_thresholds:
            result['percentile_thresholds'] = self.percentile_thresholds

        return result

    def get_all_profiles_df(self) -> pd.DataFrame:
        """
        Get all domain profiles as a DataFrame for export.

        Returns:
            DataFrame with columns: domain, tier, multiplier, bids, views, clicks, shrunk_rate
        """
        if not self.is_loaded or not self.profiles:
            return pd.DataFrame()

        rows = []
        for domain, profile in self.profiles.items():
            rows.append({
                'domain': domain,
                'tier': profile.get('tier', 'standard'),
                'multiplier': profile.get('multiplier', 1.0),
                'bids': profile.get('bids', 0),
                'views': profile.get('views', 0),
                'clicks': profile.get('clicks', 0),
                'shrunk_rate': profile.get('shrunk_rate', 0),
            })

        df = pd.DataFrame(rows)
        # Sort by multiplier descending (highest value first)
        df = df.sort_values('multiplier', ascending=False)
        return df

    def get_blocklist(self) -> pd.DataFrame:
        """
        Get blocklisted domains as a DataFrame.

        Returns:
            DataFrame with domains that have multiplier=0
        """
        if not self.is_loaded:
            return pd.DataFrame()

        blocked = [
            {'domain': domain, 'bids': profile['bids'], 'shrunk_rate': profile['shrunk_rate']}
            for domain, profile in self.profiles.items()
            if profile.get('tier') == 'blocklist'
        ]

        return pd.DataFrame(blocked)
