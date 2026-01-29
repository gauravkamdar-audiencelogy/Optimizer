"""
V5: Include ALL segments in memcache.

V5 Philosophy:
- During data collection, we want to bid on EVERYTHING
- No filtering by observations or profitability
- Learn the market through exploration
- Include bid_method and observation_count for analysis

V2 (old) approach:
- Binary filter: exclude low obs, exclude unprofitable
- This killed 78% of segments!

V5 approach:
- Include ALL segments
- Let the bid calculator handle exploration strategy
"""
import pandas as pd
import json
from typing import List, Union
from pathlib import Path
from datetime import datetime

from .config import OptimizerConfig
from .bid_calculator import BidResult


class MemcacheBuilder:
    def __init__(self, config: OptimizerConfig, v5_mode: bool = False):
        self.config = config
        self.v5_mode = v5_mode
        self.filter_stats = {
            'total_segments': 0,
            'excluded_low_observations': 0,
            'excluded_unprofitable': 0,
            'included': 0,
            # V5: Track exploration stats
            'v5_included_zero_obs': 0,
            'v5_included_low_obs': 0,
            'v5_included_medium_obs': 0,
            'v5_included_high_obs': 0
        }

    def build_memcache(
        self,
        bid_results: List,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Build memcache DataFrame.

        V5 mode: Include ALL segments, no filtering.
        V2 mode: Binary filtering by observations and profitability.
        """
        if self.v5_mode:
            return self._build_memcache_v5(bid_results, features)
        else:
            return self._build_memcache_v2(bid_results, features)

    def _build_memcache_v5(
        self,
        bid_results: List,
        features: List[str]
    ) -> pd.DataFrame:
        """
        V5: Build memcache with ALL segments included.

        MEMCACHE CONTRACT: Only features + suggested_bid_cpm
        No metadata (observation_count, bid_method, exploration_direction)

        For analysis, use build_segment_analysis() separately.
        """
        rows = []
        min_for_empirical = self.config.technical.min_observations_for_empirical
        min_for_landscape = self.config.technical.min_observations_for_landscape

        self.filter_stats['total_segments'] = len(bid_results)

        for result in bid_results:
            # V5: Include EVERYTHING but ONLY features + bid
            row = {}
            for feat in features:
                row[feat] = result.segment_key.get(feat, '')

            row['suggested_bid_cpm'] = result.final_bid

            rows.append(row)
            self.filter_stats['included'] += 1

            # Track observation tier for stats
            obs = result.observation_count
            if obs == 0:
                self.filter_stats['v5_included_zero_obs'] += 1
            elif obs < min_for_empirical:
                self.filter_stats['v5_included_low_obs'] += 1
            elif obs < min_for_landscape:
                self.filter_stats['v5_included_medium_obs'] += 1
            else:
                self.filter_stats['v5_included_high_obs'] += 1

        df = pd.DataFrame(rows)

        # Ensure column order: features first, then bid ONLY
        if len(df) > 0:
            column_order = features + ['suggested_bid_cpm']
            df = df[column_order]

        return df

    def build_segment_analysis(
        self,
        bid_results: List,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Build segment analysis CSV with full metadata for review.

        This is separate from memcache - includes all bid calculation details.
        """
        rows = []

        for result in bid_results:
            row = {}

            # Features
            for feat in features:
                row[feat] = result.segment_key.get(feat, '')

            # Bid info
            row['suggested_bid_cpm'] = result.final_bid
            row['raw_bid'] = getattr(result, 'raw_bid', result.final_bid)

            # V5 exploration info
            row['bid_method'] = getattr(result, 'bid_method', 'unknown')
            row['exploration_direction'] = getattr(result, 'exploration_direction', '')
            row['exploration_adjustment'] = getattr(result, 'exploration_adjustment', 1.0)

            # Segment stats
            row['observation_count'] = result.observation_count
            row['win_rate'] = getattr(result, 'win_rate', 0)
            row['ctr'] = getattr(result, 'ctr', 0)
            row['expected_value_cpm'] = getattr(result, 'expected_value_cpm', 0)

            # Economic
            row['is_profitable'] = getattr(result, 'is_profitable', True)
            row['npi_multiplier'] = getattr(result, 'npi_multiplier', 1.0)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by observation count descending for easier review
        if len(df) > 0:
            df = df.sort_values('observation_count', ascending=False)

        return df

    def write_segment_analysis(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        timestamp: str = None
    ) -> Path:
        """Write segment analysis to CSV file."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f'segment_analysis_{timestamp}.csv'
        filepath = output_dir / filename

        df.to_csv(filepath, index=False)

        return filepath

    def _build_memcache_v2(
        self,
        bid_results: List[BidResult],
        features: List[str]
    ) -> pd.DataFrame:
        """
        V2: Build memcache with BINARY filtering.

        Legacy mode:
        - Include if: observations >= min_observations AND is_profitable
        - Exclude otherwise
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
        """Write suggested bids to CSV file."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f'suggested_bids_{timestamp}.csv'
        filepath = output_dir / filename

        df.to_csv(filepath, index=False)

        return filepath

    def get_filter_stats(self) -> dict:
        """Return filtering statistics for metrics."""
        return self.filter_stats

    def write_selected_features(
        self,
        features: List[str],
        output_dir: Path,
        timestamp: str = None
    ) -> Path:
        """
        Write selected features to a simple text file.

        This tells the bidder which features to extract from bid requests.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f'selected_features_{timestamp}.csv'
        filepath = output_dir / filename

        df = pd.DataFrame({'feature': features})
        df.to_csv(filepath, index=False)

        return filepath

    def build_bid_summary(
        self,
        bid_results: List,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Build aggregated bid summary with tiered buckets.

        Groups segments into bid ranges for easier review:
        - $2-5: Floor to low
        - $5-8: Medium
        - $8-12: High
        - $12+: Aggressive

        For each bucket: segment count, %, avg bid, avg obs, avg WR,
        exploration direction breakdown, bid method breakdown.
        """
        # Define buckets
        buckets = [
            ('$2-5', 2.0, 5.0),
            ('$5-8', 5.0, 8.0),
            ('$8-12', 8.0, 12.0),
            ('$12+', 12.0, float('inf'))
        ]

        # Collect data per bucket
        bucket_data = {name: [] for name, _, _ in buckets}

        for result in bid_results:
            bid = result.final_bid
            for name, low, high in buckets:
                if low <= bid < high:
                    bucket_data[name].append(result)
                    break

        # Build summary rows
        total_segments = len(bid_results)
        rows = []

        for name, low, high in buckets:
            results = bucket_data[name]
            count = len(results)

            if count == 0:
                rows.append({
                    'bid_bucket': name,
                    'segments': 0,
                    'pct': '0%',
                    'avg_bid': '-',
                    'avg_obs': '-',
                    'avg_wr': '-',
                    'bid_up': 0,
                    'bid_down': 0,
                    'neutral': 0,
                    'explore_zero': 0,
                    'explore_low': 0,
                    'explore_med': 0,
                    'empirical': 0
                })
                continue

            # Calculate averages
            avg_bid = sum(r.final_bid for r in results) / count
            avg_obs = sum(r.observation_count for r in results) / count
            avg_wr = sum(getattr(r, 'win_rate', 0) for r in results) / count

            # Exploration direction counts
            bid_up = sum(1 for r in results if getattr(r, 'exploration_direction', '') == 'up')
            bid_down = sum(1 for r in results if getattr(r, 'exploration_direction', '') == 'down')
            neutral = sum(1 for r in results if getattr(r, 'exploration_direction', '') == 'neutral')

            # Bid method counts
            explore_zero = sum(1 for r in results if 'zero' in getattr(r, 'bid_method', ''))
            explore_low = sum(1 for r in results if 'low' in getattr(r, 'bid_method', '') and 'zero' not in getattr(r, 'bid_method', ''))
            explore_med = sum(1 for r in results if 'medium' in getattr(r, 'bid_method', ''))
            empirical = sum(1 for r in results if 'empirical' in getattr(r, 'bid_method', ''))

            pct = count / total_segments * 100

            rows.append({
                'bid_bucket': name,
                'segments': count,
                'pct': f'{pct:.1f}%',
                'avg_bid': f'${avg_bid:.2f}',
                'avg_obs': f'{avg_obs:.0f}',
                'avg_wr': f'{avg_wr:.1%}',
                'bid_up': bid_up,
                'bid_down': bid_down,
                'neutral': neutral,
                'explore_zero': explore_zero,
                'explore_low': explore_low,
                'explore_med': explore_med,
                'empirical': empirical
            })

        # Add TOTAL row
        total_bid_up = sum(1 for r in bid_results if getattr(r, 'exploration_direction', '') == 'up')
        total_bid_down = sum(1 for r in bid_results if getattr(r, 'exploration_direction', '') == 'down')
        total_neutral = sum(1 for r in bid_results if getattr(r, 'exploration_direction', '') == 'neutral')
        total_explore_zero = sum(1 for r in bid_results if 'zero' in getattr(r, 'bid_method', ''))
        total_explore_low = sum(1 for r in bid_results if 'low' in getattr(r, 'bid_method', '') and 'zero' not in getattr(r, 'bid_method', ''))
        total_explore_med = sum(1 for r in bid_results if 'medium' in getattr(r, 'bid_method', ''))
        total_empirical = sum(1 for r in bid_results if 'empirical' in getattr(r, 'bid_method', ''))
        total_avg_bid = sum(r.final_bid for r in bid_results) / total_segments if total_segments > 0 else 0
        total_avg_wr = sum(getattr(r, 'win_rate', 0) for r in bid_results) / total_segments if total_segments > 0 else 0

        rows.append({
            'bid_bucket': 'TOTAL',
            'segments': total_segments,
            'pct': '100%',
            'avg_bid': f'${total_avg_bid:.2f}',
            'avg_obs': '-',
            'avg_wr': f'{total_avg_wr:.1%}',
            'bid_up': total_bid_up,
            'bid_down': total_bid_down,
            'neutral': total_neutral,
            'explore_zero': total_explore_zero,
            'explore_low': total_explore_low,
            'explore_med': total_explore_med,
            'empirical': total_empirical
        })

        return pd.DataFrame(rows)

    def write_bid_summary(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        timestamp: str = None
    ) -> Path:
        """Write bid summary to CSV file."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f'bid_summary_{timestamp}.csv'
        filepath = output_dir / filename

        df.to_csv(filepath, index=False)

        return filepath

    def write_npi_cache(
        self,
        npi_model,
        output_dir: Path,
        timestamp: str = None
    ) -> tuple:
        """
        Write NPI files: multipliers (production) and summary (analysis).

        Creates two files:
        1. npi_multipliers_*.csv - Production file with only external_userid, multiplier
        2. npi_summary_*.csv - Analysis file with all columns (tier, is_recent, etc.)

        Args:
            npi_model: NPIValueModel instance
            output_dir: Output directory
            timestamp: Optional timestamp string

        Returns:
            Tuple of (multipliers_path, summary_path)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Get all profiles as DataFrame
        df = npi_model.get_all_profiles_df()

        if df.empty:
            print("    WARNING: No NPI profiles to export")
            return None, None

        # 1. Write production file (only external_userid and multiplier)
        multipliers_filename = f'npi_multipliers_{timestamp}.csv'
        multipliers_filepath = output_dir / multipliers_filename
        df[['external_userid', 'multiplier']].to_csv(multipliers_filepath, index=False)

        # 2. Write analysis file (all columns)
        summary_filename = f'npi_summary_{timestamp}.csv'
        summary_filepath = output_dir / summary_filename
        df.to_csv(summary_filepath, index=False)

        # Print summary
        tier_counts = df['tier'].value_counts().sort_index()
        recent_count = df['is_recent'].sum()

        print(f"\n  NPI Output:")
        print(f"    Total NPIs: {len(df):,}")
        for tier in sorted(tier_counts.index):
            print(f"    Tier {tier}: {tier_counts[tier]:,} ({tier_counts[tier]/len(df)*100:.1f}%)")
        print(f"    Recent clickers: {recent_count:,} ({recent_count/len(df)*100:.1f}%)")
        print(f"    Multiplier range: {df['multiplier'].min():.2f}x - {df['multiplier'].max():.2f}x")
        print(f"    Avg multiplier: {df['multiplier'].mean():.2f}x")

        return multipliers_filepath, summary_filepath

    def write_validation_report(
        self,
        validation_result,
        output_dir: Path,
        timestamp: str = None
    ) -> Path:
        """
        Write validation report to JSON file.

        Args:
            validation_result: ValidationResult instance from Validator
            output_dir: Output directory
            timestamp: Optional timestamp string

        Returns:
            Path to the validation report file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f'validation_report_{timestamp}.json'
        filepath = output_dir / filename

        # Convert to dict and add run_id
        report = {
            'run_id': timestamp,
            **validation_result.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return filepath

    def write_domain_multipliers(
        self,
        domain_model,
        output_dir: Path,
        timestamp: str = None
    ) -> tuple:
        """
        Write domain files: multipliers (production) and summary (analysis).

        Creates two files:
        1. domain_multipliers_*.csv - Production file with only domain, multiplier
        2. domain_summary_*.csv - Analysis file with all columns (tier, bids, etc.)

        Args:
            domain_model: DomainValueModel instance
            output_dir: Output directory
            timestamp: Optional timestamp string

        Returns:
            Tuple of (multipliers_path, summary_path)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Get all profiles as DataFrame
        df = domain_model.get_all_profiles_df()

        if df.empty:
            print("    WARNING: No domain profiles to export")
            return None, None

        # 1. Write production file (only domain and multiplier)
        multipliers_filename = f'domain_multipliers_{timestamp}.csv'
        multipliers_filepath = output_dir / multipliers_filename
        df[['domain', 'multiplier']].to_csv(multipliers_filepath, index=False)

        # 2. Write analysis file (all columns)
        summary_filename = f'domain_summary_{timestamp}.csv'
        summary_filepath = output_dir / summary_filename
        df.to_csv(summary_filepath, index=False)

        # 3. Write blocklist separately (if any)
        blocklist_df = domain_model.get_blocklist()
        blocklist_filepath = None
        if not blocklist_df.empty:
            blocklist_filename = f'domain_blocklist_{timestamp}.csv'
            blocklist_filepath = output_dir / blocklist_filename
            blocklist_df.to_csv(blocklist_filepath, index=False)

        # Print summary
        tier_counts = df['tier'].value_counts()

        print(f"\n  Domain Output:")
        print(f"    Total domains: {len(df):,}")
        for tier in ['premium', 'standard', 'below_avg', 'poor', 'blocklist']:
            if tier in tier_counts.index:
                count = tier_counts[tier]
                print(f"    {tier.capitalize()}: {count:,} ({count/len(df)*100:.1f}%)")
        print(f"    Multiplier range: {df['multiplier'].min():.2f}x - {df['multiplier'].max():.2f}x")
        print(f"    Avg multiplier: {df['multiplier'].mean():.2f}x")
        if blocklist_filepath:
            print(f"    Blocklisted: {len(blocklist_df):,} domains")

        return multipliers_filepath, summary_filepath
