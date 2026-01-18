"""
V2: Simplified memcache generation with binary filtering.

Changes:
- Replaced confidence scaling with binary filter
- Either a segment has enough observations AND is profitable, or it's excluded
- No half-measures that just get clipped to floor anyway
"""
import pandas as pd
from typing import List
from pathlib import Path
from datetime import datetime

from .config import OptimizerConfig
from .bid_calculator import BidResult


class MemcacheBuilder:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.filter_stats = {
            'total_segments': 0,
            'excluded_low_observations': 0,
            'excluded_unprofitable': 0,
            'included': 0
        }

    def build_memcache(
        self,
        bid_results: List[BidResult],
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
