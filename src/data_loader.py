"""
Load and clean bid/view/click data from CSV files.
Handles: malformed rows, duplicates, zero bids, date filtering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm

from .config import OptimizerConfig


class DataLoader:
    def __init__(self, data_dir: str, config: OptimizerConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.load_stats: Dict = {}

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and clean all three datasets."""
        df_bids = self._load_bids()
        df_views = self._load_views()
        df_clicks = self._load_clicks()
        return df_bids, df_views, df_clicks

    def _load_bids(self) -> pd.DataFrame:
        """
        Load and clean bid data.

        V3.1: Added floor_available handling.
        - publisher_payout = outbound bid amount (always used)
        - bid_amount = floor price from SSP (only parsed if floor_available=True)
        """
        path = self.data_dir / 'drugs_bids.csv'
        print(f"    Loading bids from {path}...")

        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        initial_count = len(df)

        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

        # Parse publisher payout from postgres array format (actual clearing price)
        tqdm.pandas(desc="    Parsing publisher payout")
        df['bid_value'] = df['publisher_payout'].progress_apply(self._parse_first_array_value)

        # V3.1: Parse floor price if available
        if self.config.technical.floor_available:
            print("    Parsing floor prices from bid_amount...")
            df['floor_price'] = df['bid_amount'].apply(self._parse_first_array_value)
            floor_count = df['floor_price'].notna().sum()
            print(f"    Found {floor_count:,} bids with floor prices")
        else:
            df['floor_price'] = np.nan  # No floor data in Phase 1

        # Filter: non-zero bids only
        df = df[df['bid_value'] > 0]

        self.load_stats['bids'] = {
            'raw': initial_count,
            'clean': len(df),
            'removed': initial_count - len(df),
            'floor_available': self.config.technical.floor_available
        }

        return df

    def _load_views(self) -> pd.DataFrame:
        """Load and clean view data."""
        path = self.data_dir / 'drugs_views.csv'
        print(f"    Loading views from {path}...")

        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        initial_count = len(df)

        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

        # Parse clearing price from publisher_payout (what we actually paid)
        df['clearing_price'] = df['publisher_payout'].apply(self._parse_first_array_value)

        # Deduplicate by log_txnid
        df = df.drop_duplicates(subset=['log_txnid'], keep='first')
        after_dedup = len(df)

        # Filter out invalid clearing prices ($0 or negative)
        # We paid something to show the ad, so $0 is invalid
        invalid_price_mask = (df['clearing_price'] <= 0) | (df['clearing_price'].isna())
        invalid_count = invalid_price_mask.sum()
        df = df[~invalid_price_mask]

        self.load_stats['views'] = {
            'raw': initial_count,
            'after_dedup': after_dedup,
            'removed_duplicates': initial_count - after_dedup,
            'removed_invalid_price': int(invalid_count),
            'clean': len(df),
            'removed': initial_count - len(df)
        }

        if invalid_count > 0:
            print(f"    Removed {invalid_count:,} views with invalid clearing price (≤$0)")

        return df

    def _load_clicks(self) -> pd.DataFrame:
        """Load click data."""
        path = self.data_dir / 'drugs_clicks.csv'
        print(f"    Loading clicks from {path}...")

        df = pd.read_csv(path, on_bad_lines='skip', engine='python')

        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

        # Parse CPC from postgres array format
        df['cpc_value'] = df['advertiser_spend'].apply(self._parse_first_array_value)

        self.load_stats['clicks'] = {
            'raw': len(df),
            'clean': len(df),
            'removed': 0
        }

        return df

    @staticmethod
    def _parse_first_array_value(val) -> float:
        """Extract first numeric value from postgres array string like {7.50000}."""
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        if val_str.startswith('{') and val_str.endswith('}'):
            inner = val_str[1:-1]
            if inner == '':
                return np.nan
            first_val = inner.split(',')[0]
            try:
                return float(first_val)
            except ValueError:
                return np.nan
        try:
            return float(val_str)
        except ValueError:
            return np.nan
