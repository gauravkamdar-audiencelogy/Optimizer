"""
Load and clean bid/view/click data from CSV files.
Handles: malformed rows, duplicates, zero bids, date filtering.

Supports two data layouts:
1. Combined: single data_{dataset}.csv with rec_type column
2. Separate: {dataset}_bids.csv, {dataset}_views.csv, {dataset}_clicks.csv

V9 Changes:
- Dataset-aware filenames (data_drugs.csv, data_nativo_consumer.csv)
- Automatic column name normalization (handles UPPERCASE from different SSPs)
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
        self._combined_data: pd.DataFrame = None

        # V9: Derive dataset name from data_dir (e.g., "data_drugs" -> "drugs")
        self.dataset_name = self.data_dir.name.replace('data_', '')

        # V10: Date filtering from config
        self.training_start_date = self._parse_date(config.data.min_bid_date)
        self.training_end_date = self._parse_date(
            config.run.training_end_date if hasattr(config, 'run') else None
        )

    def _parse_date(self, date_str: str) -> pd.Timestamp:
        """Parse date string to pandas Timestamp with UTC timezone."""
        if date_str is None:
            return None
        try:
            return pd.to_datetime(date_str, utc=True)
        except Exception:
            return None

    def _apply_date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        V10: Filter data by training date window.

        Uses:
        - training_start_date: from config.data.min_bid_date (or run.training_start_date)
        - training_end_date: from config.run.training_end_date (NEW)
        """
        initial_count = len(df)
        filtered = False

        if self.training_start_date is not None:
            before_filter = len(df)
            df = df[df['log_dt'] >= self.training_start_date]
            removed = before_filter - len(df)
            if removed > 0:
                print(f"    Date filter: removed {removed:,} rows before {self.training_start_date.date()}")
                filtered = True

        if self.training_end_date is not None:
            before_filter = len(df)
            df = df[df['log_dt'] <= self.training_end_date]
            removed = before_filter - len(df)
            if removed > 0:
                print(f"    Date filter: removed {removed:,} rows after {self.training_end_date.date()}")
                filtered = True

        if filtered:
            print(f"    After date filtering: {len(df):,} rows (from {initial_count:,})")

        # Store date range in stats
        self.load_stats['date_filter'] = {
            'start_date': str(self.training_start_date.date()) if self.training_start_date else None,
            'end_date': str(self.training_end_date.date()) if self.training_end_date else None,
            'rows_before': initial_count,
            'rows_after': len(df)
        }

        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and clean all three datasets.

        Checks for combined data_{dataset}.csv first, falls back to separate files.
        """
        # V9: Dataset-aware filename
        combined_filename = f'data_{self.dataset_name}.csv'
        combined_path = self.data_dir / combined_filename

        if combined_path.exists():
            print(f"    Found combined data file: {combined_path.name}")
            return self._load_from_combined(combined_path)
        else:
            print("    No combined file found, loading separate files...")
            df_bids = self._load_bids()
            df_views = self._load_views()
            df_clicks = self._load_clicks()
            return df_bids, df_views, df_clicks

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """V9: Normalize column names to lowercase for cross-SSP compatibility."""
        df.columns = df.columns.str.lower()
        return df

    def _load_from_combined(self, path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load bids, views, clicks from combined data file."""
        print(f"    Loading combined data from {path}...")

        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        initial_count = len(df)
        print(f"    Total rows: {initial_count:,}")

        # V9: Normalize column names (handles UPPERCASE from different SSPs)
        df = self._normalize_columns(df)

        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

        # V10: Apply date filtering
        df = self._apply_date_filter(df)

        # Standardize rec_type
        df['rec_type'] = df['rec_type'].str.lower().str.strip()

        # Split by rec_type
        # Note: nativo_consumer uses 'link' for clicks, 'lead' for conversions
        df_bids = df[df['rec_type'] == 'bid'].copy()
        df_views = df[df['rec_type'] == 'view'].copy()
        df_clicks = df[df['rec_type'].isin(['click', 'link'])].copy()

        print(f"    Split: {len(df_bids):,} bids, {len(df_views):,} views, {len(df_clicks):,} clicks")

        # Process bids
        df_bids = self._process_bids(df_bids)

        # Process views
        df_views = self._process_views(df_views)

        # Process clicks
        df_clicks = self._process_clicks(df_clicks)

        return df_bids, df_views, df_clicks

    def _process_bids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process bid data (common logic for both load paths)."""
        initial_count = len(df)

        # Parse publisher payout from postgres array format (actual clearing price)
        tqdm.pandas(desc="    Parsing bid values")
        df['bid_value'] = df['publisher_payout'].progress_apply(self._parse_first_array_value)

        # V3.1: Parse floor price if available
        if self.config.technical.floor_available:
            print("    Parsing floor prices from bid_amount...")
            df['floor_price'] = df['bid_amount'].apply(self._parse_first_array_value)
            floor_count = df['floor_price'].notna().sum()
            print(f"    Found {floor_count:,} bids with floor prices")
        else:
            df['floor_price'] = np.nan

        # Filter: non-zero bids only
        df = df[df['bid_value'] > 0]

        self.load_stats['bids'] = {
            'raw': initial_count,
            'clean': len(df),
            'removed': initial_count - len(df),
            'floor_available': self.config.technical.floor_available
        }

        return df

    def _process_views(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process view data (common logic for both load paths)."""
        initial_count = len(df)

        # Parse clearing price from publisher_payout
        df['clearing_price'] = df['publisher_payout'].apply(self._parse_first_array_value)

        # Deduplicate by log_txnid
        df = df.drop_duplicates(subset=['log_txnid'], keep='first')
        after_dedup = len(df)

        # Filter out invalid clearing prices
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
            print(f"    Removed {invalid_count:,} views with invalid clearing price")

        return df

    def _process_clicks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process click data (common logic for both load paths)."""
        # Parse CPC from postgres array format
        df['cpc_value'] = df['advertiser_spend'].apply(self._parse_first_array_value)

        self.load_stats['clicks'] = {
            'raw': len(df),
            'clean': len(df),
            'removed': 0
        }

        return df

    def _load_bids(self) -> pd.DataFrame:
        """Load and clean bid data from separate file."""
        # V9: Dataset-aware filename
        path = self.data_dir / f'{self.dataset_name}_bids.csv'
        print(f"    Loading bids from {path}...")

        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        df = self._normalize_columns(df)

        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

        # V10: Apply date filtering
        df = self._apply_date_filter(df)

        return self._process_bids(df)

    def _load_views(self) -> pd.DataFrame:
        """Load and clean view data from separate file."""
        # V9: Dataset-aware filename
        path = self.data_dir / f'{self.dataset_name}_views.csv'
        print(f"    Loading views from {path}...")

        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        df = self._normalize_columns(df)

        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

        # V10: Apply date filtering
        df = self._apply_date_filter(df)

        return self._process_views(df)

    def _load_clicks(self) -> pd.DataFrame:
        """Load click data from separate file."""
        # V9: Dataset-aware filename
        path = self.data_dir / f'{self.dataset_name}_clicks.csv'
        print(f"    Loading clicks from {path}...")

        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        df = self._normalize_columns(df)

        # Parse datetime
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

        # V10: Apply date filtering
        df = self._apply_date_filter(df)

        return self._process_clicks(df)

    @staticmethod
    def _parse_first_array_value(val) -> float:
        """Extract first numeric value from postgres array string.

        Handles formats:
        - {7.50000} - standard postgres array
        - {"0.45000"} - quoted postgres array (from some SSPs)
        - 7.50000 - plain numeric
        """
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        if val_str.startswith('{') and val_str.endswith('}'):
            inner = val_str[1:-1]
            if inner == '':
                return np.nan
            first_val = inner.split(',')[0]
            # Strip quotes if present (handles {"0.45000"} format)
            first_val = first_val.strip('"\'')
            try:
                return float(first_val)
            except ValueError:
                return np.nan
        try:
            return float(val_str)
        except ValueError:
            return np.nan
