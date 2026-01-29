"""
Feature engineering for bid optimization.
Creates derived features from raw data.

V4: Added bid_amount_cpm to training data for BidLandscapeModel.
"""
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from typing import List, Set, Dict
from tqdm import tqdm

from .config import OptimizerConfig


def validate_bid_variance(
    df_train: pd.DataFrame,
    bid_col: str = 'bid_amount_cpm',
    min_unique_bids: int = 10,
    min_std: float = 0.5
) -> Dict:
    """
    Check if data has sufficient bid variance for BidLandscapeModel.

    Args:
        df_train: Training data with bid_amount_cpm column
        bid_col: Column name for bid amount
        min_unique_bids: Minimum unique bid values required
        min_std: Minimum standard deviation required

    Returns:
        dict with 'sufficient_variance' bool and diagnostics
    """
    if bid_col not in df_train.columns:
        return {
            'sufficient_variance': False,
            'reason': f'Column {bid_col} not found in training data',
            'n_bids': len(df_train)
        }

    bid_values = df_train[bid_col].dropna()

    bid_stats = {
        'n_bids': len(df_train),
        'n_valid_bids': len(bid_values),
        'unique_bid_values': int(bid_values.nunique()),
        'bid_mean': round(float(bid_values.mean()), 4),
        'bid_std': round(float(bid_values.std()), 4),
        'bid_min': round(float(bid_values.min()), 4),
        'bid_max': round(float(bid_values.max()), 4),
    }

    bid_stats['sufficient_variance'] = (
        bid_stats['unique_bid_values'] >= min_unique_bids and
        bid_stats['bid_std'] >= min_std
    )

    if not bid_stats['sufficient_variance']:
        if bid_stats['unique_bid_values'] < min_unique_bids:
            bid_stats['reason'] = f"Only {bid_stats['unique_bid_values']} unique bids (need {min_unique_bids}+)"
        else:
            bid_stats['reason'] = f"Bid std ${bid_stats['bid_std']:.2f} < ${min_std:.2f} threshold"

    return bid_stats


class FeatureEngineer:
    def __init__(self, config: OptimizerConfig):
        self.config = config

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all derived features."""
        df = df.copy()

        # Time-based features
        if 'log_dt' in df.columns:
            df['hour_of_day'] = df['log_dt'].dt.hour
            df['day_of_week'] = df['log_dt'].dt.dayofweek

        # URL-based features from ref_bundle
        if 'ref_bundle' in df.columns:
            # pageurl_truncated: full URL without query params (matches bidder format)
            df['pageurl_truncated'] = df['ref_bundle'].apply(self._truncate_url)
            # domain: only extract from ref_bundle if domain column doesn't exist
            # (nativo_consumer already has proper domain column)
            if 'domain' not in df.columns:
                df['domain'] = df['ref_bundle'].apply(self._extract_domain)

        # Handle nulls in geo_region_name
        if 'geo_region_name' in df.columns:
            df['geo_region_name'] = df['geo_region_name'].fillna('Unknown')

        return df

    @staticmethod
    def _truncate_url(url) -> str:
        """
        Truncate URL at '?' to match bidder format.
        Bidder JS: url.split('?')[0].toLowerCase()
        """
        if pd.isna(url):
            return 'unknown'
        url_str = str(url).split('?')[0].lower()
        return url_str if url_str else 'unknown'

    @staticmethod
    def _extract_domain(url) -> str:
        """Extract domain from URL for future multi-domain SSPs."""
        if pd.isna(url):
            return 'unknown'
        try:
            parsed = urlparse(str(url))
            return parsed.netloc.lower() if parsed.netloc else 'unknown'
        except Exception:
            return 'unknown'

    def create_training_data(
        self,
        df_bids: pd.DataFrame,
        df_views: pd.DataFrame,
        df_clicks: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create joined training dataset for win rate model.

        V3.1: Fixed to use internal_txn_id instead of log_txnid.
        - Bids can have multiple internal_txn_ids (multiple ads per bid)
        - A bid is "won" if ANY of its internal_txn_ids appears in views
        - Each bid is counted once (no inflation for multi-ad bids)

        Returns DataFrame with:
        - All bid features
        - 'won' binary column (1 if bid resulted in view)
        """
        print("    Creating win rate training data (joining bids to views via internal_txn_id)...")

        # Build set of all internal_txn_ids from views
        view_txn_ids = set()
        for val in df_views['internal_txn_id']:
            view_txn_ids.update(self._parse_array_to_list(val))
        print(f"    Found {len(view_txn_ids):,} unique internal_txn_ids in views")

        # Mark bid as won if ANY of its internal_txn_ids is in views
        df_train = df_bids.copy()
        df_train['won'] = df_train['internal_txn_id'].apply(
            lambda x: int(any(txn_id in view_txn_ids for txn_id in self._parse_array_to_list(x)))
        )

        won_count = df_train['won'].sum()
        print(f"    Matched {won_count:,} bids to views ({won_count/len(df_train)*100:.1f}% win rate)")

        # V4: Add bid_amount_cpm for BidLandscapeModel
        # bid_value is already parsed from publisher_payout in data_loader.py
        if 'bid_value' in df_train.columns:
            df_train['bid_amount_cpm'] = df_train['bid_value']
            print(f"    Added bid_amount_cpm (mean: ${df_train['bid_amount_cpm'].mean():.2f}, std: ${df_train['bid_amount_cpm'].std():.2f})")

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
        print("    Creating CTR training data (matching views to clicks)...")

        # Extract click transaction IDs
        click_txn_ids = self._extract_all_txn_ids(df_clicks)
        print(f"    Found {len(click_txn_ids)} unique click transaction IDs")

        # Mark views that resulted in clicks
        # Check ALL txn_ids in array, not just first (a view can have multiple ad slots)
        df_views = df_views.copy()
        tqdm.pandas(desc="    Matching clicks to views")
        df_views['clicked'] = df_views['internal_txn_id'].progress_apply(
            lambda x: int(any(txn_id in click_txn_ids for txn_id in self._parse_array_to_list(x)))
        )

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
        for txn_id in tqdm(df_clicks['internal_txn_id'], desc="    Extracting click txn IDs"):
            txn_list = self._parse_array_to_list(txn_id)
            click_txn_ids.update(txn_list)
        return click_txn_ids
