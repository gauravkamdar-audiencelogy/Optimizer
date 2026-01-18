"""
Feature engineering for bid optimization.
Creates derived features from raw data.
"""
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from typing import List, Set
from tqdm import tqdm

from .config import OptimizerConfig


# Page category aggregation based on CTR analysis from EDA
# Reduces cardinality from ~1153 to ~10 while preserving CTR signal
PAGE_CATEGORY_GROUPS = {
    'high_intent': [
        'news', 'medical-answers', 'cg', 'monograph', 'answers'
    ],  # CTR > 0.08%

    'drug_info': [
        'dosage', 'sfx', 'mtm', 'condition'
    ],  # CTR 0.04-0.06%

    'interactions': [
        'drug-interactions', 'drug_interactions.html', 'drug-interactions-all',
        'food-interactions'
    ],  # CTR ~0.03%

    'search': [
        'search.php', 'imprints.php', 'imprints', 'alpha'
    ],  # CTR 0.02-0.04%

    'professional': [
        'pro', 'monograph'
    ],  # HCP-focused

    'interaction_check': [
        'interaction', 'interactions-check.php'
    ],  # Very low CTR (<0.02%)

    'community': [
        'comments'
    ],  # User-generated

    'drug_class': [
        'drug-class'
    ],  # Zero clicks observed
}


def aggregate_page_category(category: str) -> str:
    """
    Map raw page category to aggregated group.
    Reduces cardinality from ~1153 to ~10 while preserving CTR signal.
    """
    if pd.isna(category) or category == 'unknown':
        return 'other'

    category_lower = str(category).lower()

    for group_name, categories in PAGE_CATEGORY_GROUPS.items():
        if category_lower in [c.lower() for c in categories]:
            return group_name

    # Default for uncategorized pages
    return 'other'


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

        # Page category from ref_bundle
        if 'ref_bundle' in df.columns:
            tqdm.pandas(desc="    Extracting page categories")
            df['page_category_raw'] = df['ref_bundle'].progress_apply(self._extract_page_category)
            # Aggregated page category (reduces cardinality ~1153 -> ~10)
            df['page_category'] = df['page_category_raw'].apply(aggregate_page_category)

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

        Returns DataFrame with:
        - All bid features
        - 'won' binary column (1 if bid resulted in view)
        """
        print("    Creating win rate training data (joining bids to views)...")

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
        print("    Creating CTR training data (matching views to clicks)...")

        # Extract click transaction IDs
        click_txn_ids = self._extract_all_txn_ids(df_clicks)
        print(f"    Found {len(click_txn_ids)} unique click transaction IDs")

        # Mark views that resulted in clicks
        df_views = df_views.copy()
        tqdm.pandas(desc="    Matching clicks to views")
        df_views['first_txn_id'] = df_views['internal_txn_id'].progress_apply(
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
        for txn_id in tqdm(df_clicks['internal_txn_id'], desc="    Extracting click txn IDs"):
            txn_list = self._parse_array_to_list(txn_id)
            click_txn_ids.update(txn_list)
        return click_txn_ids
