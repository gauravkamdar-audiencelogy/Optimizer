"""
Empirical Win Rate Model: Computes P(win | features) using observed segment-level rates.

V3 Change: Replaces LogisticRegression model (ECE=0.176) with empirical observation.

Key insight: When model predictions fail, use observed data directly.
The LogisticRegression model tried to GENERALIZE win rates to unseen combinations,
but with ~22K segments and median ~9 bids/segment, it couldn't learn reliable patterns.

Empirical rates are by-definition calibrated (just counting).
Bayesian shrinkage handles low-sample segments.

Literature:
- Thompson Sampling with Bayesian updating (Chapelle & Li, 2011)
- Empirical Bayes for sparse data (Efron, 2010)
- Facebook's CTR system (McMahan et al., KDD 2013)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from ..config import OptimizerConfig


class EmpiricalWinRateModel:
    """
    Empirical win rate model using observed segment-level rates with shrinkage.

    Unlike the LogisticRegression model (ECE=0.176), this approach:
    - Uses observed data, not model predictions
    - Is by-definition calibrated (it's just counting)
    - Handles sparse segments via Bayesian shrinkage
    """

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.training_stats: Dict = {}
        self.global_win_rate: float = 0.0
        self.segment_win_rates: Optional[pd.DataFrame] = None
        self.shrinkage_k: int = 30  # Bayesian shrinkage strength

    def train(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        target: str = 'won'
    ) -> None:
        """
        Compute empirical win rates per segment with Bayesian shrinkage.

        Args:
            df_train: Training data with features and 'won' column
            features: List of feature column names for segmentation
            target: Target column name ('won')
        """
        self.feature_names = features
        self.global_win_rate = df_train[target].mean()

        # Compute segment-level empirical win rates
        print("    Computing segment-level empirical win rates...")
        self.segment_win_rates = df_train.groupby(features)[target].agg(['sum', 'count'])
        self.segment_win_rates.columns = ['wins', 'bids']
        self.segment_win_rates['empirical_win_rate'] = (
            self.segment_win_rates['wins'] / self.segment_win_rates['bids']
        )

        # Apply Bayesian shrinkage toward global win rate
        # Formula: shrunk_rate = (n * empirical + k * global) / (n + k)
        # This handles sparse segments without overfitting to small samples
        k = self.shrinkage_k
        self.segment_win_rates['shrunk_win_rate'] = (
            (self.segment_win_rates['wins'] + k * self.global_win_rate) /
            (self.segment_win_rates['bids'] + k)
        )

        # Calculate statistics
        segments_with_wins = (self.segment_win_rates['wins'] > 0).sum()
        total_segments = len(self.segment_win_rates)

        print(f"    Global win rate: {self.global_win_rate:.2%}")
        print(f"    Segments with wins: {segments_with_wins} / {total_segments}")
        print(f"    Shrinkage strength (k): {k}")

        self.is_trained = True

        # Store training stats
        self.training_stats = {
            'n_samples': len(df_train),
            'n_positive': int(df_train[target].sum()),
            'n_negative': int(len(df_train) - df_train[target].sum()),
            'global_win_rate': float(self.global_win_rate),
            'n_segments': total_segments,
            'segments_with_wins': int(segments_with_wins),
            'shrinkage_k': self.shrinkage_k,
            'features': features,
            'model_type': 'empirical_with_shrinkage'
        }

    def get_win_rate_for_segment(
        self,
        segment_values: Dict[str, any]
    ) -> float:
        """
        Get win rate for a single segment.

        Uses hierarchy:
        1. Empirical shrunk win rate if segment has observations
        2. Global win rate as fallback for unseen segments

        Args:
            segment_values: Dict mapping feature name to value

        Returns:
            Shrunk win rate for the segment
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Build segment key tuple for lookup
        segment_key = tuple(segment_values[f] for f in self.feature_names)

        # Try empirical shrunk win rate first
        if segment_key in self.segment_win_rates.index:
            return float(self.segment_win_rates.loc[segment_key, 'shrunk_win_rate'])
        else:
            # Fallback to global win rate for unseen segments
            return float(self.global_win_rate)

    def get_segment_stats(
        self,
        segment_values: Dict[str, any]
    ) -> Dict:
        """
        Get detailed statistics for a segment (for debugging/analysis).
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        segment_key = tuple(segment_values[f] for f in self.feature_names)

        if segment_key in self.segment_win_rates.index:
            row = self.segment_win_rates.loc[segment_key]
            return {
                'wins': int(row['wins']),
                'bids': int(row['bids']),
                'empirical_win_rate': float(row['empirical_win_rate']),
                'shrunk_win_rate': float(row['shrunk_win_rate']),
                'is_observed': True
            }
        else:
            return {
                'wins': 0,
                'bids': 0,
                'empirical_win_rate': None,
                'shrunk_win_rate': float(self.global_win_rate),
                'is_observed': False
            }

    def predict_win_rate(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict win rate for a DataFrame.
        Uses shrunk win rates where available, global otherwise.

        This method exists for compatibility with calibration metrics.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions = []
        for _, row in df.iterrows():
            segment_values = {f: row[f] for f in self.feature_names}
            predictions.append(self.get_win_rate_for_segment(segment_values))

        return np.array(predictions)
