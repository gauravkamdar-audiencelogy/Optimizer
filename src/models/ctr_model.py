"""
CTR Model: Predicts P(click | impression, features).

Uses empirical segment-level CTRs with Bayesian shrinkage toward global CTR.
This approach is based on Facebook's production system and academic research.

Key insight: class_weight='balanced' destroys probability calibration.
Instead, we use empirical rates + shrinkage for sparse segments.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional

from ..config import OptimizerConfig


class CTRModel:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.training_stats: Dict = {}
        self.global_ctr: float = 0.0
        self.segment_ctrs: Optional[pd.DataFrame] = None
        self.shrinkage_k: int = 30  # Bayesian shrinkage strength

    def train(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        target: str = 'clicked'
    ) -> None:
        """
        Train CTR model WITHOUT class_weight='balanced'.
        Uses empirical segment rates + Bayesian shrinkage for calibrated probabilities.
        """
        self.feature_names = features
        self.global_ctr = df_train[target].mean()

        # Compute segment-level empirical CTRs
        print("    Computing segment-level empirical CTRs...")
        self.segment_ctrs = df_train.groupby(features)[target].agg(['sum', 'count'])
        self.segment_ctrs.columns = ['clicks', 'impressions']
        self.segment_ctrs['empirical_ctr'] = (
            self.segment_ctrs['clicks'] / self.segment_ctrs['impressions']
        )

        # Apply Bayesian shrinkage toward global CTR
        # Formula: shrunk_ctr = (n * empirical + k * global) / (n + k)
        # This handles sparse segments without overfitting
        k = self.shrinkage_k
        self.segment_ctrs['shrunk_ctr'] = (
            (self.segment_ctrs['clicks'] + k * self.global_ctr) /
            (self.segment_ctrs['impressions'] + k)
        )

        segments_with_clicks = (self.segment_ctrs['clicks'] > 0).sum()
        print(f"    Segments with clicks: {segments_with_clicks} / {len(self.segment_ctrs)}")
        print(f"    Shrinkage strength (k): {k}")

        # Prepare features for fallback model
        X = df_train[features].copy()
        y = df_train[target].values

        # Identify categorical vs numeric features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()

        print(f"    Categorical features: {categorical_features}")
        print(f"    Numeric features: {numeric_features}")

        # Handle edge case: no clicks in data
        if y.sum() == 0:
            print("    WARNING: No clicks in training data - using global CTR only")
            self.model = None
            self.is_trained = True
            self.training_stats = {
                'n_samples': len(y),
                'n_clicks': 0,
                'global_ctr': 0.0,
                'mean_pred_ctr': 0.0,
                'n_segments': len(self.segment_ctrs),
                'segments_with_clicks': 0,
                'shrinkage_k': self.shrinkage_k,
                'features': features,
                'note': 'No clicks - model returns global CTR (0) for all predictions'
            }
            return

        # Create preprocessing pipeline
        transformers = []
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )

        # Train fallback model WITHOUT class_weight='balanced'
        # This preserves probability calibration
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=0.1,  # Strong regularization for sparse data
                max_iter=1000,
                solver='lbfgs'
                # NO class_weight='balanced' - preserves calibration
            ))
        ])

        print("    Training fallback logistic regression model (no class balancing)...")
        self.model.fit(X, y)
        self.is_trained = True

        # Calculate training stats
        y_pred = self.model.predict_proba(X)[:, 1]
        self.training_stats = {
            'n_samples': len(y),
            'n_clicks': int(y.sum()),
            'global_ctr': float(self.global_ctr),
            'mean_pred_ctr': float(y_pred.mean()),
            'n_segments': len(self.segment_ctrs),
            'segments_with_clicks': int(segments_with_clicks),
            'shrinkage_k': self.shrinkage_k,
            'features': features
        }

    def predict_ctr(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """Predict CTR for given features using fallback model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Handle no-clicks case: return global CTR (0) for all
        if self.model is None:
            return np.full(len(df), self.global_ctr)

        X = df[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def get_ctr_for_segment(
        self,
        segment_values: Dict[str, any],
        use_confidence_bound: bool = True  # kept for API compatibility
    ) -> float:
        """
        Get CTR prediction for a single segment.

        Uses hierarchy:
        1. Empirical shrunk CTR if segment has observations
        2. LR model prediction (properly calibrated) as fallback

        The shrunk_ctr already incorporates Bayesian smoothing,
        so no additional confidence adjustment needed.
        """
        # Build segment key tuple for lookup
        segment_key = tuple(segment_values[f] for f in self.feature_names)

        # Try empirical shrunk CTR first
        if segment_key in self.segment_ctrs.index:
            return float(self.segment_ctrs.loc[segment_key, 'shrunk_ctr'])
        else:
            # Fallback to model prediction for unseen segments
            # Handle no-clicks case
            if self.model is None:
                return self.global_ctr

            df = pd.DataFrame([segment_values])
            pred = self.predict_ctr(df)[0]
            # Clip to reasonable range around global CTR (avoid div by zero)
            if self.global_ctr > 0:
                return float(np.clip(pred, self.global_ctr * 0.1, self.global_ctr * 10))
            return float(pred)
