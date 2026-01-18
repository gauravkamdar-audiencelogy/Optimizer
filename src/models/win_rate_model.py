"""
Win Rate Model: Predicts P(win | features).

V2 NOTE: This model is for DIAGNOSTICS ONLY.
    It is NOT used in bid calculation because:
    - ECE = 0.176 (consistently overestimates by ~1.7x)
    - Using miscalibrated predictions to adjust bids adds noise

    We keep it to:
    1. Track calibration over time
    2. Identify when model quality improves enough to use
    3. Understand market dynamics

Uses logistic regression for simplicity and interpretability.
Based on: "Bid Shading by Win-Rate Estimation and Surplus Maximization" (arXiv:2009.09259)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional

from ..config import OptimizerConfig


class WinRateModel:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.training_stats: Dict = {}

    def train(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        target: str = 'won'
    ) -> None:
        """
        Train win rate model.

        Args:
            df_train: Training data with features and target
            features: List of feature column names
            target: Target column name ('won')
        """
        self.feature_names = features

        # Prepare features
        X = df_train[features].copy()
        y = df_train[target].values

        # Identify categorical vs numeric features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()

        print(f"    Categorical features: {categorical_features}")
        print(f"    Numeric features: {numeric_features}")

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

        # Create full pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver='lbfgs',
                class_weight='balanced'  # Handle imbalanced data
            ))
        ])

        # Train
        print("    Training logistic regression model...")
        self.model.fit(X, y)
        self.is_trained = True

        # Calculate training stats
        y_pred = self.model.predict_proba(X)[:, 1]
        self.training_stats = {
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum()),
            'base_rate': float(y.mean()),
            'mean_pred': float(y_pred.mean()),
            'features': features,
            'n_categorical': len(categorical_features),
            'n_numeric': len(numeric_features)
        }

    def predict_win_rate(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """Predict win probability for given features."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = df[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def get_win_rate_for_segment(
        self,
        segment_values: Dict[str, any]
    ) -> float:
        """
        Get win rate prediction for a single segment.

        Args:
            segment_values: Dict mapping feature name to value
        """
        df = pd.DataFrame([segment_values])
        return self.predict_win_rate(df)[0]
