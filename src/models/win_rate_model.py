"""
Win Rate Model: Predicts P(win | features).

V6 UPDATE: Data-agnostic pipeline with calibration gate.
    - Removed class_weight='balanced' (destroys calibration)
    - Added isotonic regression post-hoc calibration
    - Model usage decided at RUNTIME by calibration gate, not hard-coded

    The calibration gate checks ECE at runtime:
    - If ECE < threshold (default 0.10) → model is used
    - If ECE >= threshold → fall back to empirical model

    This approach is data-agnostic - pipeline adapts to data quality.

Uses logistic regression for simplicity and interpretability.
Based on: "Bid Shading by Win-Rate Estimation and Surplus Maximization" (arXiv:2009.09259)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from typing import List, Dict, Optional, Tuple

from ..config import OptimizerConfig


class WinRateModel:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.model: Optional[Pipeline] = None
        self.calibrator: Optional[IsotonicRegression] = None  # Post-hoc calibration
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.is_calibrated: bool = False
        self.training_stats: Dict = {}
        self.calibration_ece: Optional[float] = None  # ECE after calibration
        self.passes_gate: bool = False  # Whether model passes calibration gate

    def train(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        target: str = 'won'
    ) -> None:
        """
        Train win rate model with post-hoc isotonic calibration.

        V6: Removed class_weight='balanced' (destroys calibration).
        Added isotonic regression for post-hoc calibration.

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

        # V6: NO class_weight - it destroys calibration by inflating minority class ~1000x
        # Instead, we use post-hoc isotonic calibration
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver='lbfgs'
                # NOTE: class_weight='balanced' REMOVED - destroys calibration
            ))
        ])

        # Train base model
        print("    Training logistic regression model (no class_weight)...")
        self.model.fit(X, y)
        self.is_trained = True

        # Get raw predictions for calibration
        y_pred_raw = self.model.predict_proba(X)[:, 1]

        # V6 FIX: Use cross-validation for ECE evaluation
        # Fitting isotonic on training and evaluating on same data is tautological (ECE≈0)
        # Instead: use 5-fold CV to get out-of-sample calibrated predictions
        print("    Computing cross-validated calibration for ECE evaluation...")
        try:
            y_pred_cv = cross_val_predict(
                IsotonicRegression(out_of_bounds='clip'),
                y_pred_raw.reshape(-1, 1),
                y,
                cv=5,
                method='predict'
            )
            # Calculate ECE on held-out predictions (not tautological)
            self.calibration_ece = self._calculate_ece(y, y_pred_cv)
        except Exception as e:
            print(f"    Warning: CV calibration failed ({e}), using in-sample ECE")
            self.calibration_ece = self._calculate_ece(y, y_pred_raw)

        # Still fit calibrator on full data for inference
        print("    Fitting isotonic calibration on full data...")
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(y_pred_raw, y)
        self.is_calibrated = True

        # Get calibrated predictions (for stats only, not for ECE)
        y_pred_calibrated = self.calibrator.predict(y_pred_raw)

        # Check calibration gate
        ece_threshold = self.config.calibration_gate.max_ece_threshold
        self.passes_gate = self.calibration_ece < ece_threshold

        gate_status = "PASSED" if self.passes_gate else "FAILED"
        print(f"    Calibration ECE: {self.calibration_ece:.4f} (threshold: {ece_threshold})")
        print(f"    Calibration gate: {gate_status}")

        if self.config.calibration_gate.log_gate_decisions:
            if self.passes_gate:
                print(f"    → LogReg model WILL be used (ECE {self.calibration_ece:.4f} < {ece_threshold})")
            else:
                print(f"    → LogReg model will NOT be used, falling back to empirical (ECE {self.calibration_ece:.4f} >= {ece_threshold})")

        # Calculate training stats
        self.training_stats = {
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum()),
            'base_rate': float(y.mean()),
            'mean_pred_raw': float(y_pred_raw.mean()),
            'mean_pred_calibrated': float(y_pred_calibrated.mean()),
            'features': features,
            'n_categorical': len(categorical_features),
            'n_numeric': len(numeric_features),
            'calibration_ece': float(self.calibration_ece),
            'passes_calibration_gate': bool(self.passes_gate),  # Convert to Python bool
            'ece_threshold': float(ece_threshold),
            'calibration_method': 'isotonic_regression'
        }

    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE = Σ (|bin_size| / N) × |avg_pred - avg_actual|

        This is the weighted average of calibration errors across bins.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_boundaries[1:-1])

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_pred_mean = y_pred[mask].mean()
            bin_true_mean = y_true[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_pred_mean - bin_true_mean)

        return ece

    def predict_win_rate(
        self,
        df: pd.DataFrame,
        use_calibration: bool = True
    ) -> np.ndarray:
        """
        Predict win probability for given features.

        Args:
            df: DataFrame with feature columns
            use_calibration: If True and calibrator is trained, apply isotonic calibration

        Returns:
            Calibrated win probabilities (if calibration enabled and trained)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = df[self.feature_names]
        y_pred_raw = self.model.predict_proba(X)[:, 1]

        # Apply calibration if available and requested
        if use_calibration and self.is_calibrated and self.calibrator is not None:
            return self.calibrator.predict(y_pred_raw)
        else:
            return y_pred_raw

    def predict_win_rate_raw(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """Predict raw (uncalibrated) win probability. For diagnostics."""
        return self.predict_win_rate(df, use_calibration=False)

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
