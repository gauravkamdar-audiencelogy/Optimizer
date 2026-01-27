"""
Bid Landscape Model: Models P(win | bid_amount, segment_features).

V4: Adding bid_amount as a feature enables optimal bid calculation.
The bid coefficient must be POSITIVE (higher bid → higher win rate).

Based on:
- "Bid Shading by Win-Rate Estimation and Surplus Maximization" (arXiv:2009.09259)
- "Optimal Real-Time Bidding for Display Advertising" (Zhang et al., KDD 2014)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional, Tuple

from ..config import OptimizerConfig


class BidLandscapeModel:
    """
    Models P(win | bid_amount, segment_features).

    Uses logistic regression with bid_amount as explicit feature.
    Enables optimal bid calculation via golden section search.

    Key validation: bid_coefficient must be POSITIVE
    (higher bid → higher win rate). If negative, indicates
    confounding and model should not be used.
    """

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.bid_col: str = 'bid_amount_cpm'  # V6 FIX: Store bid column name for predict
        self.bid_coefficient: float = 0.0
        self.bid_scaler_mean: float = 0.0
        self.bid_scaler_std: float = 1.0
        self.is_trained: bool = False
        self.training_stats: Dict = {}

    def train(
        self,
        df_train: pd.DataFrame,
        segment_features: List[str],
        bid_col: str = 'bid_amount_cpm',
        target: str = 'won'
    ) -> None:
        """
        Train model on bid/win data.

        Args:
            df_train: DataFrame with columns [bid_amount_cpm, *segment_features, won]
            segment_features: List of segment feature column names
            bid_col: Column name for bid amount
            target: Target column name ('won')
        """
        self.feature_names = segment_features
        self.bid_col = bid_col  # V6 FIX: Store for predict methods

        # Features = bid_amount + segment features
        X = df_train[[bid_col] + segment_features].copy()
        y = df_train[target].values

        # Store bid statistics for later use
        self.bid_scaler_mean = float(X[bid_col].mean())
        self.bid_scaler_std = float(X[bid_col].std())

        # Identify feature types
        categorical = [f for f in segment_features if X[f].dtype == 'object']
        numeric = [bid_col] + [f for f in segment_features if X[f].dtype != 'object']

        print(f"    Categorical features: {categorical}")
        print(f"    Numeric features: {numeric}")

        # Preprocessing
        transformers = []
        if numeric:
            transformers.append(('num', StandardScaler(), numeric))
        if categorical:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )

        # Model: NO class_weight='balanced' to preserve calibration
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver='lbfgs'
                # NO class_weight - preserves calibration
            ))
        ])

        print("    Training bid landscape model...")
        self.model.fit(X, y)
        self.is_trained = True

        # Extract bid coefficient
        self._extract_bid_coefficient(numeric)

        # Training stats
        y_pred = self.model.predict_proba(X)[:, 1]
        self.training_stats = {
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum()),
            'global_win_rate': float(y.mean()),
            'mean_pred': float(y_pred.mean()),
            'bid_coefficient': float(self.bid_coefficient),
            'bid_coefficient_positive': self.bid_coefficient > 0,
            'features': segment_features,
            'bid_mean': float(self.bid_scaler_mean),
            'bid_std': float(self.bid_scaler_std),
            'model_type': 'logistic_regression_with_bid',
            'usage': 'USED_FOR_BIDDING' if self.bid_coefficient > 0 else 'DISABLED_NEGATIVE_COEFFICIENT'
        }

        if self.bid_coefficient > 0:
            print(f"    ✓ Bid coefficient: {self.bid_coefficient:.4f} (positive - correct)")
        else:
            print(f"    ⚠ Bid coefficient: {self.bid_coefficient:.4f} (NEGATIVE - model disabled)")

    def _extract_bid_coefficient(self, numeric_features: List[str]) -> None:
        """
        Extract coefficient on bid_amount for validation.

        The bid coefficient is the first coefficient in the numeric features
        (after StandardScaler transformation).
        """
        coefs = self.model.named_steps['classifier'].coef_[0]
        # bid_amount_cpm is first in numeric features (index 0 after scaling)
        self.bid_coefficient = float(coefs[0])

    def predict_win_rate(self, bid_amount: float, segment_values: Dict) -> float:
        """
        Predict P(win | bid_amount, segment).

        Args:
            bid_amount: Bid amount in CPM
            segment_values: Dict of segment feature values

        Returns:
            Probability of winning at this bid
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # V6 FIX: Use stored bid column name (not hardcoded 'bid_amount_cpm')
        row = {self.bid_col: bid_amount, **segment_values}
        df = pd.DataFrame([row])
        return float(self.model.predict_proba(df)[0, 1])

    def predict_win_rates_batch(
        self,
        df: pd.DataFrame,
        bid_col: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict win rates for a batch of records.

        Args:
            df: DataFrame with bid column and segment features
            bid_col: Column name for bid amount (defaults to stored name)

        Returns:
            Array of win probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # V6 FIX: Use stored bid column name if not specified
        bid_col = bid_col or self.bid_col
        X = df[[bid_col] + self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def find_optimal_bid(
        self,
        expected_value_cpm: float,
        segment_values: Dict,
        min_bid: Optional[float] = None,
        max_bid: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Find bid that maximizes expected surplus.

        optimal_bid = argmax_b [(EV - b) × P(win | b, segment)]

        Args:
            expected_value_cpm: Expected value of impression in CPM
            segment_values: Dict of segment feature values
            min_bid: Minimum bid to consider (defaults to config min)
            max_bid: Maximum bid to consider (defaults to min(EV, config max))

        Returns:
            (optimal_bid, expected_surplus)
        """
        min_bid = min_bid or self.config.technical.min_bid_cpm
        max_bid = max_bid or min(expected_value_cpm, self.config.technical.max_bid_cpm)

        # Handle edge case where max <= min
        if max_bid <= min_bid:
            win_rate = self.predict_win_rate(min_bid, segment_values)
            surplus = (expected_value_cpm - min_bid) * win_rate
            return min_bid, max(0.0, surplus)

        def surplus(b: float) -> float:
            win_prob = self.predict_win_rate(b, segment_values)
            return (expected_value_cpm - b) * win_prob

        # Golden section search for maximum
        optimal_bid = self._golden_section_search(surplus, min_bid, max_bid)
        optimal_surplus = surplus(optimal_bid)

        return round(optimal_bid, 2), round(max(0.0, optimal_surplus), 4)

    def _golden_section_search(
        self,
        f,
        a: float,
        b: float,
        tol: float = 0.1
    ) -> float:
        """
        Find maximum of unimodal function f on [a, b].

        Uses golden section search which is optimal for unimodal functions.
        """
        phi = (1 + 5**0.5) / 2  # Golden ratio
        c = b - (b - a) / phi
        d = a + (b - a) / phi

        while abs(b - a) > tol:
            if f(c) > f(d):
                b = d
            else:
                a = c
            c = b - (b - a) / phi
            d = a + (b - a) / phi

        return (a + b) / 2

    def get_win_rate_curve(
        self,
        segment_values: Dict,
        bid_range: Tuple[float, float] = (2, 30),
        resolution: float = 1.0
    ) -> Tuple[List[float], List[float]]:
        """
        Generate win rate curve for a segment (for visualization/debugging).

        Args:
            segment_values: Dict of segment feature values
            bid_range: (min_bid, max_bid) range to evaluate
            resolution: Step size for bid evaluation

        Returns:
            (bids, win_rates) lists
        """
        bids = list(np.arange(bid_range[0], bid_range[1], resolution))
        win_rates = [self.predict_win_rate(b, segment_values) for b in bids]
        return bids, win_rates

    def is_valid(self) -> bool:
        """Check if model is valid (trained with positive bid coefficient)."""
        return self.is_trained and self.bid_coefficient > 0

    def derive_exploration_multiplier(
        self,
        current_win_rate: float,
        target_win_rate: float,
        current_median_bid: float
    ) -> Dict:
        """
        Derive the exploration multiplier from the bid landscape.

        This makes the exploration parameters DATA-DRIVEN rather than arbitrary.
        Uses the bid coefficient to calculate how much bid increase is needed
        to move from current win rate to target win rate.

        Math:
            P(win) = sigmoid(β₀ + β₁ × bid_scaled + ...)
            logit(P) = β₀ + β₁ × bid_scaled + ...

            To go from current_wr to target_wr:
            delta_log_odds = logit(target_wr) - logit(current_wr)
            delta_bid_scaled = delta_log_odds / β₁
            delta_bid = delta_bid_scaled × bid_std  (unscale)

            multiplier = (current_bid + delta_bid) / current_bid

        Args:
            current_win_rate: Current global win rate (e.g., 0.30)
            target_win_rate: Target win rate (e.g., 0.65)
            current_median_bid: Current median winning bid (e.g., $12.50)

        Returns:
            Dict with derived parameters and diagnostics
        """
        from scipy.special import logit

        if not self.is_valid():
            return {
                'success': False,
                'reason': 'Model not valid (not trained or negative coefficient)',
                'fallback_multiplier': 1.3  # Conservative fallback
            }

        # Ensure win rates are in valid range for logit
        current_wr = np.clip(current_win_rate, 0.01, 0.99)
        target_wr = np.clip(target_win_rate, 0.01, 0.99)

        # Calculate delta in log-odds space
        delta_log_odds = logit(target_wr) - logit(current_wr)

        # Convert to bid space (unscale using bid_std)
        # bid_coefficient is on scaled data, so: delta_bid_scaled = delta_log_odds / coef
        # then: delta_bid = delta_bid_scaled × bid_std
        delta_bid_scaled = delta_log_odds / self.bid_coefficient
        delta_bid = delta_bid_scaled * self.bid_scaler_std

        # Calculate implied multiplier
        if current_median_bid > 0:
            new_bid = current_median_bid + delta_bid
            implied_multiplier = new_bid / current_median_bid
        else:
            implied_multiplier = 1.0 + delta_log_odds  # Fallback

        # Clip to reasonable range to avoid extreme values from extrapolation
        # Allow higher ceiling since this is data-derived
        clipped_multiplier = np.clip(implied_multiplier, 0.5, 3.0)

        # For the down multiplier (when over-winning), use the same logic
        # but it naturally comes out < 1 when target < current
        # For asymmetry, we can use a fraction of the up multiplier
        down_multiplier_ratio = 0.5  # Bid down at half the rate we bid up
        if implied_multiplier > 1.0:
            down_multiplier = 1.0 - (implied_multiplier - 1.0) * down_multiplier_ratio
        else:
            down_multiplier = implied_multiplier  # Already < 1

        down_multiplier = np.clip(down_multiplier, 0.5, 1.0)

        return {
            'success': True,
            'derived_up_multiplier': float(round(clipped_multiplier, 3)),
            'derived_down_multiplier': float(round(down_multiplier, 3)),
            'raw_multiplier': float(round(implied_multiplier, 3)),
            'delta_log_odds': float(round(delta_log_odds, 4)),
            'delta_bid': float(round(delta_bid, 2)),
            'bid_coefficient': float(round(self.bid_coefficient, 4)),
            'bid_std': float(round(self.bid_scaler_std, 2)),
            'current_wr': float(current_wr),
            'target_wr': float(target_wr),
            'current_median_bid': float(current_median_bid),
            'implied_new_bid': float(round(current_median_bid + delta_bid, 2)),
            'extrapolation_warning': bool(abs(delta_bid) > 2 * self.bid_scaler_std)  # Convert to Python bool
        }

    def find_bid_for_win_rate(
        self,
        target_win_rate: float,
        segment_values: Dict,
        min_bid: Optional[float] = None,
        max_bid: Optional[float] = None,
        tolerance: float = 0.01
    ) -> Tuple[float, float]:
        """
        Find the LOWEST bid that achieves at least target_win_rate.

        V6: This enables volume-first bidding using the bid landscape.
        Instead of heuristics ("bid 30% higher"), we use the model to answer:
        "What bid do I need to achieve 65% win rate?"

        Uses binary search since P(win|bid) is monotonically increasing.

        Args:
            target_win_rate: Target win rate to achieve (e.g., 0.65)
            segment_values: Dict of segment feature values
            min_bid: Minimum bid to consider (defaults to config min)
            max_bid: Maximum bid to consider (defaults to config max)
            tolerance: Stop when within this tolerance of target

        Returns:
            (bid_for_target, actual_win_rate)
            If target is unachievable, returns (max_bid, achieved_win_rate)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if self.bid_coefficient <= 0:
            raise ValueError("Model has non-positive bid coefficient - cannot use for bid optimization")

        min_bid = min_bid or self.config.technical.min_bid_cpm
        max_bid = max_bid or self.config.technical.max_bid_cpm

        # Check if target is achievable at max bid
        max_win_rate = self.predict_win_rate(max_bid, segment_values)
        if max_win_rate < target_win_rate:
            # Target not achievable even at max bid
            return max_bid, max_win_rate

        # Check if we already achieve target at min bid
        min_win_rate = self.predict_win_rate(min_bid, segment_values)
        if min_win_rate >= target_win_rate:
            # Already achieving target at floor
            return min_bid, min_win_rate

        # Binary search for the bid that achieves target
        low, high = min_bid, max_bid
        while high - low > 0.1:  # $0.10 precision
            mid = (low + high) / 2
            mid_win_rate = self.predict_win_rate(mid, segment_values)

            if mid_win_rate < target_win_rate:
                low = mid
            else:
                high = mid

        # Return the slightly higher value to ensure we meet target
        result_bid = round(high, 2)
        result_win_rate = self.predict_win_rate(result_bid, segment_values)

        return result_bid, result_win_rate

    def find_bid_for_win_rate_with_buffer(
        self,
        target_win_rate: float,
        segment_values: Dict,
        observation_count: int,
        min_bid: Optional[float] = None,
        max_bid: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Find bid for target win rate with uncertainty buffer.

        V6: For sparse segments, we over-bid slightly to account for
        model uncertainty. Buffer scales inversely with observation count.

        Args:
            target_win_rate: Target win rate to achieve
            segment_values: Dict of segment feature values
            observation_count: Number of observations for this segment
            min_bid: Minimum bid (defaults to config)
            max_bid: Maximum bid (defaults to config)

        Returns:
            (buffered_bid, base_bid, buffer_multiplier)
        """
        base_bid, actual_wr = self.find_bid_for_win_rate(
            target_win_rate, segment_values, min_bid, max_bid
        )

        # Buffer based on observation count (more uncertainty → higher buffer)
        if observation_count < 10:
            buffer = 1.10  # +10% for very sparse
        elif observation_count < 50:
            buffer = 1.05  # +5% for sparse
        else:
            buffer = 1.02  # +2% for well-observed

        buffered_bid = min(base_bid * buffer, max_bid or self.config.technical.max_bid_cpm)

        return round(buffered_bid, 2), round(base_bid, 2), buffer
