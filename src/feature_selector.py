"""
V3: Dynamic feature selection with auto soft-exclusion.

V3 Changes:
- Split exclusions: hard (config) vs soft (algorithm-detected)
- Auto-exclude features below signal threshold
- Log all exclusion decisions for transparency

Key principle: Algorithm should make data-driven decisions.
Hardcoding "hour_of_day is low signal" in config violates this.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .config import OptimizerConfig


@dataclass
class FeatureScore:
    """Score for a single feature."""
    name: str
    cardinality: int
    null_pct: float
    win_rate_variance: float
    signal_score: float
    coverage_at_threshold: float = 0.0
    values_with_sufficient_obs: int = 0
    effective_cardinality: int = 0        # V3.1: Count of values with >= min_share
    dominant_value: str = ""              # V3.1: Most common value
    dominant_share: float = 0.0           # V3.1: Share of most common value


@dataclass
class ExclusionRecord:
    """Record of why a feature was excluded."""
    feature: str
    exclusion_type: str  # 'hard' or 'soft'
    reason: str


class FeatureSelector:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.feature_scores: Dict[str, FeatureScore] = {}
        self.selected_features: List[str] = []
        self.excluded_features: List[ExclusionRecord] = []  # V3: Track exclusions
        self.n_samples: int = 0
        self.current_segment_count: int = 1

    def select_features(
        self,
        df_train: pd.DataFrame,
        target_col: str = 'won'
    ) -> List[str]:
        """
        Select optimal features for memcache key.

        V3 Algorithm:
        1. Start with anchor features (always included)
        2. Filter out hard exclusions (from config)
        3. Score all remaining candidates
        4. Auto soft-exclude features below thresholds (data-driven)
        5. Greedily add best remaining features

        Returns:
            List of feature names for memcache key
        """
        self.n_samples = len(df_train)
        self.excluded_features = []  # Reset

        # Start with anchor features
        self.selected_features = list(self.config.features.anchor_features)
        print(f"    Anchor features: {self.selected_features}")

        # Calculate current segment count from anchor features
        if self.selected_features:
            self.current_segment_count = df_train.groupby(self.selected_features).ngroups
        else:
            self.current_segment_count = 1

        print(f"    Initial segment count: {self.current_segment_count}")
        print(f"    Data size: {self.n_samples:,} samples")

        # Get candidates (not anchor, not hard-excluded, in data)
        hard_exclude = set(self.config.features.hard_exclude_features)
        candidates = [
            f for f in self.config.features.candidate_features
            if f not in self.selected_features
            and f not in hard_exclude
            and f in df_train.columns
        ]

        # Log hard exclusions
        for f in self.config.features.candidate_features:
            if f in hard_exclude:
                self.excluded_features.append(ExclusionRecord(
                    feature=f,
                    exclusion_type='hard',
                    reason='In hard_exclude_features config'
                ))

        print(f"    Evaluating {len(candidates)} candidate features...")

        # Score all candidates
        for feat in candidates:
            score = self._score_feature(df_train, feat, target_col)
            if score is not None:
                self.feature_scores[feat] = score
                print(f"      {feat}: score={score.signal_score:.2f}, "
                      f"eff_card={score.effective_cardinality}, "
                      f"dominant={score.dominant_share:.1%}, "
                      f"coverage={score.coverage_at_threshold:.1%}")

        # V3: Auto soft-exclude features below thresholds
        soft_excluded = []
        for feat, score in list(self.feature_scores.items()):
            exclude_reason = self._check_soft_exclusion(score)
            if exclude_reason:
                soft_excluded.append(feat)
                self.excluded_features.append(ExclusionRecord(
                    feature=feat,
                    exclusion_type='soft',
                    reason=exclude_reason
                ))
                print(f"    AUTO-EXCLUDED '{feat}': {exclude_reason}")

        # Remove soft-excluded from consideration
        for feat in soft_excluded:
            del self.feature_scores[feat]

        # Greedily add best remaining features
        remaining = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1].signal_score,
            reverse=True
        )

        for feat_name, score in remaining:
            if len(self.selected_features) >= self.config.technical.max_features:
                break

            # Check coverage with this feature added
            test_features = self.selected_features + [feat_name]
            coverage, sparse_combos = self._check_coverage(df_train, test_features)

            # Add if coverage acceptable
            if coverage >= self.config.technical.min_coverage_at_threshold:
                self.selected_features.append(feat_name)
                self.current_segment_count = df_train.groupby(self.selected_features).ngroups
                print(f"    Added '{feat_name}' (coverage={coverage:.1%}, "
                      f"segments={self.current_segment_count:,})")

        # Print exclusion summary
        self._print_exclusion_summary()

        return self.selected_features

    def _check_soft_exclusion(self, score: FeatureScore) -> str:
        """
        Check if a feature should be auto-excluded based on thresholds.

        Returns:
            Exclusion reason string if should exclude, None otherwise
        """
        min_signal = self.config.technical.min_signal_score
        min_coverage = self.config.technical.min_coverage_at_threshold
        max_null = self.config.technical.max_null_pct
        min_eff_card = self.config.technical.min_effective_cardinality

        # V3.1: Check effective cardinality first (most important filter)
        if score.effective_cardinality < min_eff_card:
            return (f"effective_cardinality={score.effective_cardinality} < {min_eff_card} "
                    f"({score.dominant_share:.1%} in '{score.dominant_value}')")

        if score.signal_score < min_signal:
            return f"signal_score={score.signal_score:.2f} < threshold={min_signal}"

        if score.coverage_at_threshold < min_coverage:
            return f"coverage={score.coverage_at_threshold:.1%} < threshold={min_coverage:.0%}"

        if score.null_pct > max_null:
            return f"null_pct={score.null_pct:.1f}% > threshold={max_null}%"

        return None

    def _score_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        target_col: str
    ) -> FeatureScore:
        """
        Calculate signal score for a feature.

        Score = variance × (1 - null_pct) × cardinality_penalty × coverage_weight
        """
        if feature not in df.columns:
            return None

        # Calculate metrics
        cardinality = df[feature].nunique()
        null_pct = df[feature].isnull().sum() / len(df) * 100

        # Skip features with extreme nulls (will be soft-excluded anyway)
        if null_pct > 50:
            return None

        # Calculate win rate variance across feature values
        try:
            win_rate_by_val = df.groupby(feature)[target_col].mean()
            win_rate_variance = win_rate_by_val.var()
            if pd.isna(win_rate_variance):
                win_rate_variance = 0
        except Exception:
            win_rate_variance = 0

        # Calculate coverage at min_observations threshold
        min_obs = self.config.technical.min_observations
        value_counts = df[feature].value_counts()
        values_with_sufficient_obs = (value_counts >= min_obs).sum()
        coverage_at_threshold = value_counts[value_counts >= min_obs].sum() / len(df)

        # V3.1: Calculate effective cardinality (values with >= min_share)
        min_share = self.config.technical.effective_cardinality_min_share
        value_shares = value_counts / len(df)
        effective_cardinality = (value_shares >= min_share).sum()
        dominant_value = str(value_counts.index[0]) if len(value_counts) > 0 else ""
        dominant_share = float(value_shares.iloc[0]) if len(value_shares) > 0 else 0.0

        # Data-driven optimal cardinality
        optimal_cardinality = self.n_samples / min_obs / max(self.current_segment_count, 1)
        optimal_cardinality = max(5, min(50, optimal_cardinality))

        # Cardinality penalty
        log_ratio = np.log(cardinality / optimal_cardinality)
        cardinality_penalty = 1.0 / (1.0 + abs(log_ratio))

        # Coverage weight (quadratic)
        coverage_weight = coverage_at_threshold ** 2

        # Final signal score
        signal_score = (
            win_rate_variance * 10000 *
            (1 - null_pct / 100) *
            cardinality_penalty *
            coverage_weight
        )

        return FeatureScore(
            name=feature,
            cardinality=cardinality,
            null_pct=null_pct,
            win_rate_variance=win_rate_variance,
            signal_score=signal_score,
            coverage_at_threshold=coverage_at_threshold,
            values_with_sufficient_obs=values_with_sufficient_obs,
            effective_cardinality=effective_cardinality,
            dominant_value=dominant_value,
            dominant_share=dominant_share
        )

    def _check_coverage(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Tuple[float, int]:
        """Check coverage and sparsity for a feature combination."""
        min_obs = self.config.technical.min_observations
        combo_sizes = df.groupby(features).size()
        sufficient = combo_sizes >= min_obs
        coverage = combo_sizes[sufficient].sum() / len(df)
        sparse_combos = (~sufficient).sum()
        return coverage, sparse_combos

    def _print_exclusion_summary(self):
        """Print summary of all exclusions."""
        hard = [e for e in self.excluded_features if e.exclusion_type == 'hard']
        soft = [e for e in self.excluded_features if e.exclusion_type == 'soft']

        if hard or soft:
            print(f"\n    Feature Selection Summary:")
            print(f"      INCLUDED: {self.selected_features}")

            if soft:
                print(f"      AUTO-EXCLUDED (data-driven):")
                for e in soft:
                    print(f"        - {e.feature}: {e.reason}")

            if hard:
                print(f"      HARD-EXCLUDED (config):")
                for e in hard:
                    print(f"        - {e.feature}")

    def get_selection_report(self) -> Dict:
        """Generate report on feature selection."""
        return {
            'selected_features': self.selected_features,
            'feature_scores': {
                name: {
                    'cardinality': int(score.cardinality),
                    'null_pct': round(float(score.null_pct), 2),
                    'win_rate_variance': round(float(score.win_rate_variance), 6),
                    'signal_score': round(float(score.signal_score), 4),
                    'coverage_at_threshold': round(float(score.coverage_at_threshold), 4),
                    'values_with_sufficient_obs': int(score.values_with_sufficient_obs),
                    'effective_cardinality': int(score.effective_cardinality),
                    'dominant_value': score.dominant_value,
                    'dominant_share': round(float(score.dominant_share), 4)
                }
                for name, score in self.feature_scores.items()
            },
            'exclusions': {
                'hard': [
                    {'feature': e.feature, 'reason': e.reason}
                    for e in self.excluded_features if e.exclusion_type == 'hard'
                ],
                'soft': [
                    {'feature': e.feature, 'reason': e.reason}
                    for e in self.excluded_features if e.exclusion_type == 'soft'
                ]
            },
            'num_features': len(self.selected_features),
            'max_features': self.config.technical.max_features,
            'total_samples': int(self.n_samples),
            'soft_exclusion_thresholds': {
                'min_signal_score': self.config.technical.min_signal_score,
                'min_coverage': self.config.technical.min_coverage_at_threshold,
                'max_null_pct': self.config.technical.max_null_pct,
                'min_effective_cardinality': self.config.technical.min_effective_cardinality,
                'effective_cardinality_min_share': self.config.technical.effective_cardinality_min_share
            }
        }
