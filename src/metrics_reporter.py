"""
V5: Metrics reporter with exploration mode support.

V5 Changes:
- Report asymmetric exploration statistics
- Track exploration direction (up/down) distribution
- Report V5 method usage stats
- Include global stats for baseline

V4 Features (retained for diagnostics):
- Report BidLandscapeModel metrics (coefficient, calibration by bid bucket)

V3 Features (retained):
- Report both LogReg (diagnostic) and Empirical win rate metrics
- Track soft vs hard feature exclusions
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, average_precision_score

from .config import OptimizerConfig
from .data_loader import DataLoader
from .feature_selector import FeatureSelector
from .models.win_rate_model import WinRateModel
from .models.empirical_win_rate_model import EmpiricalWinRateModel
from .models.ctr_model import CTRModel
from .models.bid_landscape_model import BidLandscapeModel  # V4: New
from .bid_calculator import BidResult
from .memcache_builder import MemcacheBuilder


class MetricsReporter:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.metrics: Dict[str, Any] = {}

    def _calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
        adaptive: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate calibration metrics for a model.

        Args:
            adaptive: If True, use percentile-based binning for imbalanced predictions.
                     Recommended for CTR models where 99%+ predictions cluster near 0.
                     (Nixon et al. CVPR 2019)
        """
        if len(np.unique(y_true)) < 2:
            return {'error': 'Insufficient class diversity'}

        # Adaptive binning for imbalanced predictions (e.g., CTR with 0.035% rate)
        if adaptive:
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_boundaries = np.percentile(y_pred, percentiles)
            bin_boundaries = np.unique(bin_boundaries)  # Remove duplicates from ties
            if len(bin_boundaries) < 2:
                bin_boundaries = np.array([y_pred.min(), y_pred.max()])
            actual_n_bins = len(bin_boundaries) - 1
        else:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            actual_n_bins = n_bins

        bin_indices = np.digitize(y_pred, bin_boundaries[1:-1])

        calibration_curve = {}
        ece = 0.0
        mce = 0.0

        for i in range(actual_n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_pred_mean = y_pred[mask].mean()
            bin_true_mean = y_true[mask].mean()
            bin_size = mask.sum()
            bin_weight = bin_size / len(y_true)

            calibration_error = abs(bin_pred_mean - bin_true_mean)
            ece += bin_weight * calibration_error
            mce = max(mce, calibration_error)

            # Format bin key based on prediction range (more precision for small values)
            if adaptive and bin_boundaries[i] < 0.01:
                bin_key = f"{bin_boundaries[i]:.6f}-{bin_boundaries[i+1]:.6f}"
            else:
                bin_key = f"{bin_boundaries[i]:.2f}-{bin_boundaries[i+1]:.2f}"
            calibration_curve[bin_key] = {
                'predicted': round(float(bin_pred_mean), 6),
                'actual': round(float(bin_true_mean), 6),
                'count': int(bin_size),
                'calibration_error': round(float(calibration_error), 6)
            }

        try:
            brier = brier_score_loss(y_true, y_pred)
        except Exception:
            brier = None

        return {
            'expected_calibration_error': round(float(ece), 6),
            'max_calibration_error': round(float(mce), 6),
            'brier_score': round(float(brier), 6) if brier is not None else None,
            'calibration_curve': calibration_curve,
            'n_bins_with_data': len(calibration_curve),
            'binning_method': 'adaptive_percentile' if adaptive else 'uniform'
        }

    def _calculate_ranking_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Calculate discrimination/ranking metrics.

        Includes:
        - AUC-ROC: Ranking ability (ignores calibration)
        - PR-AUC: Better for imbalanced data (baseline = prevalence)
        - Log Loss: Probabilistic accuracy
        - Normalized Entropy: Comparison to baseline (Facebook 2014)
        - RIG: Relative Information Gain = 1 - NE (more intuitive)
        - Calibration Sum Ratio: Total predicted / total actual
        """
        if len(np.unique(y_true)) < 2:
            return {'error': 'Insufficient class diversity for AUC'}

        # Sample size tracking
        n_positive = int(y_true.sum())
        sample_warning = None
        if n_positive < 1000:
            sample_warning = f"Only {n_positive} positive events - metrics may be unstable"

        # AUC-ROC
        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = None

        # PR-AUC (better for imbalanced data like CTR)
        try:
            pr_auc = average_precision_score(y_true, y_pred)
        except Exception:
            pr_auc = None

        # Log Loss
        try:
            ll = log_loss(y_true, y_pred)
        except Exception:
            ll = None

        # Baseline entropy and Normalized Entropy
        base_rate = y_true.mean()
        if base_rate > 0 and base_rate < 1:
            baseline_entropy = -base_rate * np.log(base_rate + 1e-10) - (1 - base_rate) * np.log(1 - base_rate + 1e-10)
        else:
            baseline_entropy = 0

        if ll is not None and baseline_entropy > 0:
            normalized_entropy = ll / baseline_entropy
        else:
            normalized_entropy = None

        # RIG: Relative Information Gain = 1 - NE (positive = better than baseline)
        rig = None
        if normalized_entropy is not None:
            rig = 1 - normalized_entropy

        # Calibration sum ratio: Σpredictions / Σactuals
        calibration_sum_ratio = float(y_pred.sum() / max(y_true.sum(), 1e-10))

        return {
            'auc_roc': round(float(auc), 4) if auc is not None else None,
            'pr_auc': round(float(pr_auc), 4) if pr_auc is not None else None,
            'log_loss': round(float(ll), 6) if ll is not None else None,
            'normalized_entropy': round(float(normalized_entropy), 4) if normalized_entropy is not None else None,
            'relative_information_gain': round(float(rig), 4) if rig is not None else None,
            'calibration_sum_ratio': round(calibration_sum_ratio, 4),
            'baseline_entropy': round(float(baseline_entropy), 6),
            'n_positive_events': n_positive,
            'sample_warning': sample_warning
        }

    def _calculate_economic_metrics(
        self,
        bid_results: List
    ) -> Dict[str, Any]:
        """Calculate economic impact metrics (V5: supports exploration adjustments)."""
        if not bid_results:
            return {}

        total_segments = len(bid_results)
        profitable_segments = sum(1 for r in bid_results if r.is_profitable)
        unprofitable_segments = total_segments - profitable_segments

        raw_bids = [r.raw_bid for r in bid_results]
        final_bids = [r.final_bid for r in bid_results]
        evs = [r.expected_value_cpm for r in bid_results]
        win_rates = [r.win_rate for r in bid_results]

        # V5: Get adjustments (exploration_adjustment or win_rate_adjustment)
        adjustments = []
        for r in bid_results:
            if hasattr(r, 'exploration_adjustment'):
                adjustments.append(r.exploration_adjustment)
            elif hasattr(r, 'win_rate_adjustment'):
                adjustments.append(r.win_rate_adjustment)
            else:
                adjustments.append(1.0)

        min_bid = self.config.technical.min_bid_cpm
        max_bid = self.config.technical.max_bid_cpm

        clipped_to_floor = sum(1 for r in bid_results if min_bid is not None and r.raw_bid < min_bid)
        clipped_to_ceiling = sum(1 for r in bid_results if max_bid is not None and r.raw_bid > max_bid)

        metrics = {
            'total_segments': total_segments,
            'profitable_segments': profitable_segments,
            'unprofitable_segments': unprofitable_segments,
            'pct_profitable': round(profitable_segments / total_segments * 100, 2) if total_segments > 0 else 0,

            'bid_clipping': {
                'clipped_to_floor': clipped_to_floor,
                'clipped_to_ceiling': clipped_to_ceiling,
                'pct_at_floor': round(clipped_to_floor / total_segments * 100, 2) if total_segments > 0 else 0,
                'pct_at_ceiling': round(clipped_to_ceiling / total_segments * 100, 2) if total_segments > 0 else 0,
                'pct_natural': round((total_segments - clipped_to_floor - clipped_to_ceiling) / total_segments * 100, 2) if total_segments > 0 else 0
            },

            'raw_bid_distribution': {
                'min': round(min(raw_bids), 4),
                'max': round(max(raw_bids), 4),
                'mean': round(float(np.mean(raw_bids)), 4),
                'median': round(float(np.median(raw_bids)), 4),
                'std': round(float(np.std(raw_bids)), 4)
            },

            'expected_value_distribution': {
                'min': round(min(evs), 4),
                'max': round(max(evs), 4),
                'mean': round(float(np.mean(evs)), 4),
                'median': round(float(np.median(evs)), 4)
            },

            'win_rate_distribution': {
                'min': round(min(win_rates), 4),
                'max': round(max(win_rates), 4),
                'mean': round(float(np.mean(win_rates)), 4),
                'median': round(float(np.median(win_rates)), 4)
            },

            # V5: Exploration/win rate adjustment distribution
            'adjustment_distribution': {
                'min': round(min(adjustments), 4),
                'max': round(max(adjustments), 4),
                'mean': round(float(np.mean(adjustments)), 4),
                'median': round(float(np.median(adjustments)), 4),
                'at_min_bound': sum(1 for a in adjustments if a <= self.config.technical.min_win_rate_adjustment + 0.01),
                'at_max_bound': sum(1 for a in adjustments if a >= self.config.technical.max_win_rate_adjustment - 0.01)
            }
        }

        # V5: Exploration direction breakdown
        exploration_up = sum(1 for r in bid_results if getattr(r, 'exploration_direction', '') == 'up')
        exploration_down = sum(1 for r in bid_results if getattr(r, 'exploration_direction', '') == 'down')
        exploration_neutral = sum(1 for r in bid_results if getattr(r, 'exploration_direction', '') == 'neutral')

        if exploration_up > 0 or exploration_down > 0:
            metrics['exploration_direction_breakdown'] = {
                'bid_up': exploration_up,
                'bid_down': exploration_down,
                'neutral': exploration_neutral,
                'pct_bid_up': round(exploration_up / total_segments * 100, 2) if total_segments > 0 else 0,
                'pct_bid_down': round(exploration_down / total_segments * 100, 2) if total_segments > 0 else 0
            }

        # V5: Bid method breakdown
        v5_methods = ['v5_explore_zero', 'v5_explore_low', 'v5_explore_medium', 'v5_empirical']
        v4_methods = ['landscape', 'empirical']

        method_breakdown = {}
        for method in v5_methods + v4_methods:
            count = sum(1 for r in bid_results if getattr(r, 'bid_method', '') == method)
            if count > 0:
                method_breakdown[f'{method}_count'] = count
                method_breakdown[f'{method}_pct'] = round(count / total_segments * 100, 2)

        if method_breakdown:
            metrics['bid_method_breakdown'] = method_breakdown

        return metrics

    def _calculate_bid_landscape_calibration(
        self,
        model: BidLandscapeModel,
        df_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        V4: Calculate calibration metrics for bid landscape model.

        Key check: Does higher bid → higher win rate (monotonicity)?
        """
        if not model.is_trained:
            return {'error': 'Model not trained'}

        # Predict on test set
        try:
            predictions = model.predict_win_rates_batch(df_test)
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

        y_true = df_test['won'].values

        # Overall calibration
        global_calibration_ratio = float(predictions.mean() / max(y_true.mean(), 1e-10))

        # Calibration by bid bucket
        df_cal = df_test.copy()
        df_cal['pred_win_prob'] = predictions

        try:
            df_cal['bid_bucket'] = pd.qcut(
                df_cal['bid_amount_cpm'],
                q=5,
                labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                duplicates='drop'
            )
        except ValueError:
            # Not enough unique values for 5 buckets
            df_cal['bid_bucket'] = pd.cut(
                df_cal['bid_amount_cpm'],
                bins=3,
                labels=['low', 'medium', 'high']
            )

        calibration_by_bid = df_cal.groupby('bid_bucket', observed=True).agg({
            'pred_win_prob': 'mean',
            'won': 'mean',
            'bid_amount_cpm': ['mean', 'count']
        })
        calibration_by_bid.columns = ['predicted', 'actual', 'avg_bid', 'count']
        calibration_by_bid['error'] = calibration_by_bid['predicted'] - calibration_by_bid['actual']

        # Check monotonicity: does actual win rate increase with bid?
        actual_by_bucket = calibration_by_bid['actual'].values
        is_monotonic = all(
            actual_by_bucket[i] <= actual_by_bucket[i+1]
            for i in range(len(actual_by_bucket)-1)
        )

        # Convert to serializable dict
        calibration_dict = {}
        for bucket in calibration_by_bid.index:
            row = calibration_by_bid.loc[bucket]
            calibration_dict[str(bucket)] = {
                'predicted': round(float(row['predicted']), 4),
                'actual': round(float(row['actual']), 4),
                'avg_bid': round(float(row['avg_bid']), 2),
                'count': int(row['count']),
                'error': round(float(row['error']), 4)
            }

        return {
            'bid_coefficient': round(float(model.bid_coefficient), 4),
            'bid_coefficient_positive': model.bid_coefficient > 0,
            'global_calibration_ratio': round(global_calibration_ratio, 4),
            'is_monotonic_by_bid': is_monotonic,
            'mean_absolute_calibration_error': round(
                float(abs(calibration_by_bid['error']).mean()), 4
            ),
            'calibration_by_bid_bucket': calibration_dict
        }

    def _calculate_ctr_calibration_ratio(
        self,
        mean_pred_ctr: float,
        global_ctr: float
    ) -> Dict[str, Any]:
        """Critical check: Is predicted CTR close to actual CTR?"""
        if global_ctr == 0:
            return {'error': 'Global CTR is zero'}

        ratio = mean_pred_ctr / global_ctr

        if 0.5 <= ratio <= 2.0:
            status = 'GOOD'
        elif 0.2 <= ratio <= 5.0:
            status = 'WARNING'
        else:
            status = 'CRITICAL'

        return {
            'mean_pred_ctr': round(float(mean_pred_ctr), 8),
            'global_ctr': round(float(global_ctr), 8),
            'ratio': round(float(ratio), 4),
            'status': status,
            'interpretation': f"Model predicts {ratio:.1f}x the actual CTR rate"
        }

    def compile_metrics(
        self,
        run_id: str,
        data_loader: DataLoader,
        feature_selector: FeatureSelector,
        logreg_win_rate_model: WinRateModel,      # V3: For diagnostic comparison
        empirical_win_rate_model: EmpiricalWinRateModel,  # V3/V4/V5 empirical
        ctr_model: CTRModel,
        bid_results: List,  # V5: Can be BidResult or BidResultV5
        memcache_path: Path,
        memcache_builder: MemcacheBuilder,
        df_train_wr: Optional[pd.DataFrame] = None,
        df_train_ctr: Optional[pd.DataFrame] = None,
        bid_landscape_model: Optional[BidLandscapeModel] = None,  # V4: Optional
        bid_variance_stats: Optional[Dict] = None,  # V4: Optional
        method_stats: Optional[Dict] = None,  # V4/V5: Method usage
        exploration_summary: Optional[Dict] = None,  # V5: Exploration stats
        global_stats: Optional[Dict] = None  # V5: Global baseline stats
    ) -> Dict[str, Any]:
        """Compile all metrics into single report."""

        filter_stats = memcache_builder.get_filter_stats()

        # Determine version based on exploration mode
        v5_enabled = exploration_summary is not None
        v4_enabled = bid_landscape_model is not None and bid_landscape_model.is_valid()

        if v5_enabled:
            version = 'v5_volume_first_exploration'
        elif v4_enabled:
            version = 'v4_bid_landscape_hybrid'
        else:
            version = 'v3_empirical_fallback'

        self.metrics = {
            'run_info': {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'memcache_file': str(memcache_path.name),
                'version': version,
                'v5_exploration_enabled': v5_enabled,
                'v4_landscape_enabled': v4_enabled
            },

            # Config with win rate controls
            'config': {
                'business': {
                    'target_margin': self.config.business.target_margin,
                    'target_win_rate': self.config.business.target_win_rate,
                    'win_rate_sensitivity': self.config.business.win_rate_sensitivity,
                    # V5: Exploration mode config
                    'exploration_mode': self.config.business.exploration_mode,
                    'exploration_up_multiplier': self.config.business.exploration_up_multiplier,
                    'exploration_down_multiplier': self.config.business.exploration_down_multiplier,
                    'accept_negative_margin': self.config.business.accept_negative_margin,
                    'max_negative_margin_pct': self.config.business.max_negative_margin_pct
                },
                'technical': {
                    'min_bid_cpm': self.config.technical.min_bid_cpm,
                    'max_bid_cpm': self.config.technical.max_bid_cpm,
                    'default_bid_cpm': self.config.technical.default_bid_cpm,
                    'min_observations': self.config.technical.min_observations,
                    'max_features': self.config.technical.max_features,
                    'ctr_shrinkage_k': self.config.technical.ctr_shrinkage_k,
                    'win_rate_shrinkage_k': self.config.technical.win_rate_shrinkage_k,
                    'min_signal_score': self.config.technical.min_signal_score,
                    'min_win_rate_adjustment': self.config.technical.min_win_rate_adjustment,
                    'max_win_rate_adjustment': self.config.technical.max_win_rate_adjustment,
                    # V5: Tiered observation thresholds
                    'min_observations_for_empirical': self.config.technical.min_observations_for_empirical,
                    'min_observations_for_landscape': self.config.technical.min_observations_for_landscape,
                    'exploration_bonus_zero_obs': self.config.technical.exploration_bonus_zero_obs,
                    'exploration_bonus_low_obs': self.config.technical.exploration_bonus_low_obs,
                    'exploration_bonus_medium_obs': self.config.technical.exploration_bonus_medium_obs
                }
            },

            # Bid formula documentation
            'bid_formula': {
                'version': 'v5_exploration' if v5_enabled else ('v4_hybrid' if v4_enabled else 'v3_empirical_only'),
                'v5_formula': 'bid = base_bid × exploration_adjustment × npi_multiplier',
                'v5_exploration_adjustment': {
                    'under_winning': '1.0 + wr_gap × 1.3 (bid UP)',
                    'over_winning': '1.0 + wr_gap × 0.7 (bid DOWN)'
                },
                'v4_formula': 'optimal_bid = argmax_b [(EV_cpm - b) × P(win | b, segment)]',
                'v3_fallback_formula': 'bid = EV_cpm × (1 - target_margin) × win_rate_adjustment',
                'shading_factor': 1 - self.config.business.target_margin,
                'adjustment_bounds': [self.config.technical.min_win_rate_adjustment,
                                     self.config.technical.max_win_rate_adjustment]
            },

            # V5: Exploration statistics
            'exploration_stats': exploration_summary or {},

            # V5: Global baseline stats
            'global_stats': global_stats or {},

            'data_stats': data_loader.load_stats,

            'feature_selection': feature_selector.get_selection_report(),

            'bidder_config': {
                'feature_columns': feature_selector.selected_features,
                'feature_column_order': feature_selector.selected_features,
                'num_segments': len(bid_results),
                'bid_column': 'suggested_bid_cpm'
            },

            # V4: All model types reported
            'model_performance': {
                'bid_landscape_model': (
                    {
                        **bid_landscape_model.training_stats,
                        'usage': 'USED_FOR_BIDDING - V4 optimal bid calculation'
                    } if bid_landscape_model is not None else {
                        'status': 'NOT_TRAINED',
                        'reason': bid_variance_stats.get('reason', 'Unknown') if bid_variance_stats else 'Not provided'
                    }
                ),
                'win_rate_model_empirical': {
                    **empirical_win_rate_model.training_stats,
                    'usage': 'V3_FALLBACK - empirical rates with shrinkage'
                },
                'win_rate_model_logreg': {
                    **logreg_win_rate_model.training_stats,
                    'usage': 'DIAGNOSTICS_ONLY - for calibration comparison'
                },
                'ctr_model': ctr_model.training_stats
            },

            # V4: Bid variance statistics
            'bid_variance_stats': bid_variance_stats or {},

            # V4: Method usage statistics
            'bid_method_stats': method_stats or {},

            'model_calibration': {},

            'ctr_calibration_check': self._calculate_ctr_calibration_ratio(
                ctr_model.training_stats.get('mean_pred_ctr', 0),
                ctr_model.training_stats.get('global_ctr', 0)
            ),

            'economic_analysis': self._calculate_economic_metrics(bid_results),

            'bid_summary': self._summarize_bids(bid_results),

            'segment_distribution': {
                'total_segments': filter_stats['total_segments'],
                'excluded_low_observations': filter_stats['excluded_low_observations'],
                'excluded_unprofitable': filter_stats['excluded_unprofitable'],
                'segments_in_memcache': filter_stats['included'],
                'inclusion_rate': round(
                    filter_stats['included'] /
                    max(1, filter_stats['total_segments']) * 100, 2
                ),
                # V5: Observation tier breakdown
                'v5_zero_obs': filter_stats.get('v5_included_zero_obs', 0),
                'v5_low_obs': filter_stats.get('v5_included_low_obs', 0),
                'v5_medium_obs': filter_stats.get('v5_included_medium_obs', 0),
                'v5_high_obs': filter_stats.get('v5_included_high_obs', 0)
            }
        }

        # Calculate calibration metrics
        if df_train_wr is not None:
            # V4: Bid landscape model calibration (key check: monotonicity by bid)
            if bid_landscape_model is not None and bid_landscape_model.is_trained:
                try:
                    self.metrics['model_calibration']['bid_landscape_model'] = \
                        self._calculate_bid_landscape_calibration(bid_landscape_model, df_train_wr)
                except Exception as e:
                    self.metrics['model_calibration']['bid_landscape_model'] = {'error': str(e)}

            # LogReg model calibration (for comparison)
            try:
                y_true_wr = df_train_wr['won'].values
                y_pred_logreg = logreg_win_rate_model.predict_win_rate(df_train_wr)

                self.metrics['model_calibration']['win_rate_model_logreg'] = {
                    **self._calculate_calibration_metrics(y_true_wr, y_pred_logreg),
                    **self._calculate_ranking_metrics(y_true_wr, y_pred_logreg, 'win_rate_logreg')
                }
            except Exception as e:
                self.metrics['model_calibration']['win_rate_model_logreg'] = {'error': str(e)}

            # V3: Empirical model calibration (should be very good by definition)
            try:
                y_pred_empirical = empirical_win_rate_model.predict_win_rate(df_train_wr)

                self.metrics['model_calibration']['win_rate_model_empirical'] = {
                    **self._calculate_calibration_metrics(y_true_wr, y_pred_empirical),
                    **self._calculate_ranking_metrics(y_true_wr, y_pred_empirical, 'win_rate_empirical')
                }
            except Exception as e:
                self.metrics['model_calibration']['win_rate_model_empirical'] = {'error': str(e)}

        if df_train_ctr is not None:
            try:
                y_true_ctr = df_train_ctr['clicked'].values
                y_pred_ctr = ctr_model.predict_ctr(df_train_ctr)

                # Use adaptive binning for CTR (extreme class imbalance ~0.035% CTR)
                # With uniform binning, 99.9% of predictions cluster in [0-0.1] bin
                self.metrics['model_calibration']['ctr_model'] = {
                    **self._calculate_calibration_metrics(y_true_ctr, y_pred_ctr, n_bins=5, adaptive=True),
                    **self._calculate_ranking_metrics(y_true_ctr, y_pred_ctr, 'ctr')
                }
            except Exception as e:
                self.metrics['model_calibration']['ctr_model'] = {'error': str(e)}

        # Add warnings section for proper interpretation
        self.metrics['warnings'] = []

        # Warning: Empirical model calibration on training data is tautological
        self.metrics['warnings'].append(
            "Empirical model calibration on training data is tautological by construction - "
            "segment predictions equal segment averages from same data"
        )

        # Warning: Low click count for CTR model
        if df_train_ctr is not None:
            n_clicks = df_train_ctr['clicked'].sum()
            if n_clicks < 1000:
                self.metrics['warnings'].append(
                    f"CTR model has only {n_clicks} clicks - AUC and other metrics may be unstable (recommend 1000+)"
                )

        # Clearing price analysis from VIEWS (what we actually paid for won impressions)
        if df_train_ctr is not None and 'clearing_price' in df_train_ctr.columns:
            valid_prices = df_train_ctr[df_train_ctr['clearing_price'] > 0]['clearing_price']
            if len(valid_prices) > 0:
                self.metrics['clearing_price_analysis'] = {
                    'source': 'view_records',
                    'total_views': len(df_train_ctr),
                    'views_with_valid_price': len(valid_prices),
                    'avg_clearing_price': round(float(valid_prices.mean()), 4),
                    'distribution': {
                        'min': round(float(valid_prices.min()), 4),
                        'p25': round(float(valid_prices.quantile(0.25)), 4),
                        'median': round(float(valid_prices.median()), 4),
                        'p75': round(float(valid_prices.quantile(0.75)), 4),
                        'max': round(float(valid_prices.max()), 4)
                    }
                }

        # Bid amount analysis from BIDS (our offers)
        if df_train_wr is not None and 'bid_value' in df_train_wr.columns:
            self.metrics['bid_amount_analysis'] = {
                'source': 'bid_records',
                'total_bids': len(df_train_wr),
                'won_bids': int(df_train_wr['won'].sum()),
                'win_rate': round(float(df_train_wr['won'].mean()), 4),
                'avg_bid_amount': round(float(df_train_wr['bid_value'].mean()), 4),
                'distribution': {
                    'min': round(float(df_train_wr['bid_value'].min()), 4),
                    'median': round(float(df_train_wr['bid_value'].median()), 4),
                    'max': round(float(df_train_wr['bid_value'].max()), 4)
                }
            }

        return self.metrics

    def _summarize_bids(self, bid_results: List) -> Dict:
        """Summarize bid statistics (V5: includes exploration breakdown)."""
        if not bid_results:
            return {}

        bids = [r.final_bid for r in bid_results]
        ctrs = [r.ctr for r in bid_results]
        win_rates = [r.win_rate for r in bid_results]

        # V5: Get exploration adjustments if available
        exploration_adjustments = []
        for r in bid_results:
            if hasattr(r, 'exploration_adjustment'):
                exploration_adjustments.append(r.exploration_adjustment)
            elif hasattr(r, 'win_rate_adjustment'):
                exploration_adjustments.append(r.win_rate_adjustment)

        # V4/V5: Separate by bid method
        v5_methods = ['v5_explore_zero', 'v5_explore_low', 'v5_explore_medium', 'v5_empirical']
        v4_methods = ['landscape', 'empirical']

        summary = {
            'count': len(bid_results),
            'bid_min': min(bids),
            'bid_max': max(bids),
            'bid_mean': round(sum(bids) / len(bids), 2),
            'bid_median': round(sorted(bids)[len(bids) // 2], 2),
            'ctr_mean': round(sum(ctrs) / len(ctrs), 8),
            'win_rate_mean': round(sum(win_rates) / len(win_rates), 4)
        }

        # Exploration adjustment distribution
        if exploration_adjustments:
            summary['exploration_adjustment_min'] = round(min(exploration_adjustments), 4)
            summary['exploration_adjustment_max'] = round(max(exploration_adjustments), 4)
            summary['exploration_adjustment_mean'] = round(float(np.mean(exploration_adjustments)), 4)

        # V5: Method breakdown
        for method in v5_methods:
            count = sum(1 for r in bid_results if getattr(r, 'bid_method', '') == method)
            if count > 0:
                summary[f'{method}_count'] = count

        # V4: Legacy method breakdown (for backwards compatibility)
        landscape_results = [r for r in bid_results if getattr(r, 'bid_method', '') == 'landscape']
        empirical_results = [r for r in bid_results if getattr(r, 'bid_method', '') == 'empirical']
        if landscape_results or empirical_results:
            summary['landscape_count'] = len(landscape_results)
            summary['empirical_count'] = len(empirical_results)

        # V4: Expected surplus stats (only for landscape method)
        if landscape_results:
            surpluses = [r.expected_surplus for r in landscape_results if hasattr(r, 'expected_surplus') and r.expected_surplus is not None]
            if surpluses:
                summary['expected_surplus_mean'] = round(float(np.mean(surpluses)), 4)
                summary['expected_surplus_median'] = round(float(np.median(surpluses)), 4)

        return summary

    def write_metrics(
        self,
        output_dir: Path,
        timestamp: str = None
    ) -> Path:
        """Write metrics to JSON file."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f'metrics_{timestamp}.json'
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        return filepath
