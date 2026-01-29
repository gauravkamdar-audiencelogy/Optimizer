"""
Validation Framework for Optimizer Output

Validates optimizer runs against configurable guardrails before deployment.
- Hard guardrails: Block deployment if violated
- Soft guardrails: Warn but allow deployment

Usage:
    validator = Validator(config)
    result = validator.validate(metrics, bid_results, previous_metrics)
    if not result.passed:
        print("Deployment blocked:", result.blocking_errors)
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    rule_type: str           # 'hard' or 'soft'
    passed: bool
    expected: Any
    actual: Any
    message: str


@dataclass
class ValidationResult:
    """Complete validation result."""
    passed: bool             # True if all hard checks pass
    has_warnings: bool       # True if any soft checks failed
    checks: List[ValidationCheck] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    recommendation: str = "review"  # 'deploy', 'review', 'reject'

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'validation_passed': self.passed,
            'has_warnings': self.has_warnings,
            'recommendation': self.recommendation,
            'summary': self.summary,
            'checks': [
                {
                    'name': c.name,
                    'rule_type': c.rule_type,
                    'passed': c.passed,
                    'expected': c.expected,
                    'actual': c.actual,
                    'message': c.message
                }
                for c in self.checks
            ]
        }


class Validator:
    """
    Validates optimizer output against configurable rules.

    Hard rules block deployment if violated.
    Soft rules generate warnings but allow deployment.
    """

    def __init__(self, config):
        """
        Initialize validator with config.

        Args:
            config: OptimizerConfig instance
        """
        self.config = config
        self.validation_config = config.validation
        self.hard_rules = self.validation_config.hard_rules
        self.soft_rules = self.validation_config.soft_rules

    def validate(
        self,
        metrics: dict,
        bid_results: list,
        previous_metrics: Optional[dict] = None
    ) -> ValidationResult:
        """
        Run all validation checks.

        Args:
            metrics: Metrics dict from MetricsReporter.compile_metrics()
            bid_results: List of BidResult objects from bid calculator
            previous_metrics: Optional metrics from previous run for comparison

        Returns:
            ValidationResult with pass/fail status and all check details
        """
        checks = []

        # Hard checks (blocking)
        checks.append(self._check_bid_bounds(bid_results))
        checks.append(self._check_calibration(metrics))
        checks.append(self._check_model_success(metrics))

        # Coverage check requires previous run
        if previous_metrics:
            checks.append(self._check_coverage(metrics, previous_metrics))
        else:
            # Skip coverage check if no previous run
            checks.append(ValidationCheck(
                name='coverage',
                rule_type='hard',
                passed=True,
                expected={'note': 'No previous run to compare'},
                actual={'note': 'Skipped'},
                message='Coverage check skipped (no previous run)'
            ))

        # Soft checks (warnings)
        checks.append(self._check_floor_concentration(metrics))
        checks.append(self._check_ceiling_concentration(metrics))
        checks.append(self._check_profitability(metrics))

        # Bid shift requires previous run
        if previous_metrics:
            checks.append(self._check_bid_shift(metrics, previous_metrics))

        # Calculate summary
        hard_passed = sum(1 for c in checks if c.rule_type == 'hard' and c.passed)
        hard_failed = sum(1 for c in checks if c.rule_type == 'hard' and not c.passed)
        soft_passed = sum(1 for c in checks if c.rule_type == 'soft' and c.passed)
        soft_failed = sum(1 for c in checks if c.rule_type == 'soft' and not c.passed)

        # Determine overall result
        all_hard_passed = hard_failed == 0
        has_warnings = soft_failed > 0

        # Recommendation logic
        if not all_hard_passed:
            recommendation = 'reject'
        elif has_warnings:
            recommendation = 'review'
        else:
            recommendation = 'deploy'

        return ValidationResult(
            passed=all_hard_passed,
            has_warnings=has_warnings,
            checks=checks,
            summary={
                'hard_checks_passed': hard_passed,
                'hard_checks_failed': hard_failed,
                'soft_checks_passed': soft_passed,
                'soft_checks_failed': soft_failed,
                'total_checks': len(checks)
            },
            recommendation=recommendation
        )

    # -------------------------------------------------------------------------
    # Hard Checks (Blocking)
    # -------------------------------------------------------------------------

    def _check_bid_bounds(self, bid_results: list) -> ValidationCheck:
        """Check that all bids are within configured floor/ceiling."""
        min_bid = self.config.technical.min_bid_cpm
        max_bid = self.config.technical.get_active_exploration_settings().max_bid_cpm

        if not bid_results:
            return ValidationCheck(
                name='bid_bounds',
                rule_type='hard',
                passed=False,
                expected={'min': min_bid, 'max': max_bid},
                actual={'count': 0},
                message='No bid results to validate'
            )

        bids = [r.final_bid for r in bid_results]
        actual_min = min(bids)
        actual_max = max(bids)

        # Check bounds (with small tolerance for floating point)
        tolerance = 0.001
        floor_ok = actual_min >= (min_bid - tolerance)
        ceiling_ok = actual_max <= (max_bid + tolerance)

        passed = floor_ok and ceiling_ok

        if passed:
            message = f'All bids within bounds [${min_bid:.2f}, ${max_bid:.2f}]'
        else:
            violations = []
            if not floor_ok:
                violations.append(f'min ${actual_min:.2f} < floor ${min_bid:.2f}')
            if not ceiling_ok:
                violations.append(f'max ${actual_max:.2f} > ceiling ${max_bid:.2f}')
            message = 'Bids out of bounds: ' + ', '.join(violations)

        return ValidationCheck(
            name='bid_bounds',
            rule_type='hard',
            passed=passed,
            expected={'min': min_bid, 'max': max_bid},
            actual={'min': actual_min, 'max': actual_max},
            message=message
        )

    def _check_calibration(self, metrics: dict) -> ValidationCheck:
        """Check that model calibration is within threshold."""
        max_ece = self.hard_rules.get('calibration_ece_max', 0.15)

        model_calibration = metrics.get('model_calibration', {})
        empirical_metrics = model_calibration.get('win_rate_model_empirical', {})
        ece = empirical_metrics.get('ece', None)

        if ece is None:
            return ValidationCheck(
                name='calibration',
                rule_type='hard',
                passed=True,  # Pass if no calibration data (can't evaluate)
                expected={'max_ece': max_ece},
                actual={'ece': 'N/A'},
                message='Calibration check skipped (no ECE data)'
            )

        passed = ece <= max_ece

        if passed:
            message = f'ECE {ece:.4f} within threshold {max_ece}'
        else:
            message = f'ECE {ece:.4f} exceeds threshold {max_ece}'

        return ValidationCheck(
            name='calibration',
            rule_type='hard',
            passed=passed,
            expected={'max_ece': max_ece},
            actual={'ece': ece},
            message=message
        )

    def _check_model_success(self, metrics: dict) -> ValidationCheck:
        """Check that all required models trained successfully."""
        model_calibration = metrics.get('model_calibration', {})

        # Check empirical model exists and has metrics
        empirical = model_calibration.get('win_rate_model_empirical', {})
        ctr = model_calibration.get('ctr_model', {})

        issues = []
        if not empirical:
            issues.append('empirical win rate model missing')
        if not ctr:
            issues.append('CTR model missing')

        passed = len(issues) == 0

        if passed:
            message = 'All required models trained successfully'
        else:
            message = 'Model training issues: ' + ', '.join(issues)

        return ValidationCheck(
            name='model_success',
            rule_type='hard',
            passed=passed,
            expected={'models': ['empirical_win_rate', 'ctr']},
            actual={'issues': issues if issues else 'none'},
            message=message
        )

    def _check_coverage(
        self,
        metrics: dict,
        previous_metrics: dict
    ) -> ValidationCheck:
        """Check segment coverage vs previous run."""
        min_coverage_pct = self.hard_rules.get('coverage_min_pct', 80.0)

        current_count = metrics.get('bid_summary', {}).get('count', 0)
        previous_count = previous_metrics.get('bid_summary', {}).get('count', 0)

        if previous_count == 0:
            return ValidationCheck(
                name='coverage',
                rule_type='hard',
                passed=True,
                expected={'min_pct': min_coverage_pct},
                actual={'note': 'Previous run had 0 segments'},
                message='Coverage check skipped (previous run had 0 segments)'
            )

        coverage_pct = (current_count / previous_count) * 100
        passed = coverage_pct >= min_coverage_pct

        if passed:
            message = f'Coverage {coverage_pct:.1f}% >= {min_coverage_pct}% threshold'
        else:
            message = f'Coverage {coverage_pct:.1f}% below {min_coverage_pct}% threshold'

        return ValidationCheck(
            name='coverage',
            rule_type='hard',
            passed=passed,
            expected={'min_pct': min_coverage_pct, 'previous_count': previous_count},
            actual={'current_count': current_count, 'coverage_pct': coverage_pct},
            message=message
        )

    # -------------------------------------------------------------------------
    # Soft Checks (Warnings)
    # -------------------------------------------------------------------------

    def _check_floor_concentration(self, metrics: dict) -> ValidationCheck:
        """Check percentage of bids at floor."""
        max_pct = self.soft_rules.get('pct_at_floor_max', 30.0)

        clipping = metrics.get('economic_analysis', {}).get('clipping', {})
        pct_at_floor = clipping.get('pct_clipped_floor', 0)

        passed = pct_at_floor <= max_pct

        if passed:
            message = f'{pct_at_floor:.1f}% at floor (threshold: {max_pct}%)'
        else:
            message = f'{pct_at_floor:.1f}% at floor exceeds {max_pct}% threshold'

        return ValidationCheck(
            name='floor_concentration',
            rule_type='soft',
            passed=passed,
            expected={'max_pct': max_pct},
            actual={'pct': pct_at_floor},
            message=message
        )

    def _check_ceiling_concentration(self, metrics: dict) -> ValidationCheck:
        """Check percentage of bids at ceiling."""
        max_pct = self.soft_rules.get('pct_at_ceiling_max', 30.0)

        clipping = metrics.get('economic_analysis', {}).get('clipping', {})
        pct_at_ceiling = clipping.get('pct_clipped_ceiling', 0)

        passed = pct_at_ceiling <= max_pct

        if passed:
            message = f'{pct_at_ceiling:.1f}% at ceiling (threshold: {max_pct}%)'
        else:
            message = f'{pct_at_ceiling:.1f}% at ceiling exceeds {max_pct}% threshold'

        return ValidationCheck(
            name='ceiling_concentration',
            rule_type='soft',
            passed=passed,
            expected={'max_pct': max_pct},
            actual={'pct': pct_at_ceiling},
            message=message
        )

    def _check_profitability(self, metrics: dict) -> ValidationCheck:
        """Check minimum percentage of profitable segments."""
        min_pct = self.soft_rules.get('pct_profitable_min', 40.0)

        profitability = metrics.get('economic_analysis', {}).get('profitability', {})
        pct_profitable = profitability.get('pct_profitable', 0)

        passed = pct_profitable >= min_pct

        if passed:
            message = f'{pct_profitable:.1f}% profitable (threshold: {min_pct}%)'
        else:
            message = f'{pct_profitable:.1f}% profitable below {min_pct}% threshold'

        return ValidationCheck(
            name='profitability',
            rule_type='soft',
            passed=passed,
            expected={'min_pct': min_pct},
            actual={'pct': pct_profitable},
            message=message
        )

    def _check_bid_shift(
        self,
        metrics: dict,
        previous_metrics: dict
    ) -> ValidationCheck:
        """Check bid median shift vs previous run."""
        max_change_pct = self.soft_rules.get('bid_median_change_max_pct', 50.0)

        current_median = metrics.get('bid_summary', {}).get('bid_median', 0)
        previous_median = previous_metrics.get('bid_summary', {}).get('bid_median', 0)

        if previous_median == 0:
            return ValidationCheck(
                name='bid_shift',
                rule_type='soft',
                passed=True,
                expected={'max_change_pct': max_change_pct},
                actual={'note': 'Previous median was 0'},
                message='Bid shift check skipped (previous median was 0)'
            )

        change_pct = abs((current_median - previous_median) / previous_median) * 100
        passed = change_pct <= max_change_pct

        direction = 'up' if current_median > previous_median else 'down'

        if passed:
            message = f'Median shifted {change_pct:.1f}% {direction} (threshold: {max_change_pct}%)'
        else:
            message = f'Median shifted {change_pct:.1f}% {direction} exceeds {max_change_pct}% threshold'

        return ValidationCheck(
            name='bid_shift',
            rule_type='soft',
            passed=passed,
            expected={'max_change_pct': max_change_pct, 'previous_median': previous_median},
            actual={'current_median': current_median, 'change_pct': change_pct},
            message=message
        )
