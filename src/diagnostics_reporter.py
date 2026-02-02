"""
Diagnostics Reporter - Generates human-readable model performance reports.

Outputs:
- diagnostics_{run_id}.md - Individual run diagnostics
- diagnostics_comparison_{run_id}.md - Cross-dataset comparison (when applicable)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import platform
import os


class DiagnosticsReporter:
    """Generates human-readable diagnostics reports from metrics JSON."""

    def __init__(self, metrics: Dict[str, Any], config: Any, run_id: str):
        self.metrics = metrics
        self.config = config
        self.run_id = run_id
        self.dataset_name = config.dataset.name

    def generate_report(self) -> str:
        """Generate full diagnostics report as markdown string."""
        lines = []

        # Header with meta info
        lines.extend(self._generate_header())
        lines.append("")

        # Section 1: Run Summary
        lines.extend(self._generate_run_summary())
        lines.append("")

        # Section 2: Model Performance
        lines.extend(self._generate_model_performance())
        lines.append("")

        # Section 3: Recommendations
        lines.extend(self._generate_recommendations())

        return "\n".join(lines)

    def _generate_header(self) -> list:
        """Generate report header with metadata."""
        timestamp = datetime.now().isoformat()
        run_info = self.metrics.get("run_info", {})

        return [
            f"# Optimizer Diagnostics Report",
            f"",
            f"## Run Metadata",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Run ID** | `{self.run_id}` |",
            f"| **Dataset** | `{self.dataset_name}` |",
            f"| **Timestamp** | {run_info.get('timestamp', timestamp)} |",
            f"| **Report Generated** | {timestamp} |",
            f"| **Optimizer Version** | {run_info.get('version', 'v5_volume_first_exploration')} |",
            f"| **Python Version** | {platform.python_version()} |",
            f"| **Platform** | {platform.system()} {platform.release()} |",
            f"| **Working Directory** | `{os.getcwd()}` |",
        ]

    def _generate_run_summary(self) -> list:
        """Generate Section 1: Run Summary."""
        data = self.metrics.get("data_stats", {})
        global_stats = self.metrics.get("global_stats", {})
        feature_sel = self.metrics.get("feature_selection", {})
        bid_summary = self.metrics.get("bid_summary", {})
        exploration = self.metrics.get("exploration_stats", {})
        config = self.metrics.get("config", {})

        bids = data.get("bids", {})
        views = data.get("views", {})
        clicks = data.get("clicks", {})

        # Calculate totals
        total_rows = bids.get("clean", 0) + views.get("clean", 0) + clicks.get("clean", 0)
        win_rate = global_stats.get("global_win_rate", 0)
        target_wr = config.get("business", {}).get("target_win_rate", 0.65)
        wr_gap = target_wr - win_rate

        return [
            f"---",
            f"",
            f"## 1. Run Summary",
            f"",
            f"### Data Volume",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Total Rows** | {total_rows:,} |",
            f"| **Bids** | {bids.get('clean', 0):,} |",
            f"| **Views (wins)** | {views.get('clean', 0):,} |",
            f"| **Clicks** | {clicks.get('clean', 0):,} |",
            f"| **Floor Prices Available** | {'✓' if bids.get('floor_available') else '✗'} |",
            f"",
            f"### Win Rate Analysis",
            f"",
            f"| Metric | Value | Status |",
            f"|--------|-------|--------|",
            f"| **Current Win Rate** | {win_rate:.1%} | {'🔴 Low' if win_rate < 0.3 else '🟡 Medium' if win_rate < 0.5 else '🟢 Good'} |",
            f"| **Target Win Rate** | {target_wr:.1%} | - |",
            f"| **Win Rate Gap** | {wr_gap:+.1%} | {'↑ Under-winning' if wr_gap > 0 else '↓ Over-winning' if wr_gap < 0 else '✓ On target'} |",
            f"| **Total Bids** | {global_stats.get('total_bids', 0):,} | - |",
            f"| **Total Wins** | {global_stats.get('total_wins', 0):,} | - |",
            f"",
            f"### Feature Selection",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Features Selected** | `{', '.join(feature_sel.get('selected_features', []))}` |",
            f"| **Max Features Allowed** | {feature_sel.get('max_features', 3)} |",
            f"| **Features Excluded** | {len(feature_sel.get('exclusions', {}).get('soft', []))} |",
            f"",
            f"### Bid Output",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Segments** | {bid_summary.get('count', 0):,} |",
            f"| **Bid Range** | ${bid_summary.get('bid_min', 0):.2f} - ${bid_summary.get('bid_max', 0):.2f} |",
            f"| **Median Bid** | ${bid_summary.get('bid_median', 0):.2f} |",
            f"| **Mean Bid** | ${bid_summary.get('bid_mean', 0):.2f} |",
            f"",
            f"### Exploration Direction",
            f"",
            f"| Direction | Count | Percentage |",
            f"|-----------|-------|------------|",
            f"| **Bid UP** | {exploration.get('segments_bid_up', 0)} | {exploration.get('pct_bid_up', 0):.1f}% |",
            f"| **Bid DOWN** | {exploration.get('segments_bid_down', 0)} | {exploration.get('pct_bid_down', 0):.1f}% |",
            f"| **Neutral** | {exploration.get('segments_neutral', 0)} | {100 - exploration.get('pct_bid_up', 0) - exploration.get('pct_bid_down', 0):.1f}% |",
        ]

    def _generate_model_performance(self) -> list:
        """Generate Section 2: Model Performance Metrics."""
        model_perf = self.metrics.get("model_performance", {})
        model_calib = self.metrics.get("model_calibration", {})
        global_stats = self.metrics.get("global_stats", {})

        # Win Rate Model
        wr_empirical = model_perf.get("win_rate_model_empirical", {})
        wr_logreg = model_perf.get("win_rate_model_logreg", {})
        wr_calib = model_calib.get("win_rate_model_empirical", {})

        # CTR Model
        ctr = model_perf.get("ctr_model", {})
        ctr_calib = model_calib.get("ctr_model", {})
        ctr_check = self.metrics.get("ctr_calibration_check", {})

        # Bid Landscape
        exploration = global_stats.get("exploration_derivation", {})

        lines = [
            f"---",
            f"",
            f"## 2. Model Performance Metrics",
            f"",
            f"### 2.1 Win Rate Model (Core Model)",
            f"",
            f"*Predicts probability of winning an auction given segment features.*",
            f"",
            f"| Metric | Value | Interpretation |",
            f"|--------|-------|----------------|",
            f"| **AUC-ROC** | {wr_calib.get('auc_roc', 0):.4f} | Discrimination power (0.5=random, 1.0=perfect). **{self._interpret_auc(wr_calib.get('auc_roc', 0))}** |",
            f"| **ECE** | {wr_logreg.get('calibration_ece', 0):.4f} | Expected Calibration Error. **{'✓ PASSED' if wr_logreg.get('passes_calibration_gate') else '✗ FAILED'}** (<0.1 required) |",
            f"| **Brier Score** | {wr_calib.get('brier_score', 0):.4f} | Mean squared error of probabilities. Lower = better. |",
            f"| **Log Loss** | {wr_calib.get('log_loss', 0):.4f} | Cross-entropy loss. Lower = better. |",
            f"| **Relative Info Gain** | {wr_calib.get('relative_information_gain', 0):.2%} | Improvement over naive baseline. |",
            f"| **Segments** | {wr_empirical.get('n_segments', 0)} | Total segments with data. |",
            f"| **Segments with Wins** | {wr_empirical.get('segments_with_wins', 0)} | Segments that have won at least once. |",
            f"| **Shrinkage K** | {wr_empirical.get('shrinkage_k', 30)} | Bayesian shrinkage strength (higher = more conservative). |",
            f"",
            f"### 2.2 CTR Model (Click Prediction)",
            f"",
            f"*Predicts probability of click given a won impression.*",
            f"",
            f"| Metric | Value | Interpretation |",
            f"|--------|-------|----------------|",
            f"| **AUC-ROC** | {ctr_calib.get('auc_roc', 0):.4f} | **{self._interpret_ctr_auc(ctr_calib.get('auc_roc', 0), ctr.get('n_clicks', 0))}** |",
            f"| **Global CTR** | {ctr.get('global_ctr', 0):.4%} | Baseline click rate across all impressions. |",
            f"| **Mean Predicted CTR** | {ctr.get('mean_pred_ctr', 0):.4%} | Model's average prediction. |",
            f"| **Calibration Ratio** | {ctr_check.get('ratio', 1):.2f}x | Predicted/Actual ratio (1.0 = perfect). **{ctr_check.get('status', 'UNKNOWN')}** |",
            f"| **Clicks (Training)** | {ctr.get('n_clicks', 0)} | {'⚠️ Too few (<1000)' if ctr.get('n_clicks', 0) < 1000 else '✓ Sufficient'} |",
            f"| **Segments with Clicks** | {ctr.get('segments_with_clicks', 0)} / {ctr.get('n_segments', 0)} | Segments that received clicks. |",
            f"",
            f"### 2.3 Bid Landscape Model",
            f"",
            f"*Models P(win|bid) to derive exploration multipliers.*",
            f"",
            f"| Metric | Value | Interpretation |",
            f"|--------|-------|----------------|",
            f"| **Bid Coefficient** | {exploration.get('bid_coefficient', 0):.4f} | {'✓ Positive (correct)' if exploration.get('bid_coefficient', 0) > 0 else '✗ Negative (inverted)'} - higher bid → higher P(win). |",
            f"| **Derived Up Multiplier** | {exploration.get('derived_up_multiplier', 1):.2f}x | How much to increase bid when under-winning. |",
            f"| **Derived Down Multiplier** | {exploration.get('derived_down_multiplier', 1):.2f}x | How much to decrease bid when over-winning. |",
            f"| **Current Median Bid** | ${exploration.get('current_median_bid', 0):.2f} | Current bidding level. |",
            f"| **Implied Target Bid** | ${exploration.get('implied_new_bid', 0):.2f} | Bid needed for target WR (theoretical). |",
            f"| **Extrapolation Warning** | {'⚠️ Yes' if exploration.get('extrapolation_warning') else '✓ No'} | Whether target is beyond observed data range. |",
        ]

        return lines

    def _generate_recommendations(self) -> list:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        warnings = self.metrics.get("warnings", [])
        economic = self.metrics.get("economic_analysis", {})
        ctr = self.metrics.get("model_performance", {}).get("ctr_model", {})
        global_stats = self.metrics.get("global_stats", {})

        # Check for issues
        if ctr.get("n_clicks", 0) < 1000:
            recommendations.append(
                f"- **Collect more click data**: Only {ctr.get('n_clicks', 0)} clicks. "
                f"CTR model needs 1000+ for stable metrics."
            )

        if economic.get("pct_profitable", 100) < 40:
            recommendations.append(
                f"- **Expected during exploration**: Only {economic.get('pct_profitable', 0):.1f}% "
                f"segments profitable. This is normal when learning market prices."
            )

        if economic.get("bid_clipping", {}).get("pct_at_ceiling", 0) > 20:
            recommendations.append(
                f"- **Consider raising max_bid**: {economic.get('bid_clipping', {}).get('pct_at_ceiling', 0):.1f}% "
                f"of bids at ceiling. Market may require higher bids."
            )

        wr = global_stats.get("global_win_rate", 0)
        if wr < 0.15:
            recommendations.append(
                f"- **Very low win rate ({wr:.1%})**: Consider aggressive_exploration=true "
                f"or increasing max_bid_cpm to learn market faster."
            )

        if not recommendations:
            recommendations.append("- **No critical issues detected.** Models performing as expected.")

        lines = [
            f"---",
            f"",
            f"## 3. Recommendations",
            f"",
        ]
        lines.extend(recommendations)

        if warnings:
            lines.extend([
                f"",
                f"### System Warnings",
                f"",
            ])
            for w in warnings:
                lines.append(f"- {w}")

        return lines

    def _interpret_auc(self, auc: float) -> str:
        """Interpret AUC-ROC value."""
        if auc >= 0.9:
            return "Excellent"
        elif auc >= 0.8:
            return "Good"
        elif auc >= 0.7:
            return "Fair"
        elif auc >= 0.6:
            return "Weak but useful"
        elif auc >= 0.5:
            return "Random (no signal)"
        else:
            return "Inverted (check labels)"

    def _interpret_ctr_auc(self, auc: float, n_clicks: int) -> str:
        """Interpret CTR model AUC with context."""
        if n_clicks < 200:
            return f"Unreliable (only {n_clicks} clicks)"
        elif auc >= 0.7:
            return "Good predictive power"
        elif auc >= 0.6:
            return "Some signal"
        elif auc >= 0.5:
            return "No predictive power (expected with sparse clicks)"
        else:
            return "Inverted (check implementation)"

    def save(self, output_dir: str) -> str:
        """Save report to file and return path."""
        report = self.generate_report()
        filepath = Path(output_dir) / f"diagnostics_{self.run_id}.md"
        filepath.write_text(report)
        return str(filepath)


def generate_comparison_report(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    name_a: str,
    name_b: str,
    output_path: str
) -> str:
    """Generate a comparison report between two optimizer runs."""
    timestamp = datetime.now().isoformat()

    def get_val(m, *keys, default=0):
        """Safely get nested value."""
        val = m
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    lines = [
        f"# Optimizer Comparison Report",
        f"",
        f"**Generated**: {timestamp}",
        f"",
        f"---",
        f"",
        f"## Run Comparison: {name_a} vs {name_b}",
        f"",
        f"### Data Volume",
        f"",
        f"| Metric | {name_a} | {name_b} |",
        f"|--------|----------|----------|",
        f"| **Bids** | {get_val(metrics_a, 'data_stats', 'bids', 'clean'):,} | {get_val(metrics_b, 'data_stats', 'bids', 'clean'):,} |",
        f"| **Views** | {get_val(metrics_a, 'data_stats', 'views', 'clean'):,} | {get_val(metrics_b, 'data_stats', 'views', 'clean'):,} |",
        f"| **Clicks** | {get_val(metrics_a, 'data_stats', 'clicks', 'clean'):,} | {get_val(metrics_b, 'data_stats', 'clicks', 'clean'):,} |",
        f"",
        f"### Win Rate",
        f"",
        f"| Metric | {name_a} | {name_b} |",
        f"|--------|----------|----------|",
        f"| **Current Win Rate** | {get_val(metrics_a, 'global_stats', 'global_win_rate'):.1%} | {get_val(metrics_b, 'global_stats', 'global_win_rate'):.1%} |",
        f"| **Total Wins** | {get_val(metrics_a, 'global_stats', 'total_wins'):,} | {get_val(metrics_b, 'global_stats', 'total_wins'):,} |",
        f"| **Segments** | {get_val(metrics_a, 'bid_summary', 'count'):,} | {get_val(metrics_b, 'bid_summary', 'count'):,} |",
        f"",
        f"### Model Performance",
        f"",
        f"| Metric | {name_a} | {name_b} | Interpretation |",
        f"|--------|----------|----------|----------------|",
        f"| **WR Model AUC** | {get_val(metrics_a, 'model_calibration', 'win_rate_model_empirical', 'auc_roc'):.4f} | {get_val(metrics_b, 'model_calibration', 'win_rate_model_empirical', 'auc_roc'):.4f} | Discrimination (0.5=random, 1.0=perfect) |",
        f"| **WR Model ECE** | {get_val(metrics_a, 'model_performance', 'win_rate_model_logreg', 'calibration_ece'):.4f} | {get_val(metrics_b, 'model_performance', 'win_rate_model_logreg', 'calibration_ece'):.4f} | Calibration error (<0.1 = good) |",
        f"| **CTR Model AUC** | {get_val(metrics_a, 'model_calibration', 'ctr_model', 'auc_roc'):.4f} | {get_val(metrics_b, 'model_calibration', 'ctr_model', 'auc_roc'):.4f} | Click prediction power |",
        f"| **Bid Coefficient** | {get_val(metrics_a, 'global_stats', 'exploration_derivation', 'bid_coefficient'):.4f} | {get_val(metrics_b, 'global_stats', 'exploration_derivation', 'bid_coefficient'):.4f} | Bid → P(win) relationship |",
        f"",
        f"### Bid Output",
        f"",
        f"| Metric | {name_a} | {name_b} |",
        f"|--------|----------|----------|",
        f"| **Bid Range** | ${get_val(metrics_a, 'bid_summary', 'bid_min'):.2f} - ${get_val(metrics_a, 'bid_summary', 'bid_max'):.2f} | ${get_val(metrics_b, 'bid_summary', 'bid_min'):.2f} - ${get_val(metrics_b, 'bid_summary', 'bid_max'):.2f} |",
        f"| **Median Bid** | ${get_val(metrics_a, 'bid_summary', 'bid_median'):.2f} | ${get_val(metrics_b, 'bid_summary', 'bid_median'):.2f} |",
        f"| **% Bidding UP** | {get_val(metrics_a, 'exploration_stats', 'pct_bid_up'):.1f}% | {get_val(metrics_b, 'exploration_stats', 'pct_bid_up'):.1f}% |",
        f"| **% At Ceiling** | {get_val(metrics_a, 'economic_analysis', 'bid_clipping', 'pct_at_ceiling'):.1f}% | {get_val(metrics_b, 'economic_analysis', 'bid_clipping', 'pct_at_ceiling'):.1f}% |",
        f"",
        f"### Economic Analysis",
        f"",
        f"| Metric | {name_a} | {name_b} |",
        f"|--------|----------|----------|",
        f"| **% Profitable** | {get_val(metrics_a, 'economic_analysis', 'pct_profitable'):.1f}% | {get_val(metrics_b, 'economic_analysis', 'pct_profitable'):.1f}% |",
        f"| **Avg Clearing Price** | ${get_val(metrics_a, 'clearing_price_analysis', 'avg_clearing_price'):.2f} | ${get_val(metrics_b, 'clearing_price_analysis', 'avg_clearing_price'):.2f} |",
        f"| **Global CTR** | {get_val(metrics_a, 'model_performance', 'ctr_model', 'global_ctr'):.4%} | {get_val(metrics_b, 'model_performance', 'ctr_model', 'global_ctr'):.4%} |",
        f"",
        f"---",
        f"",
        f"*Report generated by Optimizer Diagnostics Reporter*",
    ]

    report = "\n".join(lines)
    Path(output_path).write_text(report)
    return output_path
