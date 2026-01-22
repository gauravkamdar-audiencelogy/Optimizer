#!/usr/bin/env python3
"""
RTB Optimizer Pipeline - V5 Volume-First Bidding

V5 Philosophy:
- Priority is DATA COLLECTION, not margin optimization
- Losing segments → Bid HIGHER to learn ceiling
- Winning segments → Bid LOWER to find floor (don't overpay)
- Target: 65% win rate to learn the full bid landscape
- Include ALL segments (no min_observations filtering)

V5 Formula:
    wr_gap = target_wr - current_wr
    if wr_gap > 0:  # UNDER-WINNING
        adjustment = 1.0 + wr_gap * 1.3  # Bid UP
    else:  # OVER-WINNING
        adjustment = 1.0 + wr_gap * 0.7  # Bid DOWN (less aggressive)

    bid = base_bid * adjustment * npi_multiplier

Why V4 Was Wrong:
- V4 formula `argmax_b [(EV - b) × P(win)]` maximizes margin
- This outputs $2-3 bids which would CUT volume in half
- During learning phase, we need volume, not margin

Usage:
    python run_optimizer.py --config config/optimizer_config.yaml --data-dir data_drugs/ --output-dir output/
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

from src.config import OptimizerConfig
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer, validate_bid_variance
from src.feature_selector import FeatureSelector
from src.models.win_rate_model import WinRateModel  # For diagnostics comparison
from src.models.empirical_win_rate_model import EmpiricalWinRateModel
from src.models.ctr_model import CTRModel
from src.models.npi_value_model import NPIValueModel  # V5: New
from src.bid_calculator_v5 import VolumeFirstBidCalculator  # V5: New
from src.memcache_builder import MemcacheBuilder
from src.metrics_reporter import MetricsReporter


def main():
    parser = argparse.ArgumentParser(description='RTB Optimizer Pipeline V5')
    parser.add_argument('--config', type=str, default='config/optimizer_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data_drugs/',
                        help='Directory with CSV data files')
    parser.add_argument('--output-dir', type=str, default='output/',
                        help='Output directory')
    args = parser.parse_args()

    # Generate run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n{'='*60}")
    print(f"RTB Optimizer Pipeline V5 - Volume-First Bidding")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")
    print(f"Version: V5 (asymmetric exploration for data collection)")

    # Create output directory
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = OptimizerConfig.from_yaml(str(config_path))
        print(f"Config loaded from: {config_path}")
    else:
        print(f"Config file not found at {config_path}, using defaults")
        config = OptimizerConfig()

    print(f"\nConfiguration (V5):")
    print(f"  Business:")
    print(f"    target_margin: {config.business.target_margin:.0%}")
    print(f"    target_win_rate: {config.business.target_win_rate:.0%}")
    print(f"    exploration_mode: {config.business.exploration_mode}")
    print(f"    exploration_up_multiplier: {config.business.exploration_up_multiplier}")
    print(f"    exploration_down_multiplier: {config.business.exploration_down_multiplier}")
    print(f"    accept_negative_margin: {config.business.accept_negative_margin}")
    print(f"  Technical:")
    print(f"    min_bid_cpm: ${config.technical.min_bid_cpm}")
    print(f"    max_bid_cpm: ${config.technical.max_bid_cpm}")
    print(f"    default_bid_cpm: ${config.technical.default_bid_cpm}")
    print(f"    min_observations: {config.technical.min_observations}")
    print(f"    adjustment_range: [{config.technical.min_win_rate_adjustment}, {config.technical.max_win_rate_adjustment}]")

    # Step 1: Load data
    print(f"\n[1/8] Loading data...")
    data_loader = DataLoader(args.data_dir, config)
    df_bids, df_views, df_clicks = data_loader.load_all()
    print(f"  Loaded: {len(df_bids):,} bids, {len(df_views):,} views, {len(df_clicks):,} clicks")

    # Step 2: Feature engineering
    print(f"\n[2/8] Engineering features...")
    feature_engineer = FeatureEngineer(config)
    df_bids = feature_engineer.create_features(df_bids)
    df_views = feature_engineer.create_features(df_views)

    # Create training datasets
    df_train_wr = feature_engineer.create_training_data(df_bids, df_views, df_clicks)
    df_train_ctr = feature_engineer.create_ctr_data(df_views, df_clicks)
    print(f"  Win rate training: {len(df_train_wr):,} samples")
    print(f"  CTR training: {len(df_train_ctr):,} samples, {df_train_ctr['clicked'].sum()} clicks")

    # V5: Calculate global stats for exploration baseline
    print(f"\n[2.5/8] Calculating global stats for V5 exploration...")
    global_stats = {
        'median_winning_bid': float(df_views['bid_amount_cpm'].median()) if 'bid_amount_cpm' in df_views.columns else config.technical.default_bid_cpm,
        'global_win_rate': float(df_train_wr['won'].mean()),
        'total_bids': len(df_train_wr),
        'total_wins': int(df_train_wr['won'].sum())
    }
    print(f"  Median winning bid: ${global_stats['median_winning_bid']:.2f}")
    print(f"  Global win rate: {global_stats['global_win_rate']:.1%}")

    # Step 3: Feature selection (auto soft-exclusion)
    print(f"\n[3/8] Selecting features...")
    feature_selector = FeatureSelector(config)
    selected_features = feature_selector.select_features(df_train_wr, target_col='won')
    print(f"  Final selected features: {selected_features}")

    # Step 4: Train models
    print(f"\n[4/8] Training models...")

    # Empirical win rate model (for segment-level win rates)
    print(f"\n  Training empirical win rate model...")
    empirical_win_rate_model = EmpiricalWinRateModel(config)
    empirical_win_rate_model.train(df_train_wr, selected_features, target='won')

    # Legacy LogReg win rate model (for diagnostic comparison only)
    print(f"\n  Training LogReg win rate model (for diagnostics)...")
    logreg_win_rate_model = WinRateModel(config)
    logreg_win_rate_model.train(df_train_wr, selected_features, target='won')

    # CTR model
    print(f"\n  Training CTR model...")
    ctr_model = CTRModel(config)
    ctr_model.train(df_train_ctr, selected_features, target='clicked')
    print(f"  CTR model: global_ctr={ctr_model.training_stats['global_ctr']:.4%}")

    # V5: Load NPI model if configured
    npi_model = None
    if config.business.use_npi_value and config.business.npi_data_path:
        print(f"\n  Loading NPI value model...")
        npi_model = NPIValueModel.from_csv(config.business.npi_data_path)
    else:
        print(f"\n  NPI model: disabled (no data path configured)")

    # Step 5: Calculate bids
    print(f"\n[5/8] Calculating V5 bids (asymmetric exploration)...")

    # V5: VolumeFirstBidCalculator with asymmetric exploration
    bid_calculator = VolumeFirstBidCalculator(
        config=config,
        ctr_model=ctr_model,
        empirical_win_rate_model=empirical_win_rate_model,
        npi_model=npi_model,
        global_stats=global_stats
    )
    bid_calculator.set_average_cpc(df_clicks)

    # Get ALL unique segments (V5: no min_observations filter here)
    df_segments = df_train_wr.groupby(selected_features).size().reset_index(name='count')
    print(f"  Unique segments: {len(df_segments):,}")

    bid_results = bid_calculator.calculate_bids_for_segments(df_segments, selected_features)

    # Summarize bid distribution
    raw_bids = [r.raw_bid for r in bid_results]
    final_bids = [r.final_bid for r in bid_results]
    evs = [r.expected_value_cpm for r in bid_results]
    win_rates = [r.win_rate for r in bid_results]
    adjustments = [r.exploration_adjustment for r in bid_results]
    profitable_count = sum(1 for r in bid_results if r.is_profitable)

    # V5: Get method and exploration stats
    method_stats = bid_calculator.get_method_stats()
    exploration_summary = bid_calculator.get_exploration_summary()

    print(f"\n  V5 Bid Method Usage:")
    for method in ['v5_explore_zero', 'v5_explore_low', 'v5_explore_medium', 'v5_empirical']:
        count = method_stats.get(f'{method}_count', 0)
        pct = method_stats.get(f'{method}_pct', 0)
        print(f"    {method}: {count:,} ({pct:.1f}%)")

    print(f"\n  V5 Exploration Direction:")
    print(f"    Bid UP (under-winning): {exploration_summary['segments_bid_up']:,} ({exploration_summary['pct_bid_up']:.1f}%)")
    print(f"    Bid DOWN (over-winning): {exploration_summary['segments_bid_down']:,} ({exploration_summary['pct_bid_down']:.1f}%)")
    print(f"    Neutral: {exploration_summary['segments_neutral']:,}")

    print(f"\n  Bid Summary:")
    print(f"    Raw bid range: ${min(raw_bids):.2f} - ${max(raw_bids):.2f}")
    print(f"    Final bid range: ${min(final_bids):.2f} - ${max(final_bids):.2f}")
    print(f"    Raw bid median: ${sorted(raw_bids)[len(raw_bids)//2]:.2f}")
    print(f"    Final bid median: ${sorted(final_bids)[len(final_bids)//2]:.2f}")
    print(f"    EV median: ${sorted(evs)[len(evs)//2]:.2f}")
    print(f"    Win rate range: {min(win_rates):.1%} - {max(win_rates):.1%}")
    print(f"    Exploration adjustment range: {min(adjustments):.2f}x - {max(adjustments):.2f}x")
    print(f"    Profitable segments: {profitable_count:,} / {len(bid_results):,} ({profitable_count/len(bid_results)*100:.1f}%)")

    # Step 6: Build outputs
    print(f"\n[6/8] Building outputs...")

    # V5: Memcache with ALL segments included (features + bid ONLY)
    memcache_builder = MemcacheBuilder(config, v5_mode=True)
    df_memcache = memcache_builder.build_memcache(bid_results, selected_features)
    memcache_path = memcache_builder.write_memcache(df_memcache, output_dir, run_id)

    # V5: Segment analysis file (with full metadata for review)
    df_analysis = memcache_builder.build_segment_analysis(bid_results, selected_features)
    analysis_path = memcache_builder.write_segment_analysis(df_analysis, output_dir, run_id)

    # V5: Aggregated bid summary (tiered buckets for overview)
    df_bid_summary = memcache_builder.build_bid_summary(bid_results, selected_features)
    bid_summary_path = memcache_builder.write_bid_summary(df_bid_summary, output_dir, run_id)

    filter_stats = memcache_builder.get_filter_stats()
    print(f"\n  V5 Memcache Results (ALL segments included):")
    print(f"    Total segments: {filter_stats['total_segments']:,}")
    print(f"    Included in memcache: {filter_stats['included']:,}")
    print(f"    Zero obs segments: {filter_stats['v5_included_zero_obs']:,}")
    print(f"    Low obs (1-9): {filter_stats['v5_included_low_obs']:,}")
    print(f"    Medium obs (10-49): {filter_stats['v5_included_medium_obs']:,}")
    print(f"    High obs (50+): {filter_stats['v5_included_high_obs']:,}")

    print(f"\n  Memcache: {memcache_path} ({len(df_memcache):,} segments)")
    print(f"  Segment Analysis: {analysis_path} (full bid landscape)")
    print(f"  Bid Summary: {bid_summary_path} (tiered bucket overview)")

    # Step 7: Metrics
    print(f"\n[7/8] Generating metrics...")
    metrics_reporter = MetricsReporter(config)
    metrics = metrics_reporter.compile_metrics(
        run_id=run_id,
        data_loader=data_loader,
        feature_selector=feature_selector,
        logreg_win_rate_model=logreg_win_rate_model,  # For diagnostic comparison
        empirical_win_rate_model=empirical_win_rate_model,
        ctr_model=ctr_model,
        bid_results=bid_results,
        memcache_path=memcache_path,
        memcache_builder=memcache_builder,
        df_train_wr=df_train_wr,
        df_train_ctr=df_train_ctr,
        # V5: Add exploration stats
        method_stats=method_stats,
        exploration_summary=exploration_summary,
        global_stats=global_stats
    )
    metrics_path = metrics_reporter.write_metrics(output_dir, run_id)
    print(f"  Metrics: {metrics_path}")

    # Print CTR calibration status
    ctr_check = metrics.get('ctr_calibration_check', {})
    status = ctr_check.get('status', 'UNKNOWN')
    print(f"  CTR calibration: {status}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Optimizer V5 run complete: {run_id}")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - Memcache CSV: {memcache_path.name} (features + bid only)")
    print(f"  - Segment Analysis: {analysis_path.name} (full bid landscape)")
    print(f"  - Bid Summary: {bid_summary_path.name} (tiered bucket overview)")
    print(f"  - Metrics JSON: {metrics_path.name}")

    print(f"\nV5 Bid Summary:")
    bid_summary = metrics['bid_summary']
    print(f"  - Total segments calculated: {bid_summary['count']}")
    print(f"  - Segments in memcache: {len(df_memcache)} (ALL included)")
    print(f"  - Bid range: ${bid_summary['bid_min']:.2f} - ${bid_summary['bid_max']:.2f}")
    print(f"  - Bid median: ${bid_summary['bid_median']:.2f}")

    print(f"\nV5 Exploration Analysis:")
    print(f"  - Target win rate: {config.business.target_win_rate:.0%}")
    print(f"  - Segments bid UP: {exploration_summary['segments_bid_up']:,} ({exploration_summary['pct_bid_up']:.1f}%)")
    print(f"  - Segments bid DOWN: {exploration_summary['segments_bid_down']:,} ({exploration_summary['pct_bid_down']:.1f}%)")
    print(f"  - Up/Down asymmetry ratio: {exploration_summary['asymmetry_ratio']:.2f}")

    print(f"\nWin Rate Analysis:")
    print(f"  - Global win rate: {empirical_win_rate_model.global_win_rate:.1%}")
    print(f"  - Target win rate: {config.business.target_win_rate:.1%}")
    if empirical_win_rate_model.global_win_rate < config.business.target_win_rate:
        gap = config.business.target_win_rate - empirical_win_rate_model.global_win_rate
        print(f"  - Under-winning by {gap:.1%} → V5 will bid UP")
    else:
        gap = empirical_win_rate_model.global_win_rate - config.business.target_win_rate
        print(f"  - Over-winning by {gap:.1%} → V5 will bid DOWN")

    print(f"\nEconomic Analysis:")
    econ = metrics.get('economic_analysis', {})
    print(f"  - Profitable segments: {econ.get('profitable_segments', 0):,} ({econ.get('pct_profitable', 0):.1f}%)")
    print(f"  - Accept negative margin: {config.business.accept_negative_margin}")
    bid_clip = econ.get('bid_clipping', {})
    print(f"  - Bids at floor: {bid_clip.get('pct_at_floor', 0):.1f}%")
    print(f"  - Bids at ceiling: {bid_clip.get('pct_at_ceiling', 0):.1f}%")
    print(f"  - Natural bids: {bid_clip.get('pct_natural', 0):.1f}%")

    print(f"\nV5 Formula: bid = base × exploration_adjustment × npi_multiplier")
    print(f"Exploration: UNDER-winning → bid UP (×{config.business.exploration_up_multiplier})")
    print(f"             OVER-winning → bid DOWN (×{config.business.exploration_down_multiplier})")
    print(f"Target win rate: {config.business.target_win_rate:.0%}")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
