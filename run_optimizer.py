#!/usr/bin/env python3
"""
RTB Optimizer Pipeline - V3 Complete

V3 Key Changes:
- Reintroduce win rate signal via EMPIRICAL segment rates (not model predictions)
- Formula: bid = EV_cpm × (1 - margin) × win_rate_adjustment
- Auto soft-exclusion of low-signal features (algorithm-determined)
- Hard exclusions only for technical/structural reasons (config)

Usage:
    python run_optimizer.py --config config/optimizer_config.yaml --data-dir data_drugs/ --output-dir output/
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

from src.config import OptimizerConfig
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.feature_selector import FeatureSelector
from src.models.win_rate_model import WinRateModel  # For diagnostics comparison
from src.models.empirical_win_rate_model import EmpiricalWinRateModel  # V3: New
from src.models.ctr_model import CTRModel
from src.bid_calculator import BidCalculator
from src.memcache_builder import MemcacheBuilder
from src.metrics_reporter import MetricsReporter


def main():
    parser = argparse.ArgumentParser(description='RTB Optimizer Pipeline V3')
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
    print(f"RTB Optimizer Pipeline V3 - Complete with Win Rate Signal")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")
    print(f"Version: V3 (empirical win rate, auto feature exclusion)")

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

    print(f"\nConfiguration (V3):")
    print(f"  Business:")
    print(f"    target_margin: {config.business.target_margin:.0%}")
    print(f"    target_win_rate: {config.business.target_win_rate:.0%}")
    print(f"    win_rate_sensitivity: {config.business.win_rate_sensitivity}")
    print(f"  Technical:")
    print(f"    min_bid_cpm: ${config.technical.min_bid_cpm}")
    print(f"    max_bid_cpm: ${config.technical.max_bid_cpm}")
    print(f"    max_features: {config.technical.max_features}")
    print(f"    min_observations: {config.technical.min_observations}")
    print(f"    min_signal_score: {config.technical.min_signal_score}")

    # Step 1: Load data
    print(f"\n[1/7] Loading data...")
    data_loader = DataLoader(args.data_dir, config)
    df_bids, df_views, df_clicks = data_loader.load_all()
    print(f"  Loaded: {len(df_bids):,} bids, {len(df_views):,} views, {len(df_clicks):,} clicks")

    # Step 2: Feature engineering
    print(f"\n[2/7] Engineering features...")
    feature_engineer = FeatureEngineer(config)
    df_bids = feature_engineer.create_features(df_bids)
    df_views = feature_engineer.create_features(df_views)

    # Create training datasets
    df_train_wr = feature_engineer.create_training_data(df_bids, df_views, df_clicks)
    df_train_ctr = feature_engineer.create_ctr_data(df_views, df_clicks)
    print(f"  Win rate training: {len(df_train_wr):,} samples")
    print(f"  CTR training: {len(df_train_ctr):,} samples, {df_train_ctr['clicked'].sum()} clicks")

    # Step 3: Feature selection (V3: auto soft-exclusion)
    print(f"\n[3/7] Selecting features...")
    feature_selector = FeatureSelector(config)
    selected_features = feature_selector.select_features(df_train_wr, target_col='won')
    print(f"  Final selected features: {selected_features}")

    # Step 4: Train models
    print(f"\n[4/7] Training models...")

    # V3: Empirical win rate model (for bid calculation)
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

    # Step 5: Calculate bids
    print(f"\n[5/7] Calculating bids...")

    # V3: BidCalculator uses empirical win rate model
    bid_calculator = BidCalculator(config, ctr_model, empirical_win_rate_model)
    bid_calculator.set_average_cpc(df_clicks)

    # Get unique segments
    df_segments = df_train_wr.groupby(selected_features).size().reset_index(name='count')
    print(f"  Unique segments: {len(df_segments):,}")

    bid_results = bid_calculator.calculate_bids_for_segments(df_segments, selected_features)

    # Summarize bid distribution
    raw_bids = [r.raw_bid for r in bid_results]
    evs = [r.expected_value_cpm for r in bid_results]
    win_rates = [r.win_rate for r in bid_results]
    adjustments = [r.win_rate_adjustment for r in bid_results]
    profitable_count = sum(1 for r in bid_results if r.is_profitable)

    print(f"\n  Bid Summary (V3 Formula: bid = EV × (1-margin) × wr_adj):")
    print(f"    Raw bid range: ${min(raw_bids):.2f} - ${max(raw_bids):.2f}")
    print(f"    Raw bid median: ${sorted(raw_bids)[len(raw_bids)//2]:.2f}")
    print(f"    EV median: ${sorted(evs)[len(evs)//2]:.2f}")
    print(f"    Win rate range: {min(win_rates):.1%} - {max(win_rates):.1%}")
    print(f"    Win rate adjustment range: {min(adjustments):.2f}x - {max(adjustments):.2f}x")
    print(f"    Profitable segments: {profitable_count:,} / {len(bid_results):,} ({profitable_count/len(bid_results)*100:.1f}%)")

    # Step 6: Build outputs
    print(f"\n[6/7] Building outputs...")

    # Memcache with binary filtering
    memcache_builder = MemcacheBuilder(config)
    df_memcache = memcache_builder.build_memcache(bid_results, selected_features)
    memcache_path = memcache_builder.write_memcache(df_memcache, output_dir, run_id)

    filter_stats = memcache_builder.get_filter_stats()
    print(f"\n  Filtering Results:")
    print(f"    Total segments: {filter_stats['total_segments']:,}")
    print(f"    Excluded (low obs <{config.technical.min_observations}): {filter_stats['excluded_low_observations']:,}")
    print(f"    Excluded (unprofitable): {filter_stats['excluded_unprofitable']:,}")
    print(f"    Included in memcache: {filter_stats['included']:,}")

    print(f"\n  Memcache: {memcache_path} ({len(df_memcache):,} segments)")

    # Step 7: Metrics
    print(f"\n[7/7] Generating metrics...")
    metrics_reporter = MetricsReporter(config)
    metrics = metrics_reporter.compile_metrics(
        run_id=run_id,
        data_loader=data_loader,
        feature_selector=feature_selector,
        logreg_win_rate_model=logreg_win_rate_model,  # For diagnostic comparison
        empirical_win_rate_model=empirical_win_rate_model,  # V3: New
        ctr_model=ctr_model,
        bid_results=bid_results,
        memcache_path=memcache_path,
        memcache_builder=memcache_builder,
        df_train_wr=df_train_wr,
        df_train_ctr=df_train_ctr
    )
    metrics_path = metrics_reporter.write_metrics(output_dir, run_id)
    print(f"  Metrics: {metrics_path}")

    # Print CTR calibration status
    ctr_check = metrics.get('ctr_calibration_check', {})
    status = ctr_check.get('status', 'UNKNOWN')
    print(f"  CTR calibration: {status}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Optimizer V3 run complete: {run_id}")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - Memcache CSV: {memcache_path.name}")
    print(f"  - Metrics JSON: {metrics_path.name}")

    print(f"\nBid Summary:")
    bid_summary = metrics['bid_summary']
    print(f"  - Total segments calculated: {bid_summary['count']}")
    print(f"  - Segments in memcache: {len(df_memcache)}")
    print(f"  - Bid range: ${bid_summary['bid_min']:.2f} - ${bid_summary['bid_max']:.2f}")
    print(f"  - Bid median: ${bid_summary['bid_median']:.2f}")

    print(f"\nWin Rate Analysis:")
    print(f"  - Global win rate: {empirical_win_rate_model.global_win_rate:.1%}")
    print(f"  - Target win rate: {config.business.target_win_rate:.1%}")

    print(f"\nEconomic Analysis:")
    econ = metrics.get('economic_analysis', {})
    print(f"  - Profitable segments: {econ.get('profitable_segments', 0):,} ({econ.get('pct_profitable', 0):.1f}%)")
    bid_clip = econ.get('bid_clipping', {})
    print(f"  - Bids at floor: {bid_clip.get('pct_at_floor', 0):.1f}%")
    print(f"  - Bids at ceiling: {bid_clip.get('pct_at_ceiling', 0):.1f}%")
    print(f"  - Natural bids: {bid_clip.get('pct_natural', 0):.1f}%")

    print(f"\nV3 Formula: bid = EV × (1 - {config.business.target_margin}) × wr_adjustment")
    print(f"Target win rate: {config.business.target_win_rate:.0%}, Sensitivity: {config.business.win_rate_sensitivity}")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
