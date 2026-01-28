#!/usr/bin/env python3
"""
RTB Optimizer Pipeline V9

Usage:
    python run_optimizer.py --config config/optimizer_config_drugs.yaml
    python run_optimizer.py --config config/optimizer_config_nativo_consumer.yaml

    # With automatic data ingestion (checks incoming/ folder first)
    python run_optimizer.py --config config/optimizer_config_drugs.yaml --ingest
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

from src.config import OptimizerConfig
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.feature_selector import FeatureSelector
from src.models.win_rate_model import WinRateModel
from src.models.empirical_win_rate_model import EmpiricalWinRateModel
from src.models.ctr_model import CTRModel
from src.models.bid_landscape_model import BidLandscapeModel
from src.models.npi_value_model import NPIValueModel
from src.bid_calculator_v5 import VolumeFirstBidCalculator
from src.memcache_builder import MemcacheBuilder
from src.metrics_reporter import MetricsReporter


def main():
    parser = argparse.ArgumentParser(description='RTB Optimizer Pipeline V9')
    parser.add_argument('--config', type=str, default='config/optimizer_config_drugs.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--ingest', action='store_true',
                        help='Ingest new data from incoming/ folder before running optimizer')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = OptimizerConfig.from_yaml(str(config_path))
    else:
        print(f"Config file not found at {config_path}, using defaults")
        config = OptimizerConfig()

    # Derive paths from config if not explicitly provided
    data_dir = args.data_dir if args.data_dir else config.dataset.get_data_dir()
    output_base_dir = args.output_dir if args.output_dir else config.dataset.get_output_dir()

    # Step 0: Ingest new data if requested
    if args.ingest:
        from scripts.data_manager import ingest_data
        print(f"\n{'='*60}")
        print("PRE-RUN: Checking for new data to ingest")
        print(f"{'='*60}")
        data_file = config.dataset.get_data_file()
        ingested = ingest_data(Path(data_dir), dry_run=False, data_file=data_file)
        if ingested:
            print(f"\nNew data ingested. Continuing with optimizer...")
        else:
            print(f"\nNo new data to ingest. Continuing with optimizer...")

    # Generate run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n{'='*60}")
    print(f"RTB Optimizer V9")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = Path(output_base_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get active exploration preset
    exploration_preset = config.technical.get_active_exploration_settings()
    exploration_mode_name = "aggressive" if config.technical.aggressive_exploration else "gradual"

    print(f"\nConfiguration:")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Target win rate: {config.business.target_win_rate:.0%}")
    print(f"  Exploration mode: {exploration_mode_name}")
    print(f"  Bid range: ${config.technical.min_bid_cpm} - ${exploration_preset.max_bid_cpm}")
    print(f"  NPI enabled: {config.npi.enabled}")
    print(f"  Strategy: {config.bidding.strategy}")

    # Step 1: Load data
    print(f"\n[1/8] Loading data...")
    data_loader = DataLoader(data_dir, config)
    df_bids, df_views, df_clicks = data_loader.load_all()
    print(f"  Loaded: {len(df_bids):,} bids, {len(df_views):,} views, {len(df_clicks):,} clicks")

    # Step 2: Feature engineering
    print(f"\n[2/8] Engineering features...")
    feature_engineer = FeatureEngineer(config)
    df_bids = feature_engineer.create_features(df_bids)
    df_views = feature_engineer.create_features(df_views)

    df_train_wr = feature_engineer.create_training_data(df_bids, df_views, df_clicks)
    df_train_ctr = feature_engineer.create_ctr_data(df_views, df_clicks)
    print(f"  Win rate training: {len(df_train_wr):,} samples")
    print(f"  CTR training: {len(df_train_ctr):,} samples, {df_train_ctr['clicked'].sum()} clicks")

    # Calculate global stats
    print(f"\n[2.5/8] Calculating global stats...")
    global_stats = {
        'median_winning_bid': float(df_views['bid_amount_cpm'].median()) if 'bid_amount_cpm' in df_views.columns else config.technical.default_bid_cpm,
        'global_win_rate': float(df_train_wr['won'].mean()),
        'total_bids': len(df_train_wr),
        'total_wins': int(df_train_wr['won'].sum())
    }
    print(f"  Median winning bid: ${global_stats['median_winning_bid']:.2f}")
    print(f"  Global win rate: {global_stats['global_win_rate']:.1%}")

    # Step 3: Feature selection
    print(f"\n[3/8] Selecting features...")
    feature_selector = FeatureSelector(config)
    selected_features = feature_selector.select_features(df_train_wr, target_col='won')
    print(f"  Selected: {selected_features}")

    # Step 4: Train models
    print(f"\n[4/8] Training models...")

    # Empirical win rate model
    empirical_win_rate_model = EmpiricalWinRateModel(config)
    empirical_win_rate_model.train(df_train_wr, selected_features, target='won')

    # LogReg win rate model (for diagnostics)
    logreg_win_rate_model = WinRateModel(config)
    logreg_win_rate_model.train(df_train_wr, selected_features, target='won')

    # CTR model
    ctr_model = CTRModel(config)
    ctr_model.train(df_train_ctr, selected_features, target='clicked')
    print(f"  CTR: {ctr_model.training_stats['global_ctr']:.4%}")

    # Bid landscape model
    bid_landscape_model = None
    if config.bidding.strategy in ['margin_optimize', 'adaptive', 'volume_first']:
        print(f"\n  Training bid landscape model...")
        bid_landscape_model = BidLandscapeModel(config)
        try:
            bid_landscape_model.train(df_train_wr, selected_features, bid_col='bid_value', target='won')
            if bid_landscape_model.is_valid():
                print(f"  Bid landscape: VALID (coef={bid_landscape_model.bid_coefficient:.4f})")

                # Derive exploration multipliers from data
                derived_params = bid_landscape_model.derive_exploration_multiplier(
                    current_win_rate=global_stats['global_win_rate'],
                    target_win_rate=config.business.target_win_rate,
                    current_median_bid=global_stats['median_winning_bid']
                )
                if derived_params['success']:
                    global_stats['derived_up_multiplier'] = derived_params['derived_up_multiplier']
                    global_stats['derived_down_multiplier'] = derived_params['derived_down_multiplier']
                    global_stats['exploration_derivation'] = derived_params
                    print(f"  Derived multipliers: {derived_params['derived_up_multiplier']:.2f}x up, {derived_params['derived_down_multiplier']:.2f}x down")
            else:
                print(f"  Bid landscape: INVALID (negative coefficient)")
        except Exception as e:
            print(f"  Bid landscape: FAILED ({e})")
            bid_landscape_model = None

    # NPI model
    npi_model = None
    if config.npi.enabled and config.npi.data_1year:
        print(f"\n  Loading NPI model...")
        npi_model = NPIValueModel.from_click_data(
            path_1year=config.npi.data_1year,
            path_20day=config.npi.data_20day,
            max_multiplier=config.npi.max_multiplier,
            recency_boost=config.npi.recency_boost
        )
    elif config.npi.enabled:
        print(f"  NPI: enabled but no data path configured")
    else:
        print(f"  NPI: disabled")

    # Step 5: Calculate bids
    print(f"\n[5/8] Calculating bids...")
    bid_calculator = VolumeFirstBidCalculator(
        config=config,
        ctr_model=ctr_model,
        empirical_win_rate_model=empirical_win_rate_model,
        bid_landscape_model=bid_landscape_model,
        global_stats=global_stats
    )
    bid_calculator.set_average_cpc(df_clicks)

    df_segments = df_train_wr.groupby(selected_features).size().reset_index(name='count')
    print(f"  Unique segments: {len(df_segments):,}")

    bid_results = bid_calculator.calculate_bids_for_segments(df_segments, selected_features)

    # Summarize
    final_bids = [r.final_bid for r in bid_results]
    method_stats = bid_calculator.get_method_stats()
    exploration_summary = bid_calculator.get_exploration_summary()

    print(f"\n  Bid distribution:")
    print(f"    Range: ${min(final_bids):.2f} - ${max(final_bids):.2f}")
    print(f"    Median: ${sorted(final_bids)[len(final_bids)//2]:.2f}")

    print(f"\n  Exploration direction:")
    print(f"    Bid UP: {exploration_summary['segments_bid_up']:,} ({exploration_summary['pct_bid_up']:.1f}%)")
    print(f"    Bid DOWN: {exploration_summary['segments_bid_down']:,} ({exploration_summary['pct_bid_down']:.1f}%)")

    # Step 6: Build outputs
    print(f"\n[6/8] Building outputs...")
    memcache_builder = MemcacheBuilder(config, v5_mode=True)
    df_bids = memcache_builder.build_memcache(bid_results, selected_features)
    suggested_bids_path = memcache_builder.write_memcache(df_bids, output_dir, run_id)

    df_analysis = memcache_builder.build_segment_analysis(bid_results, selected_features)
    analysis_path = memcache_builder.write_segment_analysis(df_analysis, output_dir, run_id)

    npi_multipliers_path, npi_summary_path = None, None
    if npi_model and npi_model.is_loaded:
        npi_multipliers_path, npi_summary_path = memcache_builder.write_npi_cache(npi_model, output_dir, run_id)

    df_bid_summary = memcache_builder.build_bid_summary(bid_results, selected_features)
    bid_summary_path = memcache_builder.write_bid_summary(df_bid_summary, output_dir, run_id)

    features_path = memcache_builder.write_selected_features(selected_features, output_dir, run_id)

    print(f"  Suggested bids: {suggested_bids_path.name} ({len(df_bids):,} segments)")
    print(f"  Selected features: {features_path.name}")
    print(f"  Analysis: {analysis_path.name}")
    if npi_multipliers_path:
        print(f"  NPI multipliers: {npi_multipliers_path.name}")
        print(f"  NPI summary: {npi_summary_path.name}")

    # Step 7: Metrics
    print(f"\n[7/8] Generating metrics...")
    metrics_reporter = MetricsReporter(config)
    metrics = metrics_reporter.compile_metrics(
        run_id=run_id,
        data_loader=data_loader,
        feature_selector=feature_selector,
        logreg_win_rate_model=logreg_win_rate_model,
        empirical_win_rate_model=empirical_win_rate_model,
        ctr_model=ctr_model,
        bid_results=bid_results,
        memcache_path=suggested_bids_path,
        memcache_builder=memcache_builder,
        df_train_wr=df_train_wr,
        df_train_ctr=df_train_ctr,
        method_stats=method_stats,
        exploration_summary=exploration_summary,
        global_stats=global_stats
    )
    metrics_path = metrics_reporter.write_metrics(output_dir, run_id)
    print(f"  Metrics: {metrics_path.name}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Complete: {run_id}")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"\nBid Summary:")
    bid_summary = metrics['bid_summary']
    print(f"  Segments: {bid_summary['count']}")
    print(f"  Bid range: ${bid_summary['bid_min']:.2f} - ${bid_summary['bid_max']:.2f}")
    print(f"  Bid median: ${bid_summary['bid_median']:.2f}")

    print(f"\nWin Rate:")
    print(f"  Current: {global_stats['global_win_rate']:.1%}")
    print(f"  Target: {config.business.target_win_rate:.1%}")
    gap = config.business.target_win_rate - global_stats['global_win_rate']
    if gap > 0:
        print(f"  Under-winning by {gap:.1%} → bidding UP")
    else:
        print(f"  Over-winning by {-gap:.1%} → bidding DOWN")

    if npi_model and npi_model.is_loaded:
        npi_stats = npi_model.get_tier_stats()
        print(f"\nNPI Model:")
        print(f"  Total NPIs: {npi_stats.get('total_profiles', 0):,}")
        print(f"  Avg multiplier: {npi_stats.get('avg_multiplier', 1.0):.2f}x")

    print(f"{'='*60}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
