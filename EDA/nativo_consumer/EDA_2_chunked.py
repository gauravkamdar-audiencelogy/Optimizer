#!/usr/bin/env python3
"""
EDA 2: Nativo Consumer Chunked Analysis
========================================
Memory-efficient analysis of the full 14GB dataset using chunked reading.
Processes data in 500K row chunks to avoid memory issues.

Run: python EDA_nativo_consumer/EDA_2_chunked.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import gc

# Paths
DATA_DIR = Path(__file__).parent.parent / "data_nativo_consumer"
FULL_FILE = DATA_DIR / "data_nativo_consumer_full.csv"
OUTPUT_FILE = Path(__file__).parent / "EDA_2_output.md"

CHUNK_SIZE = 500_000  # Process 500K rows at a time

def parse_pg_json(val):
    """Parse PostgreSQL JSON format: {"0.73000"} -> 0.73"""
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip('{}\"'))
    except:
        return np.nan

def parse_txn_id(val):
    """Parse transaction ID: {"uuid"} -> uuid"""
    if pd.isna(val):
        return None
    return str(val).strip('{}\"')

def main():
    output = []
    output.append("# EDA 2: Nativo Consumer Full Analysis (Chunked)")
    output.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"\nData file: {FULL_FILE.name}")
    output.append(f"\nChunk size: {CHUNK_SIZE:,} rows")

    # Columns to load (minimal set for memory efficiency)
    essential_cols = [
        'LOG_DT', 'REC_TYPE', 'INTERNAL_TXN_ID',
        'DOMAIN', 'BROWSER_CODE', 'OS_CODE',
        'MEDIA_TYPE', 'GEO_REGION_NAME',
        'PUBLISHER_PAYOUT', 'BID_AMOUNT'
    ]

    # Accumulators for aggregated stats
    total_rows = 0
    rec_type_counts = Counter()
    monthly_counts = defaultdict(lambda: Counter())
    feature_stats = defaultdict(lambda: {'count': 0, 'values': Counter()})

    # For bid period analysis (Jan 2026+)
    bid_txns = set()
    view_txns = set()

    # For bid/payout stats
    payout_values = []
    floor_values = []

    min_date = None
    max_date = None

    print(f"Processing {FULL_FILE.name} in chunks of {CHUNK_SIZE:,} rows...")

    # =========================================================================
    # Pass 1: Accumulate statistics
    # =========================================================================
    chunk_num = 0
    for chunk in pd.read_csv(FULL_FILE, usecols=essential_cols,
                             chunksize=CHUNK_SIZE, low_memory=False):
        chunk_num += 1
        chunk.columns = chunk.columns.str.lower()

        # Parse datetime
        chunk['log_dt'] = pd.to_datetime(chunk['log_dt'], format='ISO8601', utc=True)

        # Update date range
        chunk_min = chunk['log_dt'].min()
        chunk_max = chunk['log_dt'].max()
        if min_date is None or chunk_min < min_date:
            min_date = chunk_min
        if max_date is None or chunk_max > max_date:
            max_date = chunk_max

        total_rows += len(chunk)

        # rec_type distribution
        for rt, count in chunk['rec_type'].value_counts().items():
            rec_type_counts[rt] += count

        # Monthly counts
        chunk['month'] = chunk['log_dt'].dt.to_period('M')
        for (month, rt), count in chunk.groupby(['month', 'rec_type']).size().items():
            monthly_counts[str(month)][rt] += count

        # Feature analysis (sample first 100K rows only)
        if total_rows <= 100_000:
            for col in ['domain', 'browser_code', 'os_code', 'media_type', 'geo_region_name']:
                if col in chunk.columns:
                    for val, count in chunk[col].value_counts().items():
                        feature_stats[col]['values'][val] += count
                    feature_stats[col]['count'] += len(chunk)

        # Bid period analysis (Jan 2026+)
        jan_2026 = chunk[chunk['log_dt'] >= '2026-01-01']

        bids = jan_2026[jan_2026['rec_type'] == 'bid']
        views = jan_2026[jan_2026['rec_type'] == 'View']

        # Collect txn_ids for win rate (limit to avoid memory explosion)
        if len(bid_txns) < 2_000_000:  # Cap at 2M
            for txn in bids['internal_txn_id'].dropna():
                bid_txns.add(parse_txn_id(txn))

        if len(view_txns) < 500_000:  # Cap at 500K
            for txn in views['internal_txn_id'].dropna():
                view_txns.add(parse_txn_id(txn))

        # Collect payout/floor values (sample)
        if len(payout_values) < 50_000:
            payout_sample = bids['publisher_payout'].apply(parse_pg_json).dropna()
            payout_values.extend(payout_sample.tolist()[:10_000])

            floor_sample = bids['bid_amount'].apply(parse_pg_json).dropna()
            floor_values.extend(floor_sample.tolist()[:10_000])

        # Progress
        print(f"  Chunk {chunk_num}: {total_rows:,} rows processed", end='\r')

        # Memory cleanup
        del chunk
        gc.collect()

    print(f"\n  Total: {total_rows:,} rows")

    # =========================================================================
    # Generate output
    # =========================================================================
    output.append("\n## 1. Overall Statistics")
    output.append(f"\n- Total records: {total_rows:,}")
    output.append(f"- Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    output.append(f"- Days of data: {(max_date - min_date).days}")

    output.append("\n### Record Type Distribution")
    output.append("\n| rec_type | Count | Percentage |")
    output.append("|----------|-------|------------|")
    for rt, count in rec_type_counts.most_common():
        output.append(f"| {rt} | {count:,} | {count/total_rows*100:.2f}% |")

    output.append("\n## 2. Monthly Breakdown")
    output.append("\n| Month | bid | View | link | lead |")
    output.append("|-------|-----|------|------|------|")
    for month in sorted(monthly_counts.keys()):
        counts = monthly_counts[month]
        output.append(f"| {month} | {counts.get('bid', 0):,} | {counts.get('View', 0):,} | {counts.get('link', 0):,} | {counts.get('lead', 0):,} |")

    output.append("\n## 3. Win Rate Analysis (January 2026)")
    won_txns = bid_txns.intersection(view_txns)
    win_rate = len(won_txns) / len(bid_txns) * 100 if len(bid_txns) > 0 else 0

    output.append(f"\n- Total bid txns tracked: {len(bid_txns):,}")
    output.append(f"- Total view txns tracked: {len(view_txns):,}")
    output.append(f"- Matched (won): {len(won_txns):,}")
    output.append(f"- **Win rate: {win_rate:.2f}%**")

    output.append("\n## 4. Bid and Floor Analysis")
    if payout_values:
        payout_arr = np.array(payout_values)
        output.append(f"\n### Our Bids (publisher_payout)")
        output.append(f"- Sample size: {len(payout_arr):,}")
        output.append(f"- Median: ${np.median(payout_arr):.3f}")
        output.append(f"- Mean: ${np.mean(payout_arr):.3f}")
        output.append(f"- Range: [${np.min(payout_arr):.3f}, ${np.max(payout_arr):.3f}]")

    if floor_values:
        floor_arr = np.array([f for f in floor_values if f < 100])  # Filter outliers
        output.append(f"\n### Floor Prices (bid_amount in bid records)")
        output.append(f"- Sample size: {len(floor_arr):,}")
        output.append(f"- Median: ${np.median(floor_arr):.3f}")
        output.append(f"- Mean: ${np.mean(floor_arr):.3f}")
        output.append(f"- Range: [${np.min(floor_arr):.3f}, ${np.max(floor_arr):.3f}]")

    output.append("\n## 5. Feature Analysis (from first 100K rows)")
    output.append("\n| Feature | Unique | Top Value | Top % |")
    output.append("|---------|--------|-----------|-------|")
    for col, stats in feature_stats.items():
        if stats['values']:
            top_val, top_count = stats['values'].most_common(1)[0]
            top_pct = top_count / stats['count'] * 100 if stats['count'] > 0 else 0
            top_val_str = str(top_val)[:20] + '...' if len(str(top_val)) > 20 else str(top_val)
            output.append(f"| {col} | {len(stats['values']):,} | {top_val_str} | {top_pct:.1f}% |")

    # =========================================================================
    # Write output
    # =========================================================================
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(output))

    print(f"\nOutput written to: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total records: {total_rows:,}")
    print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"Win rate (Jan 2026): {win_rate:.2f}%")


if __name__ == '__main__':
    main()
