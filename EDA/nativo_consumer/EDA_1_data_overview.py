#!/usr/bin/env python3
"""
EDA 1: Nativo Consumer Data Overview
=====================================
Memory-efficient exploration of the nativo_consumer dataset.
Uses chunked reading where needed.

Objectives:
1. Basic data shape and date range
2. rec_type distribution and timeline
3. Column analysis (types, cardinality, nulls, size)
4. Identify heavy columns that can be dropped
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent / "data_nativo_consumer"
DATA_FILE = DATA_DIR / "data_nativo_consumer.csv"  # 1.4GB reduced file
FULL_FILE = DATA_DIR / "data_nativo_consumer_full.csv"  # 14GB full file

OUTPUT_FILE = Path(__file__).parent / "EDA_1_output.md"

def main():
    output = []
    output.append("# EDA 1: Nativo Consumer Data Overview")
    output.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"\nData file: {DATA_FILE.name} (reduced 10% sample)")

    # =========================================================================
    # 1. Basic file info
    # =========================================================================
    output.append("\n## 1. File Information")

    import os
    reduced_size = os.path.getsize(DATA_FILE) / (1024**3)
    full_size = os.path.getsize(FULL_FILE) / (1024**3)
    output.append(f"\n- Reduced file: {reduced_size:.2f} GB")
    output.append(f"- Full file: {full_size:.2f} GB")

    # Get row count without loading
    with open(DATA_FILE, 'r') as f:
        row_count = sum(1 for _ in f) - 1  # Subtract header
    output.append(f"- Rows (reduced): {row_count:,}")
    output.append(f"- Rows (full, estimated): {row_count * 10:,}")

    # =========================================================================
    # 2. Load sample for column analysis
    # =========================================================================
    output.append("\n## 2. Column Analysis")

    # Read first chunk to get dtypes
    print("Loading sample for column analysis...")
    df_sample = pd.read_csv(DATA_FILE, nrows=10000, low_memory=False)
    df_sample.columns = df_sample.columns.str.lower()  # Normalize to lowercase

    output.append(f"\n- Total columns: {len(df_sample.columns)}")
    output.append("\n### Column Types")
    output.append("```")
    for col in df_sample.columns:
        dtype = df_sample[col].dtype
        output.append(f"{col}: {dtype}")
    output.append("```")

    # =========================================================================
    # 3. Load full reduced file for analysis
    # =========================================================================
    print("Loading full reduced file (1.4GB)...")

    # Columns we definitely need
    essential_cols = [
        'log_dt', 'rec_type', 'log_txnid', 'internal_txn_id',
        'internal_adspace_id', 'domain', 'browser_code', 'os_code',
        'carrier_code', 'make_id', 'model_id',
        'publisher_payout', 'bid_amount', 'advertiser_spend',
        'media_type', 'geo_region_name', 'geo_country_code2', 'geo_dma_code',
        'external_userid', 'ad_format'
    ]

    # Load only essential columns to save memory
    df = pd.read_csv(DATA_FILE, usecols=lambda c: c.lower() in essential_cols, low_memory=False)
    df.columns = df.columns.str.lower()

    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # =========================================================================
    # 4. Date range analysis
    # =========================================================================
    output.append("\n## 3. Date Range")

    df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

    min_date = df['log_dt'].min()
    max_date = df['log_dt'].max()
    date_range_days = (max_date - min_date).days

    output.append(f"\n- Start: {min_date.strftime('%Y-%m-%d')}")
    output.append(f"- End: {max_date.strftime('%Y-%m-%d')}")
    output.append(f"- Range: {date_range_days} days")

    # =========================================================================
    # 5. rec_type distribution
    # =========================================================================
    output.append("\n## 4. Record Type Distribution")

    rec_type_counts = df['rec_type'].value_counts()
    output.append("\n| rec_type | Count | Percentage |")
    output.append("|----------|-------|------------|")
    for rec_type, count in rec_type_counts.items():
        pct = count / len(df) * 100
        output.append(f"| {rec_type} | {count:,} | {pct:.2f}% |")

    # =========================================================================
    # 6. rec_type by date (when did bid logging start?)
    # =========================================================================
    output.append("\n## 5. Record Type Timeline")

    df['log_date'] = df['log_dt'].dt.date

    # Find first date for each rec_type
    output.append("\n### First occurrence of each rec_type")
    for rec_type in df['rec_type'].unique():
        first_date = df[df['rec_type'] == rec_type]['log_date'].min()
        output.append(f"- {rec_type}: {first_date}")

    # Monthly breakdown
    df['log_month'] = df['log_dt'].dt.to_period('M')
    monthly_rec_types = df.groupby(['log_month', 'rec_type']).size().unstack(fill_value=0)

    output.append("\n### Monthly rec_type counts")
    output.append("```")
    output.append(monthly_rec_types.to_string())
    output.append("```")

    # =========================================================================
    # 7. Feature cardinality analysis
    # =========================================================================
    output.append("\n## 6. Feature Cardinality (Potential Segment Features)")

    feature_cols = [
        'internal_adspace_id', 'domain', 'browser_code', 'os_code',
        'carrier_code', 'make_id', 'model_id', 'media_type',
        'geo_region_name', 'geo_country_code2', 'geo_dma_code', 'ad_format'
    ]

    output.append("\n| Feature | Unique Values | Top Value | Top % | Nulls % |")
    output.append("|---------|---------------|-----------|-------|---------|")

    for col in feature_cols:
        if col in df.columns:
            unique = df[col].nunique()
            null_pct = df[col].isna().mean() * 100

            top_val = df[col].value_counts().index[0] if not df[col].isna().all() else 'N/A'
            top_pct = df[col].value_counts(normalize=True).iloc[0] * 100 if not df[col].isna().all() else 0

            # Truncate long values
            top_val_str = str(top_val)[:30] + '...' if len(str(top_val)) > 30 else str(top_val)

            output.append(f"| {col} | {unique:,} | {top_val_str} | {top_pct:.1f}% | {null_pct:.1f}% |")

    # =========================================================================
    # 8. Bid/Payout analysis
    # =========================================================================
    output.append("\n## 7. Bid and Payout Analysis")

    # Parse publisher_payout (PostgreSQL array format)
    def parse_pg_array(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        if val_str.startswith('{') and val_str.endswith('}'):
            try:
                return float(val_str[1:-1].split(',')[0])
            except:
                return np.nan
        try:
            return float(val_str)
        except:
            return np.nan

    df['payout_parsed'] = df['publisher_payout'].apply(parse_pg_array)
    df['bid_parsed'] = df['bid_amount'].apply(parse_pg_array)

    output.append("\n### By rec_type")
    for rec_type in df['rec_type'].unique():
        subset = df[df['rec_type'] == rec_type]
        output.append(f"\n**{rec_type}** (n={len(subset):,})")

        payout_valid = subset['payout_parsed'].dropna()
        bid_valid = subset['bid_parsed'].dropna()

        if len(payout_valid) > 0:
            output.append(f"- publisher_payout: median=${payout_valid.median():.2f}, "
                         f"mean=${payout_valid.mean():.2f}, "
                         f"range=[${payout_valid.min():.2f}, ${payout_valid.max():.2f}]")
        else:
            output.append("- publisher_payout: all null")

        if len(bid_valid) > 0:
            output.append(f"- bid_amount: median=${bid_valid.median():.2f}, "
                         f"mean=${bid_valid.mean():.2f}, "
                         f"range=[${bid_valid.min():.2f}, ${bid_valid.max():.2f}]")
        else:
            output.append("- bid_amount: all null or zero")

    # =========================================================================
    # 9. Win rate calculation (for bid records)
    # =========================================================================
    output.append("\n## 8. Win Rate Analysis (Bid Period Only)")

    bids = df[df['rec_type'] == 'bid']
    views = df[df['rec_type'] == 'View']

    if len(bids) > 0:
        output.append(f"\n- Total bids: {len(bids):,}")
        output.append(f"- Total views: {len(views):,}")

        # Join bids to views on log_txnid
        bid_txns = set(bids['log_txnid'].dropna())
        view_txns = set(views['log_txnid'].dropna())

        won_txns = bid_txns.intersection(view_txns)
        win_rate = len(won_txns) / len(bid_txns) if len(bid_txns) > 0 else 0

        output.append(f"- Matched (won): {len(won_txns):,}")
        output.append(f"- Win rate: {win_rate:.1%}")
    else:
        output.append("\n- No bid records in this sample")

    # =========================================================================
    # 10. CTR analysis (clicks / views)
    # =========================================================================
    output.append("\n## 9. CTR and Conversion Analysis")

    clicks = df[df['rec_type'] == 'link']
    leads = df[df['rec_type'] == 'lead']

    output.append(f"\n- Views: {len(views):,}")
    output.append(f"- Clicks (link): {len(clicks):,}")
    output.append(f"- Conversions (lead): {len(leads):,}")

    if len(views) > 0:
        # Simple CTR (not deduplicated)
        ctr = len(clicks) / len(views) * 100
        cvr = len(leads) / len(views) * 100
        output.append(f"- CTR (clicks/views): {ctr:.4f}%")
        output.append(f"- CVR (leads/views): {cvr:.4f}%")

    # =========================================================================
    # 11. Heavy columns analysis (for full file optimization)
    # =========================================================================
    output.append("\n## 10. Column Size Analysis (for data reduction)")

    # Read a sample with all columns to estimate sizes
    df_all = pd.read_csv(DATA_FILE, nrows=10000, low_memory=False)
    df_all.columns = df_all.columns.str.lower()

    col_sizes = []
    for col in df_all.columns:
        # Estimate memory usage
        mem_usage = df_all[col].memory_usage(deep=True)
        avg_per_row = mem_usage / len(df_all)
        estimated_full = avg_per_row * 17_000_000 / (1024**3)  # GB for full file

        col_sizes.append({
            'column': col,
            'dtype': str(df_all[col].dtype),
            'estimated_gb': estimated_full,
            'sample_nulls_pct': df_all[col].isna().mean() * 100
        })

    col_sizes_df = pd.DataFrame(col_sizes).sort_values('estimated_gb', ascending=False)

    output.append("\n### Estimated column sizes (for 17M rows)")
    output.append("\n| Column | Type | Est. Size (GB) | Nulls % |")
    output.append("|--------|------|----------------|---------|")
    for _, row in col_sizes_df.head(20).iterrows():
        output.append(f"| {row['column']} | {row['dtype']} | {row['estimated_gb']:.2f} | {row['sample_nulls_pct']:.1f}% |")

    # =========================================================================
    # 12. Recommendations
    # =========================================================================
    output.append("\n## 11. Preliminary Observations")

    output.append("\n### Data Period Notes")
    output.append("- [To be filled based on analysis above]")

    output.append("\n### Columns to Potentially Drop")
    output.append("- [To be filled based on size analysis]")

    output.append("\n### Feature Candidates")
    output.append("- [To be filled based on cardinality analysis]")

    # =========================================================================
    # Write output
    # =========================================================================
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(output))

    print(f"\nOutput written to: {OUTPUT_FILE}")
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Rows: {row_count:,}")
    print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"rec_types: {dict(rec_type_counts)}")
    if len(bids) > 0:
        print(f"Win rate: {win_rate:.1%}")
    print(f"CTR: {ctr:.4f}%" if len(views) > 0 else "CTR: N/A")


if __name__ == '__main__':
    main()
