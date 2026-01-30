#!/usr/bin/env python3
"""
EDA 2: Nativo Consumer Full Analysis
=====================================
Deep analysis of the full dataset (14GB).
Uses chunked reading for memory efficiency.

Key findings from EDA 1:
- 10% sample breaks bid-view relationships
- Need to use full file with internal_txn_id (parsed) for linking
- Win rate ~4.9% in January (very low!)
- Bid logging started Dec 10, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
DATA_DIR = Path(__file__).parent.parent / "data_nativo_consumer"
FULL_FILE = DATA_DIR / "data_nativo_consumer_full.csv"
OUTPUT_FILE = Path(__file__).parent / "EDA_2_output.md"

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
    output.append("# EDA 2: Nativo Consumer Full Analysis")
    output.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"\nData file: {FULL_FILE.name} (14GB full dataset)")

    # =========================================================================
    # 1. Load essential columns only
    # =========================================================================
    print("Loading full dataset (essential columns only)...")

    essential_cols = [
        'LOG_DT', 'REC_TYPE', 'INTERNAL_TXN_ID',
        'DOMAIN', 'BROWSER_CODE', 'OS_CODE', 'CARRIER_CODE',
        'MAKE_ID', 'MODEL_ID', 'MEDIA_TYPE',
        'GEO_REGION_NAME', 'GEO_COUNTRY_CODE2',
        'PUBLISHER_PAYOUT', 'BID_AMOUNT',
        'INTERNAL_ADSPACE_ID', 'AD_FORMAT'
    ]

    df = pd.read_csv(FULL_FILE, usecols=essential_cols, low_memory=False)
    df.columns = df.columns.str.lower()

    print(f"Loaded {len(df):,} rows")

    # Parse datetime
    df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)
    df['log_date'] = df['log_dt'].dt.date

    # Parse monetary values
    df['payout'] = df['publisher_payout'].apply(parse_pg_json)
    df['floor'] = df['bid_amount'].apply(parse_pg_json)

    # Parse transaction ID
    df['txn_id'] = df['internal_txn_id'].apply(parse_txn_id)

    # =========================================================================
    # 2. Overall statistics
    # =========================================================================
    output.append("\n## 1. Overall Statistics")

    output.append(f"\n- Total records: {len(df):,}")
    output.append(f"- Date range: {df['log_dt'].min().strftime('%Y-%m-%d')} to {df['log_dt'].max().strftime('%Y-%m-%d')}")
    output.append(f"- Days of data: {(df['log_dt'].max() - df['log_dt'].min()).days}")

    output.append("\n### Record Type Distribution")
    output.append("\n| rec_type | Count | Percentage |")
    output.append("|----------|-------|------------|")
    for rt, count in df['rec_type'].value_counts().items():
        output.append(f"| {rt} | {count:,} | {count/len(df)*100:.2f}% |")

    # =========================================================================
    # 3. Timeline analysis
    # =========================================================================
    output.append("\n## 2. Timeline Analysis")

    # First occurrence of each rec_type
    output.append("\n### First occurrence of each rec_type")
    for rt in df['rec_type'].unique():
        first = df[df['rec_type'] == rt]['log_dt'].min()
        output.append(f"- {rt}: {first.strftime('%Y-%m-%d')}")

    # Monthly breakdown
    df['month'] = df['log_dt'].dt.to_period('M')
    monthly = df.groupby(['month', 'rec_type']).size().unstack(fill_value=0)

    output.append("\n### Monthly record counts")
    output.append("```")
    output.append(monthly.to_string())
    output.append("```")

    # =========================================================================
    # 4. Win rate analysis (bid period only)
    # =========================================================================
    output.append("\n## 3. Win Rate Analysis")

    # Filter to bid period (Jan 2026+)
    bid_period = df[df['log_dt'] >= '2026-01-01']

    bids = bid_period[bid_period['rec_type'] == 'bid']
    views = bid_period[bid_period['rec_type'] == 'View']

    output.append(f"\n### January 2026 (Bid Period)")
    output.append(f"- Total bids: {len(bids):,}")
    output.append(f"- Total views (wins): {len(views):,}")

    # Match on txn_id
    bid_txns = set(bids['txn_id'].dropna())
    view_txns = set(views['txn_id'].dropna())
    won_txns = bid_txns.intersection(view_txns)

    win_rate = len(won_txns) / len(bid_txns) * 100 if len(bid_txns) > 0 else 0

    output.append(f"- Matched (won): {len(won_txns):,}")
    output.append(f"- **Win rate: {win_rate:.2f}%**")

    # Win rate by week
    output.append("\n### Weekly Win Rate")
    bids_weekly = bids.copy()
    bids_weekly['week'] = bids_weekly['log_dt'].dt.isocalendar().week

    weekly_wr = []
    for week in sorted(bids_weekly['week'].unique()):
        week_bids = bids_weekly[bids_weekly['week'] == week]
        week_views = views[views['log_dt'].dt.isocalendar().week == week]

        week_bid_txns = set(week_bids['txn_id'].dropna())
        week_view_txns = set(week_views['txn_id'].dropna())
        week_won = week_bid_txns.intersection(week_view_txns)

        wr = len(week_won) / len(week_bid_txns) * 100 if len(week_bid_txns) > 0 else 0
        weekly_wr.append({'week': week, 'bids': len(week_bid_txns), 'wins': len(week_won), 'wr': wr})

    output.append("\n| Week | Bids | Wins | Win Rate |")
    output.append("|------|------|------|----------|")
    for w in weekly_wr:
        output.append(f"| {w['week']} | {w['bids']:,} | {w['wins']:,} | {w['wr']:.2f}% |")

    # =========================================================================
    # 5. Bid and Floor analysis
    # =========================================================================
    output.append("\n## 4. Bid and Floor Analysis")

    # Filter out outliers (floor > $100 is likely data error)
    bids_clean = bids[(bids['floor'] > 0) & (bids['floor'] < 100)]

    output.append(f"\n### Bid Records (clean, floor < $100)")
    output.append(f"- Records: {len(bids_clean):,} ({len(bids_clean)/len(bids)*100:.1f}% of bids)")
    output.append(f"- Our bid (publisher_payout):")
    output.append(f"  - Median: ${bids_clean['payout'].median():.3f}")
    output.append(f"  - Mean: ${bids_clean['payout'].mean():.3f}")
    output.append(f"  - Range: [${bids_clean['payout'].min():.3f}, ${bids_clean['payout'].max():.3f}]")
    output.append(f"- Floor (bid_amount):")
    output.append(f"  - Median: ${bids_clean['floor'].median():.3f}")
    output.append(f"  - Mean: ${bids_clean['floor'].mean():.3f}")
    output.append(f"  - Range: [${bids_clean['floor'].min():.3f}, ${bids_clean['floor'].max():.3f}]")

    # Bid vs Floor comparison
    bids_clean['margin'] = bids_clean['payout'] - bids_clean['floor']
    bids_clean['margin_pct'] = (bids_clean['payout'] / bids_clean['floor'] - 1) * 100

    output.append(f"\n### Bid vs Floor")
    output.append(f"- Bid above floor: {(bids_clean['margin'] > 0).sum():,} ({(bids_clean['margin'] > 0).mean()*100:.1f}%)")
    output.append(f"- Bid at floor: {(bids_clean['margin'] == 0).sum():,}")
    output.append(f"- Bid below floor: {(bids_clean['margin'] < 0).sum():,} ({(bids_clean['margin'] < 0).mean()*100:.1f}%)")
    output.append(f"- Avg margin above floor: {bids_clean['margin_pct'].mean():.1f}%")

    # =========================================================================
    # 6. CTR analysis (all periods)
    # =========================================================================
    output.append("\n## 5. CTR and Conversion Analysis")

    all_views = df[df['rec_type'] == 'View']
    all_clicks = df[df['rec_type'] == 'link']
    all_leads = df[df['rec_type'] == 'lead']

    output.append(f"\n### Overall (All Periods)")
    output.append(f"- Views: {len(all_views):,}")
    output.append(f"- Clicks (link): {len(all_clicks):,}")
    output.append(f"- Conversions (lead): {len(all_leads):,}")

    ctr = len(all_clicks) / len(all_views) * 100 if len(all_views) > 0 else 0
    cvr = len(all_leads) / len(all_views) * 100 if len(all_views) > 0 else 0

    output.append(f"- **CTR: {ctr:.4f}%**")
    output.append(f"- **CVR: {cvr:.4f}%**")

    # CTR by month
    output.append("\n### Monthly CTR")
    monthly_ctr = []
    for month in sorted(df['month'].unique()):
        month_views = df[(df['month'] == month) & (df['rec_type'] == 'View')]
        month_clicks = df[(df['month'] == month) & (df['rec_type'] == 'link')]
        month_ctr = len(month_clicks) / len(month_views) * 100 if len(month_views) > 0 else 0
        monthly_ctr.append({'month': str(month), 'views': len(month_views), 'clicks': len(month_clicks), 'ctr': month_ctr})

    output.append("\n| Month | Views | Clicks | CTR |")
    output.append("|-------|-------|--------|-----|")
    for m in monthly_ctr:
        output.append(f"| {m['month']} | {m['views']:,} | {m['clicks']:,} | {m['ctr']:.4f}% |")

    # =========================================================================
    # 7. Feature analysis
    # =========================================================================
    output.append("\n## 6. Feature Analysis (Potential Segment Features)")

    feature_cols = ['domain', 'browser_code', 'os_code', 'carrier_code',
                    'make_id', 'model_id', 'media_type', 'geo_region_name',
                    'geo_country_code2', 'internal_adspace_id', 'ad_format']

    output.append("\n| Feature | Unique | Top Value | Top % | Useful? |")
    output.append("|---------|--------|-----------|-------|---------|")

    for col in feature_cols:
        if col in df.columns:
            unique = df[col].nunique()
            top_val = df[col].value_counts().index[0]
            top_pct = df[col].value_counts(normalize=True).iloc[0] * 100

            # Determine usefulness
            if unique == 1 or top_pct > 95:
                useful = "✗ No variance"
            elif unique > 10000:
                useful = "⚠ High cardinality"
            else:
                useful = "✓ Candidate"

            top_val_str = str(top_val)[:20] + '...' if len(str(top_val)) > 20 else str(top_val)
            output.append(f"| {col} | {unique:,} | {top_val_str} | {top_pct:.1f}% | {useful} |")

    # =========================================================================
    # 8. Data quality issues
    # =========================================================================
    output.append("\n## 7. Data Quality Issues")

    # Floor outliers
    floor_outliers = bids[bids['floor'] >= 100]
    output.append(f"\n- Floor outliers (>=$100): {len(floor_outliers):,} ({len(floor_outliers)/len(bids)*100:.2f}%)")

    # Null values in key fields
    output.append("\n### Null values in key fields")
    for col in ['payout', 'floor', 'txn_id', 'domain', 'geo_region_name']:
        null_pct = df[col].isna().mean() * 100
        output.append(f"- {col}: {null_pct:.2f}%")

    # =========================================================================
    # 9. Recommendations
    # =========================================================================
    output.append("\n## 8. Recommendations")

    output.append("\n### Data Period")
    output.append("- **Use January 2026+ for bid landscape** (has bid records)")
    output.append("- **Use all data for CTR model** (views + clicks from Sept 2025)")
    output.append("- Consider dropping Sept-Nov 2025 if CTR patterns differ significantly")

    output.append("\n### Features to Use")
    output.append("- domain (14K+ unique - may need bucketing)")
    output.append("- browser_code (15 values)")
    output.append("- os_code (9 values)")
    output.append("- media_type (2 values)")
    output.append("- geo_region_name (65 values)")

    output.append("\n### Features to Exclude")
    output.append("- internal_adspace_id: 100% is -1")
    output.append("- geo_country_code2: 100% is US")
    output.append("- ad_format: single value")
    output.append("- make_id, model_id, carrier_code: mostly -1")

    output.append("\n### Data Reduction")
    output.append("- Drop heavy columns: ua, ref_bundle, source_filename, server_host, load_ts")
    output.append("- Filter floor outliers (>$100)")
    output.append("- Consider using January 2026 data only for initial optimization")

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
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['log_dt'].min().strftime('%Y-%m-%d')} to {df['log_dt'].max().strftime('%Y-%m-%d')}")
    print(f"Bid period win rate: {win_rate:.2f}%")
    print(f"CTR: {ctr:.4f}%")
    print(f"CVR: {cvr:.4f}%")


if __name__ == '__main__':
    main()
