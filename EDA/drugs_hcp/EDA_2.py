import pandas as pd
import numpy as np
from datetime import datetime

# Load the data (same as EDA_1)
data_folder = '../data'

# Suppress SettingWithCopyWarning for cleaner output
pd.options.mode.chained_assignment = None

print("Loading data...")
df_bids = pd.read_csv(f'{data_folder}/drugs_bids.csv', on_bad_lines='skip', engine='python')
df_clicks = pd.read_csv(f'{data_folder}/drugs_clicks.csv', on_bad_lines='skip', engine='python')
df_views = pd.read_csv(f'{data_folder}/drugs_views.csv', on_bad_lines='skip', engine='python')

df = pd.concat([df_bids, df_clicks, df_views], ignore_index=True)
df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)
print(f"Loaded {len(df):,} records\n")

print("=" * 80)
print("BATCH 2: DEEP DIVE ANALYSIS")
print("=" * 80)

# ============================================================================
# 2.1 FIX NPI VALIDATION
# ============================================================================
print("\n### 2.1 NPI VALIDATION (FIXED) ###")

if 'external_userid' in df.columns:
    def is_valid_npi_fixed(val):
        """Fixed NPI validation that handles float storage"""
        if pd.isna(val):
            return False
        # Convert to int first to remove decimal, then to string
        try:
            val_int = int(float(val))
            val_str = str(val_int)
            return val_str.isdigit() and len(val_str) in [10, 11]
        except (ValueError, TypeError):
            return False

    df['has_valid_npi'] = df['external_userid'].apply(is_valid_npi_fixed)
    npi_by_rec_type = df.groupby('rec_type')['has_valid_npi'].agg(['sum', 'count'])
    npi_by_rec_type['pct_valid'] = (npi_by_rec_type['sum'] / npi_by_rec_type['count'] * 100).round(2)
    npi_by_rec_type.columns = ['valid_npi_count', 'total_count', 'pct_valid']
    print(npi_by_rec_type)

    # Show sample of invalid NPIs if any
    invalid_npis = df[~df['has_valid_npi']]['external_userid'].head(10)
    if len(invalid_npis) > 0:
        print(f"\nSample invalid NPIs (first 10):")
        print(invalid_npis.tolist())
else:
    print("WARNING: 'external_userid' column not found in data")

# ============================================================================
# 2.2 INVESTIGATE VIEW DUPLICATES
# ============================================================================
print("\n### 2.2 VIEW DUPLICATE ANALYSIS ###")

df_views_only = df[df['rec_type'] == 'View'].copy()
dup_log_txnids = df_views_only['log_txnid'].value_counts()
dup_log_txnids = dup_log_txnids[dup_log_txnids > 1]

print(f"Total View records: {len(df_views_only):,}")
print(f"Unique log_txnids: {df_views_only['log_txnid'].nunique():,}")
print(f"Duplicated log_txnids: {len(dup_log_txnids):,}")
print(f"Records involved in duplicates: {dup_log_txnids.sum():,}")

print("\nDuplicate count distribution:")
print(dup_log_txnids.value_counts().sort_index().head(10))

# Sample a duplicated log_txnid to understand what's happening
if len(dup_log_txnids) > 0:
    sample_dup_txn = dup_log_txnids.index[0]
    print(f"\nSample duplicated log_txnid: {sample_dup_txn}")
    dup_rows = df_views_only[df_views_only['log_txnid'] == sample_dup_txn]
    print(f"Number of rows: {len(dup_rows)}")
    print("\nComparing key fields across duplicate rows:")
    compare_cols = ['log_dt', 'internal_txn_id', 'internal_adspace_id', 
                    'publisher_payout', 'campaign_code', 'external_userid']
    for col in compare_cols:
        if col in dup_rows.columns:
            unique_vals = dup_rows[col].nunique()
            print(f"  {col}: {unique_vals} unique values")
            if unique_vals <= 3:
                print(f"    Values: {dup_rows[col].tolist()}")

# ============================================================================
# 2.3 PUBLISHER PAYOUT DISTRIBUTION
# ============================================================================
print("\n### 2.3 PUBLISHER PAYOUT ANALYSIS ###")

def parse_first_array_value(val):
    """Extract first numeric value from postgres array string like {7.50000}"""
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    if val_str.startswith('{') and val_str.endswith('}'):
        inner = val_str[1:-1]
        if inner == '':
            return np.nan
        first_val = inner.split(',')[0]
        try:
            return float(first_val)
        except ValueError:
            return np.nan
    try:
        return float(val_str)
    except ValueError:
        return np.nan

# Parse publisher_payout for View records
df_views_only['payout_value'] = df_views_only['publisher_payout'].apply(parse_first_array_value)

print("Publisher payout distribution (CPM) for View records:")
print(df_views_only['payout_value'].describe())

print("\nPayout value counts (top 10):")
print(df_views_only['payout_value'].value_counts().head(10))

# Payout by month
df_views_only['month'] = df_views_only['log_dt'].dt.to_period('M')
print("\nAverage payout by month:")
print(df_views_only.groupby('month')['payout_value'].agg(['mean', 'median', 'count']))

# ============================================================================
# 2.4 ADVERTISER SPEND (CPC) DISTRIBUTION
# ============================================================================
print("\n### 2.4 ADVERTISER SPEND (CPC) ANALYSIS ###")

df_clicks_only = df[df['rec_type'] == 'link'].copy()
df_clicks_only['cpc_value'] = df_clicks_only['advertiser_spend'].apply(parse_first_array_value)

print("CPC distribution for Click records:")
print(df_clicks_only['cpc_value'].describe())

print("\nCPC value counts (top 10):")
print(df_clicks_only['cpc_value'].value_counts().head(10))

# High value clicks
high_cpc = df_clicks_only[df_clicks_only['cpc_value'] >= 50]
print(f"\nHigh-value clicks (CPC >= $50): {len(high_cpc)} ({len(high_cpc)/len(df_clicks_only)*100:.1f}%)")

# ============================================================================
# 2.5 FUNNEL ANALYSIS (DEC-JAN ONLY - WHERE WE HAVE BIDS)
# ============================================================================
print("\n### 2.5 FUNNEL ANALYSIS (Dec 2025 - Jan 2026) ###")

# Filter to Dec-Jan where we have bid data
df['month'] = df['log_dt'].dt.to_period('M')
df_dec_jan = df[df['month'].isin([pd.Period('2025-12'), pd.Period('2026-01')])]

print(f"Records in Dec-Jan window: {len(df_dec_jan):,}")
print("\nBy rec_type:")
print(df_dec_jan['rec_type'].value_counts())

# Calculate funnel metrics
bids_count = len(df_dec_jan[df_dec_jan['rec_type'] == 'bid'])
views_count = len(df_dec_jan[df_dec_jan['rec_type'] == 'View'])
clicks_count = len(df_dec_jan[df_dec_jan['rec_type'] == 'link'])

print(f"\nFunnel (Dec-Jan):")
print(f"  Bids: {bids_count:,}")
print(f"  Views: {views_count:,}")
print(f"  Clicks: {clicks_count:,}")
print(f"  Implied Win Rate (Views/Bids): {views_count/bids_count*100:.2f}%")
print(f"  CTR (Clicks/Views): {clicks_count/views_count*100:.4f}%")

# ============================================================================
# 2.6 ARRAY FIELD DEEP DIVE (NUM_ADS ANALYSIS)
# ============================================================================
print("\n### 2.6 NUM_ADS IMPACT ANALYSIS ###")

def count_array_elements(val):
    if pd.isna(val):
        return 0
    val_str = str(val)
    if val_str.startswith('{') and val_str.endswith('}'):
        inner = val_str[1:-1]
        if inner == '':
            return 0
        return len(inner.split(','))
    return 1

df_dec_jan['num_ads'] = df_dec_jan['internal_txn_id'].apply(count_array_elements)

# Bids by num_ads
df_bids_dj = df_dec_jan[df_dec_jan['rec_type'] == 'bid']
df_views_dj = df_dec_jan[df_dec_jan['rec_type'] == 'View']

print("Bid requests by num_ads:")
print(df_bids_dj['num_ads'].value_counts().sort_index())

print("\nView records by num_ads:")
print(df_views_dj['num_ads'].value_counts().sort_index())

# Is there a pattern? Do 2-ad bids convert to views at a different rate?
# This requires matching bids to views - we'll do a simpler analysis first
print("\nProportion analysis:")
bids_1ad = len(df_bids_dj[df_bids_dj['num_ads'] == 1])
bids_2ad = len(df_bids_dj[df_bids_dj['num_ads'] == 2])
views_1ad = len(df_views_dj[df_views_dj['num_ads'] == 1])
views_2ad = len(df_views_dj[df_views_dj['num_ads'] == 2])

print(f"  1-ad: {bids_1ad:,} bids, {views_1ad:,} views")
print(f"  2-ad: {bids_2ad:,} bids, {views_2ad:,} views")

# ============================================================================
# 2.7 FEATURE CARDINALITY ANALYSIS (FOR MEMCACHE KEY DESIGN)
# ============================================================================
print("\n### 2.7 FEATURE CARDINALITY ANALYSIS ###")

# Focus on bid records for feature analysis (these are our inputs)
df_bids_all = df[df['rec_type'] == 'bid']

feature_cols = [
    'browser_code', 'os_code', 'geo_region_name', 'geo_city_name',
    'domain', 'internal_adspace_id', 'internal_site_id',
    'geo_country_code2', 'geo_dma_code', 'geo_postal_code',
    'carrier_code', 'make_id', 'model_id'
]

print("Feature cardinality in bid records:")
print("-" * 50)
for col in feature_cols:
    if col in df_bids_all.columns:
        nunique = df_bids_all[col].nunique()
        null_pct = df_bids_all[col].isnull().sum() / len(df_bids_all) * 100
        print(f"{col:25} | {nunique:>8,} unique | {null_pct:>5.1f}% null")

# ============================================================================
# 2.8 TOP VALUES FOR KEY FEATURES
# ============================================================================
print("\n### 2.8 TOP VALUES FOR KEY FEATURES ###")

top_features = ['browser_code', 'os_code', 'geo_region_name', 'internal_adspace_id']

for col in top_features:
    if col in df_bids_all.columns:
        print(f"\n{col} (top 10):")
        vc = df_bids_all[col].value_counts().head(10)
        total = len(df_bids_all)
        for val, count in vc.items():
            print(f"  {val}: {count:,} ({count/total*100:.1f}%)")

# ============================================================================
# 2.9 CAMPAIGN ANALYSIS
# ============================================================================
print("\n### 2.9 CAMPAIGN ANALYSIS ###")

def parse_array_to_list(val):
    """Parse postgres array string to python list"""
    if pd.isna(val):
        return []
    val_str = str(val)
    if val_str.startswith('{') and val_str.endswith('}'):
        inner = val_str[1:-1]
        if inner == '':
            return []
        return inner.split(',')
    return [val_str]

# For clicks, get campaign codes
df_clicks_only['campaign_list'] = df_clicks_only['campaign_code'].apply(parse_array_to_list)
all_click_campaigns = [c for sublist in df_clicks_only['campaign_list'] for c in sublist]

print(f"Unique campaigns in click records: {len(set(all_click_campaigns))}")
print("\nClicks by campaign (top 10):")
from collections import Counter
campaign_counts = Counter(all_click_campaigns)
for campaign, count in campaign_counts.most_common(10):
    print(f"  Campaign {campaign}: {count} clicks")

# ============================================================================
# 2.10 MONTH-ON-MONTH NUM_ADS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n### 2.10 MONTH-ON-MONTH NUM_ADS DISTRIBUTION ###")

# Ensure num_ads is calculated for all records
if 'internal_txn_id' not in df.columns:
    print("WARNING: 'internal_txn_id' column not found - skipping num_ads analysis")
elif 'num_ads' not in df.columns:
    df['num_ads'] = df['internal_txn_id'].apply(count_array_elements)

# Add month column to main df if not present
if 'month' not in df.columns:
    df['month'] = df['log_dt'].dt.to_period('M')

if 'internal_txn_id' in df.columns and 'num_ads' in df.columns:
    print("\n--- Bids: Num_ads distribution by month ---")
    df_bids_all = df[df['rec_type'] == 'bid'].copy()
    bids_by_month_ads = df_bids_all.groupby(['month', 'num_ads']).size().unstack(fill_value=0)
    print(bids_by_month_ads)

    # Show percentages for each month
    print("\n--- Bids: Num_ads percentage by month ---")
    bids_pct = (bids_by_month_ads.div(bids_by_month_ads.sum(axis=1), axis=0) * 100).round(2)
    print(bids_pct)

    print("\n--- Views: Num_ads distribution by month ---")
    df_views_all = df[df['rec_type'] == 'View'].copy()
    views_by_month_ads = df_views_all.groupby(['month', 'num_ads']).size().unstack(fill_value=0)
    print(views_by_month_ads)

    print("\n--- Views: Num_ads percentage by month ---")
    views_pct = (views_by_month_ads.div(views_by_month_ads.sum(axis=1), axis=0) * 100).round(2)
    print(views_pct)

    print("\n--- Clicks: Num_ads distribution by month ---")
    df_clicks_all = df[df['rec_type'] == 'link'].copy()
    if len(df_clicks_all) > 0:
        clicks_by_month_ads = df_clicks_all.groupby(['month', 'num_ads']).size().unstack(fill_value=0)
        print(clicks_by_month_ads)
    else:
        print("No click records found")

# ============================================================================
# 2.11 PUBLISHER PAYOUT DISTRIBUTION (CUMULATIVE & MONTHLY)
# ============================================================================
print("\n### 2.11 PUBLISHER PAYOUT DISTRIBUTION ###")

# Parse publisher_payout for all View records
if 'publisher_payout' not in df.columns:
    print("WARNING: 'publisher_payout' column not found - skipping payout analysis")
else:
    if 'df_views_all' not in locals():
        df_views_all = df[df['rec_type'] == 'View'].copy()

    df_views_all['payout_value'] = df_views_all['publisher_payout'].apply(parse_first_array_value)

    print("\n--- Cumulative Publisher Payout Distribution (Views) ---")
    print(df_views_all['payout_value'].describe(percentiles=[.25, .50, .75, .90, .95, .99]))

    print("\n--- Payout Value Counts (Top 20) ---")
    payout_counts = df_views_all['payout_value'].value_counts().head(20)
    for val, count in payout_counts.items():
        pct = count / len(df_views_all) * 100
        print(f"  ${val:.2f}: {count:,} records ({pct:.2f}%)")

    print("\n--- Monthly Publisher Payout Statistics (Views) ---")
    if 'month' not in df_views_all.columns:
        df_views_all['month'] = df_views_all['log_dt'].dt.to_period('M')

    monthly_payout_stats = df_views_all.groupby('month')['payout_value'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ]).round(2)
    print(monthly_payout_stats)

    # Payout distribution by month (top values)
    print("\n--- Top Payout Values by Month ---")
    for month in sorted(df_views_all['month'].unique()):
        month_data = df_views_all[df_views_all['month'] == month]
        print(f"\n{month}:")
        top_payouts = month_data['payout_value'].value_counts().head(5)
        for val, count in top_payouts.items():
            pct = count / len(month_data) * 100
            print(f"  ${val:.2f}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 2.12 BID AMOUNT DISTRIBUTION (CUMULATIVE & MONTHLY)
# ============================================================================
print("\n### 2.12 BID AMOUNT DISTRIBUTION ###")

# Parse bid_amount for all bid records
if 'bid_amount' in df_bids_all.columns:
    df_bids_all['bid_value'] = df_bids_all['bid_amount'].apply(parse_first_array_value)

    print("\n--- Cumulative Bid Amount Distribution ---")
    print(df_bids_all['bid_value'].describe(percentiles=[.25, .50, .75, .90, .95, .99]))

    print("\n--- Bid Value Counts (Top 20) ---")
    bid_counts = df_bids_all['bid_value'].value_counts().head(20)
    for val, count in bid_counts.items():
        pct = count / len(df_bids_all) * 100
        print(f"  ${val:.2f}: {count:,} records ({pct:.2f}%)")

    print("\n--- Monthly Bid Amount Statistics ---")
    if 'month' not in df_bids_all.columns:
        df_bids_all['month'] = df_bids_all['log_dt'].dt.to_period('M')

    monthly_bid_stats = df_bids_all.groupby('month')['bid_value'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ]).round(2)
    print(monthly_bid_stats)

    # Bid distribution by month (top values)
    print("\n--- Top Bid Values by Month ---")
    for month in sorted(df_bids_all['month'].unique()):
        month_data = df_bids_all[df_bids_all['month'] == month]
        print(f"\n{month}:")
        top_bids = month_data['bid_value'].value_counts().head(5)
        for val, count in top_bids.items():
            pct = count / len(month_data) * 100
            print(f"  ${val:.2f}: {count:,} ({pct:.1f}%)")

    # Analysis: Non-zero bids
    print("\n--- Non-Zero Bid Analysis ---")
    non_zero_bids = df_bids_all[df_bids_all['bid_value'] > 0]
    print(f"Total bids: {len(df_bids_all):,}")
    print(f"Non-zero bids: {len(non_zero_bids):,} ({len(non_zero_bids)/len(df_bids_all)*100:.2f}%)")
    print(f"Zero bids: {len(df_bids_all) - len(non_zero_bids):,} ({(len(df_bids_all) - len(non_zero_bids))/len(df_bids_all)*100:.2f}%)")

    if len(non_zero_bids) > 0:
        print(f"\nNon-zero bid statistics:")
        print(non_zero_bids['bid_value'].describe(percentiles=[.25, .50, .75, .90, .95, .99]))
else:
    print("WARNING: 'bid_amount' column not found in bid records")

print("\n" + "=" * 80)
print("END OF BATCH 2")
print("=" * 80)