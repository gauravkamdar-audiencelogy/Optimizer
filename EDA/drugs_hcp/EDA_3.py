import pandas as pd
import numpy as np
from collections import Counter

# Load and prepare data
data_folder = '../data'
pd.options.mode.chained_assignment = None

print("Loading data...")
df_bids = pd.read_csv(f'{data_folder}/drugs_bids.csv', on_bad_lines='skip', engine='python')
df_clicks = pd.read_csv(f'{data_folder}/drugs_clicks.csv', on_bad_lines='skip', engine='python')
df_views = pd.read_csv(f'{data_folder}/drugs_views.csv', on_bad_lines='skip', engine='python')

df = pd.concat([df_bids, df_clicks, df_views], ignore_index=True)
df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)
df['month'] = df['log_dt'].dt.to_period('M')

# Helper functions
def parse_first_array_value(val):
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

def parse_array_to_list(val):
    if pd.isna(val):
        return []
    val_str = str(val)
    if val_str.startswith('{') and val_str.endswith('}'):
        inner = val_str[1:-1]
        if inner == '':
            return []
        return inner.split(',')
    return [val_str]

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

print(f"Loaded {len(df):,} records\n")

print("=" * 80)
print("BATCH 3: FUNNEL & SEGMENT ANALYSIS")
print("=" * 80)

# ============================================================================
# 3.1 DEDUPLICATE VIEWS AND RECALCULATE FUNNEL
# ============================================================================
print("\n### 3.1 DEDUPLICATED FUNNEL ANALYSIS ###")

# Separate by rec_type
df_bids_all = df[df['rec_type'] == 'bid'].copy()
df_views_all = df[df['rec_type'] == 'View'].copy()
df_clicks_all = df[df['rec_type'] == 'link'].copy()

# Deduplicate views by log_txnid (keep first occurrence)
views_before = len(df_views_all)
df_views_dedup = df_views_all.drop_duplicates(subset=['log_txnid'], keep='first')
views_after = len(df_views_dedup)

print(f"Views before dedup: {views_before:,}")
print(f"Views after dedup:  {views_after:,}")
print(f"Removed: {views_before - views_after:,} ({(views_before - views_after)/views_before*100:.1f}%)")

# Check duplicate distribution by month
print("\n--- Duplicates by Month ---")
df_views_all['is_dup'] = df_views_all.duplicated(subset=['log_txnid'], keep='first')
dup_by_month = df_views_all.groupby('month')['is_dup'].agg(['sum', 'count'])
dup_by_month['dup_pct'] = (dup_by_month['sum'] / dup_by_month['count'] * 100).round(2)
dup_by_month.columns = ['duplicates', 'total', 'dup_pct']
print(dup_by_month)

# ============================================================================
# 3.2 CORRECTED FUNNEL METRICS (DEC-JAN WITH DEDUPED VIEWS)
# ============================================================================
print("\n### 3.2 CORRECTED FUNNEL (Dec-Jan) ###")

# Filter to Dec-Jan
df_bids_dj = df_bids_all[df_bids_all['month'].isin([pd.Period('2025-12'), pd.Period('2026-01')])]
df_views_dj = df_views_dedup[df_views_dedup['month'].isin([pd.Period('2025-12'), pd.Period('2026-01')])]
df_clicks_dj = df_clicks_all[df_clicks_all['month'].isin([pd.Period('2025-12'), pd.Period('2026-01')])]

bids_count = len(df_bids_dj)
views_count = len(df_views_dj)
clicks_count = len(df_clicks_dj)

print(f"Bids:   {bids_count:,}")
print(f"Views:  {views_count:,} (deduplicated)")
print(f"Clicks: {clicks_count:,}")
print(f"\nWin Rate (Views/Bids): {views_count/bids_count*100:.2f}%")
print(f"CTR (Clicks/Views):    {clicks_count/views_count*100:.4f}%")

# Monthly breakdown
print("\n--- Monthly Funnel ---")
for month in [pd.Period('2025-12'), pd.Period('2026-01')]:
    m_bids = len(df_bids_dj[df_bids_dj['month'] == month])
    m_views = len(df_views_dj[df_views_dj['month'] == month])
    m_clicks = len(df_clicks_dj[df_clicks_dj['month'] == month])
    if m_bids > 0 and m_views > 0:
        print(f"{month}: {m_bids:,} bids → {m_views:,} views ({m_views/m_bids*100:.1f}% WR) → {m_clicks} clicks ({m_clicks/m_views*100:.4f}% CTR)")

# ============================================================================
# 3.3 WIN RATE BY NUM_ADS
# ============================================================================
print("\n### 3.3 WIN RATE BY NUM_ADS (Dec-Jan) ###")

df_bids_dj['num_ads'] = df_bids_dj['internal_txn_id'].apply(count_array_elements)
df_views_dj['num_ads'] = df_views_dj['internal_txn_id'].apply(count_array_elements)

for num_ads in [1, 2]:
    bids_n = len(df_bids_dj[df_bids_dj['num_ads'] == num_ads])
    views_n = len(df_views_dj[df_views_dj['num_ads'] == num_ads])
    if bids_n > 0:
        print(f"{num_ads}-ad requests: {bids_n:,} bids → {views_n:,} views = {views_n/bids_n*100:.1f}% win rate")

# ============================================================================
# 3.4 WIN RATE BY BID AMOUNT BUCKET
# ============================================================================
print("\n### 3.4 WIN RATE BY BID AMOUNT ###")

df_bids_dj['bid_value'] = df_bids_dj['bid_amount'].apply(parse_first_array_value)
df_views_dj['payout_value'] = df_views_dj['publisher_payout'].apply(parse_first_array_value)

# Create bid buckets
def bid_bucket(val):
    if pd.isna(val) or val == 0:
        return '00_zero'
    elif val < 5:
        return '01_under_5'
    elif val < 7.5:
        return '02_5_to_7.5'
    elif val == 7.5:
        return '03_exactly_7.5'
    elif val < 10:
        return '04_7.5_to_10'
    elif val < 15:
        return '05_10_to_15'
    elif val < 20:
        return '06_15_to_20'
    else:
        return '07_20_plus'

df_bids_dj['bid_bucket'] = df_bids_dj['bid_value'].apply(bid_bucket)
df_views_dj['payout_bucket'] = df_views_dj['payout_value'].apply(bid_bucket)

print("\nBids by bucket:")
bid_bucket_counts = df_bids_dj['bid_bucket'].value_counts().sort_index()
print(bid_bucket_counts)

print("\nViews (wins) by payout bucket:")
view_bucket_counts = df_views_dj['payout_bucket'].value_counts().sort_index()
print(view_bucket_counts)

# Note: We can't directly calculate win rate by bucket because bids and views
# aren't joined. We're comparing distributions.
print("\n--- Distribution Comparison ---")
for bucket in sorted(df_bids_dj['bid_bucket'].unique()):
    bids_b = bid_bucket_counts.get(bucket, 0)
    views_b = view_bucket_counts.get(bucket, 0)
    bid_pct = bids_b / len(df_bids_dj) * 100
    view_pct = views_b / len(df_views_dj) * 100 if bucket in view_bucket_counts else 0
    print(f"{bucket}: {bid_pct:.1f}% of bids, {view_pct:.1f}% of wins")

# ============================================================================
# 3.5 WIN RATE BY KEY FEATURES (INTERNAL_ADSPACE_ID)
# ============================================================================
print("\n### 3.5 WIN RATE BY INTERNAL_ADSPACE_ID ###")

bids_by_adspace = df_bids_dj.groupby('internal_adspace_id').size()
views_by_adspace = df_views_dj.groupby('internal_adspace_id').size()

print(f"{'Adspace':<12} | {'Bids':>10} | {'Views':>10} | {'WinRate':>10}")
print("-" * 50)
for adspace in sorted(bids_by_adspace.index):
    b = bids_by_adspace.get(adspace, 0)
    v = views_by_adspace.get(adspace, 0)
    wr = v/b*100 if b > 0 else 0
    print(f"{adspace:<12} | {b:>10,} | {v:>10,} | {wr:>9.1f}%")

# ============================================================================
# 3.6 WIN RATE BY GEO_REGION (TOP 15)
# ============================================================================
print("\n### 3.6 WIN RATE BY GEO_REGION (Top 15 by bid volume) ###")

bids_by_region = df_bids_dj.groupby('geo_region_name').size().sort_values(ascending=False)
views_by_region = df_views_dj.groupby('geo_region_name').size()

print(f"{'Region':<20} | {'Bids':>10} | {'Views':>10} | {'WinRate':>10}")
print("-" * 60)
for region in bids_by_region.head(15).index:
    b = bids_by_region.get(region, 0)
    v = views_by_region.get(region, 0)
    wr = v/b*100 if b > 0 else 0
    print(f"{str(region):<20} | {b:>10,} | {v:>10,} | {wr:>9.1f}%")

# ============================================================================
# 3.7 CTR BY KEY FEATURES (USING ALL 4 MONTHS OF VIEW/CLICK DATA)
# ============================================================================
print("\n### 3.7 CTR BY FEATURES (All available data) ###")

# Use all views and clicks (not just Dec-Jan) for CTR analysis
# since we have 4 months of view/click data

# Extract internal_txn_ids from clicks to match to views
df_clicks_all['txn_list'] = df_clicks_all['internal_txn_id'].apply(parse_array_to_list)

# Flatten click txn_ids
click_txn_ids = set()
for txn_list in df_clicks_all['txn_list']:
    click_txn_ids.update(txn_list)

print(f"Unique internal_txn_ids in clicks: {len(click_txn_ids):,}")

# For views, extract first txn_id (most have only 1)
df_views_dedup['first_txn_id'] = df_views_dedup['internal_txn_id'].apply(
    lambda x: parse_array_to_list(x)[0] if parse_array_to_list(x) else None
)

# Mark views that resulted in clicks
df_views_dedup['has_click'] = df_views_dedup['first_txn_id'].isin(click_txn_ids)

print(f"\nViews with clicks: {df_views_dedup['has_click'].sum():,}")
print(f"Overall CTR: {df_views_dedup['has_click'].sum()/len(df_views_dedup)*100:.4f}%")

# CTR by internal_adspace_id
print("\n--- CTR by internal_adspace_id ---")
ctr_by_adspace = df_views_dedup.groupby('internal_adspace_id').agg(
    views=('has_click', 'count'),
    clicks=('has_click', 'sum')
)
ctr_by_adspace['ctr_pct'] = (ctr_by_adspace['clicks'] / ctr_by_adspace['views'] * 100).round(4)
ctr_by_adspace = ctr_by_adspace.sort_values('views', ascending=False)
print(ctr_by_adspace)

# CTR by geo_region (top 15)
print("\n--- CTR by geo_region (Top 15 by volume) ---")
ctr_by_region = df_views_dedup.groupby('geo_region_name').agg(
    views=('has_click', 'count'),
    clicks=('has_click', 'sum')
)
ctr_by_region['ctr_pct'] = (ctr_by_region['clicks'] / ctr_by_region['views'] * 100).round(4)
ctr_by_region = ctr_by_region.sort_values('views', ascending=False).head(15)
print(ctr_by_region)

# ============================================================================
# 3.8 CPC BY CAMPAIGN
# ============================================================================
print("\n### 3.8 CPC BY CAMPAIGN ###")

df_clicks_all['cpc_value'] = df_clicks_all['advertiser_spend'].apply(parse_first_array_value)
df_clicks_all['campaign_first'] = df_clicks_all['campaign_code'].apply(
    lambda x: parse_array_to_list(x)[0] if parse_array_to_list(x) else None
)

cpc_by_campaign = df_clicks_all.groupby('campaign_first').agg(
    clicks=('cpc_value', 'count'),
    total_cpc=('cpc_value', 'sum'),
    avg_cpc=('cpc_value', 'mean'),
    max_cpc=('cpc_value', 'max')
).round(2)
cpc_by_campaign = cpc_by_campaign.sort_values('total_cpc', ascending=False).head(15)
print(cpc_by_campaign)

# ============================================================================
# 3.9 HIGH-VALUE CLICK ANALYSIS
# ============================================================================
print("\n### 3.9 HIGH-VALUE CLICKS (CPC >= $50) ANALYSIS ###")

high_cpc_clicks = df_clicks_all[df_clicks_all['cpc_value'] >= 50].copy()
print(f"Total high-value clicks: {len(high_cpc_clicks)}")
print(f"Total revenue from high-value: ${high_cpc_clicks['cpc_value'].sum():.2f}")

if len(high_cpc_clicks) > 0:
    print("\nDistribution by features:")
    print(f"\n  By geo_region:")
    print(high_cpc_clicks['geo_region_name'].value_counts().head(5))
    print(f"\n  By browser_code:")
    print(high_cpc_clicks['browser_code'].value_counts())
    print(f"\n  By internal_adspace_id:")
    print(high_cpc_clicks['internal_adspace_id'].value_counts())
    print(f"\n  By campaign:")
    print(high_cpc_clicks['campaign_first'].value_counts().head(5))

# ============================================================================
# 3.10 ZERO BID INVESTIGATION
# ============================================================================
print("\n### 3.10 ZERO BID ANALYSIS ###")

zero_bids = df_bids_dj[df_bids_dj['bid_value'] == 0]
non_zero_bids = df_bids_dj[df_bids_dj['bid_value'] > 0]

print(f"Zero bids: {len(zero_bids):,} ({len(zero_bids)/len(df_bids_dj)*100:.1f}%)")
print(f"Non-zero bids: {len(non_zero_bids):,}")

if len(zero_bids) > 0:
    print("\n--- Zero bids by month ---")
    print(zero_bids['month'].value_counts().sort_index())
    
    print("\n--- Zero bids by internal_adspace_id ---")
    print(zero_bids['internal_adspace_id'].value_counts())
    
    # Check if zero bids ever won (have matching views)
    zero_bid_txnids = set(zero_bids['log_txnid'].tolist())
    view_txnids = set(df_views_dj['log_txnid'].tolist())
    zero_bids_that_won = zero_bid_txnids.intersection(view_txnids)
    print(f"\n--- Zero bids that resulted in views: {len(zero_bids_that_won)} ---")

# ============================================================================
# 3.11 FEATURE COMBINATION ANALYSIS (FOR MEMCACHE KEY DESIGN)
# ============================================================================
print("\n### 3.11 FEATURE COMBINATION COVERAGE ###")

# Count unique combinations of candidate features
candidate_features = ['internal_adspace_id', 'geo_region_name', 'browser_code']

# Single features
print("\n--- Single Feature Coverage ---")
for feat in candidate_features:
    n_unique = df_bids_dj[feat].nunique()
    null_pct = df_bids_dj[feat].isnull().sum() / len(df_bids_dj) * 100
    print(f"{feat}: {n_unique} unique values, {null_pct:.1f}% null")

# Two-feature combinations
print("\n--- Two-Feature Combinations ---")
for i, f1 in enumerate(candidate_features):
    for f2 in candidate_features[i+1:]:
        combo = df_bids_dj.groupby([f1, f2]).size()
        print(f"{f1} + {f2}: {len(combo)} unique combinations")
        # Show coverage (what % of data is covered by top N combinations)
        top_10_coverage = combo.nlargest(10).sum() / len(df_bids_dj) * 100
        top_50_coverage = combo.nlargest(50).sum() / len(df_bids_dj) * 100
        print(f"  Top 10 combos cover {top_10_coverage:.1f}%, Top 50 cover {top_50_coverage:.1f}%")

# Three-feature combination
print("\n--- Three-Feature Combination ---")
combo_3 = df_bids_dj.groupby(candidate_features).size()
print(f"All 3 features: {len(combo_3)} unique combinations")
top_10_coverage = combo_3.nlargest(10).sum() / len(df_bids_dj) * 100
top_50_coverage = combo_3.nlargest(50).sum() / len(df_bids_dj) * 100
top_100_coverage = combo_3.nlargest(100).sum() / len(df_bids_dj) * 100
print(f"  Top 10 combos cover {top_10_coverage:.1f}%")
print(f"  Top 50 combos cover {top_50_coverage:.1f}%")
print(f"  Top 100 combos cover {top_100_coverage:.1f}%")

# Distribution of combo frequencies
print("\n--- Combination Frequency Distribution ---")
combo_counts = combo_3.value_counts().sort_index()
print("Combinations appearing N times:")
for n in [1, 2, 3, 5, 10, 50, 100, 500, 1000]:
    combos_at_least_n = (combo_3 >= n).sum()
    print(f"  >= {n:>4} times: {combos_at_least_n:>4} combinations")

print("\n" + "=" * 80)
print("END OF BATCH 3")
print("=" * 80)