"""
EDA_5: Final Feature Analysis
- ref_bundle (page-level URL data)
- All geo features comparison
- Time-based features
- Any remaining high-signal features
"""
import pandas as pd
import numpy as np
from urllib.parse import urlparse
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

print(f"Loaded {len(df):,} records\n")

# Print all columns to see what we have
print("=" * 80)
print("ALL AVAILABLE COLUMNS")
print("=" * 80)
print(f"\nTotal columns: {len(df.columns)}")
for i, col in enumerate(sorted(df.columns)):
    print(f"  {i+1:2}. {col}")

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

print("\n" + "=" * 80)
print("BATCH 5: FINAL FEATURE ANALYSIS")
print("=" * 80)

# ============================================================================
# 5.1 REF_BUNDLE (PAGE-LEVEL URL ANALYSIS)
# ============================================================================
print("\n### 5.1 REF_BUNDLE (PAGE-LEVEL URL ANALYSIS) ###")

if 'ref_bundle' in df.columns:
    # Sample ref_bundle values
    print("\nSample ref_bundle values:")
    sample_refs = df[df['ref_bundle'].notna()]['ref_bundle'].head(10)
    for i, ref in enumerate(sample_refs):
        print(f"  {i+1}. {ref[:100]}..." if len(str(ref)) > 100 else f"  {i+1}. {ref}")
    
    # Extract path from URL
    def extract_url_path(url):
        if pd.isna(url):
            return None
        try:
            parsed = urlparse(str(url))
            path = parsed.path
            # Get first path segment (e.g., /drug-name/ -> drug-name)
            parts = [p for p in path.split('/') if p]
            if parts:
                return parts[0]  # First path segment
            return 'root'
        except:
            return None
    
    def extract_full_path(url):
        if pd.isna(url):
            return None
        try:
            parsed = urlparse(str(url))
            return parsed.path
        except:
            return None
    
    df['page_category'] = df['ref_bundle'].apply(extract_url_path)
    df['full_path'] = df['ref_bundle'].apply(extract_full_path)
    
    print(f"\nUnique page categories: {df['page_category'].nunique()}")
    print(f"Null page categories: {df['page_category'].isnull().sum()} ({df['page_category'].isnull().sum()/len(df)*100:.1f}%)")
    
    print("\nTop 20 page categories (all data):")
    page_cats = df['page_category'].value_counts().head(20)
    for cat, count in page_cats.items():
        pct = count / len(df) * 100
        print(f"  {cat}: {count:,} ({pct:.2f}%)")
    
    # Analyze page categories by rec_type
    print("\n--- Page Categories by Record Type ---")
    for rec_type in ['bid', 'View', 'link']:
        subset = df[df['rec_type'] == rec_type]
        print(f"\n{rec_type} records - Top 10 page categories:")
        top_cats = subset['page_category'].value_counts().head(10)
        for cat, count in top_cats.items():
            pct = count / len(subset) * 100
            print(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    # CTR by page category
    print("\n--- CTR by Page Category (Views with 50+ records) ---")
    df_views_all = df[df['rec_type'] == 'View'].copy()
    df_views_dedup = df_views_all.drop_duplicates(subset=['log_txnid'], keep='first')
    df_clicks_all = df[df['rec_type'] == 'link'].copy()
    
    # Extract click txn_ids
    df_clicks_all['txn_list'] = df_clicks_all['internal_txn_id'].apply(parse_array_to_list)
    click_txn_ids = set()
    for txn_list in df_clicks_all['txn_list']:
        click_txn_ids.update(txn_list)
    
    df_views_dedup['first_txn_id'] = df_views_dedup['internal_txn_id'].apply(
        lambda x: parse_array_to_list(x)[0] if parse_array_to_list(x) else None
    )
    df_views_dedup['has_click'] = df_views_dedup['first_txn_id'].isin(click_txn_ids)
    
    ctr_by_page = df_views_dedup.groupby('page_category').agg(
        views=('has_click', 'count'),
        clicks=('has_click', 'sum')
    )
    ctr_by_page['ctr_pct'] = (ctr_by_page['clicks'] / ctr_by_page['views'] * 100).round(4)
    ctr_by_page = ctr_by_page[ctr_by_page['views'] >= 50].sort_values('views', ascending=False).head(20)
    print(ctr_by_page)
    
else:
    print("WARNING: 'ref_bundle' column not found in data")

# ============================================================================
# 5.2 ALL GEO FEATURES COMPARISON
# ============================================================================
print("\n### 5.2 ALL GEO FEATURES COMPARISON ###")

geo_features = ['geo_region_name', 'geo_city_name', 'geo_postal_code', 
                'geo_country_code2', 'geo_dma_code', 'geo_metro_code']

print("\n--- Geo Feature Cardinality and Nulls ---")
df_bids_all = df[df['rec_type'] == 'bid'].copy()

for feat in geo_features:
    if feat in df_bids_all.columns:
        nunique = df_bids_all[feat].nunique()
        null_count = df_bids_all[feat].isnull().sum()
        null_pct = null_count / len(df_bids_all) * 100
        print(f"{feat:25} | {nunique:>8,} unique | {null_pct:>5.1f}% null")
    else:
        print(f"{feat:25} | NOT FOUND")

# Win rate by each geo feature
print("\n--- Win Rate Analysis by Geo Features (Dec-Jan) ---")
df_bids_dj = df_bids_all[df_bids_all['month'].isin([pd.Period('2025-12'), pd.Period('2026-01')])]
df_views_dj = df_views_dedup[df_views_dedup['month'].isin([pd.Period('2025-12'), pd.Period('2026-01')])]

# Join bids to views
df_bids_dj['bid_value'] = df_bids_dj['bid_amount'].apply(parse_first_array_value)
df_bids_nonzero = df_bids_dj[df_bids_dj['bid_value'] > 0]

bid_view_joined = df_bids_nonzero.merge(
    df_views_dj[['log_txnid']], 
    on='log_txnid', 
    how='left',
    indicator=True
)
bid_view_joined['won'] = bid_view_joined['_merge'] == 'both'

for feat in ['geo_region_name', 'geo_country_code2']:
    if feat in bid_view_joined.columns:
        print(f"\n--- Win Rate by {feat} (Top 15) ---")
        wr_by_feat = bid_view_joined.groupby(feat).agg(
            bids=('won', 'count'),
            wins=('won', 'sum')
        )
        wr_by_feat['win_rate'] = (wr_by_feat['wins'] / wr_by_feat['bids'] * 100).round(1)
        wr_by_feat = wr_by_feat.sort_values('bids', ascending=False).head(15)
        print(wr_by_feat)

# ============================================================================
# 5.3 TIME-BASED FEATURES
# ============================================================================
print("\n### 5.3 TIME-BASED FEATURES ###")

# Hour of day analysis
df['hour'] = df['log_dt'].dt.hour
df['day_of_week'] = df['log_dt'].dt.dayofweek  # 0=Monday, 6=Sunday

print("\n--- Bids by Hour of Day ---")
bids_by_hour = df_bids_all.groupby(df_bids_all['log_dt'].dt.hour).size()
print(bids_by_hour)

print("\n--- CTR by Hour of Day ---")
df_views_dedup['hour'] = df_views_dedup['log_dt'].dt.hour
ctr_by_hour = df_views_dedup.groupby('hour').agg(
    views=('has_click', 'count'),
    clicks=('has_click', 'sum')
)
ctr_by_hour['ctr_pct'] = (ctr_by_hour['clicks'] / ctr_by_hour['views'] * 100).round(4)
print(ctr_by_hour[ctr_by_hour['views'] >= 100])

print("\n--- Bids by Day of Week ---")
df_bids_all['day_of_week'] = df_bids_all['log_dt'].dt.dayofweek
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
bids_by_dow = df_bids_all.groupby('day_of_week').size()
for dow, count in bids_by_dow.items():
    print(f"  {day_names[dow]}: {count:,}")

# ============================================================================
# 5.4 BROWSER AND OS DEEP DIVE
# ============================================================================
print("\n### 5.4 BROWSER AND OS ANALYSIS ###")

print("\n--- Win Rate by Browser Code ---")
for browser in bid_view_joined['browser_code'].value_counts().head(5).index:
    subset = bid_view_joined[bid_view_joined['browser_code'] == browser]
    wr = subset['won'].sum() / len(subset) * 100
    print(f"  Browser {browser}: {len(subset):,} bids, {wr:.1f}% win rate")

print("\n--- Win Rate by OS Code ---")
for os_code in bid_view_joined['os_code'].value_counts().head(5).index:
    subset = bid_view_joined[bid_view_joined['os_code'] == os_code]
    wr = subset['won'].sum() / len(subset) * 100
    print(f"  OS {os_code}: {len(subset):,} bids, {wr:.1f}% win rate")

# ============================================================================
# 5.5 INTERNAL_SITE_ID ANALYSIS
# ============================================================================
print("\n### 5.5 INTERNAL_SITE_ID ANALYSIS ###")

if 'internal_site_id' in df.columns:
    print("\n--- Unique internal_site_id values ---")
    print(df_bids_all['internal_site_id'].value_counts())
    
    print("\n--- Win Rate by internal_site_id ---")
    for site_id in bid_view_joined['internal_site_id'].value_counts().index:
        subset = bid_view_joined[bid_view_joined['internal_site_id'] == site_id]
        wr = subset['won'].sum() / len(subset) * 100
        print(f"  Site {site_id}: {len(subset):,} bids, {wr:.1f}% win rate")

# ============================================================================
# 5.6 CAMPAIGN ANALYSIS (DEMAND SIDE)
# ============================================================================
print("\n### 5.6 CAMPAIGN ANALYSIS ###")

# CPC by campaign
df_clicks_all['cpc_value'] = df_clicks_all['advertiser_spend'].apply(parse_first_array_value)
df_clicks_all['campaign_first'] = df_clicks_all['campaign_code'].apply(
    lambda x: parse_array_to_list(x)[0] if parse_array_to_list(x) else None
)

print("\n--- Top 15 Campaigns by Total Revenue ---")
campaign_revenue = df_clicks_all.groupby('campaign_first').agg(
    clicks=('cpc_value', 'count'),
    total_revenue=('cpc_value', 'sum'),
    avg_cpc=('cpc_value', 'mean'),
    max_cpc=('cpc_value', 'max')
).round(2)
campaign_revenue = campaign_revenue.sort_values('total_revenue', ascending=False).head(15)
print(campaign_revenue)

# ============================================================================
# 5.7 FEATURE SIGNAL SUMMARY (FOR DYNAMIC SELECTION)
# ============================================================================
print("\n### 5.7 FEATURE SIGNAL SUMMARY ###")

# Calculate information gain proxy (win rate variance) for each feature
features_to_evaluate = [
    'internal_adspace_id', 'geo_region_name', 'geo_country_code2',
    'browser_code', 'os_code', 'internal_site_id', 'page_category'
]

print("\n--- Feature Signal Analysis (Win Rate Variance) ---")
print(f"{'Feature':<25} | {'Unique':>8} | {'Null%':>6} | {'WR_Var':>8} | {'Signal':>8}")
print("-" * 70)

for feat in features_to_evaluate:
    if feat in bid_view_joined.columns and bid_view_joined[feat].notna().sum() > 0:
        nunique = bid_view_joined[feat].nunique()
        null_pct = bid_view_joined[feat].isnull().sum() / len(bid_view_joined) * 100
        
        # Calculate win rate variance across values
        wr_by_val = bid_view_joined.groupby(feat)['won'].mean()
        wr_variance = wr_by_val.var() * 10000  # Scale for readability
        
        # Signal score = variance * (1 - null_pct/100) * log(unique)
        signal = wr_variance * (1 - null_pct/100) * np.log(nunique + 1)
        
        print(f"{feat:<25} | {nunique:>8,} | {null_pct:>5.1f}% | {wr_variance:>8.2f} | {signal:>8.2f}")

# ============================================================================
# 5.8 RECOMMENDED FEATURE COMBINATIONS
# ============================================================================
print("\n### 5.8 RECOMMENDED FEATURE COMBINATIONS ###")

# Test different feature combinations for coverage and sparsity
feature_combos = [
    ['internal_adspace_id'],
    ['internal_adspace_id', 'geo_region_name'],
    ['internal_adspace_id', 'geo_region_name', 'browser_code'],
    ['internal_adspace_id', 'page_category'],
    ['internal_adspace_id', 'geo_region_name', 'page_category'],
]

print(f"\n{'Combination':<55} | {'Unique':>8} | {'>=50 obs':>10} | {'Coverage':>10}")
print("-" * 90)

for combo in feature_combos:
    if all(f in bid_view_joined.columns for f in combo):
        combo_groups = bid_view_joined.groupby(combo).size()
        n_unique = len(combo_groups)
        n_with_50 = (combo_groups >= 50).sum()
        coverage = combo_groups[combo_groups >= 50].sum() / len(bid_view_joined) * 100
        
        combo_str = ' + '.join(combo)
        print(f"{combo_str:<55} | {n_unique:>8,} | {n_with_50:>10,} | {coverage:>9.1f}%")

print("\n" + "=" * 80)
print("END OF BATCH 5")
print("=" * 80)