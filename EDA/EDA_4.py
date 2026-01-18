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

print(f"Loaded {len(df):,} records\n")

print("=" * 80)
print("BATCH 4: VERIFICATION & FINAL ANALYSIS")
print("=" * 80)

# ============================================================================
# 4.1 CLEAN DATA: FILTER OUT PROBLEMATIC RECORDS
# ============================================================================
print("\n### 4.1 DATA CLEANING ###")

# Separate by rec_type
df_bids_raw = df[df['rec_type'] == 'bid'].copy()
df_views_raw = df[df['rec_type'] == 'View'].copy()
df_clicks_raw = df[df['rec_type'] == 'link'].copy()

print(f"Raw counts: {len(df_bids_raw):,} bids, {len(df_views_raw):,} views, {len(df_clicks_raw):,} clicks")

# 1. Deduplicate views by log_txnid
df_views_dedup = df_views_raw.drop_duplicates(subset=['log_txnid'], keep='first')
print(f"Views after dedup: {len(df_views_dedup):,}")

# 2. Filter out zero bids
df_bids_raw['bid_value'] = df_bids_raw['bid_amount'].apply(parse_first_array_value)
df_bids_nonzero = df_bids_raw[df_bids_raw['bid_value'] > 0]
print(f"Bids after removing $0: {len(df_bids_nonzero):,}")

# 3. Filter to Oct 1+ for views/clicks (cleaner data)
df_views_clean = df_views_dedup[df_views_dedup['log_dt'] >= '2025-10-01']
df_clicks_clean = df_clicks_raw[df_clicks_raw['log_dt'] >= '2025-10-01']
print(f"Views from Oct 1+: {len(df_views_clean):,}")
print(f"Clicks from Oct 1+: {len(df_clicks_clean):,}")

# ============================================================================
# 4.2 VERIFIED NUM_ADS ANALYSIS (WITH PROPER JOIN)
# ============================================================================
print("\n### 4.2 VERIFIED NUM_ADS WIN RATE (Joined Analysis) ###")

# Filter to Dec-Jan where we have bids
df_bids_dj = df_bids_nonzero[df_bids_nonzero['month'].isin([pd.Period('2025-12'), pd.Period('2026-01')])]
df_views_dj = df_views_dedup[df_views_dedup['month'].isin([pd.Period('2025-12'), pd.Period('2026-01')])]

# Calculate num_ads
df_bids_dj['num_ads_bid'] = df_bids_dj['internal_txn_id'].apply(count_array_elements)
df_views_dj['num_ads_view'] = df_views_dj['internal_txn_id'].apply(count_array_elements)

print(f"Dec-Jan non-zero bids: {len(df_bids_dj):,}")
print(f"Dec-Jan deduped views: {len(df_views_dj):,}")

# Join bids to views on log_txnid
bid_view_joined = df_bids_dj.merge(
    df_views_dj[['log_txnid', 'num_ads_view']], 
    on='log_txnid', 
    how='left',
    indicator=True
)

bid_view_joined['won'] = bid_view_joined['_merge'] == 'both'

print(f"\nJoined records: {len(bid_view_joined):,}")
print(f"Bids that won (matched to view): {bid_view_joined['won'].sum():,}")
print(f"Win rate (joined): {bid_view_joined['won'].sum()/len(bid_view_joined)*100:.2f}%")

print("\n--- Win Rate by Bid num_ads (Properly Joined) ---")
for num_ads in [1, 2]:
    subset = bid_view_joined[bid_view_joined['num_ads_bid'] == num_ads]
    bids = len(subset)
    wins = subset['won'].sum()
    wr = wins/bids*100 if bids > 0 else 0
    print(f"{num_ads}-ad bids: {bids:,} bids → {wins:,} wins = {wr:.1f}% win rate")

# Check if num_ads changes between bid and view
print("\n--- Does num_ads change from bid to view? ---")
won_bids = bid_view_joined[bid_view_joined['won'] == True].copy()
won_bids['num_ads_match'] = won_bids['num_ads_bid'] == won_bids['num_ads_view']
print(f"Total won bids: {len(won_bids):,}")
print(f"num_ads matches: {won_bids['num_ads_match'].sum():,} ({won_bids['num_ads_match'].sum()/len(won_bids)*100:.1f}%)")

# Cross-tab
print("\n--- Bid num_ads vs View num_ads (won bids only) ---")
crosstab = pd.crosstab(won_bids['num_ads_bid'], won_bids['num_ads_view'], margins=True)
print(crosstab)

# ============================================================================
# 4.3 UNMATCHED VIEWS INVESTIGATION
# ============================================================================
print("\n### 4.3 UNMATCHED VIEWS INVESTIGATION ###")

# Views in Dec-Jan that don't match to any bid
view_txnids = set(df_views_dj['log_txnid'].tolist())
bid_txnids = set(df_bids_dj['log_txnid'].tolist())

matched_views = view_txnids.intersection(bid_txnids)
unmatched_views = view_txnids - bid_txnids

print(f"Total views (Dec-Jan): {len(view_txnids):,}")
print(f"Views matching a bid: {len(matched_views):,}")
print(f"Views NOT matching any bid: {len(unmatched_views):,} ({len(unmatched_views)/len(view_txnids)*100:.1f}%)")

# Check if unmatched views are from before bid recording started
unmatched_view_df = df_views_dj[df_views_dj['log_txnid'].isin(unmatched_views)]
print(f"\nUnmatched views by date range:")
print(f"  Min date: {unmatched_view_df['log_dt'].min()}")
print(f"  Max date: {unmatched_view_df['log_dt'].max()}")

# ============================================================================
# 4.4 WIN RATE BY BID AMOUNT (JOINED)
# ============================================================================
print("\n### 4.4 WIN RATE BY BID AMOUNT (Joined) ###")

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

bid_view_joined['bid_bucket'] = bid_view_joined['bid_value'].apply(bid_bucket)

print(f"{'Bid Bucket':<20} | {'Bids':>10} | {'Wins':>10} | {'Win Rate':>10}")
print("-" * 60)
for bucket in sorted(bid_view_joined['bid_bucket'].unique()):
    subset = bid_view_joined[bid_view_joined['bid_bucket'] == bucket]
    bids = len(subset)
    wins = subset['won'].sum()
    wr = wins/bids*100 if bids > 0 else 0
    print(f"{bucket:<20} | {bids:>10,} | {wins:>10,} | {wr:>9.1f}%")

# ============================================================================
# 4.5 WIN RATE BY FEATURE SEGMENT (JOINED)
# ============================================================================
print("\n### 4.5 WIN RATE BY SEGMENT (Joined) ###")

print("\n--- By internal_adspace_id ---")
print(f"{'Adspace':<12} | {'Bids':>10} | {'Wins':>10} | {'Win Rate':>10}")
print("-" * 50)
for adspace in sorted(bid_view_joined['internal_adspace_id'].unique()):
    subset = bid_view_joined[bid_view_joined['internal_adspace_id'] == adspace]
    bids = len(subset)
    wins = subset['won'].sum()
    wr = wins/bids*100 if bids > 0 else 0
    print(f"{adspace:<12} | {bids:>10,} | {wins:>10,} | {wr:>9.1f}%")

print("\n--- By geo_region (Top 15) ---")
region_stats = bid_view_joined.groupby('geo_region_name').agg(
    bids=('won', 'count'),
    wins=('won', 'sum')
).sort_values('bids', ascending=False).head(15)
region_stats['win_rate'] = (region_stats['wins'] / region_stats['bids'] * 100).round(1)
print(region_stats)

# ============================================================================
# 4.6 CTR ANALYSIS (CLEANED DATA - OCT ONWARDS)
# ============================================================================
print("\n### 4.6 CTR ANALYSIS (Oct 2025 onwards) ###")

# Extract internal_txn_ids from clicks
df_clicks_clean['txn_list'] = df_clicks_clean['internal_txn_id'].apply(parse_array_to_list)
click_txn_ids = set()
for txn_list in df_clicks_clean['txn_list']:
    click_txn_ids.update(txn_list)

print(f"Unique click txn_ids: {len(click_txn_ids):,}")

# Mark views with clicks
df_views_clean['first_txn_id'] = df_views_clean['internal_txn_id'].apply(
    lambda x: parse_array_to_list(x)[0] if parse_array_to_list(x) else None
)
df_views_clean['has_click'] = df_views_clean['first_txn_id'].isin(click_txn_ids)

print(f"Views (Oct+): {len(df_views_clean):,}")
print(f"Views with clicks: {df_views_clean['has_click'].sum():,}")
print(f"CTR: {df_views_clean['has_click'].sum()/len(df_views_clean)*100:.4f}%")

# CTR by month
print("\n--- CTR by Month ---")
ctr_by_month = df_views_clean.groupby('month').agg(
    views=('has_click', 'count'),
    clicks=('has_click', 'sum')
)
ctr_by_month['ctr_pct'] = (ctr_by_month['clicks'] / ctr_by_month['views'] * 100).round(4)
print(ctr_by_month)

# CTR by adspace
print("\n--- CTR by Adspace ---")
ctr_by_adspace = df_views_clean.groupby('internal_adspace_id').agg(
    views=('has_click', 'count'),
    clicks=('has_click', 'sum')
)
ctr_by_adspace['ctr_pct'] = (ctr_by_adspace['clicks'] / ctr_by_adspace['views'] * 100).round(4)
ctr_by_adspace = ctr_by_adspace.sort_values('views', ascending=False)
print(ctr_by_adspace)

# ============================================================================
# 4.7 EXPECTED VALUE CALCULATION PER SEGMENT
# ============================================================================
print("\n### 4.7 EXPECTED VALUE BY SEGMENT ###")

# For each segment, calculate:
# - Win Rate (from joined bid-view)
# - CTR (from view-click)
# - Avg CPC (from clicks)

# Get avg CPC
df_clicks_clean['cpc_value'] = df_clicks_clean['advertiser_spend'].apply(parse_first_array_value)
avg_cpc = df_clicks_clean['cpc_value'].mean()
print(f"Overall Avg CPC: ${avg_cpc:.2f}")

# For adspace segments with enough data
print("\n--- Expected Value by Adspace ---")
print(f"{'Adspace':<12} | {'WinRate':>8} | {'CTR':>10} | {'E[Revenue]':>12}")
print("-" * 50)

# Get win rates from joined data (Dec-Jan only, but representative)
wr_by_adspace = bid_view_joined.groupby('internal_adspace_id')['won'].mean()

# Get CTR from view-click data
ctr_by_adspace_dict = (df_views_clean.groupby('internal_adspace_id')['has_click'].mean() * 100).to_dict()

for adspace in sorted(wr_by_adspace.index):
    wr = wr_by_adspace.get(adspace, 0) * 100
    ctr = ctr_by_adspace_dict.get(adspace, 0)
    # Expected revenue per bid = WR * CTR * CPC / 100 / 100 (convert percentages)
    expected_rev = (wr/100) * (ctr/100) * avg_cpc
    print(f"{adspace:<12} | {wr:>7.1f}% | {ctr:>9.4f}% | ${expected_rev:>10.4f}")

# ============================================================================
# 4.8 SUMMARY STATISTICS FOR MODEL TRAINING
# ============================================================================
print("\n### 4.8 SUMMARY FOR MODEL TRAINING ###")

print("\n--- Recommended Training Data ---")
print(f"Bids (Dec-Jan, non-zero): {len(df_bids_dj):,}")
print(f"Views (Oct+, deduped): {len(df_views_clean):,}")
print(f"Clicks (Oct+): {len(df_clicks_clean):,}")

print("\n--- Feature Summary ---")
print("Candidate memcache key features:")
print(f"  internal_adspace_id: {df_bids_dj['internal_adspace_id'].nunique()} unique")
print(f"  geo_region_name: {df_bids_dj['geo_region_name'].nunique()} unique (11.4% null)")
print(f"  browser_code: {df_bids_dj['browser_code'].nunique()} unique (94% = code 14)")

print("\n--- Key Metrics ---")
overall_wr = bid_view_joined['won'].mean() * 100
overall_ctr = df_views_clean['has_click'].mean() * 100
print(f"Overall Win Rate: {overall_wr:.1f}%")
print(f"Overall CTR: {overall_ctr:.4f}%")
print(f"Avg CPC: ${avg_cpc:.2f}")
print(f"Expected Revenue per Bid: ${(overall_wr/100) * (overall_ctr/100) * avg_cpc:.4f}")

print("\n" + "=" * 80)
print("END OF BATCH 4")
print("=" * 80)

# Add to EDA_4 or run separately
print("\n### SEPTEMBER CLICK QUALITY CHECK ###")

df_clicks_sep = df_clicks_raw[df_clicks_raw['month'] == pd.Period('2025-09')]
df_clicks_oct_plus = df_clicks_raw[df_clicks_raw['month'] >= pd.Period('2025-10')]

print(f"September clicks: {len(df_clicks_sep)}")
print(f"Oct+ clicks: {len(df_clicks_oct_plus)}")

# Check CPC distribution - are Sep clicks systematically different?
df_clicks_sep['cpc'] = df_clicks_sep['advertiser_spend'].apply(parse_first_array_value)
df_clicks_oct_plus['cpc'] = df_clicks_oct_plus['advertiser_spend'].apply(parse_first_array_value)

print(f"\nSep CPC: mean=${df_clicks_sep['cpc'].mean():.2f}, median=${df_clicks_sep['cpc'].median():.2f}")
print(f"Oct+ CPC: mean=${df_clicks_oct_plus['cpc'].mean():.2f}, median=${df_clicks_oct_plus['cpc'].median():.2f}")

# Check if Sep clicks come from same campaigns
print(f"\nSep campaigns: {df_clicks_sep['campaign_code'].nunique()} unique")
print(f"Oct+ campaigns: {df_clicks_oct_plus['campaign_code'].nunique()} unique")

# Check adspace distribution
print(f"\nSep clicks by adspace:")
print(df_clicks_sep['internal_adspace_id'].value_counts())
print(f"\nOct+ clicks by adspace:")
print(df_clicks_oct_plus['internal_adspace_id'].value_counts())