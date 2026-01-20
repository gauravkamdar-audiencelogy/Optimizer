import pandas as pd
import numpy as np
from datetime import datetime

# Load the data from 3 CSV files in the data folder
data_folder = '../data'

print("Loading CSV files...")
print("Note: Using on_bad_lines='skip' to handle malformed rows\n")

# Helper function to count total lines in a file
def count_file_lines(filepath):
    with open(filepath, 'r') as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header

# Load with error handling for malformed lines
# Note: engine='python' doesn't support low_memory parameter
bids_total = count_file_lines(f'{data_folder}/drugs_bids.csv')
df_bids = pd.read_csv(f'{data_folder}/drugs_bids.csv',
                      on_bad_lines='skip',
                      engine='python')
bids_skipped = bids_total - len(df_bids)

clicks_total = count_file_lines(f'{data_folder}/drugs_clicks.csv')
df_clicks = pd.read_csv(f'{data_folder}/drugs_clicks.csv',
                        on_bad_lines='skip',
                        engine='python')
clicks_skipped = clicks_total - len(df_clicks)

views_total = count_file_lines(f'{data_folder}/drugs_views.csv')
df_views = pd.read_csv(f'{data_folder}/drugs_views.csv',
                       on_bad_lines='skip',
                       engine='python')
views_skipped = views_total - len(df_views)

print(f"Bids: Loaded {len(df_bids):,} records (skipped {bids_skipped:,})")
print(f"Clicks: Loaded {len(df_clicks):,} records (skipped {clicks_skipped:,})")
print(f"Views: Loaded {len(df_views):,} records (skipped {views_skipped:,})")

total_skipped = bids_skipped + clicks_skipped + views_skipped
print(f"\nTotal skipped rows: {total_skipped:,}")

# Combine all dataframes
df = pd.concat([df_bids, df_clicks, df_views], ignore_index=True)
print(f"Combined total: {len(df):,} records\n")

print("=" * 80)
print("BATCH 1: BASELINE METRICS & DATA QUALITY")
print("=" * 80)

# 1.1 Basic shape and date range
print("\n### 1.1 DATA OVERVIEW ###")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Date range: {df['log_dt'].min()} to {df['log_dt'].max()}")

# 1.2 Record type distribution
print("\n### 1.2 RECORD TYPE DISTRIBUTION ###")
rec_type_counts = df['rec_type'].value_counts()
print(rec_type_counts)
print(f"\nPercentages:")
print((rec_type_counts / len(df) * 100).round(2))

# 1.3 Records by month and type
print("\n### 1.3 RECORDS BY MONTH AND TYPE ###")
# Use format='ISO8601' and utc=True to handle mixed timezones
df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)
df['month'] = df['log_dt'].dt.to_period('M')
monthly_by_type = df.groupby(['month', 'rec_type']).size().unstack(fill_value=0)
print(monthly_by_type)

# 1.4 Check for NPI validity in external_userid
print("\n### 1.4 NPI VALIDATION ###")
if 'external_userid' in df.columns:
    def is_valid_npi(val):
        if pd.isna(val):
            return False
        val_str = str(val).strip()
        return val_str.isdigit() and len(val_str) in [10, 11]

    df['has_valid_npi'] = df['external_userid'].apply(is_valid_npi)
    npi_by_rec_type = df.groupby('rec_type')['has_valid_npi'].agg(['sum', 'count'])
    npi_by_rec_type['pct_valid'] = (npi_by_rec_type['sum'] / npi_by_rec_type['count'] * 100).round(2)
    npi_by_rec_type.columns = ['valid_npi_count', 'total_count', 'pct_valid']
    print(npi_by_rec_type)
else:
    print("WARNING: 'external_userid' column not found in data")

# 1.5 Array field inspection (internal_txn_id)
print("\n### 1.5 ARRAY FIELD ANALYSIS (internal_txn_id) ###")
if 'internal_txn_id' in df.columns and df['internal_txn_id'].notna().sum() > 0:
    # Check if it's stored as string representation of array
    sample_txn = df['internal_txn_id'].dropna().iloc[0]
    print(f"Sample internal_txn_id value: {sample_txn}")
    print(f"Type: {type(sample_txn)}")
else:
    print("WARNING: 'internal_txn_id' column not found or all values are null")
    sample_txn = None

# Parse array and count elements
if 'internal_txn_id' in df.columns:
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

    df['num_ads'] = df['internal_txn_id'].apply(count_array_elements)
    print("\nNumber of ads per request distribution:")
    # Use try-except to handle cases where value_counts might fail
    try:
        num_ads_dist = df.groupby('rec_type')['num_ads'].value_counts().unstack(fill_value=0)
        print(num_ads_dist)
    except Exception as e:
        print(f"Could not create cross-tabulation: {e}")
        print("\nAlternative view - value counts by rec_type:")
        for rt in df['rec_type'].unique():
            print(f"\n{rt}:")
            print(df[df['rec_type'] == rt]['num_ads'].value_counts().sort_index())
else:
    print("\nSkipping num_ads analysis - internal_txn_id column not available")

# 1.6 Unique identifier check
print("\n### 1.6 UNIQUE IDENTIFIER ANALYSIS ###")
print(f"Unique log_txnid count: {df['log_txnid'].nunique():,}")
print(f"Total rows: {len(df):,}")
print(f"Ratio (should be ~1.0 if log_txnid is unique per row): {df['log_txnid'].nunique() / len(df):.4f}")

# Check uniqueness by rec_type
for rt in df['rec_type'].unique():
    subset = df[df['rec_type'] == rt]
    ratio = subset['log_txnid'].nunique() / len(subset)
    print(f"  {rt}: {subset['log_txnid'].nunique():,} unique / {len(subset):,} rows = {ratio:.4f}")

# 1.7 Key columns null check
print("\n### 1.7 NULL CHECK FOR KEY COLUMNS ###")
key_cols = ['log_txnid', 'internal_txn_id', 'external_userid', 'publisher_payout',
            'advertiser_spend', 'browser_code', 'os_code', 'geo_region_name',
            'domain', 'internal_adspace_id', 'campaign_code']
# Filter to only columns that exist in the dataframe
existing_key_cols = [col for col in key_cols if col in df.columns]
missing_key_cols = [col for col in key_cols if col not in df.columns]

if missing_key_cols:
    print(f"WARNING: These expected columns are missing: {missing_key_cols}\n")

null_check = df[existing_key_cols].isnull().sum()
null_pct = (null_check / len(df) * 100).round(2)
null_df = pd.DataFrame({'null_count': null_check, 'null_pct': null_pct})
print(null_df)

# 1.8 Sample of actual data (first row of each rec_type)
print("\n### 1.8 SAMPLE DATA BY REC_TYPE ###")
# Use actual rec_type values from the data instead of hardcoded list
sample_cols = ['log_txnid', 'internal_txn_id', 'external_userid',
               'publisher_payout', 'advertiser_spend', 'browser_code',
               'os_code', 'geo_region_name', 'campaign_code']

for rt in sorted(df['rec_type'].unique()):
    print(f"\n--- {rt} sample ---")
    sample = df[df['rec_type'] == rt].head(1)
    if len(sample) > 0:
        for col in sample_cols:
            if col in sample.columns:
                print(f"  {col}: {sample[col].values[0]}")
    else:
        print(f"  No records found for {rt}")