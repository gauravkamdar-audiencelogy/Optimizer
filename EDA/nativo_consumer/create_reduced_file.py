#!/usr/bin/env python3
"""
Create Reduced Nativo Consumer Data File
=========================================
Drops heavy columns to create a smaller working file.
Reduces ~14GB to ~5GB by removing unnecessary columns.

Run: python EDA_nativo_consumer/create_reduced_file.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent / "data_nativo_consumer"
FULL_FILE = DATA_DIR / "data_nativo_consumer_full.csv"
OUTPUT_FILE = DATA_DIR / "data_nativo_consumer_reduced.csv"

# Columns to DROP (heavy + not needed for optimizer)
DROP_COLUMNS = [
    # Heavy string columns (saves ~9GB)
    'ua',                    # User agent - 2.7GB
    'ref_bundle',            # Referrer - 2.2GB
    'source_filename',       # Internal - 1.9GB
    'server_host',           # Internal - 1.2GB
    'load_ts',               # Load timestamp - 1.1GB

    # Internal tracking (not needed)
    'source_feed',
    'source_row_number',
    'billing_status',
    'fraud_status',

    # Geo redundancy (keep geo_region_name only)
    'geo_city_name',
    'geo_dma_code',
    'geo_timezone',
    'geo_latitude',
    'geo_longitude',
    'geo_postal_code',
    'geo_country_code3',
    'geo_country_name',
    'geo_region_code',
    'geo_continent_code',

    # IDs we don't use
    'nid',
    'ip',
    'userid',
    'click_tracking_id',
    'subid',

    # Campaign/demand-side (ignore per project_notes)
    'advertiser_id',
    'campaign_code',
    'banner_code',
    'advertiser_spend',
    'rule_id',
    'data_cost',
    'data_revenue',
    'list_id',
    'conversion_payout',

    # SSP internals
    'matched_cats',
    'matched_kw',
    'content_cats',
    'adx_custom',
    'category_codes',
    'ae_endpoint_codes',
    'ae_response_codes',
    'ae_response_bids',
]

CHUNK_SIZE = 500_000

def main():
    print(f"Creating reduced data file...")
    print(f"Input: {FULL_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Dropping {len(DROP_COLUMNS)} columns")

    # First pass: determine which columns exist
    print("\nReading header...")
    sample = pd.read_csv(FULL_FILE, nrows=1)
    sample.columns = sample.columns.str.lower()

    existing_cols = set(sample.columns)
    drop_cols = [c.lower() for c in DROP_COLUMNS if c.lower() in existing_cols]
    keep_cols = [c for c in sample.columns if c not in drop_cols]

    print(f"Total columns: {len(existing_cols)}")
    print(f"Dropping: {len(drop_cols)}")
    print(f"Keeping: {len(keep_cols)}")

    # Process in chunks
    print(f"\nProcessing in chunks of {CHUNK_SIZE:,}...")
    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(FULL_FILE, usecols=keep_cols,
                             chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.lower()
        total_rows += len(chunk)

        # Write to output
        if first_chunk:
            chunk.to_csv(OUTPUT_FILE, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

        print(f"  Processed {total_rows:,} rows", end='\r')

    print(f"\n\nDone! Wrote {total_rows:,} rows to {OUTPUT_FILE}")

    # Show file size comparison
    import os
    full_size = os.path.getsize(FULL_FILE) / (1024**3)
    reduced_size = os.path.getsize(OUTPUT_FILE) / (1024**3)
    reduction = (1 - reduced_size/full_size) * 100

    print(f"\nFile size comparison:")
    print(f"  Full: {full_size:.2f} GB")
    print(f"  Reduced: {reduced_size:.2f} GB")
    print(f"  Reduction: {reduction:.1f}%")


if __name__ == '__main__':
    main()
