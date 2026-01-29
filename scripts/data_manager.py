#!/usr/bin/env python3
"""
Data Management CLI for Bid Optimizer

Production Workflow:
    1. Drop new data file(s) in data/drugs/incoming/
    2. Run: python scripts/data_manager.py ingest --data-dir data/drugs/
    3. Script appends to data_drugs.csv, moves processed files to processed/

Directory Structure:
    data/
    ├── NPI_click_data_*.csv    # Shared NPI files (used across datasets)
    ├── drugs/                  # drugs.com dataset
    │   ├── data_drugs.csv      # Main data file (optimizer reads this)
    │   ├── incoming/           # Drop new files here
    │   ├── processed/          # Processed incoming files (audit trail)
    │   └── archive/            # Original separate files (historical)
    └── nativo_consumer/        # nativo consumer dataset
        └── data_nativo_consumer_*.csv

Commands:
    ingest    - Process all files in incoming/, append to main, move to processed/
    combine   - One-time: merge separate bids/views/clicks into data_{name}.csv
    info      - Show data statistics and directory status
    init      - Initialize directory structure

Usage:
    python scripts/data_manager.py init --data-dir data/drugs/
    python scripts/data_manager.py ingest --data-dir data/drugs/
    python scripts/data_manager.py info --data-dir data/drugs/
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import sys


def parse_first_array_value(val) -> float:
    """Extract first numeric value from postgres array string like {7.50000}."""
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


def standardize_rec_type(rec_type: str) -> str:
    """Standardize rec_type values to lowercase."""
    if pd.isna(rec_type):
        return 'unknown'
    rt = str(rec_type).lower().strip()
    # Map variations
    if rt in ['bid', 'bids']:
        return 'bid'
    elif rt in ['view', 'views']:
        return 'view'
    elif rt in ['click', 'clicks', 'link']:
        return 'click'
    return rt


def init_directories(data_dir: Path) -> None:
    """Initialize the directory structure for data management."""
    print("=" * 60)
    print("INITIALIZING DIRECTORY STRUCTURE")
    print("=" * 60)

    dirs_to_create = [
        data_dir / 'incoming',
        data_dir / 'processed',
        data_dir / 'archive',
    ]

    for d in dirs_to_create:
        if d.exists():
            print(f"  [EXISTS] {d.relative_to(data_dir.parent)}/")
        else:
            d.mkdir(parents=True)
            print(f"  [CREATED] {d.relative_to(data_dir.parent)}/")

    print("\n" + "=" * 60)
    print("DIRECTORY STRUCTURE READY")
    print("=" * 60)
    print(f"""
Directory structure:
    {data_dir.name}/
    ├── drugs_data.csv          # Main data (optimizer reads this)
    ├── incoming/               # Drop new files here
    ├── processed/              # Processed files (audit trail)
    └── archive/                # Original separate files

Workflow:
    1. Drop new CSV file(s) in {data_dir.name}/incoming/
    2. Run: python scripts/data_manager.py ingest --data-dir {data_dir}
    3. Files are processed and moved to processed/
""")


def ingest_data(data_dir: Path, dry_run: bool = False, data_file: str = None) -> bool:
    """
    Process all files in incoming/, append to main data, move to processed/.

    This is the main production workflow command.

    Args:
        data_dir: Path to data directory
        dry_run: If True, show what would be done without doing it
        data_file: Name of main data file (default: derived from dir name, e.g., data_drugs.csv)

    Returns:
        True if any data was ingested, False otherwise
    """
    print("=" * 60)
    print("INGESTING NEW DATA")
    print("=" * 60)

    incoming_dir = data_dir / 'incoming'
    processed_dir = data_dir / 'processed'

    # Derive data file name from directory if not specified
    # data_drugs/ -> data_drugs.csv, data_nativo_consumer/ -> data_nativo_consumer.csv
    if data_file is None:
        data_file = f"{data_dir.name}.csv"
    main_data_path = data_dir / data_file

    # Ensure directories exist
    if not incoming_dir.exists():
        print(f"Creating incoming directory: {incoming_dir}")
        incoming_dir.mkdir(parents=True)

    if not processed_dir.exists():
        print(f"Creating processed directory: {processed_dir}")
        processed_dir.mkdir(parents=True)

    # Find files to process
    incoming_files = list(incoming_dir.glob('*.csv'))

    if not incoming_files:
        print(f"\nNo CSV files found in {incoming_dir.relative_to(data_dir.parent)}/")
        print("Drop new data files there and run this command again.")
        return False

    print(f"\nFound {len(incoming_files)} file(s) to process:")
    for f in incoming_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.1f} MB)")

    if dry_run:
        print("\n[DRY RUN] Would process these files. Run without --dry-run to execute.")
        return False

    # Check main data exists
    if not main_data_path.exists():
        print(f"\nERROR: Main data file {main_data_path.name} does not exist.")
        print("Run 'combine' command first to create it from separate files.")
        sys.exit(1)

    # Load existing data
    print(f"\n1. Loading existing data from {main_data_path.name}...")
    df_existing = pd.read_csv(main_data_path, on_bad_lines='skip', engine='python')
    df_existing['log_dt'] = pd.to_datetime(df_existing['log_dt'], format='ISO8601', utc=True)
    rows_before = len(df_existing)
    print(f"   Existing rows: {rows_before:,}")

    existing_min = df_existing['log_dt'].min()
    existing_max = df_existing['log_dt'].max()
    print(f"   Date range: {existing_min.strftime('%Y-%m-%d')} to {existing_max.strftime('%Y-%m-%d')}")

    # Process each incoming file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_new_data = []
    processed_files = []

    for i, incoming_file in enumerate(incoming_files, 1):
        print(f"\n2.{i}. Processing {incoming_file.name}...")

        try:
            df_new = pd.read_csv(incoming_file, on_bad_lines='skip', engine='python')

            # Normalize column names to lowercase (handle UPPERCASE exports)
            df_new.columns = df_new.columns.str.lower()

            df_new['log_dt'] = pd.to_datetime(df_new['log_dt'], format='ISO8601', utc=True)
            df_new['rec_type'] = df_new['rec_type'].apply(standardize_rec_type)

            rows_in_file = len(df_new)
            new_min = df_new['log_dt'].min()
            new_max = df_new['log_dt'].max()

            print(f"     Rows: {rows_in_file:,}")
            print(f"     Date range: {new_min.strftime('%Y-%m-%d')} to {new_max.strftime('%Y-%m-%d')}")

            # Check rec_type distribution
            rec_counts = df_new['rec_type'].value_counts()
            print(f"     Rec types: {dict(rec_counts)}")

            all_new_data.append(df_new)
            processed_files.append(incoming_file)

        except Exception as e:
            print(f"     ERROR: Failed to process {incoming_file.name}: {e}")
            print(f"     Skipping this file.")
            continue

    if not all_new_data:
        print("\nNo files were successfully processed.")
        return False

    # Concatenate all new data
    print(f"\n3. Concatenating {len(all_new_data)} file(s)...")
    df_all_new = pd.concat(all_new_data, ignore_index=True)
    total_new_rows = len(df_all_new)
    print(f"   Total new rows: {total_new_rows:,}")

    # Merge with existing
    print("\n4. Merging with existing data...")
    df_combined = pd.concat([df_existing, df_all_new], ignore_index=True)
    print(f"   Rows before dedup: {len(df_combined):,}")

    # Deduplicate
    print("\n5. Deduplicating by (log_txnid, rec_type)...")
    df_combined = df_combined.drop_duplicates(subset=['log_txnid', 'rec_type'], keep='first')
    rows_after = len(df_combined)
    dedup_removed = rows_before + total_new_rows - rows_after
    rows_added = rows_after - rows_before
    print(f"   Duplicates removed: {dedup_removed:,}")
    print(f"   Net rows added: {rows_added:,}")
    print(f"   Final row count: {rows_after:,}")

    # Sort by date
    print("\n6. Sorting by date...")
    df_combined = df_combined.sort_values('log_dt').reset_index(drop=True)

    # Write main data
    print(f"\n7. Writing to {main_data_path.name}...")
    df_combined.to_csv(main_data_path, index=False)
    file_size_mb = main_data_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.1f} MB")

    # Move processed files
    print(f"\n8. Moving processed files to processed/...")
    for incoming_file in processed_files:
        # Add timestamp to filename for audit trail
        new_name = f"{incoming_file.stem}_{timestamp}{incoming_file.suffix}"
        processed_path = processed_dir / new_name
        shutil.move(str(incoming_file), str(processed_path))
        print(f"   {incoming_file.name} -> processed/{new_name}")

    # Summary
    print("\n" + "=" * 60)
    print("INGEST COMPLETE")
    print("=" * 60)

    final_min = df_combined['log_dt'].min()
    final_max = df_combined['log_dt'].max()
    print(f"\nMain data summary:")
    print(f"   File: {main_data_path.name}")
    print(f"   Total rows: {rows_after:,}")
    print(f"   Date range: {final_min.strftime('%Y-%m-%d')} to {final_max.strftime('%Y-%m-%d')}")

    print(f"\nChanges:")
    print(f"   Files processed: {len(processed_files)}")
    print(f"   Rows added: {rows_added:,}")
    print(f"   Duplicates removed: {dedup_removed:,}")

    print(f"\nRec type distribution:")
    rec_counts = df_combined['rec_type'].value_counts()
    for rec_type, count in rec_counts.items():
        pct = count / len(df_combined) * 100
        print(f"   {rec_type}: {count:,} ({pct:.1f}%)")

    return True


def combine_data(data_dir: Path, archive: bool = True) -> None:
    """
    One-time operation: Combine separate bids/views/clicks files into single drugs_data.csv.

    Use this when migrating from separate files to combined format.
    """
    print("=" * 60)
    print("COMBINING DATA FILES (One-time migration)")
    print("=" * 60)

    bids_path = data_dir / 'drugs_bids.csv'
    views_path = data_dir / 'drugs_views.csv'
    clicks_path = data_dir / 'drugs_clicks.csv'
    output_path = data_dir / 'drugs_data.csv'

    # Check files exist
    missing = []
    for path in [bids_path, views_path, clicks_path]:
        if not path.exists():
            missing.append(path.name)

    if missing:
        print(f"ERROR: Missing files: {', '.join(missing)}")
        print("\nThis command is for one-time migration from separate files.")
        print("If you already have drugs_data.csv, use 'ingest' command instead.")
        sys.exit(1)

    if output_path.exists():
        print(f"WARNING: {output_path.name} already exists!")
        response = input("Overwrite? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Aborted.")
            sys.exit(0)

    # Load each file
    print(f"\n1. Loading bids from {bids_path.name}...")
    df_bids = pd.read_csv(bids_path, on_bad_lines='skip', engine='python')
    df_bids['rec_type'] = df_bids['rec_type'].apply(standardize_rec_type)
    print(f"   Loaded {len(df_bids):,} rows")

    print(f"\n2. Loading views from {views_path.name}...")
    df_views = pd.read_csv(views_path, on_bad_lines='skip', engine='python')
    df_views['rec_type'] = df_views['rec_type'].apply(standardize_rec_type)
    print(f"   Loaded {len(df_views):,} rows")

    print(f"\n3. Loading clicks from {clicks_path.name}...")
    df_clicks = pd.read_csv(clicks_path, on_bad_lines='skip', engine='python')
    df_clicks['rec_type'] = df_clicks['rec_type'].apply(standardize_rec_type)
    print(f"   Loaded {len(df_clicks):,} rows")

    # Concatenate
    print("\n4. Concatenating...")
    total_before = len(df_bids) + len(df_views) + len(df_clicks)
    df_combined = pd.concat([df_bids, df_views, df_clicks], ignore_index=True)
    print(f"   Total rows before dedup: {len(df_combined):,}")

    # Deduplicate
    print("\n5. Deduplicating by (log_txnid, rec_type)...")
    df_combined = df_combined.drop_duplicates(subset=['log_txnid', 'rec_type'], keep='first')
    dedup_removed = total_before - len(df_combined)
    print(f"   Removed {dedup_removed:,} duplicates")
    print(f"   Final row count: {len(df_combined):,}")

    # Parse datetime and sort
    print("\n6. Sorting by date...")
    df_combined['log_dt'] = pd.to_datetime(df_combined['log_dt'], format='ISO8601', utc=True)
    df_combined = df_combined.sort_values('log_dt').reset_index(drop=True)

    # Write output
    print(f"\n7. Writing to {output_path.name}...")
    df_combined.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.1f} MB")

    # Archive originals
    if archive:
        print("\n8. Archiving original files...")
        archive_dir = data_dir / 'archive'
        archive_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for path in [bids_path, views_path, clicks_path]:
            archive_path = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
            shutil.move(str(path), str(archive_path))
            print(f"   Moved {path.name} -> archive/{archive_path.name}")

    # Initialize other directories
    (data_dir / 'incoming').mkdir(exist_ok=True)
    (data_dir / 'processed').mkdir(exist_ok=True)

    # Summary
    print("\n" + "=" * 60)
    print("COMBINE COMPLETE")
    print("=" * 60)
    print(f"\nRec type distribution:")
    rec_counts = df_combined['rec_type'].value_counts()
    for rec_type, count in rec_counts.items():
        print(f"   {rec_type}: {count:,}")

    min_date = df_combined['log_dt'].min()
    max_date = df_combined['log_dt'].max()
    print(f"\nDate range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    print(f"""
Next steps:
    1. For new data, drop files in {data_dir.name}/incoming/
    2. Run: python scripts/data_manager.py ingest --data-dir {data_dir}
""")


def show_info(data_dir: Path, data_file: str = None) -> None:
    """Show data statistics and directory status."""
    print("=" * 60)
    print("DATA INFO")
    print("=" * 60)

    # Directory status
    incoming_dir = data_dir / 'incoming'
    processed_dir = data_dir / 'processed'
    archive_dir = data_dir / 'archive'

    # Derive data file name from directory if not specified
    if data_file is None:
        data_file = f"{data_dir.name}.csv"
    main_data_path = data_dir / data_file

    print("\n[Directory Status]")
    print(f"   incoming/:  ", end="")
    if incoming_dir.exists():
        incoming_files = list(incoming_dir.glob('*.csv'))
        if incoming_files:
            print(f"{len(incoming_files)} file(s) waiting to be processed")
            for f in incoming_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"               - {f.name} ({size_mb:.1f} MB)")
        else:
            print("empty (ready for new files)")
    else:
        print("NOT CREATED (run 'init' command)")

    print(f"   processed/: ", end="")
    if processed_dir.exists():
        processed_files = list(processed_dir.glob('*.csv'))
        print(f"{len(processed_files)} file(s)")
    else:
        print("NOT CREATED")

    print(f"   archive/:   ", end="")
    if archive_dir.exists():
        archive_files = list(archive_dir.glob('*.csv'))
        print(f"{len(archive_files)} file(s)")
    else:
        print("NOT CREATED")

    # Main data file
    print("\n[Main Data File]")
    if main_data_path.exists():
        file_size_mb = main_data_path.stat().st_size / (1024 * 1024)
        print(f"   File: {main_data_path.name} ({file_size_mb:.1f} MB)")

        # Load and analyze
        print("   Loading...")
        df = pd.read_csv(main_data_path, usecols=['log_dt', 'rec_type', 'log_txnid'])
        df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)

        print(f"   Total rows: {len(df):,}")

        min_date = df['log_dt'].min()
        max_date = df['log_dt'].max()
        date_range_days = (max_date - min_date).days
        print(f"   Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range_days} days)")

        # Days since last data
        today = pd.Timestamp.now(tz='UTC')
        days_old = (today - max_date).days
        print(f"   Data freshness: {days_old} day(s) old")

        print(f"\n   Rec type distribution:")
        rec_counts = df['rec_type'].value_counts()
        for rec_type, count in rec_counts.items():
            pct = count / len(df) * 100
            print(f"      {rec_type}: {count:,} ({pct:.1f}%)")

        print(f"\n   Date ranges by rec_type:")
        for rec_type in rec_counts.index:
            mask = df['rec_type'] == rec_type
            rt_min = df.loc[mask, 'log_dt'].min()
            rt_max = df.loc[mask, 'log_dt'].max()
            print(f"      {rec_type}: {rt_min.strftime('%Y-%m-%d')} to {rt_max.strftime('%Y-%m-%d')}")

    else:
        print(f"   NOT FOUND: {main_data_path.name}")
        print("\n   Checking for separate files...")

        files = {
            'bids': data_dir / 'drugs_bids.csv',
            'views': data_dir / 'drugs_views.csv',
            'clicks': data_dir / 'drugs_clicks.csv'
        }

        found_any = False
        for name, path in files.items():
            if path.exists():
                found_any = True
                file_size_mb = path.stat().st_size / (1024 * 1024)
                df = pd.read_csv(path, usecols=['log_dt'])
                df['log_dt'] = pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)
                min_date = df['log_dt'].min()
                max_date = df['log_dt'].max()
                print(f"   {name}: {len(df):,} rows, {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        if found_any:
            print("\n   Run 'combine' to merge these into drugs_data.csv")

    # Processed files detail
    if processed_dir.exists():
        processed_files = sorted(processed_dir.glob('*.csv'))
        if processed_files:
            print(f"\n[Processed Files (last 5)]")
            for f in processed_files[-5:]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   {f.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Data Management CLI for Bid Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Production Workflow:
    1. Initialize:  python scripts/data_manager.py init --data-dir data/drugs/
    2. Drop files:  Copy new CSV files to data/drugs/incoming/
    3. Ingest:      python scripts/data_manager.py ingest --data-dir data/drugs/
    4. Verify:      python scripts/data_manager.py info --data-dir data/drugs/

One-time Migration (from separate files):
    python scripts/data_manager.py combine --data-dir data/drugs/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize directory structure')
    init_parser.add_argument('--data-dir', type=str, required=True, help='Data directory')

    # Ingest command (main production workflow)
    ingest_parser = subparsers.add_parser('ingest', help='Process incoming files and append to main data')
    ingest_parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    ingest_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')

    # Combine command (one-time migration)
    combine_parser = subparsers.add_parser('combine', help='One-time: combine separate bids/views/clicks files')
    combine_parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    combine_parser.add_argument('--no-archive', action='store_true', help='Do not archive original files')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show data statistics and directory status')
    info_parser.add_argument('--data-dir', type=str, required=True, help='Data directory')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir} does not exist.")
        sys.exit(1)

    if args.command == 'init':
        init_directories(data_dir)
    elif args.command == 'ingest':
        ingest_data(data_dir, dry_run=args.dry_run)
    elif args.command == 'combine':
        combine_data(data_dir, archive=not args.no_archive)
    elif args.command == 'info':
        show_info(data_dir)


if __name__ == '__main__':
    main()
