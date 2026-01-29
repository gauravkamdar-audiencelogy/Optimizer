#!/usr/bin/env python3
"""
Check Integration Status

Verifies that all integration credentials are properly configured
and tests connections to each service.

Usage:
    python scripts/check_integrations.py
    python scripts/check_integrations.py --test-connections
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations import (
    load_env,
    print_integration_status,
    get_integration_status,
    S3Client,
    SnowflakeClient,
    MySQLClient
)


def check_status():
    """Check and print integration status."""
    print("=" * 60)
    print("Integration Status Check")
    print("=" * 60)

    # Load environment
    load_env()

    # Print status
    print_integration_status()

    # Summary
    status = get_integration_status()
    enabled_count = sum([
        status['s3']['enabled'],
        status['snowflake']['enabled'],
        status['mysql']['enabled']
    ])

    print(f"\nSummary: {enabled_count}/3 integrations enabled")

    if status['environment'] == 'local':
        print("\n[NOTE] OPTIMIZER_ENV=local - integrations will be skipped during runs")
        print("       Set OPTIMIZER_ENV=qa or OPTIMIZER_ENV=production to enable")

    return enabled_count


def test_connections():
    """Test actual connections to each service."""
    print("\n" + "=" * 60)
    print("Testing Connections")
    print("=" * 60)

    results = {}

    # Test S3
    print("\n[S3]")
    try:
        s3 = S3Client()
        if s3.enabled:
            # Try to list buckets to verify credentials
            s3._client.list_buckets()
            print("  Connection: SUCCESS")
            results['s3'] = True
        else:
            print("  Connection: SKIPPED (not enabled)")
            results['s3'] = None
    except Exception as e:
        print(f"  Connection: FAILED - {e}")
        results['s3'] = False

    # Test Snowflake
    print("\n[Snowflake]")
    try:
        sf = SnowflakeClient()
        if sf.enabled:
            conn = sf._get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                print("  Connection: SUCCESS")
                results['snowflake'] = True
            else:
                print("  Connection: FAILED - could not connect")
                results['snowflake'] = False
        else:
            print("  Connection: SKIPPED (not enabled)")
            results['snowflake'] = None
    except Exception as e:
        print(f"  Connection: FAILED - {e}")
        results['snowflake'] = False

    # Test MySQL
    print("\n[MySQL]")
    try:
        mysql = MySQLClient()
        if mysql.enabled:
            conn = mysql._get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                print("  Connection: SUCCESS")
                results['mysql'] = True
            else:
                print("  Connection: FAILED - could not connect")
                results['mysql'] = False
        else:
            print("  Connection: SKIPPED (not enabled)")
            results['mysql'] = None
    except Exception as e:
        print(f"  Connection: FAILED - {e}")
        results['mysql'] = False

    # Summary
    print("\n" + "=" * 60)
    print("Connection Test Summary")
    print("=" * 60)

    for service, result in results.items():
        if result is True:
            status = "PASS"
        elif result is False:
            status = "FAIL"
        else:
            status = "SKIP"
        print(f"  {service.upper()}: {status}")

    failed = sum(1 for r in results.values() if r is False)
    if failed > 0:
        print(f"\n[WARNING] {failed} connection(s) failed!")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description='Check Integration Status')
    parser.add_argument('--test-connections', action='store_true',
                        help='Test actual connections to each service')
    args = parser.parse_args()

    # Always check status
    enabled = check_status()

    # Optionally test connections
    if args.test_connections:
        return test_connections()

    return 0 if enabled > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
