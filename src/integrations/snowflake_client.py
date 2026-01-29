"""
Snowflake Client for Data Fetching

Handles:
- Fetching latest bid/view/click data from Snowflake
- Fetching live production metrics for validation

Environment Variables:
    SNOWFLAKE_ACCOUNT: Snowflake account identifier
    SNOWFLAKE_USER: Snowflake username
    SNOWFLAKE_PASSWORD: Snowflake password (for password auth)
    SNOWFLAKE_PRIVATE_KEY_PATH: Path to PEM file (for key-pair auth)
    SNOWFLAKE_WAREHOUSE: Snowflake warehouse name
    SNOWFLAKE_DATABASE: Snowflake database name
    SNOWFLAKE_SCHEMA: Snowflake schema name
    SNOWFLAKE_ROLE: Snowflake role (optional)

Supports both password authentication and key-pair authentication.
If SNOWFLAKE_PRIVATE_KEY_PATH is set, key-pair auth is used.

Usage:
    client = SnowflakeClient()
    if client.enabled:
        rows = client.fetch_latest_data("drugs", output_path, days_back=7)
        metrics = client.fetch_production_metrics("drugs", days_back=7)
"""
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta


class SnowflakeClient:
    """
    Snowflake client for data fetching.

    Works in two modes:
    - Local mode: No credentials, all operations are no-ops with logging
    - Enabled mode: Full Snowflake operations with snowflake-connector-python

    Supports both password and key-pair authentication.
    """

    def __init__(self):
        """Initialize Snowflake client."""
        self.account = os.environ.get('SNOWFLAKE_ACCOUNT')
        self.user = os.environ.get('SNOWFLAKE_USER')
        self.password = os.environ.get('SNOWFLAKE_PASSWORD')
        self.private_key_path = os.environ.get('SNOWFLAKE_PRIVATE_KEY_PATH')
        self.warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE')
        self.database = os.environ.get('SNOWFLAKE_DATABASE')
        self.schema = os.environ.get('SNOWFLAKE_SCHEMA')
        self.role = os.environ.get('SNOWFLAKE_ROLE')

        self.enabled = self._check_credentials()
        self._connection = None
        self._private_key = None

    def _check_credentials(self) -> bool:
        """Check if Snowflake credentials are available."""
        # Must have account, user, warehouse, database
        required = [
            self.account,
            self.user,
            self.warehouse,
            self.database
        ]
        if not all(v is not None for v in required):
            return False

        # Must have either password OR private key path
        has_auth = (self.password is not None) or (self.private_key_path is not None)
        return has_auth

    def _load_private_key(self):
        """Load private key from PEM file."""
        if self._private_key is not None:
            return self._private_key

        if not self.private_key_path:
            return None

        key_path = Path(self.private_key_path)
        if not key_path.exists():
            print(f"  [ERROR] Private key file not found: {key_path}")
            return None

        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            with open(key_path, 'rb') as key_file:
                self._private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,  # Assuming unencrypted key
                    backend=default_backend()
                )
            return self._private_key
        except ImportError:
            print("  [WARNING] cryptography package not installed. Key-pair auth requires: pip install cryptography")
            return None
        except Exception as e:
            print(f"  [ERROR] Failed to load private key: {e}")
            return None

    def _get_connection(self):
        """Get or create Snowflake connection."""
        if self._connection is not None:
            return self._connection

        try:
            import snowflake.connector

            # Build connection parameters
            conn_params = {
                'account': self.account,
                'user': self.user,
                'warehouse': self.warehouse,
                'database': self.database,
                'schema': self.schema
            }

            # Add role if specified
            if self.role:
                conn_params['role'] = self.role

            # Use key-pair auth if private key path is set, otherwise password
            if self.private_key_path:
                private_key = self._load_private_key()
                if private_key is None:
                    self.enabled = False
                    return None

                # Get private key bytes in DER format for Snowflake
                from cryptography.hazmat.primitives import serialization
                private_key_bytes = private_key.private_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                conn_params['private_key'] = private_key_bytes
            else:
                conn_params['password'] = self.password

            self._connection = snowflake.connector.connect(**conn_params)
            return self._connection

        except ImportError:
            print("  [WARNING] snowflake-connector-python not installed. Run: pip install snowflake-connector-python")
            self.enabled = False
            return None
        except Exception as e:
            print(f"  [ERROR] Snowflake connection failed: {e}")
            self.enabled = False
            return None

    def close(self):
        """Close Snowflake connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def fetch_latest_data(
        self,
        dataset: str,
        output_path: Path,
        days_back: int = 7
    ) -> int:
        """
        Fetch latest bid/view/click data from Snowflake.

        Writes data to a CSV file in the incoming/ directory for ingestion.

        Args:
            dataset: Dataset name (e.g., "drugs", "nativo_consumer")
            output_path: Path to write the CSV file
            days_back: Number of days of data to fetch

        Returns:
            Number of rows fetched, or 0 if local mode
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] Snowflake data fetch skipped")
            return 0

        conn = self._get_connection()
        if conn is None:
            return 0

        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)

            # Build query based on dataset
            # Note: Actual table names and columns should be configured per environment
            query = self._build_data_query(dataset, start_date, end_date)

            print(f"  Fetching data for {dataset} from {start_date.date()} to {end_date.date()}...")

            cursor = conn.cursor()
            cursor.execute(query)

            # Fetch to pandas and write CSV
            import pandas as pd
            df = cursor.fetch_pandas_all()
            cursor.close()

            if len(df) == 0:
                print(f"  No new data found")
                return 0

            # Write to output path
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

            print(f"  Fetched {len(df):,} rows to {output_path}")
            return len(df)

        except Exception as e:
            print(f"  [ERROR] Snowflake data fetch failed: {e}")
            return 0

    def _build_data_query(
        self,
        dataset: str,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """
        Build SQL query for data fetch.

        Note: This is a placeholder. Actual queries should be configured
        per environment and dataset.

        Args:
            dataset: Dataset name
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            SQL query string
        """
        # Placeholder query structure
        # Actual implementation should use proper table names
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        if dataset == 'drugs':
            table = 'RTB_LOGS_DRUGS'
        elif dataset == 'nativo_consumer':
            table = 'RTB_LOGS_NATIVO'
        else:
            table = f'RTB_LOGS_{dataset.upper()}'

        query = f"""
        SELECT *
        FROM {table}
        WHERE log_dt >= '{start_str}'
          AND log_dt <= '{end_str}'
        ORDER BY log_dt DESC
        """

        return query

    def fetch_production_metrics(
        self,
        dataset: str,
        days_back: int = 7
    ) -> Optional[dict]:
        """
        Fetch live production metrics for validation comparison.

        Returns aggregated metrics over the specified period.

        Args:
            dataset: Dataset name
            days_back: Number of days to aggregate

        Returns:
            Dict with production metrics, or None if local mode
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] Production metrics fetch skipped")
            return None

        conn = self._get_connection()
        if conn is None:
            return None

        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Placeholder query - actual implementation depends on data model
            query = f"""
            SELECT
                COUNT(CASE WHEN rec_type = 'bid' THEN 1 END) as total_bids,
                COUNT(CASE WHEN rec_type = 'view' THEN 1 END) as total_wins,
                COUNT(CASE WHEN rec_type = 'click' THEN 1 END) as total_clicks,
                AVG(CASE WHEN rec_type = 'view' THEN bid_amount_cpm END) as avg_winning_bid
            FROM RTB_LOGS_{dataset.upper()}
            WHERE log_dt >= '{start_str}'
              AND log_dt <= '{end_str}'
            """

            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            cursor.close()

            if row is None:
                return None

            total_bids, total_wins, total_clicks, avg_winning_bid = row

            metrics = {
                'period_start': start_str,
                'period_end': end_str,
                'total_bids': total_bids or 0,
                'total_wins': total_wins or 0,
                'total_clicks': total_clicks or 0,
                'win_rate': (total_wins / total_bids) if total_bids > 0 else 0,
                'avg_winning_bid': float(avg_winning_bid) if avg_winning_bid else 0
            }

            print(f"  Production metrics ({days_back}d): WR={metrics['win_rate']:.1%}, "
                  f"Avg bid=${metrics['avg_winning_bid']:.2f}")

            return metrics

        except Exception as e:
            print(f"  [WARNING] Production metrics fetch failed: {e}")
            return None
