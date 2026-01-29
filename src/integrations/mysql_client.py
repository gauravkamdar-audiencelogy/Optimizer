"""
MySQL Client for Optimizer Audit Logging

Handles:
- Recording optimizer run metadata
- Updating run status (pending, running, completed, deployed)
- Storing validation results
- Tracking active deployments

Environment Variables:
    MYSQL_HOST: MySQL server hostname
    MYSQL_PORT: MySQL server port (default: 3306)
    MYSQL_USER: MySQL username
    MYSQL_PASSWORD: MySQL password
    MYSQL_DATABASE: Database name

Usage:
    client = MySQLClient()
    if client.enabled:
        run_id = client.create_run(dataset, config)
        client.update_run_status(run_id, 'completed', metrics)
"""
import os
import json
from typing import Optional, Dict, Any
from datetime import datetime


class MySQLClient:
    """
    MySQL client for optimizer audit logging.

    Works in two modes:
    - Local mode: No credentials, all operations are no-ops with logging
    - Enabled mode: Full MySQL operations with mysql-connector-python
    """

    def __init__(self):
        """Initialize MySQL client."""
        self.host = os.environ.get('MYSQL_HOST')
        self.port = int(os.environ.get('MYSQL_PORT', '3306'))
        self.user = os.environ.get('MYSQL_USER')
        self.password = os.environ.get('MYSQL_PASSWORD')
        self.database = os.environ.get('MYSQL_DATABASE')

        self.enabled = self._check_credentials()
        self._connection = None

    def _check_credentials(self) -> bool:
        """Check if MySQL credentials are available."""
        required = [
            self.host,
            self.user,
            self.password,
            self.database
        ]
        return all(v is not None for v in required)

    def _get_connection(self):
        """Get or create MySQL connection."""
        if self._connection is not None:
            try:
                self._connection.ping(reconnect=True)
                return self._connection
            except Exception:
                self._connection = None

        try:
            import mysql.connector
            self._connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return self._connection
        except ImportError:
            print("  [WARNING] mysql-connector-python not installed. MySQL operations disabled.")
            self.enabled = False
            return None
        except Exception as e:
            print(f"  [ERROR] MySQL connection failed: {e}")
            self.enabled = False
            return None

    def close(self):
        """Close MySQL connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def create_run(
        self,
        run_id: str,
        dataset: str,
        config: dict,
        created_by: str = None
    ) -> Optional[int]:
        """
        Create a new optimizer run record.

        Args:
            run_id: Run identifier (e.g., "20260128_143000")
            dataset: Dataset name (e.g., "drugs")
            config: Full config dict
            created_by: User who initiated the run

        Returns:
            Database ID of created record, or None if local mode
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] MySQL create_run skipped: {run_id}")
            return None

        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor()

            # Extract key config values
            business = config.get('business', {})
            technical = config.get('technical', {})
            bidding = config.get('bidding', {})

            query = """
            INSERT INTO optimizer_runs (
                run_id, dataset, created_by, status,
                config_json, target_win_rate, strategy,
                aggressive_exploration, min_bid_cpm, max_bid_cpm
            ) VALUES (
                %s, %s, %s, 'pending',
                %s, %s, %s,
                %s, %s, %s
            )
            """

            cursor.execute(query, (
                run_id,
                dataset,
                created_by or 'system',
                json.dumps(config),
                business.get('target_win_rate'),
                bidding.get('strategy'),
                technical.get('aggressive_exploration', False),
                technical.get('min_bid_cpm'),
                technical.get('max_bid_cpm')
            ))

            conn.commit()
            record_id = cursor.lastrowid
            cursor.close()

            print(f"  Created run record: ID={record_id}")
            return record_id

        except Exception as e:
            print(f"  [ERROR] MySQL create_run failed: {e}")
            return None

    def update_run_status(
        self,
        run_id: str,
        status: str,
        metrics: dict = None,
        s3_path: str = None,
        validation_result: dict = None,
        error_message: str = None
    ) -> bool:
        """
        Update run status and metrics.

        Args:
            run_id: Run identifier
            status: New status (pending, running, completed, failed, validated, deployed)
            metrics: Metrics dict from optimizer
            s3_path: S3 path where output was uploaded
            validation_result: Validation result dict
            error_message: Error message if failed

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] MySQL update_run_status skipped: {run_id} -> {status}")
            return False

        conn = self._get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()

            # Build update query dynamically
            updates = ['status = %s']
            values = [status]

            if metrics:
                updates.append('metrics_json = %s')
                values.append(json.dumps(metrics))

                # Extract key metrics
                bid_summary = metrics.get('bid_summary', {})
                if bid_summary:
                    updates.extend([
                        'segments_count = %s',
                        'bid_median = %s',
                        'bid_min = %s',
                        'bid_max = %s'
                    ])
                    values.extend([
                        bid_summary.get('count'),
                        bid_summary.get('bid_median'),
                        bid_summary.get('bid_min'),
                        bid_summary.get('bid_max')
                    ])

                # Extract features
                features = metrics.get('feature_selection', {}).get('selected_features', [])
                if features:
                    updates.append('features_used = %s')
                    values.append(json.dumps(features))

            if s3_path:
                updates.append('s3_path = %s')
                values.append(s3_path)

            if validation_result:
                updates.append('validation_json = %s')
                values.append(json.dumps(validation_result))
                updates.append('validation_status = %s')
                values.append('passed' if validation_result.get('validation_passed') else 'failed')

            if error_message:
                updates.append('error_message = %s')
                values.append(error_message)

            if status == 'deployed':
                updates.append('deployed_at = %s')
                values.append(datetime.utcnow())

            # Add run_id to values for WHERE clause
            values.append(run_id)

            query = f"""
            UPDATE optimizer_runs
            SET {', '.join(updates)}
            WHERE run_id = %s
            """

            cursor.execute(query, values)
            conn.commit()
            cursor.close()

            print(f"  Updated run status: {run_id} -> {status}")
            return True

        except Exception as e:
            print(f"  [ERROR] MySQL update_run_status failed: {e}")
            return False

    def set_active_deployment(
        self,
        run_id: str,
        dataset: str
    ) -> bool:
        """
        Set a run as the active deployment for a dataset.

        Deactivates any previous active run for the same dataset.

        Args:
            run_id: Run to set as active
            dataset: Dataset name

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] MySQL set_active_deployment skipped: {run_id}")
            return False

        conn = self._get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()

            # Deactivate previous active run
            cursor.execute("""
                UPDATE optimizer_runs
                SET is_active = FALSE
                WHERE dataset = %s AND is_active = TRUE
            """, (dataset,))

            # Activate new run
            cursor.execute("""
                UPDATE optimizer_runs
                SET is_active = TRUE, deployed_at = %s
                WHERE run_id = %s
            """, (datetime.utcnow(), run_id))

            conn.commit()
            cursor.close()

            print(f"  Set active deployment: {run_id} for {dataset}")
            return True

        except Exception as e:
            print(f"  [ERROR] MySQL set_active_deployment failed: {e}")
            return False

    def get_active_run(self, dataset: str) -> Optional[dict]:
        """
        Get the currently active run for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Run record dict, or None if not found
        """
        if not self.enabled:
            return None

        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM optimizer_runs
                WHERE dataset = %s AND is_active = TRUE
                LIMIT 1
            """, (dataset,))

            result = cursor.fetchone()
            cursor.close()
            return result

        except Exception as e:
            print(f"  [WARNING] MySQL get_active_run failed: {e}")
            return None

    def get_previous_run(self, dataset: str) -> Optional[dict]:
        """
        Get the most recent completed run for a dataset.

        Useful for validation comparison.

        Args:
            dataset: Dataset name

        Returns:
            Run record dict, or None if not found
        """
        if not self.enabled:
            return None

        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM optimizer_runs
                WHERE dataset = %s
                  AND status IN ('completed', 'validated', 'deployed')
                ORDER BY created_at DESC
                LIMIT 1
            """, (dataset,))

            result = cursor.fetchone()
            cursor.close()
            return result

        except Exception as e:
            print(f"  [WARNING] MySQL get_previous_run failed: {e}")
            return None

    def get_run_config(self, run_id: str) -> Optional[dict]:
        """
        Get config for a specific run.

        Used when optimizer is triggered from frontend -
        fetches the config stored in the database.

        Args:
            run_id: Run identifier

        Returns:
            Config dict, or None if not found
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] MySQL get_run_config skipped")
            return None

        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT config_json FROM optimizer_runs
                WHERE run_id = %s
            """, (run_id,))

            result = cursor.fetchone()
            cursor.close()

            if result and result.get('config_json'):
                return json.loads(result['config_json'])
            return None

        except Exception as e:
            print(f"  [WARNING] MySQL get_run_config failed: {e}")
            return None
