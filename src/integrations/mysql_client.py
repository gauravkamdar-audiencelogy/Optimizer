"""
MySQL Client for Optimizer

Handles:
- Loading run configs from opt_run_configs table (Phase 3)
- Loading entity configs from opt_entity_configs table
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

    SSH Tunnel (optional - for bastion access):
    MYSQL_SSH_HOST: Bastion host (e.g., bastion.example.com)
    MYSQL_SSH_PORT: SSH port (default: 22)
    MYSQL_SSH_USER: SSH username
    MYSQL_SSH_KEY_PATH: Path to SSH private key (.pem file)

Usage:
    client = MySQLClient()
    if client.enabled:
        # Load configs for UI-triggered run
        run_configs = client.get_run_configs_by_run_id(run_id)
        entity = client.get_entity_by_code('nativo_consumer')

        # Audit logging
        run_id = client.create_run(dataset, config)
        client.update_run_status(run_id, 'completed', metrics)
"""
import os
import json
from typing import Optional, Dict, Any, List
from datetime import datetime


# =============================================================================
# Config Value Type Conversion
# =============================================================================
# MySQL stores all config values as strings. These helpers convert to proper types.

_BOOLEAN_KEYS = {
    'fast_learning', 'floor_available', 'npi_enabled', 'domain_enabled',
    'exploration_mode', 'aggressive_exploration'
}
_FLOAT_KEYS = {'target_win_rate', 'max_bid_cpm', 'min_bid_cpm'}
_DATE_KEYS = {'training_start_date', 'training_end_date', 'data_start_date', 'data_end_date'}
_LIST_KEYS = {'user_disabled_features', 'ssp_exclusions'}


def _parse_config_value(key: str, value: str) -> Any:
    """Parse config value from MySQL string to proper Python type."""
    if value is None or value == '':
        if key in _LIST_KEYS:
            return []
        return None

    if key in _BOOLEAN_KEYS:
        return value.lower() in ('true', '1', 'yes')

    if key in _FLOAT_KEYS:
        try:
            return float(value)
        except ValueError:
            return None

    if key in _DATE_KEYS:
        return value if value else None

    if key in _LIST_KEYS:
        if not value:
            return []
        return [v.strip() for v in value.split(',') if v.strip()]

    return value


class MySQLClient:
    """
    MySQL client for optimizer audit logging.

    Works in two modes:
    - Local mode: No credentials, all operations are no-ops with logging
    - Enabled mode: Full MySQL operations with mysql-connector-python

    Supports SSH tunneling for bastion access.
    """

    def __init__(self):
        """Initialize MySQL client."""
        self.host = os.environ.get('MYSQL_HOST')
        self.port = int(os.environ.get('MYSQL_PORT', '3306'))
        self.user = os.environ.get('MYSQL_USER')
        self.password = os.environ.get('MYSQL_PASSWORD')
        self.database = os.environ.get('MYSQL_DATABASE')

        # SSH tunnel configuration (optional)
        self.ssh_host = os.environ.get('MYSQL_SSH_HOST')
        self.ssh_port = int(os.environ.get('MYSQL_SSH_PORT', '22'))
        self.ssh_user = os.environ.get('MYSQL_SSH_USER')
        self.ssh_key_path = os.environ.get('MYSQL_SSH_KEY_PATH')

        self.enabled = self._check_credentials()
        self._connection = None
        self._tunnel = None

    def _check_credentials(self) -> bool:
        """Check if MySQL credentials are available."""
        required = [
            self.host,
            self.user,
            self.password,
            self.database
        ]
        return all(v is not None for v in required)

    def _needs_ssh_tunnel(self) -> bool:
        """Check if SSH tunnel is configured."""
        return all([
            self.ssh_host,
            self.ssh_user,
            self.ssh_key_path
        ])

    def _start_ssh_tunnel(self):
        """Start SSH tunnel to bastion host."""
        if self._tunnel is not None:
            return self._tunnel

        try:
            from sshtunnel import SSHTunnelForwarder
            import paramiko

            # Resolve key path (handle ./ prefix)
            key_path = os.path.expanduser(self.ssh_key_path)
            if not os.path.isabs(key_path):
                # Relative path - resolve from current working directory
                key_path = os.path.abspath(key_path)

            if not os.path.exists(key_path):
                print(f"  [ERROR] SSH key not found: {key_path}")
                return None

            # Load the private key manually to handle different key types
            pkey = None
            key_errors = []

            # Try RSA first (most common)
            try:
                pkey = paramiko.RSAKey.from_private_key_file(key_path)
            except Exception as e:
                key_errors.append(f"RSA: {e}")

            # Try Ed25519
            if pkey is None:
                try:
                    pkey = paramiko.Ed25519Key.from_private_key_file(key_path)
                except Exception as e:
                    key_errors.append(f"Ed25519: {e}")

            # Try ECDSA
            if pkey is None:
                try:
                    pkey = paramiko.ECDSAKey.from_private_key_file(key_path)
                except Exception as e:
                    key_errors.append(f"ECDSA: {e}")

            if pkey is None:
                print(f"  [ERROR] Could not load SSH key: {key_path}")
                for err in key_errors:
                    print(f"    {err}")
                return None

            self._tunnel = SSHTunnelForwarder(
                (self.ssh_host, self.ssh_port),
                ssh_username=self.ssh_user,
                ssh_pkey=pkey,
                remote_bind_address=(self.host, self.port),
                local_bind_address=('127.0.0.1', 0)  # Auto-assign local port
            )
            self._tunnel.start()
            print(f"  SSH tunnel: {self.ssh_host} -> {self.host}:{self.port} (local port: {self._tunnel.local_bind_port})")
            return self._tunnel

        except ImportError:
            print("  [ERROR] sshtunnel not installed. Run: pip install sshtunnel")
            return None
        except Exception as e:
            print(f"  [ERROR] SSH tunnel failed: {e}")
            return None

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

            # Determine connection parameters
            connect_host = self.host
            connect_port = self.port

            # Use SSH tunnel if configured
            if self._needs_ssh_tunnel():
                tunnel = self._start_ssh_tunnel()
                if tunnel is None:
                    return None
                connect_host = '127.0.0.1'
                connect_port = tunnel.local_bind_port

            self._connection = mysql.connector.connect(
                host=connect_host,
                port=connect_port,
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
        """Close MySQL connection and SSH tunnel."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

        if self._tunnel is not None:
            self._tunnel.stop()
            self._tunnel = None

    # =========================================================================
    # CONFIG LOADING METHODS (Phase 3)
    # =========================================================================

    def get_entity_by_code(self, entity_code: str) -> Optional[Dict[str, Any]]:
        """
        Get entity record from opt_entities table.

        Args:
            entity_code: Entity identifier (e.g., 'nativo_consumer', 'drugs_hcp')

        Returns:
            Dict with opt_entity_id, entity_code, entity_name, s3_base_path, etc.
            Returns None if not found or not connected.
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] get_entity_by_code skipped: {entity_code}")
            return None

        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT opt_entity_id, entity_code, entity_name, object_id, object_type,
                       current_version_id, active_run_id, snowflake_table,
                       s3_base_path, status, created_on
                FROM opt_entities
                WHERE entity_code = %s AND status = 'A'
                LIMIT 1
            """, (entity_code,))

            result = cursor.fetchone()
            cursor.close()

            if result:
                print(f"  Found entity: {entity_code} (opt_entity_id={result['opt_entity_id']})")
            else:
                print(f"  [WARNING] Entity not found: {entity_code}")

            return result

        except Exception as e:
            print(f"  [ERROR] get_entity_by_code failed: {e}")
            return None

    def get_entity_configs(self, opt_entity_id: int) -> Dict[str, Any]:
        """
        Get entity configs from opt_entity_configs table.

        Args:
            opt_entity_id: Entity ID from opt_entities.opt_entity_id

        Returns:
            Dict of config_key -> parsed config_value
            Example: {'floor_available': True, 'targeting_type': 'consumer', ...}
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] get_entity_configs skipped: opt_entity_id={opt_entity_id}")
            return {}

        conn = self._get_connection()
        if conn is None:
            return {}

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT config_key, config_value
                FROM opt_entity_configs
                WHERE opt_entity_id = %s AND status = 'A'
            """, (opt_entity_id,))

            rows = cursor.fetchall()
            cursor.close()

            configs = {}
            for row in rows:
                key = row['config_key']
                value = _parse_config_value(key, row['config_value'])
                configs[key] = value

            print(f"  Loaded {len(configs)} entity configs from MySQL")
            return configs

        except Exception as e:
            print(f"  [ERROR] get_entity_configs failed: {e}")
            return {}

    def get_run_configs_by_run_id(self, run_id: int) -> Dict[str, Any]:
        """
        Get run configs from opt_run_configs table.

        This is the primary method for loading user-set configs when
        optimizer is triggered from the UI.

        Args:
            run_id: Run ID from opt_runs table

        Returns:
            Dict of config_key -> parsed config_value
            Example: {'target_win_rate': 0.65, 'max_bid_cpm': 20.0, ...}
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] get_run_configs_by_run_id skipped: run_id={run_id}")
            return {}

        conn = self._get_connection()
        if conn is None:
            return {}

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT config_key, config_value
                FROM opt_run_configs
                WHERE run_id = %s
            """, (run_id,))

            rows = cursor.fetchall()
            cursor.close()

            configs = {}
            for row in rows:
                key = row['config_key']
                value = _parse_config_value(key, row['config_value'])
                configs[key] = value

            print(f"  Loaded {len(configs)} run configs from MySQL for run_id={run_id}")
            return configs

        except Exception as e:
            print(f"  [ERROR] get_run_configs_by_run_id failed: {e}")
            return {}

    def get_run_by_id(self, run_id: int) -> Optional[Dict[str, Any]]:
        """
        Get run record from opt_runs table.

        Args:
            run_id: Run ID

        Returns:
            Dict with full run record, or None if not found
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] get_run_by_id skipped: run_id={run_id}")
            return None

        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT run_id, opt_entity_id, version_id, run_code, status,
                       triggered_by, trigger_type, error_message,
                       data_start_date, data_end_date, total_bids,
                       total_views, total_clicks, segments_count, domains_count,
                       npis_count, features_used, bid_median, bid_min, bid_max,
                       s3_output_path, validation_status, config_snapshot,
                       total_segments, segments_in_memcache, global_win_rate, global_ctr,
                       created_on, started_at, completed_at
                FROM opt_runs
                WHERE run_id = %s
                LIMIT 1
            """, (run_id,))

            result = cursor.fetchone()
            cursor.close()

            if result:
                print(f"  Found run: run_id={run_id}, status={result.get('status')}")
            else:
                print(f"  [WARNING] Run not found: run_id={run_id}")

            return result

        except Exception as e:
            print(f"  [ERROR] get_run_by_id failed: {e}")
            return None

    def create_run_with_configs(
        self,
        entity_code: str,
        configs: Dict[str, Any],
        triggered_by: str = 'system',
        trigger_type: str = 'manual'
    ) -> Optional[int]:
        """
        Create a new run record and store associated configs.

        This is used by the UI to create a run before triggering the optimizer.

        Args:
            entity_code: Entity code (e.g., 'nativo_consumer')
            configs: Dict of config_key -> config_value
            triggered_by: Who triggered this run (user_id or 'system')
            trigger_type: How it was triggered ('manual', 'scheduled')

        Returns:
            run_id of created record, or None if failed
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] create_run_with_configs skipped")
            return None

        conn = self._get_connection()
        if conn is None:
            return None

        try:
            # First get opt_entity_id
            entity = self.get_entity_by_code(entity_code)
            if not entity:
                print(f"  [ERROR] Cannot create run - entity not found: {entity_code}")
                return None

            opt_entity_id = entity['opt_entity_id']
            version_id = entity.get('current_version_id')

            cursor = conn.cursor()

            # Generate run_code (timestamp-based)
            from datetime import datetime
            run_code = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Insert run record
            cursor.execute("""
                INSERT INTO opt_runs (opt_entity_id, version_id, run_code, status, triggered_by, trigger_type, created_by)
                VALUES (%s, %s, %s, 'queued', %s, %s, %s)
            """, (opt_entity_id, version_id, run_code, triggered_by, trigger_type, triggered_by))

            run_id = cursor.lastrowid

            # Insert config values
            for key, value in configs.items():
                # Convert value to string for storage
                if isinstance(value, bool):
                    str_value = 'true' if value else 'false'
                elif isinstance(value, list):
                    str_value = ','.join(str(v) for v in value)
                elif value is None:
                    str_value = ''
                else:
                    str_value = str(value)

                cursor.execute("""
                    INSERT INTO opt_run_configs (run_id, config_key, config_value, created_by)
                    VALUES (%s, %s, %s, %s)
                """, (run_id, key, str_value, triggered_by))

            conn.commit()
            cursor.close()

            print(f"  Created run: run_id={run_id}, run_code={run_code} with {len(configs)} configs")
            return run_id

        except Exception as e:
            print(f"  [ERROR] create_run_with_configs failed: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test MySQL connection and return status.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled:
            print("  [LOCAL MODE] MySQL not configured")
            return False

        conn = self._get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            print("  MySQL connection: OK")
            return True
        except Exception as e:
            print(f"  MySQL connection: FAILED ({e})")
            return False

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

            # Add timestamps based on status
            if status == 'running':
                updates.append('started_at = %s')
                values.append(datetime.utcnow())
            elif status in ('completed', 'failed'):
                updates.append('completed_at = %s')
                values.append(datetime.utcnow())

            if metrics:
                # Extract key metrics from bid_summary
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

                # Extract global stats if provided
                global_stats = metrics.get('global_stats', {})
                if global_stats:
                    if 'global_win_rate' in global_stats:
                        updates.append('global_win_rate = %s')
                        values.append(global_stats['global_win_rate'])
                    if 'global_ctr' in global_stats:
                        updates.append('global_ctr = %s')
                        values.append(global_stats['global_ctr'])

                # Extract data counts if provided
                data_stats = metrics.get('data_stats', {})
                if data_stats:
                    if 'total_bids' in data_stats:
                        updates.append('total_bids = %s')
                        values.append(data_stats['total_bids'])
                    if 'total_views' in data_stats:
                        updates.append('total_views = %s')
                        values.append(data_stats['total_views'])
                    if 'total_clicks' in data_stats:
                        updates.append('total_clicks = %s')
                        values.append(data_stats['total_clicks'])
                    if 'domains_count' in data_stats:
                        updates.append('domains_count = %s')
                        values.append(data_stats['domains_count'])
                    if 'npis_count' in data_stats:
                        updates.append('npis_count = %s')
                        values.append(data_stats['npis_count'])

            if s3_path:
                updates.append('s3_output_path = %s')
                values.append(s3_path)

            if validation_result:
                updates.append('validation_status = %s')
                values.append('passed' if validation_result.get('validation_passed') else 'failed')

            if error_message:
                updates.append('error_message = %s')
                values.append(error_message)

            # Add run_id to values for WHERE clause
            values.append(run_id)

            query = f"""
            UPDATE opt_runs
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

