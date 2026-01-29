"""
S3 Client for Optimizer Output Management

Handles:
- Uploading optimizer output to S3
- Managing manifest.json for bidder deployment
- Fetching previous run metrics for validation comparison

Environment Variables:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    OPTIMIZER_S3_BUCKET: S3 bucket name

Usage:
    client = S3Client()
    if client.enabled:
        s3_path = client.upload_directory(output_dir, "drugs/20260128_143000")
        client.update_manifest("drugs", "20260128_143000", s3_path)
"""
import os
import json
from pathlib import Path
from typing import Optional
from datetime import datetime


class S3Client:
    """
    S3 client for optimizer output management.

    Works in two modes:
    - Local mode: No credentials, all operations are no-ops with logging
    - Enabled mode: Full S3 operations with boto3

    S3 Structure:
        s3://{bucket}/optimizer/{dataset}/
        ├── runs/
        │   ├── 20260125_100000/
        │   ├── 20260126_100000/
        │   └── 20260128_143000/  (current)
        └── active/
            └── manifest.json  (points to current run)
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None
    ):
        """
        Initialize S3 client.

        Args:
            bucket: S3 bucket name (or from OPTIMIZER_S3_BUCKET env var)
            region: AWS region (or from AWS_REGION env var, default us-east-1)
        """
        self.bucket = bucket or os.environ.get('OPTIMIZER_S3_BUCKET')
        self.region = region or os.environ.get('AWS_REGION', 'us-east-1')
        self.enabled = self._check_credentials()
        self._client = None

        if self.enabled:
            self._init_boto3()

    def _check_credentials(self) -> bool:
        """Check if AWS credentials are available."""
        has_key = os.environ.get('AWS_ACCESS_KEY_ID') is not None
        has_secret = os.environ.get('AWS_SECRET_ACCESS_KEY') is not None
        has_bucket = self.bucket is not None

        return has_key and has_secret and has_bucket

    def _init_boto3(self):
        """Initialize boto3 client (only called if enabled)."""
        try:
            import boto3
            self._client = boto3.client('s3', region_name=self.region)
        except ImportError:
            print("  [WARNING] boto3 not installed. S3 operations disabled.")
            self.enabled = False

    def upload_directory(
        self,
        local_path: Path,
        s3_prefix: str
    ) -> Optional[str]:
        """
        Upload output directory to S3.

        Args:
            local_path: Local directory path to upload
            s3_prefix: S3 prefix (e.g., "drugs/runs/20260128_143000")

        Returns:
            Full S3 path if successful, None if local mode
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] S3 upload skipped: {local_path}")
            return None

        s3_path = f"s3://{self.bucket}/optimizer/{s3_prefix}/"

        try:
            local_path = Path(local_path)
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"optimizer/{s3_prefix}/{relative_path}"
                    self._client.upload_file(str(file_path), self.bucket, s3_key)

            print(f"  Uploaded to: {s3_path}")
            return s3_path

        except Exception as e:
            print(f"  [ERROR] S3 upload failed: {e}")
            return None

    def update_manifest(
        self,
        dataset: str,
        run_id: str,
        s3_path: str,
        previous_run_id: Optional[str] = None,
        previous_s3_path: Optional[str] = None
    ) -> bool:
        """
        Update active/manifest.json for bidder.

        The manifest tells the bidder which optimizer run is active.

        Args:
            dataset: Dataset name (e.g., "drugs")
            run_id: Current run ID (e.g., "20260128_143000")
            s3_path: S3 path to current run
            previous_run_id: Previous run ID for rollback
            previous_s3_path: S3 path to previous run for rollback

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] Manifest update skipped")
            return False

        manifest = {
            'active_run_id': run_id,
            'active_path': s3_path,
            'rollback_run_id': previous_run_id,
            'rollback_path': previous_s3_path,
            'deployed_at': datetime.utcnow().isoformat() + 'Z',
            'dataset': dataset
        }

        try:
            manifest_key = f"optimizer/{dataset}/active/manifest.json"
            self._client.put_object(
                Bucket=self.bucket,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType='application/json'
            )

            print(f"  Manifest updated: s3://{self.bucket}/{manifest_key}")
            return True

        except Exception as e:
            print(f"  [ERROR] Manifest update failed: {e}")
            return False

    def get_previous_run_metrics(self, dataset: str) -> Optional[dict]:
        """
        Fetch metrics.json from the currently active run.

        Useful for validation comparison against previous run.

        Args:
            dataset: Dataset name (e.g., "drugs")

        Returns:
            Metrics dict if found, None otherwise
        """
        if not self.enabled:
            print(f"  [LOCAL MODE] Previous metrics fetch skipped")
            return None

        try:
            # First get the manifest to find active run
            manifest_key = f"optimizer/{dataset}/active/manifest.json"
            response = self._client.get_object(Bucket=self.bucket, Key=manifest_key)
            manifest = json.loads(response['Body'].read().decode('utf-8'))

            active_run_id = manifest.get('active_run_id')
            if not active_run_id:
                return None

            # Now get the metrics file from active run
            # Find the metrics file (has timestamp in name)
            prefix = f"optimizer/{dataset}/runs/{active_run_id}/"
            response = self._client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )

            metrics_key = None
            for obj in response.get('Contents', []):
                if 'metrics_' in obj['Key'] and obj['Key'].endswith('.json'):
                    metrics_key = obj['Key']
                    break

            if not metrics_key:
                return None

            response = self._client.get_object(Bucket=self.bucket, Key=metrics_key)
            metrics = json.loads(response['Body'].read().decode('utf-8'))

            print(f"  Loaded previous metrics from: s3://{self.bucket}/{metrics_key}")
            return metrics

        except Exception as e:
            print(f"  [WARNING] Could not fetch previous metrics: {e}")
            return None

    def get_manifest(self, dataset: str) -> Optional[dict]:
        """
        Fetch current manifest for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Manifest dict if found, None otherwise
        """
        if not self.enabled:
            return None

        try:
            manifest_key = f"optimizer/{dataset}/active/manifest.json"
            response = self._client.get_object(Bucket=self.bucket, Key=manifest_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception:
            return None
