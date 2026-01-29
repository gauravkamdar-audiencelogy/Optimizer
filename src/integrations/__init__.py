"""
Integration modules for external services.

These modules provide stubs that work in local mode (no credentials)
and can be activated on QA/production servers with environment variables.

Modules:
- env_loader: Load credentials from .env file
- s3_client: Upload outputs to S3, manage manifest for bidder
- snowflake_client: Fetch bid data and production metrics

Setup:
    1. Copy .env.template to .env
    2. Fill in credentials
    3. Integrations auto-enable when credentials are present

Usage:
    from src.integrations import load_env, S3Client, SnowflakeClient

    load_env()  # Load credentials from .env
    s3 = S3Client()
    if s3.enabled:
        s3.upload_directory(output_dir, "drugs/runs/20260128")
"""
from .env_loader import (
    load_env,
    get_env,
    is_local_mode,
    is_debug,
    get_integration_status,
    print_integration_status
)
from .s3_client import S3Client
from .snowflake_client import SnowflakeClient
from .mysql_client import MySQLClient

__all__ = [
    'load_env',
    'get_env',
    'is_local_mode',
    'is_debug',
    'get_integration_status',
    'print_integration_status',
    'S3Client',
    'SnowflakeClient',
    'MySQLClient'
]
