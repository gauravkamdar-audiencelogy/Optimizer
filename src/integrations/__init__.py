"""
Integration modules for external services.

These modules provide stubs that work in local mode (no credentials)
and can be activated on QA/production servers with environment variables.

Modules:
- s3_client: Upload outputs to S3, manage manifest for bidder
- snowflake_client: Fetch bid data and production metrics

Usage:
    from src.integrations.s3_client import S3Client
    from src.integrations.snowflake_client import SnowflakeClient
"""
from .s3_client import S3Client
from .snowflake_client import SnowflakeClient

__all__ = ['S3Client', 'SnowflakeClient']
