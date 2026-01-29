"""
Environment Variable Loader

Loads credentials from .env file for integrations.
Uses python-dotenv if available, falls back to os.environ.

Usage:
    from src.integrations.env_loader import load_env, get_env

    load_env()  # Call once at startup
    bucket = get_env('OPTIMIZER_S3_BUCKET')
"""
import os
from pathlib import Path
from typing import Optional


_env_loaded = False


def load_env(env_file: str = None) -> bool:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file (default: project root/.env)

    Returns:
        True if .env was loaded, False otherwise
    """
    global _env_loaded

    if _env_loaded:
        return True

    # Find .env file
    if env_file is None:
        # Look in project root
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'
    else:
        env_file = Path(env_file)

    if not env_file.exists():
        print(f"  [INFO] No .env file found at {env_file}")
        print(f"         Copy .env.template to .env and fill in credentials")
        return False

    try:
        # Try python-dotenv first
        from dotenv import load_dotenv
        load_dotenv(env_file)
        _env_loaded = True
        print(f"  [INFO] Loaded environment from {env_file}")
        return True
    except ImportError:
        # Fall back to manual parsing
        _load_env_manual(env_file)
        _env_loaded = True
        print(f"  [INFO] Loaded environment from {env_file} (manual parser)")
        return True


def _load_env_manual(env_file: Path):
    """Manually parse .env file if python-dotenv not installed."""
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                # Only set if not empty
                if value:
                    os.environ[key] = value


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable value.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Value or default
    """
    return os.environ.get(key, default)


def is_local_mode() -> bool:
    """Check if running in local mode (skip integrations)."""
    env = get_env('OPTIMIZER_ENV', 'local')
    return env.lower() == 'local'


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    debug = get_env('OPTIMIZER_DEBUG', 'false')
    return debug.lower() in ('true', '1', 'yes')


def get_integration_status() -> dict:
    """
    Get status of all integration credentials.

    Returns:
        Dict with enabled status for each integration
    """
    return {
        's3': {
            'enabled': all([
                get_env('AWS_ACCESS_KEY_ID'),
                get_env('AWS_SECRET_ACCESS_KEY'),
                get_env('OPTIMIZER_S3_BUCKET')
            ]),
            'bucket': get_env('OPTIMIZER_S3_BUCKET'),
            'region': get_env('AWS_REGION', 'us-east-1')
        },
        'snowflake': {
            'enabled': all([
                get_env('SNOWFLAKE_ACCOUNT'),
                get_env('SNOWFLAKE_USER'),
                get_env('SNOWFLAKE_PASSWORD'),
                get_env('SNOWFLAKE_WAREHOUSE'),
                get_env('SNOWFLAKE_DATABASE')
            ]),
            'account': get_env('SNOWFLAKE_ACCOUNT'),
            'database': get_env('SNOWFLAKE_DATABASE')
        },
        'mysql': {
            'enabled': all([
                get_env('MYSQL_HOST'),
                get_env('MYSQL_USER'),
                get_env('MYSQL_PASSWORD'),
                get_env('MYSQL_DATABASE')
            ]),
            'host': get_env('MYSQL_HOST'),
            'database': get_env('MYSQL_DATABASE')
        },
        'environment': get_env('OPTIMIZER_ENV', 'local')
    }


def print_integration_status():
    """Print status of all integrations."""
    status = get_integration_status()

    print("\n[Integration Status]")
    print(f"  Environment: {status['environment']}")

    print(f"\n  S3:")
    if status['s3']['enabled']:
        print(f"    Status: ENABLED")
        print(f"    Bucket: {status['s3']['bucket']}")
        print(f"    Region: {status['s3']['region']}")
    else:
        print(f"    Status: DISABLED (missing credentials)")

    print(f"\n  Snowflake:")
    if status['snowflake']['enabled']:
        print(f"    Status: ENABLED")
        print(f"    Account: {status['snowflake']['account']}")
        print(f"    Database: {status['snowflake']['database']}")
    else:
        print(f"    Status: DISABLED (missing credentials)")

    print(f"\n  MySQL:")
    if status['mysql']['enabled']:
        print(f"    Status: ENABLED")
        print(f"    Host: {status['mysql']['host']}")
        print(f"    Database: {status['mysql']['database']}")
    else:
        print(f"    Status: DISABLED (missing credentials)")
    print()
