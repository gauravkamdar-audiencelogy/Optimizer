#!/usr/bin/env python3
"""Query MySQL database to see current optimizer data."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

# Load env manually
env_file = Path('.env')
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, val = line.split('=', 1)
            os.environ[key.strip()] = val.strip()

from src.integrations import MySQLClient

client = MySQLClient()
if not client.enabled:
    print('MySQL not configured')
    sys.exit(1)

conn = client._get_connection()
if not conn:
    print('Failed to connect')
    sys.exit(1)

cursor = conn.cursor(dictionary=True)

# 1. Query opt_entities
print('=' * 80)
print('1. opt_entities - What SSPs/entities exist?')
print('=' * 80)
cursor.execute('''
    SELECT opt_entity_id, entity_code, entity_name, object_id, object_type,
           current_version_id, snowflake_table, s3_base_path, status
    FROM opt_entities
''')
rows = cursor.fetchall()
for row in rows:
    print()
    print(f"  opt_entity_id: {row['opt_entity_id']}")
    print(f"  entity_code:   {row['entity_code']}")
    print(f"  entity_name:   {row['entity_name']}")
    print(f"  object_type:   {row['object_type']}")
    print(f"  object_id:     {row['object_id']}")
    print(f"  snowflake:     {row['snowflake_table']}")
    print(f"  s3_base_path:  {row['s3_base_path']}")
    print(f"  status:        {row['status']}")

# 2. Query opt_entity_configs
print()
print('=' * 80)
print('2. opt_entity_configs - What entity-level configs exist?')
print('=' * 80)
cursor.execute('''
    SELECT e.entity_code, c.config_key, c.config_value
    FROM opt_entity_configs c
    JOIN opt_entities e ON c.opt_entity_id = e.opt_entity_id
    ORDER BY e.entity_code, c.config_key
''')
rows = cursor.fetchall()
current_entity = None
for row in rows:
    if row['entity_code'] != current_entity:
        current_entity = row['entity_code']
        print(f"\n  [{current_entity}]")
    print(f"    {row['config_key']}: {row['config_value']}")

if not rows:
    print("  (no entity configs found)")

# 3. Query recent opt_runs
print()
print('=' * 80)
print('3. opt_runs - Recent runs')
print('=' * 80)
cursor.execute('''
    SELECT r.run_id, e.entity_code, r.status, r.triggered_by, r.trigger_type,
           r.segments_count, r.bid_median, r.validation_status, r.created_on
    FROM opt_runs r
    JOIN opt_entities e ON r.opt_entity_id = e.opt_entity_id
    ORDER BY r.run_id DESC
    LIMIT 5
''')
rows = cursor.fetchall()
if rows:
    for row in rows:
        print(f"\n  run_id: {row['run_id']} | entity: {row['entity_code']}")
        print(f"    status: {row['status']} | triggered_by: {row['triggered_by']} ({row['trigger_type']})")
        print(f"    segments: {row['segments_count']} | bid_median: {row['bid_median']}")
        print(f"    validation: {row['validation_status']} | created: {row['created_on']}")
else:
    print("  (no runs found)")

# 4. Query opt_run_configs for most recent run
print()
print('=' * 80)
print('4. opt_run_configs - Configs for most recent run')
print('=' * 80)
cursor.execute('SELECT run_id FROM opt_runs ORDER BY run_id DESC LIMIT 1')
latest = cursor.fetchone()
if latest:
    run_id = latest['run_id']
    print(f"\n  [run_id={run_id}]")
    cursor.execute('SELECT config_key, config_value FROM opt_run_configs WHERE run_id = %s ORDER BY config_key', (run_id,))
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"    {row['config_key']}: {row['config_value']}")
    else:
        print('    (no configs found)')
else:
    print("  (no runs found)")

# 5. Check opt_versions
print()
print('=' * 80)
print('5. opt_versions - What optimizer versions exist?')
print('=' * 80)
cursor.execute('SELECT version_id, opt_id, version_code, version_name, status FROM opt_versions')
rows = cursor.fetchall()
if rows:
    for row in rows:
        print(f"  version_id: {row['version_id']} | code: {row['version_code']} | name: {row['version_name']} | status: {row['status']}")
else:
    print("  (no versions found)")

cursor.close()
client.close()
print()
print('=' * 80)
print('Done.')
