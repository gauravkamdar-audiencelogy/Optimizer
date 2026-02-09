# Optimizer Database Schema

## Tables and Columns

### 1. opt_master
```
opt_id, opt_code, opt_name, description, status, created_at
```

### 2. opt_versions
```
version_id, opt_id, version_code, version_name, supported_configs, description,
status, created_at
```

### 3. opt_entities
```
opt_entity_id, entity_code, entity_name, targeting_type, object_id, object_type,
current_version_id, s3_base_path, snowflake_table, status, created_at
```

**New column: `targeting_type`** (VARCHAR 20)
- Values: `'hcp'` or `'consumer'`
- Determines what type of traffic this entity optimizes
- If an SSP (e.g., Nativo) has both HCP and consumer, create separate entity rows

### 4. opt_entity_configs
```
id, opt_entity_id, config_key, config_value, status
```

**Standard config keys:**
| config_key | Values | Description |
|------------|--------|-------------|
| `npi_enabled` | true/false | Can this entity use NPI targeting? |
| `floor_available` | true/false | Does this entity provide floor prices? |
| `domain_enabled` | true/false | Are domain multipliers enabled? |

### 5. opt_runs
```
run_id, opt_entity_id, version_id, run_code, status, triggered_by, trigger_type,
data_start_date, data_end_date, total_bids, total_views, total_clicks,
segments_count, domains_count, npis_count, features_used, bid_median, bid_min,
bid_max, s3_output_path, validation_status, error_message, config_snapshot,
total_segments, segments_in_memcache, global_win_rate, global_ctr,
created_on, started_at, completed_at
```

**Status values:** `queued` → `running` → `completed` / `failed`

### 6. opt_run_configs
```
id, run_id, config_key, config_value, created_by
```

**Standard config keys:**
| config_key | Type | Description |
|------------|------|-------------|
| `target_win_rate` | float | Target win rate (e.g., 0.60) |
| `max_bid_cpm` | float | Bid ceiling in CPM |
| `min_bid_cpm` | float | Bid floor in CPM |
| `training_start_date` | date | Data window start |
| `training_end_date` | date | Data window end |
| `exploration_mode` | bool | Enable exploration bonuses |
| `user_disabled_features` | list | Features to exclude |

### 7. opt_run_metrics
```
id, run_id, metric_key, metric_value
```

### 8. opt_feature_macros
```
macro_id, feature_name, macro_template, status
```

### 9. opt_deployments
```
deployment_id, run_id, opt_entity_id, traffic_pct, is_active, deployed_by,
deployed_at, deactivated_at
```

---

## Table Descriptions

| Table | Purpose |
|-------|---------|
| **opt_master** | Different optimizer types (RTB, formula-based, etc.) |
| **opt_versions** | Versions of each optimizer and what configs they support |
| **opt_entities** | SSP + targeting type combinations we optimize for |
| **opt_entity_configs** | Static capability flags per entity (floor_available, npi_enabled) |
| **opt_runs** | Every optimizer execution - inputs, outputs, metrics |
| **opt_run_configs** | User-set config values for each run |
| **opt_run_metrics** | Model performance metrics (AUC, ECE, calibration) |
| **opt_feature_macros** | Maps feature names to bidder macros |
| **opt_deployments** | Which runs get live traffic and at what % |

---

## Schema Diagram

```
┌─────────────────┐
│   opt_master    │  "What optimizer types exist"
│─────────────────│
│ opt_id (PK)     │
│ opt_code        │
│ opt_name        │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐
│  opt_versions   │  "What versions exist, what configs they support"
│─────────────────│
│ version_id (PK) │
│ opt_id (FK)     │───────────────────────────────┐
│ version_code    │                               │
│ supported_configs                               │
└────────┬────────┘                               │
         │ 1:N                                    │
         ▼                                        │
┌─────────────────┐                               │
│  opt_entities   │  "SSP + targeting combos"     │
│─────────────────│                               │
│ opt_entity_id   │                               │
│ entity_code     │                               │
│ targeting_type  │  ← NEW: 'hcp' or 'consumer'   │
│ current_version_id (FK)─────────────────────────┘
│ s3_base_path    │
└────────┬────────┘
         │
         ├──────────────────┐
         │ 1:N              │ 1:N
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│opt_entity_configs│  │    opt_runs     │  "Every execution"
│─────────────────│  │─────────────────│
│ opt_entity_id   │  │ run_id (PK)     │
│ config_key      │  │ opt_entity_id   │
│ config_value    │  │ version_id (FK) │
└─────────────────┘  │ targeting_type  │
                     │ total_bids/views│
                     │ segments_count  │
                     │ s3_output_path  │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │ 1:N           │ 1:N           │ 1:N
              ▼               ▼               ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │opt_run_configs│ │opt_run_metrics│ │opt_deployments│
      │──────────────│ │──────────────│ │──────────────│
      │ run_id (FK)  │ │ run_id (FK)  │ │ run_id (FK)  │
      │ config_key   │ │ metric_key   │ │ opt_entity_id │
      │ config_value │ │ metric_value │ │ traffic_pct  │
      └──────────────┘ └──────────────┘ │ is_active    │
                                        └──────────────┘

┌─────────────────┐
│opt_feature_macros│  "Feature → bidder macro mapping (standalone)"
│─────────────────│
│ macro_id (PK)   │
│ feature_name    │
│ macro_template  │
└─────────────────┘
```

---

## Data Flow

### UI-Triggered Run
```
1. User on UI: Nativo page → Optimization tab → HCP menu
2. User sets configs (target_win_rate, max_bid, etc.)
3. UI creates opt_runs record (status='queued')
4. UI inserts into opt_run_configs
5. UI triggers: python run_optimizer.py --entity nativo --run-id 123
6. Optimizer reads from opt_run_configs
7. Optimizer updates opt_runs (status, metrics, s3_path)
8. Optimizer uploads files to S3
```

### Cron Job Run
```
1. GitHub cron triggers
2. Query opt_entities for all active entities
3. For each entity:
   a. Get latest run's configs from opt_run_configs
   b. Create new opt_runs record
   c. Trigger optimizer
4. Optimizer updates opt_runs with results
```

---

## Current Data (QA)

### opt_entities
| opt_entity_id | entity_code | targeting_type | s3_base_path |
|---------------|-------------|----------------|--------------|
| 12 | nativo | consumer | s3://tn-optimizer-data/nativo |
| 13 | drugs.com | hcp | s3://tn-optimizer-data/drugs_com |
| 14 | media.net | consumer | s3://tn-optimizer-data/medianet |

### opt_entity_configs
| entity | config_key | config_value |
|--------|------------|--------------|
| drugs.com | npi_enabled | true |
| drugs.com | floor_available | false |
| drugs.com | domain_enabled | true |
| nativo | npi_enabled | false |
| nativo | floor_available | true |
| nativo | domain_enabled | true |
