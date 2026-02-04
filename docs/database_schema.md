Tables and Columns 

  1. opt_master

  opt_id, opt_code, opt_name, description, status, created_at

  2. opt_versions

  version_id, opt_id, version_code, version_name, supported_configs, description,
  status, created_at

  3. opt_entities

  entity_id, entity_code, entity_name, object_id, object_type, current_version_id,
  s3_base_path, status, created_at

  4. opt_entity_configs

  id, entity_id, config_key, config_value

  5. opt_runs

  run_id, entity_id, version_id, status, triggered_by, trigger_type, data_start_dt,
  data_end_dt, total_bids, total_views, total_clicks, segments_count, domains_count,
  npis_count, features_used, bid_median, bid_min, bid_max, s3_output_path,
  validation_status, error_message, created_at, started_at, completed_at

  6. opt_run_configs

  id, run_id, config_key, config_value

  7. opt_run_metrics

  id, run_id, metric_key, metric_value

  8. opt_feature_macros

  macro_id, feature_name, macro_template, status

  9. opt_deployments

  deployment_id, run_id, entity_id, traffic_pct, is_active, deployed_by, deployed_at,
  deactivated_at



Table Descriptions
  Table: opt_master
  What it tracks: Different optimizer types (RTB, formula-based, etc.)
  ────────────────────────────────────────
  Table: opt_versions
  What it tracks: Versions of each optimizer and what configs they support
  ────────────────────────────────────────
  Table: opt_entities
  What it tracks: SSP clients we optimize for (drugs_hcp, nativo_consumer)
  ────────────────────────────────────────
  Table: opt_entity_configs
  What it tracks: Static settings per SSP that don't change between runs
    (floor_available, npi_enabled)
  ────────────────────────────────────────
  Table: opt_runs
  What it tracks: Every optimizer execution - what ran, on what data, what came out,
    when
  ────────────────────────────────────────
  Table: opt_run_configs
  What it tracks: Config values user set for each run (target_win_rate, max_bid)
  ────────────────────────────────────────
  Table: opt_run_metrics
  What it tracks: Model performance metrics after run completes (AUC, ECE, etc.)
  ────────────────────────────────────────
  Table: opt_feature_macros
  What it tracks: Maps feature names to bidder macros (geo_region_name →
    [ADX_GEO_STATE])
  ────────────────────────────────────────
  Table: opt_deployments
  What it tracks: Which completed runs get live traffic and at what % (A/B testing)
  ---
  
Schema Diagram

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
  │ supported_configs│                              │
  └────────┬────────┘                               │
           │ 1:N                                    │
           ▼                                        │
  ┌─────────────────┐                               │
  │  opt_entities   │  "Who we optimize for"        │
  │─────────────────│                               │
  │ entity_id (PK)  │                               │
  │ entity_code     │                               │
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
  │ entity_id (FK)  │  │ run_id (PK)     │
  │ config_key      │  │ entity_id (FK)  │
  │ config_value    │  │ version_id (FK) │
  └─────────────────┘  │ total_bids/views/clicks │
                       │ segments/domains/npis_count │
                       │ features_used   │
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
        │ config_key   │ │ metric_key   │ │ entity_id (FK)│
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

Flow:
  opt_master → opt_versions → opt_entities → opt_runs → opt_run_configs
                                        ↓              → opt_run_metrics
                                 opt_entity_configs    → opt_deployments

  opt_feature_macros (standalone, managed by bidder team)