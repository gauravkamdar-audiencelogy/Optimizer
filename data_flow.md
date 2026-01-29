┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              OPTIMIZER RUN FLOW                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────┐    ┌─────────────────────────────────────────────────────────────────────┐ │
│  │   UI    │    │                         MYSQL DATABASE                              │ │
│  │ (User)  │    │                                                                     │ │
│  └────┬────┘    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │ │
│       │         │  │opt_entities │     │opt_entity_  │     │opt_version_ │            │ │
│       │ 1. Set  │  │             │     │configs      │     │configs      │            │ │
│       │ configs │  │ entity_id=2 │     │             │     │             │            │ │
│       ├────────▶│  │ drugs_com   │────▶│ overrides   │────▶│ defaults    │            │ │
│       │         │  │ version=v3  │     │             │     │             │            │ │
│       │         │  └─────────────┘     └─────────────┘     └─────────────┘            │ │
│       │         │         │                   │                   │                   │ │
│       │         │         │                   ▼                   ▼                   │ │
│       │         │         │         ┌─────────────────────────────────────┐           │ │
│       │         │         │         │     MERGED CONFIG (JSON)            │           │ │
│       │         │         │         │  {                                  │           │ │
│       │         │         │         │    "business": {                    │           │ │
│       │         │         │         │      "target_margin": 0.30,         │           │ │
│       │         │         │         │      "target_win_rate": 0.50        │           │ │
│       │         │         │         │    },                               │           │ │
│       │         │         │         │    "technical": {...}               │           │ │
│       │         │         │         │  }                                  │           │ │
│       │         │         │         └───────────────┬─────────────────────┘           │ │
│       │         │         │                         │                                 │ │
│       │         │         ▼                         ▼                                 │ │
│  ┌────┴────┐    │  ┌─────────────────────────────────────────────────────────┐        │ │
│  │  CRON   │    │  │                     opt_runs                            │        │ │
│  │  (Job)  │    │  │─────────────────────────────────────────────────────────│        │ │
│  └────┬────┘    │  │ run_id=47                                               │        │ │
│       │         │  │ entity_id=2                                             │        │ │
│       │ 2. Trigger │ version_id=3                                            │        │ │
│       │ run     │  │ run_code="20260128_090000"                              │        │ │
│       ├────────▶│  │ status="queued"  ──▶  "running"  ──▶  "success"         │        │ │
│       │         │  │ triggered_by="cron"                                     │        │ │
│       │         │  │ config_snapshot={...FULL JSON...}                       │        │ │
│       │         │  │ data_start_date="2025-09-15"                            │        │ │
│       │         │  │ data_end_date="2026-01-28"                              │        │ │
│       │         │  └─────────────────────────────────────────────────────────┘        │ │
│       │         │                                                                     │ │
│       │         └─────────────────────────────────────────────────────────────────────┘ │
│       │                                                                                 │
│       ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                           OPTIMIZER COMPONENT                                       ││
│  │                                                                                     ││
│  │  3. Read config from opt_runs WHERE run_id=47                                       ││
│  │     ┌──────────────────────────────────────┐                                        ││
│  │     │ config = json.loads(config_snapshot) │                                        ││
│  │     └──────────────────────────────────────┘                                        ││
│  │                         │                                                           ││
│  │  4. Fetch data from Snowflake (data_start_date → data_end_date)                     ││
│  │                         │                                                           ││
│  │  5. Run optimizer pipeline                                                          ││
│  │     - Feature selection → [internal_adspace_id, geo_region_name, os_code]           ││
│  │     - Model training                                                                ││
│  │     - Bid calculation                                                               ││
│  │                         │                                                           ││
│  │  6. Generate outputs                                                                ││
│  │     - memcache_20260128_090000.csv                                                  ││
│  │     - metrics_20260128_090000.json                                                  ││
│  │                         │                                                           ││
│  └─────────────────────────┼───────────────────────────────────────────────────────────┘│
│                            │                                                            │
│                            ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                              S3 UPLOAD                                              ││
│  │                                                                                     ││
│  │  s3://tn-optimizer-data/drugs_com/v3/20260128_090000/                               ││
│  │    ├── memcache.csv                                                                 ││
│  │    └── metrics.json                                                                 ││
│  │                                                                                     ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                            │                                                            │
│                            ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                         UPDATE MYSQL (Post-run)                                     ││
│  │                                                                                     ││
│  │  UPDATE opt_runs SET status='success', completed_at=NOW(),                          ││
│  │         total_segments=1454, segments_in_memcache=93                                ││
│  │  WHERE run_id=47;                                                                   ││
│  │                                                                                     ││
│  │  INSERT INTO opt_run_features (run_id, feature_name, feature_order, signal_score)   ││
│  │  VALUES (47, 'internal_adspace_id', 1, 999.0),                                      ││
│  │         (47, 'geo_region_name', 2, 184.0),                                          ││
│  │         (47, 'os_code', 3, 78.2);                                                   ││
│  │                                                                                     ││
│  │  INSERT INTO opt_run_outputs (run_id, output_type, s3_bucket, s3_key, row_count)    ││
│  │  VALUES (47, 'memcache', 'tn-optimizer-data',                                       ││
│  │          'drugs_com/v3/20260128_090000/memcache.csv', 93);                          ││
│  │                                                                                     ││
│  │  INSERT INTO opt_run_feature_macros (run_id, macro_combination)                     ││
│  │  VALUES (47, '[ADX_PLACEMENT_ID]|[ADX_GEO_STATE]|[ADX_USER_OS]');                   ││
│  │                                                                                     ││
│  │  UPDATE opt_entities SET active_run_id=47 WHERE entity_id=2;                        ││
│  │                                                                                     ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

