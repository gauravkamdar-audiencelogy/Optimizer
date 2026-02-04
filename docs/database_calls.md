  Example Flow: Running Optimizer for drugs_hcp

  Stage 1: UI Creates a New Run

  Actor: UI/Frontend
  Action: User configures and triggers a new optimizer run

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ UI: User clicks "Run Optimizer" for drugs_hcp                           │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  READS:                                                                 │
  │  ┌──────────────────┐                                                   │
  │  │ opt_entities     │ ── Get entity_id, current_version_id for         │
  │  └──────────────────┘    drugs_hcp                                      │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_entity_configs│ ── Get static settings (floor_available=false,  │
  │  └──────────────────┘    npi_enabled=true) to show in UI               │
  │                                                                         │
  │  WRITES:                                                                │
  │  ┌──────────────────┐                                                   │
  │  │ opt_runs         │ ── INSERT new run with status='pending'          │
  │  └──────────────────┘                                                   │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_run_configs  │ ── INSERT user's config values                   │
  │  └──────────────────┘    (target_win_rate, max_bid, etc.)              │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  SQL Operations:
  -- READ: Get entity info
  SELECT entity_id, entity_code, current_version_id, s3_base_path
  FROM opt_entities WHERE entity_code = 'drugs_hcp';
  -- Returns: entity_id=1, current_version_id=5

  -- READ: Get static configs (to display in UI)
  SELECT config_key, config_value
  FROM opt_entity_configs WHERE entity_id = 1;
  -- Returns: floor_available=false, npi_enabled=true, domain_enabled=true

  -- WRITE: Create run record
  INSERT INTO opt_runs (run_id, entity_id, version_id, status, triggered_by,
  trigger_type, created_at)
  VALUES ('20260203_150000', 1, 5, 'pending', 'gaurav@company.com', 'manual', NOW());

  -- WRITE: Store user's config values
  INSERT INTO opt_run_configs (run_id, config_key, config_value, config_section)
  VALUES
    ('20260203_150000', 'target_win_rate', '0.65', 'business'),
    ('20260203_150000', 'max_bid_cpm', '30.00', 'technical'),
    ('20260203_150000', 'exploration_mode', 'true', 'business'),
    ('20260203_150000', 'aggressive_exploration', 'true', 'technical');

  ---
  Stage 2: Optimizer Picks Up and Processes Run

  Actor: Optimizer
  Action: Poll for pending runs, execute optimization, write results

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ OPTIMIZER: Polls for pending runs                                       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  READS:                                                                 │
  │  ┌──────────────────┐                                                   │
  │  │ opt_runs         │ ── Find runs with status='pending'               │
  │  └──────────────────┘                                                   │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_entities     │ ── Get s3_base_path, entity_code                 │
  │  └──────────────────┘                                                   │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_entity_configs│ ── Get SSP static settings                      │
  │  └──────────────────┘    (floor_available, npi_enabled)                │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_run_configs  │ ── Get user's config values for this run         │
  │  └──────────────────┘    (target_win_rate, max_bid, etc.)              │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_feature_macros│ ── Get feature→macro mapping                    │
  │  └──────────────────┘    (for writing to opt_run_features)             │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ OPTIMIZER: Executes (loads data, trains models, generates bids)         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  WRITES (progress):                                                     │
  │  ┌──────────────────┐                                                   │
  │  │ opt_runs         │ ── UPDATE status='running', started_at=NOW()     │
  │  └──────────────────┘                                                   │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ OPTIMIZER: Completes successfully                                       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  WRITES:                                                                │
  │  ┌──────────────────┐                                                   │
  │  │ opt_runs         │ ── UPDATE status='completed', segments_count,    │
  │  └──────────────────┘    bid_median, s3_output_path, completed_at      │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_run_features │ ── INSERT selected features with macro_ids       │
  │  └──────────────────┘    (geo_region_name, os_code, browser_code)      │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_run_metrics  │ ── INSERT model metrics                          │
  │  └──────────────────┘    (wr_model_auc, ctr_model_auc, etc.)           │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  SQL Operations:
  -- READ: Poll for pending runs
  SELECT r.run_id, r.entity_id, r.version_id, e.entity_code, e.s3_base_path
  FROM opt_runs r
  JOIN opt_entities e ON r.entity_id = e.entity_id
  WHERE r.status = 'pending'
  ORDER BY r.created_at LIMIT 1;

  -- READ: Get SSP static configs
  SELECT config_key, config_value FROM opt_entity_configs WHERE entity_id = 1;
  -- Returns: floor_available=false, npi_enabled=true, domain_enabled=true

  -- READ: Get run configs
  SELECT config_key, config_value, config_section FROM opt_run_configs WHERE run_id =
  '20260203_150000';
  -- Returns: target_win_rate=0.65, max_bid_cpm=30.00, etc.

  -- READ: Get macro mapping (for features)
  SELECT macro_id, feature_name, macro_template FROM opt_feature_macros WHERE status =
   'A';

  -- WRITE: Mark as running
  UPDATE opt_runs SET status = 'running', started_at = NOW() WHERE run_id =
  '20260203_150000';

  -- ... OPTIMIZER EXECUTES (loads Snowflake data, trains models, generates bids,
  uploads to S3) ...

  -- WRITE: Mark as completed with results
  UPDATE opt_runs SET
    status = 'completed',
    segments_count = 541,
    bid_median = 30.00,
    bid_min = 19.63,
    bid_max = 30.00,
    s3_output_path =
  's3://tn-optimizer-data/optimizer/drugs_hcp/runs/20260203_150000/',
    validation_status = 'passed',
    completed_at = NOW()
  WHERE run_id = '20260203_150000';

  -- WRITE: Record features used
  INSERT INTO opt_run_features (run_id, feature_name, feature_order, is_anchor,
  signal_score, macro_id) VALUES
    ('20260203_150000', 'geo_region_name', 1, 1, NULL, 2),
    ('20260203_150000', 'os_code', 2, 0, 95.72, 3);

  -- WRITE: Record model metrics
  INSERT INTO opt_run_metrics (run_id, metric_key, metric_value, metric_section)
  VALUES
    ('20260203_150000', 'wr_model_auc', 0.6174, 'win_rate_model'),
    ('20260203_150000', 'wr_model_ece', 0.0226, 'win_rate_model'),
    ('20260203_150000', 'ctr_model_auc', 0.4976, 'ctr_model'),
    ('20260203_150000', 'bid_coefficient', 0.2779, 'bid_landscape'),
    ('20260203_150000', 'global_win_rate', 0.312, 'global_stats');

  ---
  Stage 3: User Configures A/B Testing

  Actor: UI/Frontend
  Action: User selects from COMPLETED runs and configures traffic split
  Note: Runs must already exist as completed before user can configure A/B testing.
        Bidder picks up changes automatically (via its own polling/scheduled refresh).

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ UI: User configures traffic split across completed runs                 │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  READS:                                                                 │
  │  ┌──────────────────┐                                                   │
  │  │ opt_runs         │ ── List completed runs available for selection   │
  │  └──────────────────┘    (status='completed', validation_status='passed')│
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_run_metrics  │ ── Show metrics to help user decide              │
  │  └──────────────────┘    (AUC, ECE, bid ranges, etc.)                  │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_deployments  │ ── Show current active traffic split             │
  │  └──────────────────┘                                                   │
  │                                                                         │
  │  WRITES:                                                                │
  │  ┌──────────────────┐                                                   │
  │  │ opt_deployments  │ ── UPDATE/INSERT traffic split config            │
  │  └──────────────────┘    (e.g., Run A = 80%, Run B = 20%)              │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  SQL Operations:
  -- READ: List completed runs for selection
  SELECT r.run_id, r.created_at, r.segments_count, r.bid_median
  FROM opt_runs r
  WHERE r.entity_id = 1 AND r.status = 'completed' AND r.validation_status = 'passed'
  ORDER BY r.created_at DESC;

  -- READ: Show metrics for each run (to help user decide)
  SELECT metric_key, metric_value FROM opt_run_metrics WHERE run_id IN ('20260203_150000', '20260201_120000');

  -- READ: Show current traffic split
  SELECT run_id, traffic_pct FROM opt_deployments WHERE entity_id = 1 AND is_active = 1;

  -- WRITE: Update traffic split (deactivate old, insert new)
  -- Option A: Replace all active deployments
  UPDATE opt_deployments SET is_active = 0, deactivated_at = NOW()
  WHERE entity_id = 1 AND is_active = 1;

  INSERT INTO opt_deployments (run_id, entity_id, traffic_pct, is_active, deployed_by, deployed_at)
  VALUES
    ('20260203_150000', 1, 80, 1, 'gaurav@company.com', NOW()),
    ('20260201_120000', 1, 20, 1, 'gaurav@company.com', NOW());

  -- Bidder will pick up this change on its next refresh cycle (no manual trigger needed)

  ---
  Stage 4: Bidder Loads Active Deployment

  Actor: Bidder
  Action: Load active run data for real-time bidding

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ BIDDER: Startup / Refresh cycle                                         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  READS:                                                                 │
  │  ┌──────────────────┐                                                   │
  │  │ opt_deployments  │ ── Get active deployment(s) for entity           │
  │  └──────────────────┘    (supports A/B with traffic_pct)               │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_runs         │ ── Get s3_output_path for the run                │
  │  └──────────────────┘                                                   │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_run_features │ ── Get features + macro mapping                  │
  │  └──────────────────┘    (to build memcache keys from bid request)     │
  │                                                                         │
  │  ┌──────────────────┐                                                   │
  │  │ opt_feature_macros│ ── Get macro templates for each feature         │
  │  └──────────────────┘                                                   │
  │                                                                         │
  │  WRITES: None (read-only)                                               │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  SQL Operations:
  -- READ: Get active deployment(s)
  SELECT d.deployment_id, d.run_id, d.traffic_pct,
         r.s3_output_path, e.entity_code
  FROM opt_deployments d
  JOIN opt_runs r ON d.run_id = r.run_id
  JOIN opt_entities e ON d.entity_id = e.entity_id
  WHERE e.entity_code = 'drugs_hcp' AND d.is_active = 1;

  -- READ: Get features and macros for memcache key building
  SELECT f.feature_name, f.feature_order, m.macro_template
  FROM opt_run_features f
  JOIN opt_feature_macros m ON f.macro_id = m.macro_id
  WHERE f.run_id = '20260203_150000'
  ORDER BY f.feature_order;
  -- Returns:
  -- geo_region_name | 1 | [ADX_GEO_STATE]
  -- os_code         | 2 | [ADX_USER_OS]

  -- Bidder then:
  -- 1. Downloads suggested_bids.csv from S3 → loads into memcache
  -- 2. Downloads domain_multipliers.csv → loads into domain cache
  -- 3. On each bid request: extracts [ADX_GEO_STATE]|[ADX_USER_OS] → looks up bid

  ---
  Summary: Table Access by Component
  ┌────────────────────────┬─────┬───────────┬────────┐
  │         Table          │ UI  │ Optimizer │ Bidder │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_master             │  R  │     R     │   -    │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_versions           │  R  │     R     │   -    │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_entities           │  R  │     R     │   R    │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_entity_configs     │  R  │     R     │   -    │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_runs               │ R/W │    R/W    │   R    │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_run_configs        │  W  │     R     │   -    │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_run_metrics        │  R  │     W     │   -    │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_feature_macros     │  -  │     R     │   R    │
  ├────────────────────────┼─────┼───────────┼────────┤
  │ opt_deployments        │ R/W │     -     │   R    │
  └────────────────────────┴─────┴───────────┴────────┘
  R = Read, W = Write, R/W = Both

  **Total: 9 tables**

  ---