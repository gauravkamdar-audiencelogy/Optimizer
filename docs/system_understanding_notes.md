# System Understanding Notes

**Document Purpose**: Persistent notes capturing understanding of the RTB Optimizer system architecture, database schema, and orchestration flow. Updated as understanding deepens.

**Last Updated**: 2026-02-03

---

## 1. System Actors

| Actor | Role | DB Operations |
|-------|------|---------------|
| **UI/Frontend** | Users configure optimizers, trigger runs, view status | WRITE: create pending runs, config values. READ: run status, metrics |
| **Optimizer** | Executes optimization logic, produces output files | READ: pending runs, config. WRITE: status updates, metrics, features, S3 paths |
| **Bidder** | Real-time bidding using optimizer outputs | READ ONLY: active runs, features, S3 paths |
| **Scheduler** | Triggers periodic runs | WRITE: create pending runs |

---

## 2. Hierarchy of Concepts

```
OPTIMIZER TYPE (fundamentally different approaches)
│   Examples: Formula-based, ML-based, ML+RL, DL-based
│   Meta: when built, actively maintained, description
│
└── OPTIMIZER VERSION (capability iterations within a type)
    │   Examples: ML_v1 (WR only), ML_v2 (WR+CTR), ML_v3 (+NPI), ML_v4 (+domain)
    │   Meta: release date, status, what configs it supports
    │
    └── VERSION CONFIG SCHEMA (what config keys this version allows)
            Examples: target_win_rate, max_bid, exploration_mode
            Meta: data type, default, min/max, UI visibility

SSP CLIENT (business customer)
│   Examples: drugs_hcp (obj_id: 1334), nativo_consumer (obj_id: 723), media.net
│   Has: static SSP-specific configs (floor_available, npi_enabled)
│
└── RUN (actual execution instance)
    │   Has: config values, status, metrics, features, outputs
    │
    └── DEPLOYMENT (which run is active for bidder)
            Supports: A/B testing (multiple active runs with traffic split)
```

---

## 3. Config Inheritance Model

| Level | What it defines | Example |
|-------|-----------------|---------|
| **Version** | What config keys EXIST | "This version supports NPI targeting" |
| **SSP** | Which keys are ENABLED + static defaults | "drugs_hcp has floor_available=false, npi_enabled=true" |
| **Run** | What VALUES were used | "target_win_rate=65%, max_bid=30" |

**Key Insight**: Versions are SHARED - any SSP can use any version. The SSP just picks which version to use and provides its own config values.

---

## 4. The Complete Flow

### 4.1 Run Initiation (UI or Scheduler)

```
1. User opens UI for SSP "drugs_hcp"
2. UI fetches: available versions, version's config schema, SSP defaults
3. UI renders config form with defaults pre-filled
4. User adjusts values (target_win_rate=65%, max_bid=30)
5. User clicks "Run Optimizer"
6. UI creates DB record:
   - ssp_client_id = drugs_hcp
   - version_id = ML_v4
   - config_snapshot = {target_win_rate: 0.65, max_bid: 30, ...}
   - status = 'pending'
   - triggered_by = 'user@company.com'
```

### 4.2 Optimizer Execution

```
1. Optimizer polls for pending runs (or receives trigger)
2. Reads run config from DB:
   - config_snapshot JSON
   - SSP's static settings (floor_available, etc.)
3. Executes optimization:
   - Loads data from Snowflake/S3
   - Selects features
   - Trains models
   - Generates bids
4. Writes back to DB:
   - status = 'running' → 'completed'
   - features_used = ['geo_region_name', 'os_code']
   - metrics_snapshot = {auc: 0.61, ece: 0.02, ...}
   - s3_path = 's3://bucket/optimizer/drugs_hcp/runs/20260203_121359/'
5. Creates output file records:
   - suggested_bids.csv → run_outputs table
   - domain_multipliers.csv → run_outputs table
```

### 4.3 A/B Testing Configuration

```
1. User opens A/B testing UI for SSP "drugs_hcp"
2. UI shows list of COMPLETED runs (status='completed', validation='passed')
3. User selects runs and assigns traffic split:
   - run_id = 20260203_121359 → 80%
   - run_id = 20260201_120000 → 20%
4. UI writes to opt_deployments table
5. Bidder picks up changes on next refresh (no manual trigger)
```

**Key Point**: Runs must exist as completed BEFORE user can configure A/B testing.
The bidder has its own logic (polling/scheduled jobs) to pick up deployment changes.

### 4.4 Bidder Consumption

```
1. Bidder queries active deployments for SSP
2. Gets: run_id, features_used, s3_path
3. Loads memcache from S3 files:
   - suggested_bids.csv → segment→bid lookup
   - domain_multipliers.csv → domain→multiplier lookup
4. On bid request:
   - Extracts feature values from request
   - Builds memcache key
   - Looks up bid
   - Applies multipliers
```

---

## 5. SSP Client Details

| SSP | object_id | Has Floor | Has NPI | Notes |
|-----|-----------|-----------|---------|-------|
| drugs_hcp | 1334 | No | Yes | Healthcare provider targeting |
| nativo_consumer | 723 | Yes | No | Consumer targeting |
| media.net | 1335 | TBD | TBD | Future integration |

---

## 6. Current Optimizer Versions (as of 2026-02)

| Version | Capabilities | Status |
|---------|--------------|--------|
| v1 | Win rate model only | Deprecated |
| v2 | Win rate + CTR model | Deprecated |
| v3 | + Bid landscape model | Deprecated |
| v4 | + NPI targeting | Deprecated |
| v5 | + Domain targeting, IQR tiering | **Active** |
| v6 (planned) | + Floor price optimization | Development |

---

## 7. Output Files Per Run

| File | Purpose | Consumer |
|------|---------|----------|
| `suggested_bids_{run_id}.csv` | segment → bid lookup | Bidder (memcache) |
| `domain_multipliers_{run_id}.csv` | domain → multiplier | Bidder (domain cache) |
| `domain_blocklist_{run_id}.csv` | domains to exclude | Bidder |
| `npi_multipliers_{run_id}.csv` | NPI → multiplier | Bidder (drugs_hcp only) |
| `metrics_{run_id}.json` | Model diagnostics | UI, Analytics |
| `diagnostics_{run_id}.md` | Human-readable report | UI, Operators |
| `validation_report_{run_id}.json` | Pass/fail checks | Deployment gate |

---

## 8. Run Status Lifecycle

```
pending → running → completed → validated → deployed
                 ↘ failed
```

| Status | Meaning |
|--------|---------|
| pending | Created by UI/scheduler, waiting for optimizer |
| running | Optimizer has picked up, executing |
| completed | Optimizer finished successfully |
| failed | Optimizer encountered error |
| validated | Passed hard validation rules |
| deployed | Active for bidder traffic |

---

## 9. A/B Testing Support

- Multiple runs can be active simultaneously for one SSP
- Each deployment has `traffic_pct` (e.g., 80% vs 20%)
- Bidder routes requests based on traffic split
- Allows comparing: different versions, different configs, different data windows

---

## 10. Open Questions / Future Considerations

1. **Feature-to-macro mapping**: Currently backend handles it. May need DB table if bidder needs dynamic mapping.

2. **Data source tracking**: Should runs track which Snowflake query / S3 path was used for input data?

3. **Rollback mechanism**: How to quickly revert to previous run if new deployment causes issues?

4. **Config validation**: Should DB enforce config value constraints, or leave to optimizer?

5. **Run dependencies**: Can a run depend on another run's outputs? (e.g., domain model trained separately)

---

## 11. Database Documentation

**Schema**: `docs/database_schema_v1_adapted.md` - 12 tables, all flat MySQL
**Flow**: `docs/database_calls.md` - Which tables are read/written at each stage

## 12. Database Schema Summary (v1 Simplified)

```
opt_master (2-5 rows) ← Different optimizer types
    └── opt_versions (5-10 rows) ← Versions + supported_configs
            └── opt_entities (3-10 rows) ← SSP clients
                    ├── opt_entity_configs (10-30 rows) ← SSP static settings
                    └── opt_runs (1000s over time) ← Includes features_used
                            ├── opt_run_configs (5-10 per run) ← Config values
                            └── opt_run_metrics (10-20 per run) ← Model metrics

opt_feature_macros (20-50 rows) ← Managed by bidder team
opt_deployments (1-3 per entity for A/B)
```

**Total Tables: 9** (all flat MySQL, no JSON)

**Key Design Decisions:**
- No JSON columns - everything in flat relational tables
- supported_configs on opt_versions, features_used on opt_runs (comma-separated)
- S3 path stored on opt_runs, not individual file tracking
- ETL metrics deferred for now

---

*End of Notes*
