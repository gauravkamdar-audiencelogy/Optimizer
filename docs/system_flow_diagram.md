# System Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                  EXTERNAL SYSTEMS                                    │
├─────────────────────┬─────────────────────┬─────────────────────────────────────────┤
│                     │                     │                                         │
│  ┌───────────────┐  │  ┌───────────────┐  │  ┌───────────────┐                      │
│  │   SNOWFLAKE   │  │  │      S3       │  │  │     MYSQL     │                      │
│  │  (Data Source)│  │  │   (Storage)   │  │  │ (Audit Log)   │                      │
│  └───────┬───────┘  │  └───────┬───────┘  │  └───────┬───────┘                      │
│          │          │          │          │          │                              │
│          │ Export   │          │ Upload   │          │ Log                          │
│          │ Data     │          │ Results  │          │ Runs                         │
│          ▼          │          ▲          │          ▲                              │
└──────────┼──────────┴──────────┼──────────┴──────────┼──────────────────────────────┘
           │                     │                     │
           ▼                     │                     │
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│                              RTB OPTIMIZER                                          │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           DATA INGESTION                                     │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │   │
│  │  │  Snowflake  │───▶│   Ingest    │───▶│  Combined   │                      │   │
│  │  │   Export    │    │   Script    │    │  Data File  │                      │   │
│  │  │  (CSV)      │    │             │    │             │                      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘                      │   │
│  │                                               │                              │   │
│  │        data/incoming/                         │   data/{dataset}/            │   │
│  │                                               ▼                              │   │
│  └───────────────────────────────────────────────┼──────────────────────────────┘   │
│                                                  │                                  │
│  ┌───────────────────────────────────────────────┼──────────────────────────────┐   │
│  │                           OPTIMIZER PIPELINE  │                              │   │
│  │                                               ▼                              │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ [1] DATA LOADER                                                      │    │   │
│  │  │     • Load bids, views, clicks from combined CSV                     │    │   │
│  │  │     • Parse PostgreSQL arrays ({7.50} or {"0.45"})                   │    │   │
│  │  │     • Parse floor prices (nativo only)                               │    │   │
│  │  │     • Handle rec_type: bid/view/click/link                           │    │   │
│  │  └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                         │   │
│  │                                    ▼                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ [2] FEATURE ENGINEERING                                              │    │   │
│  │  │     • Join bids → views via internal_txn_id (win/loss labels)        │    │   │
│  │  │     • Join views → clicks (CTR labels)                               │    │   │
│  │  │     • Create time features (hour, day_of_week)                       │    │   │
│  │  │     • Preserve domain column (nativo) or extract from URL (drugs)    │    │   │
│  │  └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                         │   │
│  │                                    ▼                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ [3] FEATURE SELECTION                                                │    │   │
│  │  │     • Score features by predictive power                             │    │   │
│  │  │     • Auto-exclude low-variance features (>90% single value)         │    │   │
│  │  │     • Auto-exclude low-signal features (score < threshold)           │    │   │
│  │  │     • Select top N features for segmentation                         │    │   │
│  │  └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                         │   │
│  │                                    ▼                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ [4] MODEL TRAINING                                                   │    │   │
│  │  │                                                                      │    │   │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │    │   │
│  │  │  │ Win Rate Model  │  │   CTR Model     │  │ Bid Landscape   │      │    │   │
│  │  │  │ (Empirical +    │  │ (Empirical +    │  │ Model           │      │    │   │
│  │  │  │  LogReg)        │  │  LogReg)        │  │ (P(win|bid))    │      │    │   │
│  │  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘      │    │   │
│  │  │           │                    │                    │               │    │   │
│  │  │           └────────────────────┼────────────────────┘               │    │   │
│  │  │                                │                                    │    │   │
│  │  │                    Calibration Check (ECE)                          │    │   │
│  │  │                                │                                    │    │   │
│  │  └────────────────────────────────┼────────────────────────────────────┘    │   │
│  │                                   │                                          │   │
│  │                                   ▼                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ [5] VALUE MODELS (Multipliers)                                       │    │   │
│  │  │                                                                      │    │   │
│  │  │  ┌─────────────────────────┐    ┌─────────────────────────┐         │    │   │
│  │  │  │ NPI Model (drugs only)  │    │ Domain Model (nativo)   │         │    │   │
│  │  │  │ • Load 1yr + 20day data │    │ • IQR tiering on WR     │         │    │   │
│  │  │  │ • IQR tiering on clicks │    │ • 5 tiers + blocklist   │         │    │   │
│  │  │  │ • Recency boost         │    │ • Bayesian shrinkage    │         │    │   │
│  │  │  │ • 5 tiers (0.7x - 3.0x) │    │ • 5 tiers (0.6x - 1.5x) │         │    │   │
│  │  │  └─────────────────────────┘    └─────────────────────────┘         │    │   │
│  │  │                                                                      │    │   │
│  │  └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                   │                                          │   │
│  │                                   ▼                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ [6] BID CALCULATOR                                                   │    │   │
│  │  │     • Strategy: adaptive (volume_first ↔ margin_optimize)            │    │   │
│  │  │     • Exploration: bid UP on losing, DOWN on winning                 │    │   │
│  │  │     • Multipliers derived from bid landscape                         │    │   │
│  │  │     • Per-segment bid calculation                                    │    │   │
│  │  └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                   │                                          │   │
│  │                                   ▼                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ [7] OUTPUT GENERATION                                                │    │   │
│  │  │     • suggested_bids_*.csv (segment → bid)                           │    │   │
│  │  │     • npi_multipliers_*.csv (NPI → multiplier)                       │    │   │
│  │  │     • domain_multipliers_*.csv (domain → multiplier)                 │    │   │
│  │  │     • domain_blocklist_*.csv (domains to exclude)                    │    │   │
│  │  │     • metrics_*.json (model diagnostics)                             │    │   │
│  │  └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                   │                                          │   │
│  │                                   ▼                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ [8] VALIDATION                                                       │    │   │
│  │  │     • Hard rules: bid bounds, calibration ECE, model success         │    │   │
│  │  │     • Soft rules: floor/ceiling concentration, profitability         │    │   │
│  │  │     • PASS/FAIL gate for deployment                                  │    │   │
│  │  └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                   │                                          │   │
│  └───────────────────────────────────┼──────────────────────────────────────┘   │
│                                      │                                          │
│  ┌───────────────────────────────────┼──────────────────────────────────────┐   │
│  │                           INTEGRATIONS                                    │   │
│  │                                   │                                       │   │
│  │         ┌─────────────────────────┼─────────────────────────┐            │   │
│  │         │                         │                         │            │   │
│  │         ▼                         ▼                         ▼            │   │
│  │  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐    │   │
│  │  │  S3 Upload  │           │  MySQL Log  │           │  Manifest   │    │   │
│  │  │             │           │             │           │  Update     │    │   │
│  │  │ runs/{id}/  │           │ run_history │           │ active/     │    │   │
│  │  │             │           │ table       │           │ manifest    │    │   │
│  │  └─────────────┘           └─────────────┘           └─────────────┘    │   │
│  │                                                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    BIDDER                                            │
│                                                                                      │
│   Loads from S3:                                                                     │
│   • suggested_bids_*.csv  →  segment_bid lookup                                      │
│   • npi_multipliers_*.csv →  NPI multiplier lookup                                   │
│   • domain_multipliers_*.csv → domain multiplier lookup                              │
│                                                                                      │
│   At request time:                                                                   │
│   final_bid = segment_bid × npi_multiplier × domain_multiplier                       │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Detail

### 1. Data Acquisition

```
┌─────────────────────────────────────────────────────────────────┐
│                         SNOWFLAKE                               │
│                                                                 │
│  RTB_LOG table contains:                                        │
│  • rec_type: bid, view, click/link, lead                        │
│  • publisher_payout: our bid                                    │
│  • bid_amount: floor price (nativo) or campaign bid             │
│  • internal_txn_id: links bid→view→click                        │
│                                                                 │
│  Export query filters by:                                       │
│  • Date range (min_bid_date, min_view_date)                     │
│  • Traffic source / publisher                                   │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ CSV Export
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL FILE SYSTEM                           │
│                                                                 │
│  data/                                                          │
│  ├── NPI_click_data_1year.csv    ← Shared NPI data              │
│  ├── NPI_click_data_20days.csv   ← Recent clickers              │
│  ├── drugs/                                                     │
│  │   └── data_drugs.csv          ← Combined bid/view/click      │
│  └── nativo_consumer/                                           │
│      └── data_nativo_consumer.csv                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Join Logic (Win/Loss Determination)

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│    BIDS     │         │    VIEWS    │         │   CLICKS    │
│             │         │             │         │             │
│ internal_   │         │ internal_   │         │ internal_   │
│ txn_id      │────────▶│ txn_id      │────────▶│ txn_id      │
│             │  LEFT   │             │  LEFT   │             │
│ publisher_  │  JOIN   │ clearing_   │  JOIN   │ (link/click │
│ payout      │         │ price       │         │  rec_type)  │
└─────────────┘         └─────────────┘         └─────────────┘
      │                       │                       │
      │                       │                       │
      ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING DATA                            │
│                                                             │
│  won = 1 if bid.internal_txn_id IN views.internal_txn_id    │
│  clicked = 1 if view.internal_txn_id IN clicks.internal_txn │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Three-Level Bid Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BID REQUEST ARRIVES                        │
│                                                                 │
│  Contains:                                                      │
│  • Segment features (geo, os, browser, etc.)                    │
│  • NPI (external_userid) - drugs.com only                       │
│  • Domain (request domain)                                      │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LEVEL 1: SEGMENT BID                        │
│                                                                 │
│  Lookup: segment_key → base_bid                                 │
│  Example: (adspace=123, geo=CA, os=iOS) → $7.50                 │
│                                                                 │
│  Source: suggested_bids_*.csv (from optimizer)                  │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LEVEL 2: DOMAIN MULTIPLIER                   │
│                                                                 │
│  Lookup: domain → multiplier                                    │
│  Example: "premium-site.com" → 1.30x                            │
│                                                                 │
│  Source: domain_multipliers_*.csv (from optimizer)              │
│  Tiers: extreme_stars(1.5x), stars(1.3x), baseline(1.0x),       │
│         poor(0.6x), blocklist(0.0x)                             │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LEVEL 3: NPI MULTIPLIER                     │
│                                                                 │
│  Lookup: NPI → multiplier (drugs.com only)                      │
│  Example: "1234567890" → 2.50x                                  │
│                                                                 │
│  Source: npi_multipliers_*.csv (from optimizer)                 │
│  Tiers: extreme_elite(3.0x), elite(2.5x), cream(1.8x),          │
│         baseline(1.0x), poor(0.7x)                              │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FINAL BID                                 │
│                                                                 │
│  final_bid = base_bid × domain_multiplier × npi_multiplier      │
│                                                                 │
│  Example: $7.50 × 1.30 × 2.50 = $24.38                          │
│  (capped at max_bid from config)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## S3 Storage Structure

```
s3://tn-optimizer-data/
└── optimizer/
    ├── drugs/
    │   ├── runs/
    │   │   ├── 20260129_190416/
    │   │   │   ├── suggested_bids_20260129_190416.csv
    │   │   │   ├── npi_multipliers_20260129_190416.csv
    │   │   │   ├── selected_features_20260129_190416.csv
    │   │   │   ├── segment_analysis_20260129_190416.csv
    │   │   │   ├── metrics_20260129_190416.json
    │   │   │   └── validation_report_20260129_190416.json
    │   │   └── 20260128_.../
    │   └── active/
    │       └── manifest.json  ← Points bidder to latest run
    │
    └── nativo_consumer/
        ├── runs/
        │   └── 20260129_192434/
        │       ├── suggested_bids_*.csv
        │       ├── domain_multipliers_*.csv
        │       ├── domain_blocklist_*.csv
        │       └── ...
        └── active/
            └── manifest.json
```

### Manifest Format

```json
{
  "run_id": "20260129_192434",
  "timestamp": "2026-01-29T19:24:34",
  "files": {
    "suggested_bids": "runs/20260129_192434/suggested_bids_20260129_192434.csv",
    "domain_multipliers": "runs/20260129_192434/domain_multipliers_20260129_192434.csv",
    "npi_multipliers": null
  },
  "validation": {
    "passed": true,
    "recommendation": "review"
  }
}
```

---

## Environment Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                     OPTIMIZER_ENV                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "local"      → Skip all integrations (dev/testing)             │
│  "qa"         → Full integrations with QA credentials           │
│  "production" → Full integrations with prod credentials         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     .env File                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  # Environment                                                  │
│  OPTIMIZER_ENV=local                                            │
│                                                                 │
│  # Snowflake                                                    │
│  SNOWFLAKE_ACCOUNT=xxx                                          │
│  SNOWFLAKE_USER=xxx                                             │
│  SNOWFLAKE_PASSWORD=xxx  (or SNOWFLAKE_PRIVATE_KEY_PATH)        │
│  SNOWFLAKE_WAREHOUSE=xxx                                        │
│  SNOWFLAKE_DATABASE=xxx                                         │
│                                                                 │
│  # S3                                                           │
│  AWS_ACCESS_KEY_ID=xxx                                          │
│  AWS_SECRET_ACCESS_KEY=xxx                                      │
│  S3_BUCKET=tn-optimizer-data                                    │
│                                                                 │
│  # MySQL                                                        │
│  MYSQL_HOST=xxx                                                 │
│  MYSQL_USER=xxx                                                 │
│  MYSQL_PASSWORD=xxx                                             │
│  MYSQL_DATABASE=optimizer_audit                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## User Input → Config → Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT SOURCES                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐      │
│  │   YAML Config File  │    │   Command Line Args │    │   Environment Vars  │      │
│  │                     │    │                     │    │                     │      │
│  │ optimizer_config_   │    │ --config            │    │ .env file           │      │
│  │ drugs.yaml          │    │ --data-dir          │    │ OPTIMIZER_ENV       │      │
│  │                     │    │ --ingest            │    │ AWS_*, MYSQL_*      │      │
│  └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘      │
│             │                          │                          │                  │
│             └──────────────────────────┼──────────────────────────┘                  │
│                                        │                                             │
│                                        ▼                                             │
└────────────────────────────────────────┼─────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CONFIG LOADING                                          │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         src/config.py                                       │    │
│  │                                                                             │    │
│  │  @dataclass OptimizerConfig:                                                │    │
│  │  ├── dataset: DatasetConfig      (name, paths)                              │    │
│  │  ├── business: BusinessConfig    (target_win_rate)                          │    │
│  │  ├── technical: TechnicalConfig  (min_bid, max_bid, floor_available)        │    │
│  │  ├── bidding: BiddingConfig      (strategy: adaptive/volume_first)          │    │
│  │  ├── npi: NPIConfig              (enabled, tiering_method, multipliers)     │    │
│  │  ├── domain: DomainConfig        (enabled, tiering_method, multipliers)     │    │
│  │  ├── features: FeaturesConfig    (anchor, candidates, exclude)              │    │
│  │  ├── advanced: AdvancedConfig    (thresholds, shrinkage_k)                  │    │
│  │  └── validation: ValidationConfig (hard_rules, soft_rules)                  │    │
│  │                                                                             │    │
│  │  load_config(yaml_path) → OptimizerConfig                                   │    │
│  │                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                             │
│                                        ▼                                             │
└────────────────────────────────────────┼─────────────────────────────────────────────┘
                                         │
                                         │ config object passed to all components
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RUN_OPTIMIZER.PY                                        │
│                                                                                      │
│  def main():                                                                         │
│      config = load_config(args.config)                                              │
│      data_loader = DataLoader(data_dir, config)                                     │
│      feature_engineer = FeatureEngineer(config)                                     │
│      feature_selector = FeatureSelector(config)                                     │
│      win_rate_model = EmpiricalWinRateModel(config)                                 │
│      ctr_model = CTRModel(config)                                                   │
│      bid_landscape_model = BidLandscapeModel(config)                                │
│      npi_model = NPIValueModel(config.npi)           # if enabled                   │
│      domain_model = DomainValueModel(config)         # if enabled                   │
│      bid_calculator = VolumeFirstBidCalculator(config, models...)                   │
│      memcache_builder = MemcacheBuilder(config)                                     │
│      validator = OutputValidator(config)                                            │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Memcache Building Process

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           MEMCACHE BUILDER                                           │
│                           src/memcache_builder.py                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  INPUT: bid_results (list of BidResult objects from calculator)                      │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  BidResult:                                                                  │    │
│  │  • segment_key: tuple (adspace, geo, os, ...)                               │    │
│  │  • raw_bid: float (before clipping)                                         │    │
│  │  • final_bid: float (after floor/ceiling)                                   │    │
│  │  • win_rate: float (current segment WR)                                     │    │
│  │  • observations: int                                                        │    │
│  │  • strategy: str (volume_first/margin_optimize)                             │    │
│  │  • exploration_direction: str (up/down/stable)                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                            │
│                                         ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                    OUTPUT FILES GENERATED                                    │    │
│  │                                                                             │    │
│  │  1. suggested_bids_*.csv (PRODUCTION - loaded by bidder)                    │    │
│  │     ┌────────────────────────────────────────────────────────┐              │    │
│  │     │ internal_adspace_id | geo_region_name | os_code | bid  │              │    │
│  │     ├────────────────────────────────────────────────────────┤              │    │
│  │     │ 123                 | California      | 8       | 7.50 │              │    │
│  │     │ 123                 | California      | 1       | 8.20 │              │    │
│  │     │ 123                 | Texas           | 8       | 6.80 │              │    │
│  │     └────────────────────────────────────────────────────────┘              │    │
│  │                                                                             │    │
│  │  2. selected_features_*.csv (PRODUCTION - tells bidder which features)      │    │
│  │     ┌────────────────────────────────────────────────────────┐              │    │
│  │     │ feature_name       | position                          │              │    │
│  │     ├────────────────────────────────────────────────────────┤              │    │
│  │     │ internal_adspace_id| 0                                 │              │    │
│  │     │ geo_region_name    | 1                                 │              │    │
│  │     │ os_code            | 2                                 │              │    │
│  │     └────────────────────────────────────────────────────────┘              │    │
│  │                                                                             │    │
│  │  3. npi_multipliers_*.csv (PRODUCTION - NPI lookup)                         │    │
│  │     ┌────────────────────────────────────────────────────────┐              │    │
│  │     │ npi        | multiplier                                │              │    │
│  │     ├────────────────────────────────────────────────────────┤              │    │
│  │     │ 1234567890 | 2.50                                      │              │    │
│  │     │ 0987654321 | 1.80                                      │              │    │
│  │     └────────────────────────────────────────────────────────┘              │    │
│  │                                                                             │    │
│  │  4. domain_multipliers_*.csv (PRODUCTION - domain lookup)                   │    │
│  │     ┌────────────────────────────────────────────────────────┐              │    │
│  │     │ domain              | multiplier                       │              │    │
│  │     ├────────────────────────────────────────────────────────┤              │    │
│  │     │ premium-site.com    | 1.30                             │              │    │
│  │     │ good-site.com       | 1.15                             │              │    │
│  │     │ bad-site.com        | 0.60                             │              │    │
│  │     └────────────────────────────────────────────────────────┘              │    │
│  │                                                                             │    │
│  │  5. segment_analysis_*.csv (ANALYSIS - full segment details)                │    │
│  │     All columns including win_rate, observations, strategy, etc.            │    │
│  │                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Bidder Memcache Loading

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              BIDDER (Node.js)                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ON STARTUP / REFRESH:                                                               │
│                                                                                      │
│  1. Read manifest.json from S3                                                       │
│     ┌────────────────────────────────────────────────────────────────┐              │
│     │  GET s3://tn-optimizer-data/optimizer/drugs/active/manifest.json│              │
│     │  → { "run_id": "20260129_190416", "files": {...} }              │              │
│     └────────────────────────────────────────────────────────────────┘              │
│                                         │                                            │
│                                         ▼                                            │
│  2. Download files referenced in manifest                                            │
│     ┌────────────────────────────────────────────────────────────────┐              │
│     │  GET s3://.../runs/20260129_190416/suggested_bids_*.csv        │              │
│     │  GET s3://.../runs/20260129_190416/selected_features_*.csv     │              │
│     │  GET s3://.../runs/20260129_190416/npi_multipliers_*.csv       │              │
│     │  GET s3://.../runs/20260129_190416/domain_multipliers_*.csv    │              │
│     └────────────────────────────────────────────────────────────────┘              │
│                                         │                                            │
│                                         ▼                                            │
│  3. Build in-memory lookup structures                                                │
│     ┌────────────────────────────────────────────────────────────────┐              │
│     │                                                                │              │
│     │  segmentBids = Map<string, number>                             │              │
│     │  // key = "123|California|8" (pipe-delimited feature values)   │              │
│     │  // value = 7.50 (bid CPM)                                     │              │
│     │                                                                │              │
│     │  npiMultipliers = Map<string, number>                          │              │
│     │  // key = "1234567890" (NPI)                                   │              │
│     │  // value = 2.50 (multiplier)                                  │              │
│     │                                                                │              │
│     │  domainMultipliers = Map<string, number>                       │              │
│     │  // key = "example.com"                                        │              │
│     │  // value = 1.30 (multiplier)                                  │              │
│     │                                                                │              │
│     └────────────────────────────────────────────────────────────────┘              │
│                                         │                                            │
│                                         ▼                                            │
│  ON BID REQUEST:                                                                     │
│     ┌────────────────────────────────────────────────────────────────┐              │
│     │                                                                │              │
│     │  // Extract features from request                              │              │
│     │  segmentKey = `${adspace}|${geo}|${os}`;                       │              │
│     │                                                                │              │
│     │  // Lookup base bid                                            │              │
│     │  baseBid = segmentBids.get(segmentKey) || defaultBid;          │              │
│     │                                                                │              │
│     │  // Apply multipliers                                          │              │
│     │  domainMult = domainMultipliers.get(domain) || 1.0;            │              │
│     │  npiMult = npiMultipliers.get(npi) || 1.0;                     │              │
│     │                                                                │              │
│     │  // Calculate final bid                                        │              │
│     │  finalBid = Math.min(baseBid * domainMult * npiMult, maxBid);  │              │
│     │                                                                │              │
│     │  return finalBid;                                              │              │
│     │                                                                │              │
│     └────────────────────────────────────────────────────────────────┘              │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## MySQL Audit Logging

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MYSQL AUDIT DATABASE                                    │
│                              optimizer_audit                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  TABLE: optimizer_runs                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  Column              │ Type         │ Description                           │    │
│  ├──────────────────────┼──────────────┼───────────────────────────────────────┤    │
│  │  id                  │ INT AUTO_INC │ Primary key                           │    │
│  │  run_id              │ VARCHAR(50)  │ Timestamp ID (20260129_190416)        │    │
│  │  dataset             │ VARCHAR(50)  │ drugs / nativo_consumer               │    │
│  │  started_at          │ DATETIME     │ Run start time                        │    │
│  │  completed_at        │ DATETIME     │ Run end time                          │    │
│  │  status              │ ENUM         │ running/completed/failed              │    │
│  │  validation_passed   │ BOOLEAN      │ Did validation pass?                  │    │
│  │  deployed            │ BOOLEAN      │ Was manifest updated?                 │    │
│  │  s3_path             │ VARCHAR(255) │ S3 output location                    │    │
│  │  segments_count      │ INT          │ Number of segments                    │    │
│  │  bid_median          │ DECIMAL      │ Median bid                            │    │
│  │  win_rate_current    │ DECIMAL      │ Current win rate                      │    │
│  │  win_rate_target     │ DECIMAL      │ Target win rate                       │    │
│  │  config_hash         │ VARCHAR(64)  │ SHA256 of config (for reproducibility)│    │
│  │  error_message       │ TEXT         │ Error details if failed               │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  INSERT happens at:                                                                  │
│  • Run start → status='running'                                                      │
│  • Run end → status='completed' or 'failed', fill metrics                            │
│  • Deployment → deployed=TRUE                                                        │
│                                                                                      │
│  QUERY examples:                                                                     │
│  • Latest successful run: WHERE status='completed' ORDER BY completed_at DESC        │
│  • Failed runs this week: WHERE status='failed' AND started_at > NOW() - 7 DAY       │
│  • Win rate trend: SELECT run_id, win_rate_current FROM ... ORDER BY started_at      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Validation Gates

```
┌─────────────────────────────────────────────────────────────────┐
│                      HARD RULES (Block)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • bid_floor_respected: All bids >= min_bid_cpm                 │
│  • bid_ceiling_respected: All bids <= max_bid_cpm               │
│  • calibration_ece_max: Model ECE < threshold (0.15)            │
│  • coverage_min_pct: Segments >= 80% of previous run            │
│                                                                 │
│  ❌ FAIL → Do not deploy                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      SOFT RULES (Warn)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • pct_at_floor_max: < 30% bids at floor                        │
│  • pct_at_ceiling_max: < 30% bids at ceiling                    │
│  • pct_profitable_min: > 40% segments profitable                │
│                                                                 │
│  ⚠️ WARN → Deploy with review                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
