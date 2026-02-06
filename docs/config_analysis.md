# Config Analysis: User vs System Settings

**Created**: 2026-02-04
**Last Updated**: 2026-02-05
**Purpose**: Deep review of all config settings to bifurcate user-facing vs system-level controls

---

## Executive Summary

| Category | Count | Description |
|----------|-------|-------------|
| **User-Facing** | 6 | Settings business users adjust per run |
| **SSP-Type Entity** | 4 | Settings per SSP-Type combo (one YAML per entity) |
| **System/Global** | 25+ | Technical settings locked in backend (data-agnostic) |
| **Unused/Dead** | 4 | Can be removed from codebase |
| **Future (Roadmap)** | 1 | Planned user-facing addition |

---

## 0. ARCHITECTURE OVERVIEW

### Config Sources

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONFIG ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] SSP-Type YAML File (one per entity)                           │
│      optimizer_config_drugs_hcp.yaml                                │
│      optimizer_config_nativo_consumer.yaml                          │
│      optimizer_config_nativo_hcp.yaml (future)                      │
│          ↓                                                          │
│          Contains: floor_available, targeting_type,                 │
│                    ssp_exclusions, system configs                   │
│                                                                     │
│  [2] Run Configs from MySQL (per run)                              │
│      opt_run_configs table                                          │
│          ↓                                                          │
│          Contains: target_win_rate, max_bid_cpm, fast_learning,    │
│                    training dates, user_disabled_features           │
│                                                                     │
│  [3] Merge: YAML + MySQL → Final Config → Execute                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature Processing Flow

```
Raw Data (all columns from Snowflake/CSV)
              ↓
[1] SYSTEM EXCLUSIONS (hardcoded in code)
    Remove: log_txnid, internal_txn_id, log_dt
    These are NEVER features (join keys, timestamps)
              ↓
DataFrame with all potential signal columns
              ↓
[2] SSP-TYPE EXCLUSIONS (from entity YAML)
    Remove: columns with bad data quality in THIS entity
    Example: internal_adspace_id (100% -1 in nativo)
              ↓
[3] USER DISABLED FEATURES (from run config, optional)
    Remove: features user wants to exclude for experimentation
              ↓
[4] MODEL ROUTING (logic in code)
    Based on targeting_type:
    - If hcp: external_userid → NPI Model
    - If consumer: external_userid already excluded (no NPI)
    - domain column → Domain Model (if enabled)
    - Remaining → Segment Model candidates
              ↓
[5] FEATURE SELECTION (per model)
    Each model picks best features by min_signal_score
              ↓
Training
```

---

## 1. USER-FACING CONFIGS (Show in UI)

These settings are entered by business users for each run. **No defaults** - user must provide values.

### 1.1 Core Run Settings (Required)

| Config | Type | UI Element | Tooltip |
|--------|------|------------|---------|
| `target_win_rate` | float | Slider (40-85%) | "What % of bids should win? Higher = more volume, lower = better margins" |
| `max_bid_cpm` | float | Number input ($) | "Ceiling price - won't bid above this amount" |
| `fast_learning` | boolean | Toggle | "ON = learn quickly with higher risk, OFF = learn gradually and safely" |

### 1.2 Optional Settings (Advanced)

| Config | Type | UI Element | Tooltip |
|--------|------|------------|---------|
| `training_start_date` | date | Date picker | "Use data from this date forward (leave empty for all available)" |
| `training_end_date` | date | Date picker | "Use data up to this date (leave empty for latest)" |
| `user_disabled_features` | list | Multi-select | "Exclude these features from training (for experimentation)" |

### 1.3 Removed from User-Facing (with rationale)

| Config | Why Removed |
|--------|-------------|
| ~~`min_bid_cpm`~~ | **Trust the model.** In first-price auctions, a hard floor causes overpayment. For floor-available SSPs, the floor price from data IS the minimum. |
| ~~`validation.enabled`~~ | **Always on.** Validation is a safety net. Users shouldn't bypass protection. |

### 1.4 Recommended UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  RUN OPTIMIZER                              [nativo_consumer ▼] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Target Win Rate        [====65%====]  ℹ️                       │
│                         ← Better margins    More volume →       │
│                                                                 │
│  Maximum Bid            $[30.00]  ℹ️                            │
│                         Won't bid above this                    │
│                                                                 │
│  Learning Mode          ○ Gradual (safer)  ℹ️                   │
│                         ● Fast (higher risk)                    │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  Advanced ▼                                                     │
│    Training Window                                              │
│      Start: [2025-12-10]    End: [2026-01-31]                   │
│                                                                 │
│    Disable Features (experimental)                              │
│      [ geo_region_name ] [x]                                    │
│      [+ Add feature    ]                                        │
│                                                                 │
│                                         [Cancel]  [Run ▶]       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.5 Future User-Facing Configs (Roadmap)

| Config | Description | Use Case |
|--------|-------------|----------|
| `exploration_budget_pct` | "Max % of bids on exploration" | Risk control - limit spend on unknown segments |

---

## 2. SSP-TYPE ENTITY CONFIGS

**Each SSP-Type combination = 1 YAML file.** This file contains all entity-specific settings.

### 2.1 Entity Examples

| Entity | YAML File | SSP | Type | Floor | NPI |
|--------|-----------|-----|------|-------|-----|
| drugs_hcp | `optimizer_config_drugs_hcp.yaml` | drugs | hcp | No | Yes |
| nativo_consumer | `optimizer_config_nativo_consumer.yaml` | nativo | consumer | Yes | No |
| nativo_hcp | `optimizer_config_nativo_hcp.yaml` (future) | nativo | hcp | Yes | Yes |

### 2.2 Entity Config Fields (4 fields only)

| Config | Type | Purpose |
|--------|------|---------|
| `entity_name` | string | Unique identifier (e.g., "nativo_consumer") |
| `floor_available` | boolean | Does bid data include floor prices? |
| `targeting_type` | enum | "hcp" or "consumer" - determines NPI model usage |
| `ssp_exclusions` | list | Columns to exclude (bad data quality in THIS entity) |

### 2.3 SSP Exclusions (What Goes Here)

**ONLY columns with data quality issues specific to this entity.**

| Entity | ssp_exclusions | Reason |
|--------|----------------|--------|
| nativo_consumer | `internal_adspace_id` | 100% is -1 (useless) |
| nativo_consumer | `geo_country_code2` | 100% is US (no variance) |
| drugs_hcp | `pageurl_truncated` | 78K+ values, causes OOM |

**NOT in ssp_exclusions** (handled by code):
- `log_txnid`, `log_dt` → System exclusions (hardcoded)
- `domain` → Model routing (goes to Domain Model)
- `external_userid` → Model routing (goes to NPI Model if hcp)

### 2.4 Proposed Entity YAML Structure

```yaml
# =============================================================================
# optimizer_config_nativo_consumer.yaml
# =============================================================================

# Entity identity
entity:
  name: "nativo_consumer"
  floor_available: true
  targeting_type: "consumer"    # → no NPI model

  # Columns with bad data quality in THIS entity's data
  ssp_exclusions:
    - internal_adspace_id       # 100% is -1
    - geo_country_code2         # 100% is US

# User-facing configs (values come from MySQL at runtime)
# These are placeholders - actual values set per run
run:
  target_win_rate: null         # REQUIRED - user must enter
  max_bid_cpm: null             # REQUIRED - user must enter
  fast_learning: null           # REQUIRED - user must enter
  training_start_date: null     # Optional
  training_end_date: null       # Optional
  user_disabled_features: []    # Optional

# System configs (same across all entities, included here for completeness)
system:
  # ... (see Section 3)
```

### 2.5 Derived Settings (computed from entity config)

| Setting | Logic |
|---------|-------|
| NPI model enabled | `targeting_type == 'hcp'` |
| Domain model enabled | `targeting_type == 'consumer'` |
| Min bid source | `floor_available ? 'from_data' : 'from_model'` |
| external_userid handling | If hcp → route to NPI Model. If consumer → excluded. |

---

## 3. SYSTEM/GLOBAL CONFIGS (Backend Only)

These are **data-agnostic** algorithm parameters. Defaults are OK here because they don't depend on entity characteristics.

### 3.0 System Exclusions (Hardcoded)

These columns are NEVER features. Hardcoded in preprocessing code, not in config.

```python
SYSTEM_EXCLUSIONS = [
    'log_txnid',        # Join key
    'internal_txn_id',  # Join key
    'log_dt',           # Timestamp
]
```

### 3.1 Shared Data Sources

| Config | Value | Purpose |
|--------|-------|---------|
| `npi_data_1year` | "data/NPI_click_data_1year.csv" | 1-year NPI click history (shared across all HCP entities) |
| `npi_data_20day` | "data/NPI_click_data_20days.csv" | 20-day NPI click history (recency signal) |

### 3.2 Model Parameters

| Config | Value | Purpose |
|--------|-------|---------|
| `shrinkage_k` | 30 | Bayesian shrinkage strength |
| `calibration_ece_threshold` | 0.10 | Model calibration gate |
| `min_observations` | 1 | Min obs to include segment |
| `min_observations_for_empirical` | 10 | Min obs for empirical model |
| `min_observations_for_landscape` | 50 | Min obs for bid landscape |
| `min_win_rate_for_margin` | 0.55 | WR to switch to margin mode |
| `min_observations_for_margin` | 100 | Obs to switch to margin mode |

### 3.3 Feature Selection

| Config | Value | Purpose |
|--------|-------|---------|
| `min_signal_score` | 50.0 | Min score to select feature |
| `max_features` | 3 | Max features per model |
| `min_coverage_at_threshold` | 0.5 | Min coverage required |
| `max_null_pct` | 30.0 | Max null % allowed |
| `min_effective_cardinality` | 2 | Min distinct values |

### 3.4 Exploration Presets

| Preset | bonus_zero_obs | bonus_low_obs | bonus_medium_obs | max_adjustment |
|--------|---------------|---------------|------------------|----------------|
| Gradual | 0.25 | 0.15 | 0.08 | 1.4 |
| Aggressive | 0.50 | 0.35 | 0.15 | 1.8 |

### 3.5 NPI Tiering (when targeting_type='hcp')

| Config | Value | Purpose |
|--------|-------|---------|
| `tiering_method` | 'iqr' | IQR-based tiering |
| `iqr_multiplier_stars` | 1.5 | Q3 + 1.5×IQR = elite |
| `iqr_multiplier_extreme` | 3.0 | Q3 + 3.0×IQR = extreme |
| `max_multiplier` | 3.0 | Cap on NPI multiplier |
| `recency_boost` | 1.2 | Boost for recent activity |

### 3.6 Domain Tiering (when targeting_type='consumer')

| Config | Value | Purpose |
|--------|-------|---------|
| `tiering_method` | 'iqr' | IQR-based tiering |
| `iqr_multiplier_stars` | 1.5 | Q3 + 1.5×IQR = stars |
| `iqr_multiplier_extreme` | 3.0 | Q3 + 3.0×IQR = extreme |
| `blocklist_rate_factor` | 0.1 | Block if rate < global×0.1 |

### 3.7 Validation Rules (Always On)

**Hard Rules** (block deployment):
| Rule | Value |
|------|-------|
| `coverage_min_pct` | 80.0 |
| `calibration_ece_max` | 0.15 |
| `bid_floor_respected` | true |
| `bid_ceiling_respected` | true |

**Soft Rules** (warn only):
| Rule | Value |
|------|-------|
| `bid_median_change_max_pct` | 50.0 |
| `pct_at_floor_max` | 30.0 |
| `pct_at_ceiling_max` | 30.0 |
| `pct_profitable_min` | 40.0 |

---

## 4. UNUSED/DEAD CONFIGS (Cleaned Up)

**Status: Cleanup completed 2026-02-05**

| Config | Location | Status |
|--------|----------|--------|
| ~~`business.exploration_mode`~~ | config.py:123 | **KEEP** - actively used in bid_calculator_v5.py:467 |
| `business.target_margin` | config.py:124 | Keep for now - used in deprecated bid_calculator.py |
| `advanced.target_margin` | config.py:249 | **REMOVED** - was unused |
| `advanced.default_bid_cpm` | config.py:247 | **REMOVED** - was duplicate of technical |
| `advanced.floor_available` | config.py:248 | **REMOVED** - was duplicate of technical |

**Note:** Original analysis incorrectly marked `exploration_mode` as dead. Code analysis showed it's actively used to control exploration behavior in the V5 bid calculator.

---

## 5. MISSING CONFIGS (Hardcoded Values)

### High Priority

| Location | Value | Suggested Config |
|----------|-------|------------------|
| bid_calculator_v5.py:460 | 0.05 | `win_rate_gap_tolerance` |
| bid_landscape_model.py:338 | 3.0 | `exploration_max_multiplier` |

### Medium Priority

| Location | Value | Suggested Config |
|----------|-------|------------------|
| bid_landscape_model.py:228 | 0.1 | `golden_section_tolerance` |
| bid_calculator_v5.py:484 | 0.5 | `uncertainty_weights` |

---

## 6. DATABASE MAPPING

### User-Facing Configs → `opt_run_configs` (per run)

| config_key | Example Value | Notes |
|------------|---------------|-------|
| target_win_rate | 0.65 | Required |
| max_bid_cpm | 30.00 | Required |
| fast_learning | true | Required |
| training_start_date | 2025-12-10 | Optional |
| training_end_date | 2026-01-31 | Optional |
| user_disabled_features | geo_region_name,os_code | Optional, comma-separated |

### Entity Configs → `opt_entity_configs` (per entity)

| config_key | Example Value | Notes |
|------------|---------------|-------|
| floor_available | true | Entity capability |
| targeting_type | consumer | "hcp" or "consumer" |
| ssp_exclusions | internal_adspace_id,geo_country_code2 | Comma-separated |

### System Configs → Not in database

System configs are either:
1. Hardcoded in optimizer code
2. In the entity YAML file (system section)
3. In `opt_versions.supported_configs` (if version-specific)

---

## 7. SUMMARY OF CHANGES FROM ORIGINAL

| What Changed | Before | After |
|--------------|--------|-------|
| min_bid_cpm | User-facing | Removed (trust model / use floor) |
| validation toggle | User-facing | Removed (always on) |
| training_end_date | Not available | Added to user-facing |
| user_disabled_features | Not available | Added to user-facing (Advanced) |
| feature inclusion lists | anchor + candidates | Removed (auto-detect + exclusion) |
| NPI data paths | Per entity | System-level (shared) |
| SSP vs Entity | SSP-only | SSP-Type combo (entity) |

---

## 8. NEXT STEPS

1. **Clean up dead configs** - Remove unused fields from config.py
2. **Add user_disabled_features** - Add to run config and feature selection pipeline
3. **Add training_end_date** - Add to data filtering
4. **Restructure YAML files** - Adopt entity-based structure
5. **Update preprocessing** - Implement 5-stage feature flow
6. **Create UI spec** - Use Section 1.4 as starting point

---

*End of Analysis*
