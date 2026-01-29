# CLAUDE.md

## Core Principles

### Question Everything
- Do not take things at face value, question everything

### Economic Reasoning First
- Every technical decision must serve the goal: **build an economically viable optimizer**
- When a component underperforms, ask "how do I fix this to capture economic value?" not "should I remove this?"
- Removing functionality means losing signal. Lost signal = lost money.

### Research-Backed Decisions
- Search literature (arxiv, production papers) before implementing non-trivial solutions
- Cite at least one paper or production system where the approach worked
- If you can't find evidence it works, be explicit about the uncertainty

### Measure What Matters
- If you build a model, output its diagnostics (calibration, discrimination, distributions)
- If you can't measure it, you can't improve it
- Never ship a metrics file without the metrics needed to evaluate the thing it describes

### Data-Agnostic Model, Data-Driven Config
**The model should be data-agnostic; the config should be data-driven.**

- **Model** = formulas, logic, code structure → works on ANY data without modification
- **Config** = parameters fed into the model → DERIVED from the data automatically

This separation means:
- No hardcoded "magic numbers" in the model
- Parameters adapt automatically when data changes
- Model code is stable; only derived parameters change
- Same model works across different markets/datasets

**Example:**
```
BAD:  exploration_multiplier = 1.3  # Why 1.3? Someone made it up.
GOOD: exploration_multiplier = derive_from_bid_landscape(data)  # Derived from P(win|bid)
```

---

## Technical Guardrails

### Do
- Parse PostgreSQL arrays: `{value1,value2}` → extract first numeric value
- Deduplicate views by `log_txnid` before analysis
- Filter zero bids before training
- Use `pd.to_datetime(df['log_dt'], format='ISO8601', utc=True)` for mixed timezones
- Join bids→views on `log_txnid` for win rate (not `len(views)/len(bids)`)
- Fill geo nulls with 'Unknown' before encoding

### Avoid
- `class_weight='balanced'` with extreme class imbalance (<1% positive rate) - destroys probability calibration
- Removing models/features without first attempting to fix them
- Hardcoding feature selections - use data-driven selection
- Outputting predictions without calibration diagnostics

### Prefer
- Bayesian shrinkage toward global rates over raw segment rates (sparse data)
- Simpler models with proper calibration over complex models with poor calibration
- Binary filtering (include/exclude) over continuous confidence scaling when data is sparse

---

## Architecture Overview

### Auction Type: First-Price
- We pay what we bid (no second-price discount)
- Bid shading is critical — bid landscape model is essential
- Need to find market clearing price, not bid true value

### Two-Level Bidding Architecture
```
Memcache:   segment → base_bid     (e.g., $19.20)
NPI cache:  npi → multiplier       (e.g., 3.0x)
Bidder:     final_bid = base_bid × npi_multiplier  (applied at request time)
```

**Why separate?**
- Can't cross NPI with segments (explosion: 1,500 × 65K = 97M rows)
- NPI comes in bid request (`external_userid` field)
- Keeps segment learning separate from prescriber value

### Calibration Gate Pattern
Build models optimally, then gate their usage at runtime:
```python
ece = evaluate_calibration(model, validation_data)
if ece < config.calibration_threshold:
    use_model = True
else:
    use_model = False  # Fall back to empirical
    log.warning(f"Model ECE={ece:.3f} exceeds threshold")
```

### Asymmetric Exploration
- **Under-winning (WR < target)**: Bid HIGHER - lost bids tell us market values them higher
- **Over-winning (WR > target)**: Bid LOWER - find floor, don't overpay
- Lost bids ARE the learning signal

---

## Current Configuration

### Supported Datasets
| Dataset | Targeting | NPI Data | Floor Prices |
|---------|-----------|----------|--------------|
| drugs | HCP (Healthcare Providers) | Yes | No |
| nativo_consumer | Consumer | No | Yes |

### Bidding Strategy
- Strategy: `adaptive` (per-segment automatic switching)
- Volume-first until segment matures (WR >= 55%, obs >= 100)
- Then switches to margin optimization

### Exploration Mode
- Toggle: `aggressive_exploration: true/false`
- Gradual (default): Conservative step sizes, $20 max bid
- Aggressive: Faster learning, $30 max bid, higher risk

### Feature Selection
- Automatic data-driven selection
- Currently selected: `internal_adspace_id`, `geo_region_name`, `os_code`
- Auto-excludes low-variance features (single dominant value)

---

## Integrations

### Environment Setup
```bash
cp .env.template .env  # Fill in credentials
```

Set `OPTIMIZER_ENV=local` for development (skips integrations), `qa` or `production` for full integration.

### S3 (Output Storage)
- Bucket: `s3://tn-optimizer-data/optimizer/{dataset}/`
- Structure: `runs/{run_id}/` for outputs, `active/manifest.json` for bidder
- Auto-uploads after successful validation

### Snowflake (Data Source)
- Supports password and key-pair authentication
- Set `SNOWFLAKE_PRIVATE_KEY_PATH` for PEM file auth

### MySQL (Audit Logging)
- Tracks run history, deployment status
- Schema design in `docs/database_schema.md`

### Validation Framework
Hard guardrails (block deployment):
- `bid_floor_respected`: All bids >= min_bid
- `bid_ceiling_respected`: All bids <= max_bid
- `calibration_ece_max`: Model ECE below threshold

Soft guardrails (warn but allow):
- `pct_at_floor_max`: Not too many bids at floor
- `pct_at_ceiling_max`: Not too many bids at ceiling
- `pct_profitable_min`: Minimum profitable segments

Toggle with `validation.enabled: true/false` in config.

---

## Data Model

### Record Types
| rec_type | Description | Key Fields |
|----------|-------------|------------|
| `bid` | Bid request submitted | `publisher_payout` = our bid offer |
| `view` | Won impression (ad shown) | `publisher_payout` = clearing price |
| `click` | Click on impression | `advertiser_spend` = CPC revenue |

### Win/Loss Determination
- **WON**: Bid has matching view via `log_txnid`
- **LOST**: Bid has NO matching view (we bid too low)

### Key Fields
- `publisher_payout`: PostgreSQL array `{7.50000}` - extract first value
- `log_txnid`: Links bid → view → click records
- `external_userid`: NPI (10-digit healthcare provider ID) for drugs.com

### Data Pipeline
```bash
# Drop new file in incoming/
cp export.csv data_drugs/incoming/

# Ingest + optimize in one command
python run_optimizer.py --config config/optimizer_config_drugs.yaml --ingest

# Or separate steps
python scripts/data_manager.py ingest --data-dir data_drugs/
python run_optimizer.py --config config/optimizer_config_drugs.yaml
```

---

## Quick Reference

### Commands
```bash
# Activate environment
source ./venv/bin/activate

# Run optimizer (drugs.com)
python run_optimizer.py --config config/optimizer_config_drugs.yaml

# Run optimizer (nativo)
python run_optimizer.py --config config/optimizer_config_nativo_consumer.yaml

# With data ingestion
python run_optimizer.py --config config/optimizer_config_drugs.yaml --ingest

# Check integration status
python scripts/check_integrations.py --test-connections

# Data management
python scripts/data_manager.py info --data-dir data_drugs/
```

### Output Files
| File | Purpose |
|------|---------|
| `suggested_bids_*.csv` | Production: segment → bid (load into bidder) |
| `selected_features_*.csv` | Production: features used in this run |
| `npi_multipliers_*.csv` | Production: NPI → multiplier lookup |
| `validation_report_*.json` | Deployment gate: pass/fail + details |
| `segment_analysis_*.csv` | Analysis: full segment metadata |
| `metrics_*.json` | Diagnostics: model calibration, distributions |

### Project Structure
```
optimizer_drugs_hcp/
├── config/                    # Dataset-specific configs
├── data_drugs/               # drugs.com data + incoming/
├── data_nativo_consumer/     # nativo data
├── docs/                     # Architecture, schemas, plans
├── output_drugs/             # Timestamped run outputs
├── scripts/                  # CLI tools (data_manager, check_integrations)
├── src/                      # Core optimizer code
│   ├── models/              # ML models (CTR, win rate, bid landscape, NPI)
│   ├── integrations/        # S3, Snowflake, MySQL clients
│   └── validator.py         # Output validation
├── run_optimizer.py          # Main entry point
├── CLAUDE.md                 # This file
└── README.md                 # User documentation
```

---

## Key Learnings (Reference)

### NPI Valuation: Use Click COUNT, Not Revenue
- Revenue = Click × Campaign_Payout (demand-side confounded)
- Click count = "Does this NPI engage?" (pure supply-side signal)
- Revenue makes valuations dependent on advertiser behavior we don't control

### Model Calibration
- Empirical model with Bayesian shrinkage is well-calibrated by definition
- LogReg needs post-hoc isotonic calibration + cross-validation for ECE
- Always evaluate ECE on held-out data, never training data

### Exploration Multipliers
- Derived from bid landscape coefficient (data-driven)
- Capped at 3.0x to limit extrapolation risk
- Falls back to config values if derivation fails

---

## Documentation Index

- `README.md` - User guide, quick start, configuration options
- `docs/architecture_diagram.md` - System architecture visualization
- `docs/data_flow.md` - End-to-end data flow diagram
- `docs/database_schema.md` - MySQL schema for audit logging
- `docs/PLAN_ARCHIVE.md` - Completed work history
- `docs/meta_data/` - Bid request field documentation
