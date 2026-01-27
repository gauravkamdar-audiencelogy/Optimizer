# RTB Bid Optimizer V9

Real-time bidding optimizer for programmatic advertising. Uses machine learning to calculate optimal bid prices per segment, with support for multiple datasets and NPI (healthcare provider) value multipliers.

## Overview

**Current Phase: Data Collection**
- Target: 65% win rate (currently ~30%)
- Strategy: Bid UP on losing segments to learn market prices
- Accepts negative margins during learning phase

**Supported Datasets:**
| Dataset | Targeting | NPI Data | Floor Prices |
|---------|-----------|----------|--------------|
| drugs.com | HCP (Healthcare Providers) | ✓ | ✗ |
| nativo_consumer | Consumer | ✗ | ✓ |

**Core Formula:**
```
final_bid = segment_bid × npi_multiplier  (NPI applied at request time by bidder)
```

## Quick Start

```bash
# 1. Activate virtual environment
source ./venv/bin/activate

# 2. Run optimizer for drugs.com
python run_optimizer.py --config config/optimizer_config_drugs.yaml

# 3. Or run for nativo_consumer
python run_optimizer.py --config config/optimizer_config_nativo_consumer.yaml
```

The optimizer auto-derives data and output paths from the config's dataset name:
- Config: `optimizer_config_drugs.yaml` → Data: `data_drugs/` → Output: `output_drugs/`
- Config: `optimizer_config_nativo_consumer.yaml` → Data: `data_nativo_consumer/` → Output: `output_nativo_consumer/`

To override auto-derived paths:
```bash
python run_optimizer.py --config config/optimizer_config_drugs.yaml \
    --data-dir custom_data/ --output-dir custom_output/
```

## Project Structure

```
optimizer_drugs_hcp/
├── run_optimizer.py                          # Main entry point
├── config/
│   ├── optimizer_config_drugs.yaml           # drugs.com settings
│   └── optimizer_config_nativo_consumer.yaml # nativo settings
├── src/
│   ├── data_loader.py                        # Load and clean bid/view/click data
│   ├── feature_engineering.py                # Feature creation and win rate joining
│   ├── feature_selector.py                   # Data-driven feature selection
│   ├── bid_calculator_v5.py                  # V5+ asymmetric exploration calculator
│   ├── memcache_builder.py                   # Output file generation
│   ├── metrics_reporter.py                   # Diagnostics and metrics
│   ├── config.py                             # Configuration dataclasses
│   └── models/
│       ├── empirical_model.py                # Segment-level empirical rates
│       ├── win_rate_model.py                 # Win rate prediction (with calibration gate)
│       ├── ctr_model.py                      # Click-through rate prediction
│       ├── bid_landscape_model.py            # P(win|bid) model
│       └── npi_value_model.py                # NPI value tiering (by click count)
├── scripts/
│   └── data_manager.py                       # Data ingestion CLI
├── data_drugs/                               # drugs.com data
│   ├── data_drugs.csv                        # Main data file
│   ├── incoming/                             # Drop new files here
│   ├── processed/                            # Processed files (audit trail)
│   ├── NPI_click_data_1year.csv              # NPI historical clicks
│   └── NPI_click_data_20days.csv             # NPI recent clicks
├── data_nativo_consumer/                     # nativo data
│   └── data_nativo_consumer.csv
├── output_drugs/                             # drugs.com outputs (timestamped)
└── output_nativo_consumer/                   # nativo outputs (timestamped)
```

## Data Management

### Adding New Data (drugs.com example)

```bash
# 1. Drop new CSV file in incoming/
cp new_export.csv data_drugs/incoming/

# 2. Run ingest (deduplicates, normalizes columns)
python scripts/data_manager.py ingest --data-dir data_drugs/

# 3. Verify
python scripts/data_manager.py info --data-dir data_drugs/
```

### Data Manager Commands

```bash
# Initialize directory structure
python scripts/data_manager.py init --data-dir data_drugs/

# Process incoming files (main workflow)
python scripts/data_manager.py ingest --data-dir data_drugs/

# Show data statistics and freshness
python scripts/data_manager.py info --data-dir data_drugs/

# One-time: combine separate files (bids/views/clicks → single file)
python scripts/data_manager.py combine --data-dir data_drugs/
```

## Output Files

Each run creates a timestamped folder (e.g., `output_drugs/20260127_143000/`):

| File | Purpose |
|------|---------|
| `suggested_bids_*.csv` | **Production**: segment features + bid (load into bidder) |
| `selected_features_*.csv` | **Production**: list of features used in this run |
| `npi_multipliers_*.csv` | **Production**: NPI → multiplier lookup (drugs.com only) |
| `npi_summary_*.csv` | Analysis: full NPI details (tier, recency, etc.) |
| `segment_analysis_*.csv` | Analysis: full segment metadata |
| `bid_summary_*.csv` | Analysis: tiered bucket overview |
| `metrics_*.json` | Diagnostics: model calibration, distributions |

### Suggested Bids Format
```csv
internal_adspace_id,geo_region_name,os_code,suggested_bid_cpm
111563,California,8,19.20
111564,Texas,14,20.00
```

### NPI Multipliers Format (drugs.com only)
```csv
external_userid,multiplier
1234567890,2.5
9876543210,1.0
```

## Configuration

### Dataset-Specific Config

Each dataset has its own config file with a `dataset` section:

```yaml
# config/optimizer_config_drugs.yaml
dataset:
  name: "drugs"
  # Auto-derived:
  #   data_dir: data_drugs/
  #   output_dir: output_drugs/
  #   data_file: data_drugs.csv

business:
  npi_exists: true           # Has NPI data
  floor_available: false     # No floor prices in bid requests
```

```yaml
# config/optimizer_config_nativo_consumer.yaml
dataset:
  name: "nativo_consumer"

business:
  npi_exists: false          # No NPI data
  floor_available: true      # Has floor prices
```

### Key Configuration Options

```yaml
business:
  target_win_rate: 0.65          # Target for exploration
  exploration_mode: true         # Enable asymmetric bidding
  use_npi_value: true            # Enable NPI multipliers (if npi_exists)
  npi_max_multiplier: 3.0        # Max NPI boost

technical:
  min_bid_cpm: 3.00              # Floor
  max_bid_cpm: 20.00             # Ceiling (gradual mode)
  min_observations: 1            # Include all segments
  max_features: 3                # Segment granularity

  # V8: Exploration aggressiveness toggle
  aggressive_exploration: false  # true = faster learning, higher risk

bidding:
  strategy: "adaptive"           # volume_first, margin_optimize, or adaptive
```

### Exploration Presets (V8)

Toggle between aggressive and gradual exploration:

| Setting | Aggressive | Gradual (Default) |
|---------|------------|-------------------|
| Zero-obs bonus | +50% | +25% |
| Low-obs bonus | +35% | +15% |
| Medium-obs bonus | +15% | +8% |
| Max adjustment | 1.8x | 1.4x |
| Max bid | $30 | $20 |

Set `aggressive_exploration: true` in config to use aggressive mode.

## Bidding Logic

### Asymmetric Exploration (V5+)
- **Under-winning (WR < 65%)**: Bid **UP** - lost bids tell us market values them higher
- **Over-winning (WR > 65%)**: Bid **DOWN** - find floor, don't overpay

### Data-Driven Multipliers (V8)
Exploration multipliers are derived from the bid landscape model:
```
Derived from: bid coefficient, current WR, target WR
Current: 3.0x up (capped), 0.5x down
```

### Strategy Selection (Adaptive)
- **volume_first**: Maximize win rate, accept negative margins
- **margin_optimize**: Maximize profit per impression
- **adaptive**: Auto-switch per segment based on maturity (WR >= 55%, obs >= 100)

### NPI Value Tiers (drugs.com, by click count)

| Tier | Threshold | Multiplier |
|------|-----------|------------|
| 1 (Elite) | ≥6 clicks | 2.5x |
| 2 (High) | ≥3 clicks | 1.8x |
| 3 (Medium) | ≥2 clicks | 1.3x |
| 4 (Standard) | <2 clicks | 1.0x |

Recent clickers (last 20 days) get +20% boost.

## Bidder Integration

At request time:
```python
# 1. Look up segment bid from memcache
segment_key = (internal_adspace_id, geo_region_name, os_code)
base_bid = memcache.get(segment_key, default_bid)

# 2. Look up NPI multiplier (drugs.com only)
npi = bid_request.external_userid
npi_mult = npi_cache.get(npi, 1.0)

# 3. Calculate final bid
final_bid = base_bid * npi_mult
```

## Daily Operations

### Recommended Cadence
Run optimizer **daily** during exploration phase:
- Market conditions change (campaigns start/end)
- Faster iteration = faster learning
- Weekly is too slow for exploration

### Typical Workflow
```bash
# 1. Ingest any new data
python scripts/data_manager.py ingest --data-dir data_drugs/

# 2. Run optimizer
python run_optimizer.py --config config/optimizer_config_drugs.yaml

# 3. Review output
ls -la output_drugs/$(ls -t output_drugs/ | head -1)/

# 4. Deploy memcache and NPI files to bidder
```

## Requirements

```
pandas
numpy
scikit-learn
tqdm
pyyaml
```

Install: `pip install -r requirements.txt`

## Development Notes

See `CLAUDE.md` for detailed implementation notes, architectural decisions, and learnings.

Key design principles:
- **Data-agnostic model, data-driven config**: Model formulas are fixed; parameters derived from data
- **Calibration gates**: Models used only if ECE < threshold
- **NPI at request time**: Segment bids don't include NPI; bidder applies multiplier
