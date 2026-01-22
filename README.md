# RTB Bid Optimizer V5

Real-time bidding optimizer for programmatic advertising. Uses machine learning to calculate optimal bid prices per segment, with NPI (healthcare provider) value multipliers.

## Overview

**Current Phase: Data Collection**
- Target: 65% win rate (currently ~30%)
- Strategy: Bid UP on losing segments to learn market prices
- Accepts negative margins during learning phase

**Core Formula:**
```
final_bid = segment_bid × npi_multiplier
```

## Quick Start

```bash
# 1. Activate virtual environment
source ./venv/bin/activate

# 2. Run optimizer
python run_optimizer.py --config config/optimizer_config.yaml --data-dir data_drugs/ --output-dir output/
```

## Project Structure

```
optimizer_drugs_hcp/
├── run_optimizer.py              # Main entry point
├── config/
│   └── optimizer_config.yaml     # All configuration
├── src/
│   ├── data_loader.py            # Load and clean bid/view/click data
│   ├── feature_engineering.py    # Feature creation and win rate joining
│   ├── feature_selector.py       # Data-driven feature selection
│   ├── bid_calculator_v5.py      # V5 asymmetric exploration calculator
│   ├── memcache_builder.py       # Output file generation
│   ├── metrics_reporter.py       # Diagnostics and metrics
│   ├── config.py                 # Configuration dataclasses
│   └── models/
│       ├── empirical_model.py    # Segment-level empirical rates
│       ├── win_rate_model.py     # Win rate prediction
│       ├── ctr_model.py          # Click-through rate prediction
│       └── npi_value_model.py    # NPI value tiering
├── scripts/
│   └── data_manager.py           # Data ingestion CLI
└── data_drugs/
    ├── drugs_data.csv            # Main data file
    ├── incoming/                 # Drop new files here
    ├── processed/                # Processed files (audit trail)
    └── archive/                  # Original separate files
```

## Data Management

### Adding New Data

```bash
# 1. Drop new CSV file in incoming/
cp new_export.csv data_drugs/incoming/

# 2. Run ingest
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

# Show data statistics
python scripts/data_manager.py info --data-dir data_drugs/

# One-time: combine separate files (bids/views/clicks → drugs_data.csv)
python scripts/data_manager.py combine --data-dir data_drugs/
```

## Output Files

Each run creates a timestamped folder in `output/`:

| File | Purpose |
|------|---------|
| `memcache_*.csv` | **Production**: segment features + bid (load into bidder) |
| `npi_multipliers_*.csv` | **Production**: NPI → multiplier lookup |
| `segment_analysis_*.csv` | Analysis: full segment metadata |
| `bid_summary_*.csv` | Analysis: tiered bucket overview |
| `metrics_*.json` | Diagnostics: model calibration, distributions |

### Memcache Format
```csv
internal_adspace_id,geo_region_name,os_code,suggested_bid_cpm
111563,California,8,7.25
111564,Texas,14,8.50
```

### NPI Multipliers Format
```csv
external_userid,multiplier,tier,is_recent
1234567890,2.5,1,True
9876543210,1.0,4,False
```

## Configuration

Key settings in `config/optimizer_config.yaml`:

```yaml
business:
  target_win_rate: 0.65          # Target for exploration
  exploration_mode: true         # Enable asymmetric bidding
  use_npi_value: true            # Enable NPI multipliers
  npi_max_multiplier: 3.0        # Max NPI boost

technical:
  min_bid_cpm: 3.00              # Floor
  max_bid_cpm: 30.00             # Ceiling
  min_observations: 1            # Include all segments
  max_features: 3                # Segment granularity
```

## V5 Bidding Logic

**Asymmetric Exploration:**
- Under-winning (WR < 65%): Bid **UP** aggressively (×1.3)
- Over-winning (WR > 65%): Bid **DOWN** cautiously (×0.7)

**NPI Value Tiers:**
| Tier | Percentile | RPU Threshold | Multiplier |
|------|------------|---------------|------------|
| 1 | Top 1% | >$135 | 2.5x |
| 2 | Top 5% | >$45 | 1.8x |
| 3 | Top 20% | >$15 | 1.3x |
| 4 | Rest | <$15 | 1.0x |

Recent clickers (last 20 days) get +20% boost.

## Bidder Integration

At request time:
```python
# 1. Look up segment bid
segment_key = (internal_adspace_id, geo_region_name, os_code)
base_bid = memcache.get(segment_key, default_bid)

# 2. Look up NPI multiplier
npi = bid_request.external_userid
npi_mult = npi_cache.get(npi, 1.0)

# 3. Calculate final bid
final_bid = base_bid * npi_mult
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

See `CLAUDE.md` for detailed implementation notes, decisions, and learnings.
