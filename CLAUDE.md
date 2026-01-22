# CLAUDE.md

## Core Principles
- Do not take things on face value, question everything

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

## Business Context (January 2026)

### Cold Start Problem
- Started with flat $7.50 bids, no loss logging (only logged wins/views)
- December 10th: Added `rec_type=bid` logging - can now see win/loss
- Currently exploring with bids up to $12.50 to learn market

### Current Phase: DATA COLLECTION (Not Margin Optimization)
- **Priority**: Volume > Margin
- **Goal**: Learn market prices (bid landscape)
- **Strategy**: Accept negative margins during learning
- **Budget**: No constraint - can bid aggressively
- **Target**: 65% win rate (currently ~30%)

### NPI Data
- `external_userid` field contains NPI numbers (10-digit healthcare provider IDs)
- 30-day revenue/click likelihood data is AVAILABLE
- High-value NPIs should get higher bids
- This is HCP (Healthcare Provider) targeting data

### V4 vs V5 Strategy
| Aspect | V4 (Wrong) | V5 (Correct) |
|--------|------------|--------------|
| Objective | Maximize margin `(EV-b)×P(win)` | Target 65% win rate |
| Losing segments | Ignored | Bid UP to learn ceiling |
| Winning segments | Keep same bid | Bid DOWN to find floor |
| Output bids | $2-3 median (too low) | $8-12 expected |
| Segments included | 93 (6%) | ~1,500 (100%) |

### Asymmetric Exploration Principle
- **Under-winning (WR < 65%)**: Bid HIGHER - lost bids tell us market values them higher
- **Over-winning (WR > 65%)**: Bid LOWER - find floor, don't overpay
- The lost bids ARE the learning signal

## Key Data Insights (from EDA)

### Bid Distribution (Dec-Jan)
- 207K bids, 92K views (44% win rate)
- 71% of bids at $7.50 (legacy flat bid)
- Only 1.9% of bids at $20+
- $20+ bids = 8.1% of wins (4x over-representation!) → room to explore higher

### Win Rate by Segment
- By adspace: 26% to 56% (wide variance)
- By region: California 55%, Virginia 35%
- By OS: iOS 47%, Unknown (-1) 21%

### High-Value Signals
- Some campaigns pay $130/click (vs avg $12)
- Page category matters: `news` 0.26% CTR vs `interaction` 0.004%
- NPI-level targeting is the key unlock

### Volume-Killing Filter
- `min_observations=100` excluded 78% of segments
- V5 removes this filter - include ALL segments with uncertainty-adjusted bids

## Working Memory

### CTR Model Miscalibration (Resolved)
- **Symptom**: Model predicted 35% CTR when actual was 0.037%
- **Cause**: `class_weight='balanced'` inflates minority class by ~1000x
- **Fix**: Remove class_weight, implement Bayesian shrinkage toward global CTR
- **Lesson**: With extreme imbalance, class weighting destroys the calibration you need for bidding

### Win Rate Model Miscalibration (Open)
- **Symptom**: ECE=0.176, consistently overestimates by ~1.7x
- **Current state**: Removed from bid formula (diagnostic only)
- **TODO**: Investigate if same class_weight issue; fix and restore to capture market signals

### Bid Floor Clipping (Resolved)
- **Symptom**: 92.67% of bids clipped to $5 floor
- **Cause**: Floor too high relative to expected values
- **Fix**: Lower floor to $2, increase target_margin to 30%

## V5 Implementation Notes (January 21, 2026)

### User Requirements Gathered
1. **Priority**: Data collection over margin optimization
2. **Target Win Rate**: 60-70% (we chose 65%)
3. **Budget Constraint**: None - can bid aggressively
4. **NPI Data**: Available in `external_userid` field
5. **Loss Strategy**: Bid HIGHER to learn ceiling (lost bids = learning signal)
6. **Asymmetric Exploration**:
   - Losing segments → Bid UP (aggressive, 1.3x multiplier)
   - Winning segments → Bid DOWN (cautious, 0.7x multiplier)

### Memcache Contract (CRITICAL)
**Memcache can ONLY contain:**
- Feature columns (e.g., internal_adspace_id, geo_region_name, os_code)
- `suggested_bid_cpm`

**DO NOT include in memcache:**
- observation_count
- exploration_direction
- bid_method
- Any other metadata

For analysis purposes, create a separate `segment_analysis_*.csv` file.

### V5 Files Created/Modified
| File | Change |
|------|--------|
| `src/bid_calculator_v5.py` | NEW - VolumeFirstBidCalculator with asymmetric exploration |
| `src/models/npi_value_model.py` | NEW - NPI value lookup (ready when data available) |
| `src/config.py` | Updated with V5 params, removed external_userid from hard exclusions |
| `config/optimizer_config.yaml` | V5 config: 65% WR, exploration mode, tiered obs thresholds |
| `src/memcache_builder.py` | Updated to include all segments, v5_mode flag |
| `run_optimizer.py` | Integrated V5 calculator, NPI model, global stats |
| `src/metrics_reporter.py` | Added exploration metrics |

### V5 Configuration Parameters
```yaml
business:
  target_win_rate: 0.65         # Up from 0.50
  exploration_mode: true
  exploration_up_multiplier: 1.3
  exploration_down_multiplier: 0.7
  accept_negative_margin: true
  max_negative_margin_pct: 0.50  # Cap at 1.5x EV

technical:
  min_observations: 1            # Was 100!
  min_observations_for_empirical: 10
  min_observations_for_landscape: 50
  min_win_rate_adjustment: 0.6   # Was 0.8
  max_win_rate_adjustment: 1.8   # Was 1.2
  exploration_bonus_zero_obs: 0.50
  exploration_bonus_low_obs: 0.35
  exploration_bonus_medium_obs: 0.15
```

### V5 Core Formula
```python
wr_gap = target_wr - current_wr  # 0.65 - current

if wr_gap > 0:  # UNDER-WINNING
    adjustment = 1.0 + wr_gap * 1.3  # Bid UP
else:  # OVER-WINNING
    adjustment = 1.0 + wr_gap * 0.7  # Bid DOWN

# Scale by uncertainty
uncertainty = 1.0 / (1.0 + np.log1p(observation_count))
adjustment = 1.0 + (adjustment - 1.0) * (0.5 + 0.5 * uncertainty)

# Clip to bounds
adjustment = np.clip(adjustment, 0.6, 1.8)

bid = base_bid * adjustment * npi_multiplier
```

### V5 Run Results (20260121_162058)
**Input Data:**
- 207,013 bids
- 577,586 views
- 248 clicks
- Global win rate: 30.3%

**Feature Selection:**
- Selected: `['internal_adspace_id', 'geo_region_name', 'os_code']`
- Auto-excluded: geo_country_code2 (98.7% US), domain (100% drugs.com), browser_code, hour_of_day, day_of_week
- Hard-excluded: pageurl_truncated (78K+ segments, OOM issues)

**Output:**
- 1,454 segments (vs 93 in V4 - 15x more!)
- Bid median: $7.25 (vs $2.82 in V4 - 2.5x higher!)
- Bid range: $2.01 - $13.86
- 98.8% segments bid UP (global WR 30% < target 65%)
- 0% segments bid DOWN
- 72.8% profitable segments

**Method Distribution:**
- v5_explore_low (1-9 obs): 633 (43.5%)
- v5_explore_medium (10-49 obs): 389 (26.8%)
- v5_empirical (50+ obs): 432 (29.7%)

### Bug Fixes During Implementation
1. **pageurl_truncated OOM**: Created 78K+ segments, crashed system. Fixed by adding to hard exclusions.
2. **Guardrails floor bug**: max_overpay was overriding min_bid floor. Fixed to always respect floor regardless of EV cap.

### Output Files Structure
- `memcache_*.csv` - Production file: features + suggested_bid_cpm ONLY
- `segment_analysis_*.csv` - Analysis file: full bid landscape with metadata
- `metrics_*.json` - Run metrics and diagnostics

## NPI Value Model Integration (January 22, 2026)

### Data Sources
- `data_drugs/NPI_click_data_1year.csv`: 64,783 valid NPIs with historical RPU (revenue per user)
- `data_drugs/NPI_click_data_20days.csv`: 6,770 recent clickers (recency signal)
- Some non-NPI IDs in 1-year data (32-36 char hashes) - filter to 10-digit NPIs only

### Key Data Findings
- **Value concentration is EXTREME**:
  - Top 1% NPIs (647) = 13.4% of revenue
  - Top 5% NPIs (3,239) = 42.4% of revenue
  - Top 10% = 55.9% of revenue
  - Top 20% = 73.5% of revenue
- **Recency matters**: Recent clickers avg $17.98 RPU vs $11.69 (54% higher!)
- All 20-day NPIs exist in 1-year data (it's a subset)

### Tiering Logic (Percentile-based)
| Tier | Percentile | RPU Threshold | Base Multiplier | With Recency (+20%) |
|------|------------|---------------|-----------------|---------------------|
| 1 (Elite) | Top 1% | >$135 | 2.5x | 3.0x (capped) |
| 2 (High) | Top 5% | >$45 | 1.8x | 2.16x |
| 3 (Medium) | Top 20% | >$15 | 1.3x | 1.56x |
| 4 (Standard) | Rest | <$15 | 1.0x | 1.2x |

### User Decisions (Ask Questions!)
1. **Recency handling**: Chose "Recency boost" - use 1-year RPU as base, add boost if in 20-day data
2. **Max multiplier**: Chose 3.0x (aggressive) - top prescribers are worth capturing
3. **Output format**: Chose separate `npi_multipliers_*.csv` - NOT embedded in segment memcache

### Architecture Decision: NPI is REQUEST-TIME Multiplier
**NPI is NOT part of segment key.** This is critical:
- Memcache: segment → base_bid
- NPI cache: npi → multiplier
- Bidder combines at request time: `final_bid = segment_bid × npi_multiplier`

**Why?**
- Can't cross NPI with segments (1,454 × 64K = 93M rows - explosion)
- NPI comes in bid request (`external_userid` field)
- Keeps segment learning separate from prescriber value

### Config Changes
```yaml
business:
  use_npi_value: true
  npi_1year_path: "data_drugs/NPI_click_data_1year.csv"
  npi_20day_path: "data_drugs/NPI_click_data_20days.csv"
  npi_max_multiplier: 3.0
  npi_recency_boost: 1.2  # +20%
```

### NPI Model Results
- 64,783 NPIs loaded
- Tier 1: 710 (1.1%), Tier 2: 2,583 (4.0%), Tier 3: 11,084 (17.1%), Tier 4: 50,406 (77.8%)
- 6,770 recent clickers (10.5%) get +20% boost
- Avg multiplier: 1.12x
- 203 NPIs get max 3.0x multiplier (Tier 1 + recent)

### Output Files (Complete List)
```
output/YYYYMMDD_HHMMSS/
├── memcache_*.csv           # Segment → bid (production, features + bid ONLY)
├── npi_multipliers_*.csv    # NPI → multiplier (bidder lookup)
├── segment_analysis_*.csv   # Full segment metadata (analysis)
├── bid_summary_*.csv        # Tiered bucket overview
└── metrics_*.json           # All diagnostics
```

### Bidder Implementation
```python
# At request time:
segment_key = extract_features(bid_request)
base_bid = memcache.get(segment_key)

npi = bid_request.external_userid
npi_mult = npi_cache.get(npi, 1.0)  # Default 1.0 if unknown

final_bid = base_bid * npi_mult
```

## Config Evaluation Notes (January 22, 2026)

### max_features=3 → OPTIMAL
- 4th best feature (`hour_of_day`) has signal score 21.80 vs threshold 50.0
- Adding it would explode segments: 1,454 → ~35,000
- Would collapse avg observations: 142 → 6 per segment
- 99.9% would be sparse (<10 obs)
- **Conclusion**: Limit is bounded by signal quality, not arbitrary cap

### min_observations=1 → OPTIMAL
- Low-obs segments (1-9) = 43% of segments but only 1.1% of observations
- They're NOT drowning out signal from high-obs segments
- Win rates stable across tiers (~31%) - no noise corruption
- Empirical model ECE = 0.0143 (well-calibrated despite sparse segments)
- Shrinkage (k=30) successfully handles uncertainty
- **Conclusion**: Enables volume-first exploration as designed

### Features Evaluated (11 total, 3 selected)
Selected: `internal_adspace_id`, `geo_region_name`, `os_code`

Auto-excluded (data-driven):
- `geo_country_code2`: eff_card=1 (98.7% US)
- `domain`: eff_card=1 (100% www.drugs.com)
- `browser_code`: eff_card=1 (93.9% is '14')
- `hour_of_day`: signal=21.80 < 50 threshold
- `day_of_week`: signal=10.56 < 50 threshold
- `media_type`: eff_card=1 (97.5% display)
- `make_id`, `model_id`, `carrier_code`: eff_card=1 (100% is '-1')

### Candidate Features in Config
User added more features to evaluate (even if they get rejected):
- `media_type`, `make_id`, `model_id`, `carrier_code`
- These are correctly auto-excluded due to single dominant value

### Features NOT Worth Adding (Demand-Side)
- `list_id`, `campaign_code`, `banner_code` - These are demand-side (campaign-specific)
- `total_ads`, `rule_id` - Internal controls, not from SSP bid request

## Data Management Pipeline (January 22, 2026)

### Directory Structure
```
data_drugs/
├── drugs_data.csv          # Main data file (optimizer reads this)
├── incoming/               # Drop new files here
├── processed/              # Processed files with timestamp (audit trail)
├── archive/                # Original separate files (historical)
├── NPI_click_data_1year.csv
└── NPI_click_data_20days.csv
```

### Production Workflow
```bash
# 1. Drop new CSV in incoming/
cp new_export.csv data_drugs/incoming/

# 2. Run ingest
python scripts/data_manager.py ingest --data-dir data_drugs/

# 3. Run optimizer
python run_optimizer.py --config config/optimizer_config.yaml --data-dir data_drugs/ --output-dir output/
```

### Data Manager Commands
```bash
# Initialize directory structure
python scripts/data_manager.py init --data-dir data_drugs/

# Process incoming files (main workflow)
python scripts/data_manager.py ingest --data-dir data_drugs/

# Show data statistics and freshness
python scripts/data_manager.py info --data-dir data_drugs/

# One-time: combine separate files (migration)
python scripts/data_manager.py combine --data-dir data_drugs/
```

### Key Learnings

**Column Name Normalization:**
- Database exports may have UPPERCASE columns (LOG_DT, REC_TYPE)
- Script normalizes to lowercase automatically
- Don't assume column case matches

**Deduplication Key:**
- Key: `(log_txnid, rec_type)`
- Same transaction can have bid + view events (different rec_type)
- Dedup removes exact duplicates, keeps distinct event types

**Data Overlap Handling:**
- New exports may overlap with existing data (e.g., Jan 1-22 overlaps with existing Jan 1-11)
- Script reports overlap and deduplicates automatically
- Net rows added = new rows - duplicates

**rec_type Values:**
- `bid` - Bid placed (win or loss)
- `view` - Ad was shown (win)
- `click` - User clicked (conversion)
- Standardized to lowercase from variations (View, link, etc.)

### Current Data State (January 22, 2026)
```
Total rows: 900,111
Date range: Sep 15, 2025 → Jan 22, 2026 (129 days)
Data freshness: 0 days old

Rec type distribution:
  view: 606,705 (67.4%)
  bid: 293,144 (32.6%)
  click: 262 (0.0%)
```

### Data Loader Auto-Detection
`src/data_loader.py` automatically detects data format:
1. First checks for `drugs_data.csv` (combined format)
2. Falls back to separate files (`drugs_bids.csv`, `drugs_views.csv`, `drugs_clicks.csv`)
3. Backward compatible - no config change needed

### Files Modified for Data Management
| File | Change |
|------|--------|
| `scripts/data_manager.py` | NEW: CLI for init, ingest, info, combine |
| `src/data_loader.py` | Support combined file, refactored processing |
| `config/optimizer_config.yaml` | Added data file settings |

## Virtual Environment
- Located at `./venv/`
- Activate: `source ./venv/bin/activate`
- Run optimizer: `python run_optimizer.py --config config/optimizer_config.yaml --data-dir data_drugs/ --output-dir output/`

## Git Workflow
- Always use good commit messages
- Co-author line: `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>`
- Push after committing

## Plan File Management
- Active plans: `/Users/gaurav_kamdar/.claude/plans/foamy-frolicking-lemon.md`
- Completed work: `PLAN_ARCHIVE.md`
- Keep active plan file SHORT - only pending/future work
- Archive completed tasks with full details

## Reference
- `README.md` - Project overview and quick start guide
- `PLAN_ARCHIVE.md` - Completed work history with full implementation details
- `SCRATCHPAD.md` - Scratch notes and findings
- `scripts/data_manager.py --help` - Data management CLI help