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
- **Target**: 60-70% win rate (currently ~44%)

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

## Reference
- `PIPELINE_SPEC.md` - Full technical specification and code examples
- `EDA_*.md` - Data analysis findings
- `SCRATCHPAD.md` - Feel free to use this to make notes for yourself and your findings. Use it to offload things from working memory, and context into this file for you to come back and refernce it later.