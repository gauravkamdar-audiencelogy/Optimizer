# Plan Archive - Completed Work

## V5 Implementation (COMPLETED - Jan 2026)

### Summary
Replaced margin-maximizing V4 with **volume-first** strategy using asymmetric exploration.

**Core Principle:**
- Losing segments (WR < 65%) → Bid HIGHER to learn ceiling
- Winning segments (WR > 65%) → Bid LOWER to find floor
- Target: 65% win rate to learn the full bid landscape

**Why V4 Was Wrong:**
- V4 formula `argmax_b [(EV - b) × P(win)]` maximizes margin → outputs $2-3 bids
- This would CUT volume in half during learning phase

### Files Created/Modified
| File | Change |
|------|--------|
| `src/bid_calculator_v5.py` | NEW: Asymmetric exploration calculator |
| `src/models/npi_value_model.py` | NEW: NPI value lookup |
| `src/config.py` | V5 params, remove external_userid from exclusions |
| `config/optimizer_config.yaml` | V5 parameters |
| `src/memcache_builder.py` | Include all segments, add bid_summary |
| `run_optimizer.py` | Integrate V5 calculator |
| `src/metrics_reporter.py` | Exploration direction metrics |

### V5 Results
- 1,454 segments (vs 93 in V4 - 15x more)
- Bid median: $7.25 (vs $2.82 in V4 - 2.5x higher)
- 98.8% segments bid UP (global WR 30.3% < target 65%)

---

## Tiered Bucket Summary (COMPLETED - Jan 2026)

Added `bid_summary_*.csv` output grouping segments into bid buckets:
- $2-5, $5-8, $8-12, $12+

Shows: segment count, %, avg bid, avg obs, avg WR, exploration direction breakdown.

---

## Config Evaluation (COMPLETED - Jan 2026)

### max_features=3 → OPTIMAL
- 4th best feature (hour_of_day) scores 21.80 vs 50.0 threshold
- Adding it would explode segments 1,454 → ~35,000
- Limit bounded by signal quality, not arbitrary cap

### min_observations=1 → OPTIMAL
- Low-obs segments = 43% of segments but only 1.1% of observations
- Win rates stable across tiers (~31%)
- Empirical model ECE = 0.0143 (well-calibrated)
- Shrinkage (k=30) handles uncertainty

---

## Memcache Contract Fix (COMPLETED - Jan 2026)

**Rule:** Memcache can ONLY have features + suggested_bid_cpm (no metadata)

Created separate files:
- `segment_analysis_*.csv` - Full metadata per segment
- `bid_summary_*.csv` - Aggregated tiered overview

---

## NPI Value Model Integration (COMPLETED - Jan 2026)

### Goal
Bid higher for high-value prescribers based on historical click/revenue data.

### Data Sources
- `NPI_click_data_1year.csv`: 64,783 NPIs with historical RPU
- `NPI_click_data_20days.csv`: 6,770 recent clickers

### Tiering Logic (Percentile-based on 1-year RPU)
| Tier | Percentile | RPU Threshold | Base Multiplier | With Recency |
|------|------------|---------------|-----------------|--------------|
| 1 (Elite) | Top 1% | >$135 | 2.5x | 3.0x |
| 2 (High) | Top 5% | >$45 | 1.8x | 2.16x |
| 3 (Medium) | Top 20% | >$15 | 1.3x | 1.56x |
| 4 (Standard) | Rest | <$15 | 1.0x | 1.2x |

### Key Findings
- **Value concentration**: Top 5% NPIs = 42% of revenue
- **Recency signal**: Recent clickers avg $17.98 RPU vs $11.69 (54% higher)
- Recency boost: +20% for NPIs that clicked in last 20 days

### Output
- `npi_multipliers_*.csv`: NPI → multiplier lookup for bidder
- Bidder applies: `final_bid = segment_bid × npi_multiplier`

### Files Modified
| File | Change |
|------|--------|
| `src/models/npi_value_model.py` | Add `from_click_data()`, tiering, recency |
| `src/config.py` | Add npi_1year_path, npi_20day_path, npi_max_multiplier |
| `config/optimizer_config.yaml` | Configure NPI data paths |
| `src/memcache_builder.py` | Add `write_npi_cache()` method |
| `run_optimizer.py` | Load NPI model, output NPI cache |

### Results
- 64,783 NPIs with multipliers
- 710 Tier 1 (1.1%), 2,583 Tier 2 (4.0%), 11,084 Tier 3 (17.1%)
- 6,770 recent clickers get +20% boost
- Avg multiplier: 1.12x, Max: 3.0x
