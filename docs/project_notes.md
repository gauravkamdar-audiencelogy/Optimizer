# Project Notes

External truths about the project that don't change often. Not config values, not TODOs.

---

## Supply-Side vs Demand-Side Signals

**We are building a SUPPLY-SIDE optimizer.**

| Signal Type | Examples | Use in Optimizer |
|-------------|----------|------------------|
| Supply-side | Click count, CTR, win rate, impressions | ✓ Use |
| Demand-side | Revenue, CPC, campaign payout | ✗ Avoid |

**Why avoid demand-side signals:**
- Revenue = Click × Campaign_Payout
- Campaign_Payout depends on which advertisers are bidding
- Same user clicking generates different revenue depending on active campaigns
- Makes valuations unstable and outside our control

**Implication:** Use binary outcomes (clicked: yes/no, converted: yes/no), not monetary values.

---

## Dataset: nativo_consumer

### Schema Facts
- **rec_type values**: `bid`, `view`, `link`, `lead`
- `link` = click event
- `lead` = conversion event (binary, no value)

### Field Semantics (CRITICAL - different meaning by rec_type)

**In `bid` records:**
| Field | Meaning |
|-------|---------|
| `bid_amount` | Floor price from SSP (what we must beat) |
| `publisher_payout` | Our bid (what we offered) |

**In `view` records:**
| Field | Meaning |
|-------|---------|
| `bid_amount` | Campaign bid (demand-side, IGNORE THIS) |
| `publisher_payout` | Our bid / clearing price |

### Structural Facts
- No NPI data (consumer targeting, not HCP)
- Floor prices available (in bid records)
- `rec_type=bid` logging started **January 16, 2026** (BID feed, 14.9M records)
  - Note: RTB feed has ~10K bid records from Dec 10, 2025 (test/legacy data, ignore)
- `internal_txn_id` links bid → view → link records (same as drugs)

### Data Volume
- Full file: 14GB, ~17.3M rows
- 10% sample file: 1.4GB, ~1.7M rows
- **WARNING**: 10% random sample breaks bid-view relationships (use full file)

### EDA Findings (January 2026)
- January 2026 win rate: ~4.9% (very low vs drugs.com 30%)
- CTR: ~0.56%
- `internal_adspace_id` = -1 for 100% of records (not useful as feature)
- `geo_country_code2` = US for ~100% (not useful)
- Domains: 14K+ unique (high cardinality, may need bucketing)
- Heavy columns to drop: `ua` (2.7GB), `ref_bundle` (2.2GB), `source_filename` (1.9GB)

### Known Issues
- Full-file EDA script hangs/runs indefinitely (memory issue with 14GB file)
- Need chunked processing for full analysis

---

## Dataset: drugs

### Schema Facts
- **rec_type values**: `bid`, `view`, `click`
- `click` = click event (equivalent to `link` in nativo)
- `external_userid` contains NPI (10-digit healthcare provider ID)

### Field Semantics
- `publisher_payout` = Our bid (PostgreSQL array format `{7.50000}`)
- `internal_txn_id` = Links bid → view → click records (NOT log_txnid!)
  - Format: `{uuid}` or `{uuid1,uuid2}` for multi-slot
  - Must parse/unnest before joining

### Structural Facts
- HCP targeting (100% NPI coverage in requests)
- No floor prices in bid requests
- `rec_type=bid` logging added December 10, 2025

---

## Bidding Strategy Definitions

**What "adaptive" strategy means:**
- Per-segment automatic switching based on maturity
- Immature segments (low WR, low obs): volume_first mode
- Mature segments (WR >= threshold, obs >= threshold): margin_optimize mode
- Thresholds defined in config file

**What "volume_first" means:**
- Prioritize win rate over margin
- Accept negative margins during learning
- Bid UP on losing segments to learn market prices

**What "margin_optimize" means:**
- Prioritize profit per impression
- Use bid landscape to find optimal bid
- Only for segments with sufficient data

---

## Hierarchical IQR Tiering (January 2026)

### Why IQR over Percentiles
- **43x more stable** than percentiles (0.05% vs 2.13% flip-flop rate)
- Thresholds derived from data distribution (Q1, Q3, IQR), not arbitrary percentile cuts
- Small data changes don't shift tier boundaries
- Captures statistical outliers mathematically

### Tier Structure

**Domain Tiers (5-tier system):**
| Tier | Threshold | Multiplier | Expected % |
|------|-----------|------------|------------|
| extreme_stars | > Q3 + 3.0×IQR | 1.50x | ~3% |
| stars | > Q3 + 1.5×IQR | 1.30x | ~4% |
| cream | > Middle_Q3 + 1.5×Middle_IQR | 1.15x | ~2% |
| baseline | > poor_threshold | 1.00x | ~70% |
| poor | ≤ poor_threshold | 0.60x | ~21% |
| blocklist | < global × 0.1 | 0.00x | varies |

**NPI Tiers (higher multipliers due to direct prescriber value):**
| Tier | Threshold | Multiplier |
|------|-----------|------------|
| extreme_elite | > Q3 + 3.0×IQR | 3.00x (capped at max_multiplier) |
| elite | > Q3 + 1.5×IQR | 2.50x |
| cream | > Middle_Q3 + 1.5×Middle_IQR | 1.80x |
| baseline | > poor_threshold | 1.00x |
| poor | ≤ poor_threshold | 0.70x |

### Critical Implementation Detail: Cream Threshold

The cream threshold is computed via **recursive IQR** on the "middle" data (domains/NPIs not in stars/elite tiers).

**Bug discovered & fixed (Jan 2026):** With discrete data (integer click counts), the cream threshold could equal the elite threshold, making the cream tier unusable.

**Fix:** Ensure `cream_threshold < elite_threshold` by capping at 95% of elite (or -1 for integers).

```python
# In _compute_cream_threshold():
if raw_cream >= elite_threshold:
    return max(elite_threshold - 1, elite_threshold * 0.95)
```

### Config Settings
```yaml
# In domain or npi section:
tiering_method: iqr  # 'iqr' or 'percentile'
iqr_multiplier_stars: 1.5
iqr_multiplier_extreme: 3.0
poor_rate_factor: 0.3
min_bids_for_tiering: 30  # or min_clicks_for_tiering for NPI
```

---

## Model Performance Expectations

### AUC-ROC Interpretation

| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect discrimination |
| 0.7-0.9 | Good discrimination |
| 0.5-0.7 | Weak discrimination |
| 0.5 | Random (no predictive power) |
| < 0.5 | Usually indicates label inversion bug |

### Expected Model Performance (drugs.com dataset)

| Model | AUC-ROC | Notes |
|-------|---------|-------|
| Win Rate (Empirical) | ~0.70 | Good - used for bid calculations |
| Win Rate (LogReg) | ~0.68 | Decent - used for calibration comparison |
| CTR Model | ~0.50 | **Expected** - only 225 clicks in 577K views |

**CTR AUC near 0.5 is NOT a bug:** With only 225 positive events (0.039% rate), the model cannot learn meaningful patterns. This is data sparsity, not implementation error.

### Model Interactions (Bid Calculation Flow)

```
1. EmpiricalWinRateModel → Current segment win rate
2. Win Rate Gap = Target WR - Current WR
3. BidLandscapeModel → Derived exploration multipliers
4. Base Bid × Exploration Adjustment = Segment Bid
   (Stored in suggested_bids_*.csv)

5. At request time (by bidder):
   Final Bid = Segment Bid × NPI Multiplier × Domain Multiplier
   (NPI/Domain multipliers stored separately)
```

**Key insight:** NPI and Domain multipliers are NOT applied during segment bid calculation. They're stored in separate lookup files and applied by the bidder at request time. This keeps segment learning independent from prescriber/domain value.

---

## Data Sparsity Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| min_observations_for_empirical | 10 | Below this, use global + exploration bonus |
| min_observations_for_landscape | 50 | Below this, use heuristic instead of bid landscape |
| min_observations_for_margin | 100 | Below this, stay in volume_first mode |
| shrinkage_k | 30 | Bayesian shrinkage strength |
| min_bids_for_tiering (domain) | 30 | Below this, domain gets insufficient_data tier |
| min_clicks_for_tiering (NPI) | 5 | Below this, NPI gets insufficient_data tier |
