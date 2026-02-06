# System Config Reference Manual

**Purpose:** Technical documentation of system-level configs, how they affect model behavior, and when to adjust them.

**Last Updated:** 2026-02-05

---

## 1. Model Parameters

### 1.1 Shrinkage (shrinkage_k)

| Setting | Value |
|---------|-------|
| **Location** | `advanced.shrinkage_k` (default: 30) |
| **Override** | `domain.shrinkage_k` for domain model |

**Formula:**
```
shrunk_rate = (observations × empirical_rate + k × global_rate) / (observations + k)
```

**Effect:**
- Higher k = more conservative (pull segment rates toward global mean)
- Lower k = more aggressive (trust segment-level data more)

**When to adjust:**
- Increase if seeing too much variance in sparse segments
- Decrease if model is too conservative for data-rich segments

---

### 1.2 Calibration Gate

| Setting | Value |
|---------|-------|
| **Location** | `advanced.calibration_ece_threshold` (default: 0.10) |

**Effect:**
- If model ECE (Expected Calibration Error) > threshold, fall back to empirical model
- Lower threshold = stricter calibration requirements

**Gate behavior:**
```python
if model_ece > threshold:
    use_empirical_model()  # Fall back to empirical
else:
    use_calibrated_model()
```

**When to adjust:**
- Tighten (lower) for production systems where calibration is critical
- Relax (higher) during development or when data is sparse

---

### 1.3 Observation Thresholds

| Config | Default | Purpose |
|--------|---------|---------|
| `min_observations` | 1 | Include segment in output |
| `min_observations_for_empirical` | 10 | Use empirical model vs global rate |
| `min_observations_for_landscape` | 50 | Use bid landscape model vs formula |

**Decision flow:**
```
observations < 1              → Exclude segment
observations < 10             → Use global rate
observations < 50             → Use empirical model
observations >= 50            → Use bid landscape model
```

---

## 2. Feature Selection

| Config | Default | Purpose |
|--------|---------|---------|
| `min_signal_score` | 50.0 | Min score for feature to be selected |
| `max_features` | 3 | Max features per model |
| `min_coverage_at_threshold` | 0.5 | Min data coverage required |
| `max_null_pct` | 30.0 | Max null percentage allowed |
| `min_effective_cardinality` | 2 | Min distinct values required |

**Signal score components:**
- AUC (discrimination power)
- Coverage (% of data represented)
- Cardinality (number of unique values)

**When to adjust:**
- Lower `min_signal_score` if too few features selected
- Increase `max_features` if model underperforms (careful: more features = more segments = sparser data)

---

## 3. Exploration Presets

Controlled by `technical.aggressive_exploration` toggle.

### 3.1 Gradual (Conservative)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bonus_zero_obs` | 0.25 | +25% bid for unknown segments |
| `bonus_low_obs` | 0.15 | +15% for 1-9 observations |
| `bonus_medium_obs` | 0.08 | +8% for 10-49 observations |
| `max_adjustment` | 1.4 | Max 40% total adjustment |
| `max_bid_cpm` | $20 | Bid ceiling |

### 3.2 Aggressive (Fast Learning)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bonus_zero_obs` | 0.50 | +50% bid for unknown segments |
| `bonus_low_obs` | 0.35 | +35% for 1-9 observations |
| `bonus_medium_obs` | 0.15 | +15% for 10-49 observations |
| `max_adjustment` | 1.8 | Max 80% total adjustment |
| `max_bid_cpm` | $30 | Bid ceiling |

### 3.3 Exploration Zones

| Zone | Observations | Bonus Applied | Rationale |
|------|-------------|---------------|-----------|
| Unknown | 0 | `bonus_zero_obs` | No data - explore aggressively |
| Low | 1-9 | `bonus_low_obs` | Limited data - still learning |
| Medium | 10-49 | `bonus_medium_obs` | Some signal - moderate exploration |
| Mature | 50+ | 0 (no bonus) | Enough data - exploit knowledge |

---

## 4. NPI Tiering (HCP Only)

Applies when `npi.enabled: true` and `npi.tiering_method: iqr`.

### 4.1 IQR Threshold Parameters

| Config | Default | Description |
|--------|---------|-------------|
| `iqr_multiplier_stars` | 1.5 | Elite threshold = Q3 + 1.5×IQR |
| `iqr_multiplier_extreme` | 3.0 | Extreme threshold = Q3 + 3.0×IQR |
| `poor_rate_factor` | 0.3 | Poor = bottom 30th percentile |

### 4.2 Tier Multipliers

| Tier | Multiplier | Threshold |
|------|-----------|-----------|
| Extreme Elite | 3.00x (capped at `max_multiplier`) | > Q3 + 3.0×IQR |
| Elite | 2.50x | > Q3 + 1.5×IQR |
| Cream | 1.80x | > Middle_Q3 + 1.5×Middle_IQR |
| Baseline | 1.00x | > poor_threshold |
| Poor | 0.70x | ≤ poor_threshold |

### 4.3 Other NPI Settings

| Config | Default | Description |
|--------|---------|-------------|
| `max_multiplier` | 3.0 | Hard cap on NPI multiplier |
| `recency_boost` | 1.2 | +20% for recent clicks |
| `min_clicks_for_tiering` | 5 | Min clicks to assign tier |

---

## 5. Domain Tiering (Consumer)

Applies when `domain.enabled: true` and `domain.tiering_method: iqr`.

### 5.1 Tier Multipliers

| Tier | Multiplier | Threshold |
|------|-----------|-----------|
| Extreme Stars | 1.50x | > Q3 + 3.0×IQR |
| Stars | 1.30x | > Q3 + 1.5×IQR |
| Cream | 1.15x | > Middle_Q3 + 1.5×Middle_IQR |
| Baseline | 1.00x | > poor_threshold |
| Poor | 0.60x | ≤ poor_threshold |
| Blocklist | 0.00x | < global × `blocklist_rate_factor` |

### 5.2 Blocklist Criteria

| Config | Default | Description |
|--------|---------|-------------|
| `blocklist_rate_factor` | 0.1 | Block if rate < global × 0.1 |
| `min_bids_for_blocklist` | 100 | Need enough data to judge |
| `min_bids_for_tiering` | 30 | Min bids to assign tier |

---

## 6. Validation Rules

Controlled by `validation.enabled`. Always-on in production.

### 6.1 Hard Rules (Block Deployment)

| Rule | Default | Description |
|------|---------|-------------|
| `coverage_min_pct` | 80.0 | Min % segments vs previous run |
| `calibration_ece_max` | 0.15 | Max ECE for models used |
| `bid_floor_respected` | true | All bids >= min_bid |
| `bid_ceiling_respected` | true | All bids <= max_bid |

**If any hard rule fails:** Deployment blocked. Must fix and re-run.

### 6.2 Soft Rules (Warn Only)

| Rule | Default | Description |
|------|---------|-------------|
| `bid_median_change_max_pct` | 50.0 | Max median bid change vs previous |
| `pct_at_floor_max` | 30.0 | Max % of bids at floor |
| `pct_at_ceiling_max` | 30.0 | Max % of bids at ceiling |
| `pct_profitable_min` | 40.0 | Min % of profitable segments |

**If soft rule fails:** Warning logged but deployment allowed.

---

## 7. Adaptive Bidding Strategy

Controlled by `bidding.strategy: adaptive`.

### 7.1 Mode Switching

| Mode | Condition | Behavior |
|------|-----------|----------|
| Volume | Default | Optimize for win rate |
| Margin | WR >= 55% AND obs >= 100 | Optimize for profit margin |

### 7.2 Switching Thresholds

| Config | Default | Description |
|--------|---------|-------------|
| `min_win_rate_for_margin` | 0.55 | WR threshold to switch |
| `min_observations_for_margin` | 100 | Obs threshold to switch |

**Why adaptive?**
- New segments need volume to learn market prices
- Mature segments can optimize for margin once we understand the landscape

---

## 8. Bid Calculation Flow

```
1. Get segment empirical win rate
2. Apply Bayesian shrinkage (if sparse)
3. Calculate base bid from bid landscape
4. Apply exploration bonus (based on zone)
5. Apply NPI multiplier (if HCP)
6. Apply domain multiplier (if Consumer)
7. Clamp to [min_bid, max_bid]
```

---

## 9. Quick Reference

### Must-Know Defaults

| Config | Default | Impact |
|--------|---------|--------|
| `shrinkage_k` | 30 | Segment rate smoothing |
| `calibration_ece_threshold` | 0.10 | Model trust threshold |
| `min_signal_score` | 50.0 | Feature selection gate |
| `max_features` | 3 | Segment granularity |
| `aggressive_exploration` | false | Learning speed vs risk |

### File Locations

| Config Level | Location |
|-------------|----------|
| Entity-specific | `config/optimizer_config_{entity}.yaml` |
| User-facing | MySQL `opt_run_configs` table (per run) |
| System defaults | `src/config.py` dataclasses |

---

*End of System Config Reference*
