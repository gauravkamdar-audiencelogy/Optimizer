# UPDATE_3: Reintroduce Win Rate Signal & Data-Driven Feature Selection

**Purpose**: Fix two conceptual errors in V2 that compromise the optimizer's ability to align with market reality  
**Priority**: HIGH - V2 is deployable but fundamentally incomplete without win rate signal  
**Philosophy**: Use observed data, not model predictions, when models fail

---

## Executive Summary

### Two Problems Identified

| Problem | What Went Wrong | Impact |
|---------|-----------------|--------|
| Win rate removal | Threw out the signal because the MODEL was bad | Blind to market prices |
| Hardcoded feature exclusions | Data-driven decisions made by human, not algorithm | Violates core principle |

### The Insight

**V2 conflated two things:**
1. The LogisticRegression model's predictions → BAD (ECE=0.176)
2. The actual win rate signal from data → STILL USEFUL

We threw out the baby with the bathwater. We should have replaced the bad model with observed empirical rates, not removed win rate entirely.

---

## CHANGE 1: Reintroduce Win Rate via Empirical Segment-Level Rates

### The Problem

V2's bid formula ignores win rate:
```
bid = EV × (1 - margin)
```

This makes us blind to market dynamics. Consider two segments with identical CTR:
- Segment A: 80% win rate at current bids (we're overpaying)
- Segment B: 10% win rate at current bids (we're losing valuable inventory)

V2 bids the same for both. That's economically wrong.

### Why the LogisticRegression Model Failed

The model tried to GENERALIZE win rates to unseen feature combinations. With:
- 193,756 training samples
- 21,825+ unique segments
- Many segments with <10 observations

The model couldn't learn reliable patterns. It consistently overestimated by ~1.7x.

### The Solution: Empirical Rates with Shrinkage

Instead of predicting win rate, we OBSERVE it directly:

```
For each segment:
    observed_wins = count of views matching bids in segment
    observed_bids = count of bids in segment
    
    # Bayesian shrinkage (same as CTR model)
    k = 30  # shrinkage strength
    global_win_rate = 0.304  # from data
    
    shrunk_win_rate = (observed_wins + k × global_win_rate) / (observed_bids + k)
```

**Why this works:**
1. No model predictions, just counting
2. Shrinkage handles low-sample segments (pulls toward global average)
3. Same technique achieved ECE=0.00003 for CTR

**Literature support:**
- Thompson Sampling with Bayesian updating (Chapelle & Li, 2011, "An Empirical Evaluation of Thompson Sampling")
- Empirical Bayes for sparse data (Efron, 2010, "Large-Scale Inference")
- Same shrinkage approach used in Facebook's CTR system (McMahan et al., KDD 2013)

### The Updated Bid Formula

```
# Expected Value
EV = P(click) × E[CPC] × 1000  # Convert to CPM

# Win Rate Adjustment (using empirical rates)
# If win rate is HIGH → we're overpaying → bid LOWER
# If win rate is LOW → we're losing auctions → bid HIGHER
target_win_rate = 0.50  # We want to win ~50% of bids
win_rate_adjustment = 1 + (target_win_rate - empirical_win_rate) × sensitivity

# sensitivity controls how aggressively we adjust
# 0.5 means: if we're 20pp below target, bid 10% higher
sensitivity = 0.5

# Final bid
bid = EV × (1 - margin) × win_rate_adjustment
```

### Key Difference from V1

| Aspect | V1 (Failed) | V3 (Proposed) |
|--------|-------------|---------------|
| Win rate source | ML model predictions | Observed segment counts |
| Calibration | ECE=0.176 (bad) | Empirical (exact by definition) |
| Generalization | Tries to predict unseen | Only uses what we've observed |
| Complexity | Feature engineering + LogReg | Simple counting + shrinkage |

### Why This Is Economically Correct

The bid shading literature (VerizonMedia, arXiv:2009.09259) shows optimal bid:

```
optimal_bid = argmax_b [ (value - b) × P(win|b) ]
```

V2 removed P(win|b) entirely. That's mathematically incorrect for auction optimization. We need SOME signal about win likelihood, even if imperfect.

### Confidence: 80%

This approach should work because:
- Identical technique (shrinkage) worked for CTR
- We're using observed data, not model predictions
- Theoretically required for optimal bidding

Risk: Low-volume segments will still be noisy, but shrinkage handles this gracefully.

---

## CHANGE 2: Data-Driven Feature Exclusion

### The Problem

Current config hardcodes:
```yaml
exclude_features:
  - geo_country_code2    # "98% US, not useful"
  - hour_of_day          # "Low signal score"
  - day_of_week          # "Low signal score"
```

These are data-driven decisions made by a human looking at EDA output. This violates our core principle:

> "The model has to be such that we just give it the input tables, rest it figures out by data driven decisions which features to keep"

### The Solution

**Split exclusions into two categories:**

1. **Hard Exclusions** (user-configured, business/technical reasons):
   - `geo_postal_code` - Too sparse for stable segments (7,027 unique)
   - `geo_city_name` - Same issue (3,333 unique)
   - Any PII columns
   - Technical columns (IDs, timestamps used for joins)

2. **Soft Exclusions** (algorithm-determined, data-driven):
   - Features with signal_score below threshold
   - Features where no value has sufficient observations
   - Features that would create too many sparse segments

### Updated Feature Selection Algorithm

```
Input: all columns from data, hard_exclude list from config
Output: selected features for memcache key

1. FILTER: Remove hard_exclude columns (user-specified, never use)

2. SCORE: For each remaining candidate feature:
   - Calculate cardinality
   - Calculate null percentage
   - Calculate win_rate variance across values
   - Calculate signal_score = variance × (1 - null%) × log(cardinality)
   - Calculate coverage (% of data with sufficient observations per value)

3. SOFT EXCLUDE: Automatically exclude features where:
   - signal_score < minimum_signal_threshold (e.g., 50)
   - coverage_at_threshold < 0.5 (too sparse)
   - null_pct > 30% (too many missing values)
   
   Log these exclusions with reasons (for transparency)

4. SELECT: Greedily add features by signal_score until:
   - max_features reached, OR
   - combined coverage drops below threshold

5. OUTPUT: Selected features + report of what was excluded and why
```

### Config Changes

**Before (V2):**
```yaml
exclude_features:
  - geo_postal_code
  - geo_city_name
  - geo_country_code2    # DATA-DRIVEN, shouldn't be here
  - hour_of_day          # DATA-DRIVEN, shouldn't be here
  - day_of_week          # DATA-DRIVEN, shouldn't be here
```

**After (V3):**
```yaml
# Hard exclusions: columns that should NEVER be used as features
# (business/technical reasons, not signal-based)
hard_exclude_features:
  - geo_postal_code      # Too sparse (7,027 unique)
  - geo_city_name        # Too sparse (3,333 unique)
  - log_txnid            # Technical: join key
  - internal_txn_id      # Technical: join key
  - log_dt               # Technical: timestamp

# Soft exclusion thresholds (algorithm uses these to auto-exclude)
feature_selection:
  min_signal_score: 50           # Below this = auto-exclude
  min_coverage_at_threshold: 0.5 # Must cover 50% of data
  max_null_pct: 30               # Above this = auto-exclude
```

### What Gets Logged

When the algorithm auto-excludes a feature:

```
Feature Selection Report:
  
  INCLUDED:
    internal_adspace_id: score=93.89, coverage=100%
    geo_region_name: score=152.17, coverage=99.9%
    os_code: score=80.47, coverage=100%
  
  AUTO-EXCLUDED (data-driven):
    hour_of_day: score=22.86 < min_threshold(50)
    day_of_week: score=12.11 < min_threshold(50)
    geo_country_code2: coverage=99.9% but only 5 values with sufficient obs
  
  HARD-EXCLUDED (config):
    geo_postal_code: in hard_exclude list
    geo_city_name: in hard_exclude list
```

This gives transparency: user sees what was excluded and WHY, without having to make those decisions manually.

---

## CHANGE 3: Add Target Win Rate Back to Config

### Reasoning

V2 removed `target_win_rate` because we couldn't optimize for it with a miscalibrated model. With empirical win rates, we can now use it meaningfully.

### What It Controls

```
target_win_rate = 0.50  # We want to win 50% of our bids

If empirical win rate = 0.70 (winning too much):
  → We're probably overbidding
  → Adjustment factor < 1.0
  → Bid gets reduced

If empirical win rate = 0.30 (winning too little):
  → We're probably underbidding
  → Adjustment factor > 1.0
  → Bid gets increased
```

### Config Update

```yaml
business:
  target_margin: 0.30        # Bid shading (keep from V2)
  target_win_rate: 0.50      # REINTRODUCE: target win rate
  win_rate_sensitivity: 0.5  # How aggressively to adjust (0.3-0.7 range)
```

---

## Summary of V3 Changes

| Component | V2 State | V3 Change | Rationale |
|-----------|----------|-----------|-----------|
| Win rate signal | Removed entirely | Reintroduce via empirical segment rates | Can't optimize bids without market signal |
| Win rate model | LogisticRegression (bad calibration) | Simple counting + shrinkage | Observed data > model predictions |
| Feature exclusions | Mix of hard + data-driven in config | Split into hard (config) vs soft (algorithm) | Algorithm should make data-driven decisions |
| target_win_rate | Removed | Reintroduce | Now usable with empirical rates |

---

## Expected Impact

| Metric | V2 | V3 Expected | Reasoning |
|--------|-------|-------------|-----------|
| Market alignment | None (blind to prices) | Yes (uses observed win rates) | Win rate signal tells us market clearing prices |
| User configuration | Must decide feature exclusions | Only hard exclusions | Algorithm handles data-driven decisions |
| Bid stability | Stable but suboptimal | Stable AND market-aware | Empirical rates are stable, model predictions aren't |
| Segments in memcache | 87 | ~150-300 | Win rate adjustment may qualify more segments |

---

## Philosophical Note

### V2's Mistake

V2 said: "The win rate model is bad, so let's ignore win rate."

This is like saying: "My thermometer is broken, so I'll ignore temperature when deciding what to wear."

### V3's Correction

V3 says: "The win rate model is bad, so let's use a better estimate: observed empirical rates."

We don't need to PREDICT win rate if we can OBSERVE it directly for each segment.

### The Core Principle

> "The model must reflect the objective economic reality from the data."

Economic reality includes:
1. How much a click is worth (CTR × CPC) ✓ V2 had this
2. What the market is willing to pay (win rate signal) ✗ V2 was missing this

V3 completes the picture.

---

## Implementation Priority

1. **HIGH**: Empirical win rate calculation with shrinkage
2. **HIGH**: Updated bid formula with win rate adjustment
3. **MEDIUM**: Split feature exclusions into hard vs soft
4. **LOW**: Enhanced logging for feature selection decisions

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Empirical win rates too noisy | Low | Shrinkage handles this (proven with CTR) |
| Win rate adjustment overshoots | Medium | Use conservative sensitivity (0.5) + clip adjustment to [0.8, 1.2] |
| Algorithm excludes useful features | Low | Log all decisions, allow human override |

---

## Test Plan for V3

1. Run optimizer with V3 changes
2. Compare bid distribution to V2:
   - Expect more variation (segments adjusted by win rate)
   - Expect fewer segments at floor (win rate adjustment creates natural spread)
3. Verify empirical win rates match actual data (should be exact, by definition)
4. Check that low-signal features are auto-excluded without config changes

---

## Appendix: Why Empirical > Predicted

The LogisticRegression model tried to answer: "Given features X, what's P(win)?"

This requires learning patterns across segments. With:
- High segment cardinality (21,825 unique)
- Low samples per segment (median ~9 bids per segment)
- Class imbalance (30% wins)

The model couldn't generalize reliably.

**Empirical rates ask a simpler question:** "For this exact segment, how often did we win?"

This doesn't require generalization. It's just arithmetic:
```
win_rate(segment) = wins(segment) / bids(segment)
```

For segments with few observations, shrinkage pulls toward global average. This is mathematically sound and computationally trivial.

**The tradeoff:**
- Predicted: Can estimate win rate for unseen segments, but poorly calibrated
- Empirical: Can only estimate for seen segments, but well calibrated

Given our data volume, empirical wins.
