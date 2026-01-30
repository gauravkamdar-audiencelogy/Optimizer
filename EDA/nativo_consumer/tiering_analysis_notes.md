# Domain Tiering Analysis - Final Notes

## User Context (from discussion)

1. **Economic impact**: Domains are meaningful signal for profitability
2. **Stability preference**: Stability > accuracy, but must converge over time
3. **Run cadence**: Daily, cumulative data (not just daily snapshot)
4. **Middle segment**: Concerned about missing signal if we collapse to baseline
5. **Generalization**: OK to have different models, but sparse features should share logic
6. **Debuggability**: Interpretable output is key

---

## Key Findings from Analysis

### 1. Cumulative Data Changes Everything
- K-means "instability" is a cold-start problem, not steady-state
- With cumulative data, cluster centers stabilize over time (law of large numbers)
- Early volatility acceptable; long-term convergence guaranteed

### 2. IQR Method is 43x More Stable Than Percentiles
- Tested with small perturbation (simulating daily data addition)
- IQR: 0.05% tier changes
- Percentile: 2.13% tier changes
- Clear winner for stability

### 3. 93.4% of Domains Are "Middle"
- Only 6.6% are statistical outliers (stars)
- 0% qualify as dogs using standard IQR (lower fence goes negative)
- Trying to split the middle into 2 tiers is over-engineering

### 4. NPI Model Uses Asymmetric Approach
- Focus on TOP performers (1%, 5%, 20%)
- Everyone else is "standard"
- This is economically smarter than symmetric percentiles

---

## Recommended Approach: IQR-Based Outlier Detection

### Tier Structure (3 tiers)

| Tier | Criteria | Multiplier | Count | % |
|------|----------|------------|-------|---|
| Premium | rate > Q3 + 1.5*IQR | 1.3x | 410 | 6.6% |
| Baseline | everything else | 1.0x | 5,774 | 93.4% |
| Poor | rate < global * 0.3 | 0.6x | TBD | TBD |
| Blocklist | rate < global * 0.1 | 0.0x | TBD | TBD |

### Why This Works

1. **Stars**: IQR upper fence identifies genuine statistical outliers
2. **Dogs**: Performance threshold (% of global) for underperformers
3. **Baseline**: Don't over-segment the middle
4. **Stable**: Only clear outliers change tier → minimal flip-flopping

### Data-Derived Parameters
- Q1, Q3, IQR: Computed from data
- global_rate: Computed from data
- min_obs: 30 (configurable)

---

## Rejected Approaches

| Approach | Why Rejected |
|----------|--------------|
| 4-tier percentiles | Creates artificial distinctions in homogeneous middle |
| K-means | Unstable for edge cases, k is magic number |
| Adaptive method selection | Over-complex, hard to debug |
| Jenks Natural Breaks | O(n²) complexity, overkill for 1D |

---

## Implementation Notes

1. Same logic can apply to NPI and Domain (both sparse features)
2. Multipliers are config-driven, thresholds are data-derived
3. Output should include diagnostic metrics (Q1, Q3, IQR, fence values)
4. Consider "extreme stars" (>Q3 + 3*IQR) for even higher multiplier
