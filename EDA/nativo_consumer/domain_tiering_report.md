# Domain Tiering Analysis Report

Generated: 2026-01-29 17:12:05

## Executive Summary

**Recommendation: Consider K-Means**

K-means creates more homogeneous tiers (within-var: 0.000000 vs 0.000001) with better separation.

---

## Distribution Summary

| Metric | Value |
|--------|-------|
| Total domains | 15,041 |
| Mean CTR | 0.1149% |
| Median CTR | 0.1177% |
| Std Dev | 0.1191% |
| Range | 0.0015% - 3.4497% |
| Skewness | 20.15 |
| Kurtosis | 462.81 |

### Shape Interpretation

- **Skewness (20.15)**: Distribution is right-skewed (long tail of high performers)
- **Kurtosis (462.81)**: Distribution has heavy tails (more outliers than normal)

### Key Percentiles

| Percentile | CTR Value |
|------------|-----------|
| 1st | 0.0171% |
| 5th | 0.0579% |
| 10th | 0.0861% |
| 25th | 0.1104% |
| 50th (median) | 0.1177% |
| 75th | 0.1177% |
| 90th | 0.1177% |
| 95th | 0.1177% |
| 99th | 0.1177% |

### Outliers

- Low outliers (< Q1 - 1.5×IQR): 2,204
- High outliers (> Q3 + 1.5×IQR): 113

---

## Percentile Tiers (Current Implementation)

| Tier | Threshold | Domain Count | % of Total |
|------|-----------|--------------|------------|
| Premium (Top 5%) | >= 0.1177% | 8,093 | 53.8% |
| Standard (5-50%) | >= 0.1177% | 0 | 0.0% |
| Below Avg (50-90%) | >= 0.0861% | 5,519 | 36.7% |
| Poor (Bottom 10%) | < 0.0861% | 1,429 | 9.5% |

---

## K-Means Analysis

### Optimal k Detection

| Method | Suggested k | Notes |
|--------|-------------|-------|
| Elbow | 3 | Point where adding clusters yields diminishing returns |
| Silhouette | 2 | Highest cluster separation score |
| Current | 4 | Matches 4-tier percentile system |

### Clustering Quality by k

| k | Inertia | Silhouette | Calinski-Harabasz |
|---|---------|------------|-------------------|
| 2 | 0 | 0.990 | 60,534 |
| 3 | 0 | 0.985 | 85,019 |
| 4 | 0 | 0.846 | 82,475 |
| 5 | 0 | 0.850 | 104,124 |
| 6 | 0 | 0.856 | 125,425 |
| 7 | 0 | 0.856 | 142,235 |
| 8 | 0 | 0.857 | 153,998 |

### K-Means Cluster Centers (k=4)

| Cluster | Center CTR | Domain Count |
|---------|------------|--------------|
| 0 | 0.0487% | 1,297 |
| 1 | 0.1144% | 13,691 |
| 2 | 1.3399% | 36 |
| 3 | 3.0001% | 17 |

---

## Method Comparison

| Metric | Percentile | K-Means | Winner |
|--------|------------|---------|--------|
| Within-tier variance | 0.000001 | 0.000000 | K-Means |
| Between-tier separation | 0.000000 | 0.000001 | K-Means |
| Agreement | 45.4% | - | - |

### Interpretation

- **Within-tier variance**: Lower is better (domains in same tier are more similar)
- **Between-tier separation**: Higher is better (tiers are more distinct)
- **Agreement**: How often both methods assign the same tier

---

## Tier Assignment Comparison

| Tier | Percentile | K-Means | Difference |
|------|------------|---------|------------|
| Poor | 1,429 | 1,297 | -132 |
| Below Avg | 5,519 | 13,691 | +8,172 |
| Standard | 0 | 36 | +36 |
| Premium | 8,093 | 17 | -8,076 |

---

## Recommendation

### **Consider K-Means**

**Reasoning:** K-means creates more homogeneous tiers (within-var: 0.000000 vs 0.000001) with better separation.

### Decision Criteria Applied

| Criterion | Result |
|-----------|--------|
| Agreement > 90% | ✗ (45.4%) |
| K-means better within-var | ✓ |
| K-means better separation | ✓ |
| Elbow at k=4 | ✗ (elbow at k=3) |
| Silhouette at k=4 | ✗ (best at k=2) |

---

## Files Generated

- `domain_tiering_plots.png` - Visual diagnostics
- `domain_tiering_report.md` - This report

## Next Steps

1. Consider updating `domain_value_model.py` to support k-means mode
2. Add `tiering_method: kmeans` config option
3. Test impact on bid distribution and win rates
4. Monitor tier stability over time with k-means
