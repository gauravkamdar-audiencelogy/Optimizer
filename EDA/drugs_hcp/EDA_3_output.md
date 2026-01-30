# BATCH 3: FUNNEL & SEGMENT ANALYSIS

**Data loaded:** 830,806 records

---

## 3.1 DEDUPLICATED FUNNEL ANALYSIS

**Views Summary:**
- Before deduplication: 623,545
- After deduplication: 577,815
- Removed: 45,730 (7.3%)

**Duplicates by Month:**

| Month | Duplicates | Total | Duplicate % |
|-------|-----------|-------|-------------|
| 2025-09 | 7,832 | 106,131 | 7.38% |
| 2025-10 | 22,284 | 290,359 | 7.67% |
| 2025-11 | 9,908 | 128,953 | 7.68% |
| 2025-12 | 3,684 | 65,070 | 5.66% |
| 2026-01 | 2,022 | 33,032 | 6.12% |

---

## 3.2 CORRECTED FUNNEL (Dec-Jan)

**Aggregate Metrics:**
- Bids: 207,013
- Views (deduplicated): 92,396
- Clicks: 38
- Win Rate (Views/Bids): 44.63%
- CTR (Clicks/Views): 0.0411%

**Monthly Breakdown:**

| Month | Bids | Views | Win Rate | Clicks | CTR |
|-------|------|-------|----------|--------|-----|
| 2025-12 | 151,159 | 61,386 | 40.6% | 21 | 0.0342% |
| 2026-01 | 55,854 | 31,010 | 55.5% | 17 | 0.0548% |

---

## 3.3 WIN RATE BY NUM_ADS (Dec-Jan)

| Ad Request Type | Bids | Views | Win Rate |
|-----------------|------|-------|----------|
| 1-ad requests | 57,173 | 40,962 | 71.6% |
| 2-ad requests | 149,840 | 51,410 | 34.3% |

---

## 3.4 WIN RATE BY BID AMOUNT

**Bids by Bucket:**

| Bid Bucket | Count |
|-----------|-------|
| 00_zero | 13,257 |
| 01_under_5 | 1,828 |
| 02_5_to_7.5 | 33,834 |
| 03_exactly_7.5 | 147,739 |
| 06_15_to_20 | 6,497 |
| 07_20_plus | 3,858 |

**Views (Wins) by Payout Bucket:**

| Payout Bucket | Count |
|--------------|-------|
| 00_zero | 48 |
| 01_under_5 | 2,376 |
| 02_5_to_7.5 | 9,528 |
| 03_exactly_7.5 | 51,023 |
| 04_7.5_to_10 | 9,760 |
| 05_10_to_15 | 11,932 |
| 06_15_to_20 | 250 |
| 07_20_plus | 7,479 |

**Distribution Comparison:**

| Bucket | % of Bids | % of Wins |
|--------|----------|----------|
| 00_zero | 6.4% | 0.1% |
| 01_under_5 | 0.9% | 2.6% |
| 02_5_to_7.5 | 16.3% | 10.3% |
| 03_exactly_7.5 | 71.4% | 55.2% |
| 06_15_to_20 | 3.1% | 0.3% |
| 07_20_plus | 1.9% | 8.1% |

---

## 3.5 WIN RATE BY INTERNAL_ADSPACE_ID

| Adspace | Bids | Views | Win Rate |
|---------|------|-------|----------|
| 111563 | 47,231 | 26,236 | 55.5% |
| 111564 | 69,376 | 34,272 | 49.4% |
| 111565 | 57,558 | 18,343 | 31.9% |
| 111566 | 15,448 | 6,964 | 45.1% |
| 111567 | 6,124 | 2,838 | 46.3% |
| 111568 | 4,211 | 1,655 | 39.3% |
| 111569 | 2,885 | 909 | 31.5% |
| 111570 | 2,069 | 535 | 25.9% |
| 111571 | 2,111 | 596 | 28.2% |

---

## 3.6 WIN RATE BY GEO_REGION (Top 15 by bid volume)

| Region | Bids | Views | Win Rate |
|--------|------|-------|----------|
| California | 28,804 | 15,800 | 54.9% |
| Texas | 12,886 | 5,282 | 41.0% |
| Florida | 12,584 | 5,772 | 45.9% |
| New York | 12,112 | 5,414 | 44.7% |
| Pennsylvania | 8,537 | 3,937 | 46.1% |
| New Jersey | 7,852 | 3,618 | 46.1% |
| Virginia | 6,797 | 2,385 | 35.1% |
| Ohio | 5,893 | 2,304 | 39.1% |
| Michigan | 5,677 | 2,468 | 43.5% |
| Tennessee | 5,466 | 2,233 | 40.9% |
| North Carolina | 5,206 | 2,242 | 43.1% |
| Massachusetts | 4,883 | 2,184 | 44.7% |
| Maryland | 4,865 | 1,753 | 36.0% |
| Illinois | 4,835 | 1,966 | 40.7% |
| Georgia | 4,384 | 2,013 | 45.9% |

---

## 3.7 CTR BY FEATURES (All available data)

**Overall Summary:**
- Unique internal_txn_ids in clicks: 231
- Views with clicks: 214
- Overall CTR: 0.0370%

**CTR by internal_adspace_id:**

| Adspace | Views | Clicks | CTR % |
|---------|-------|--------|-------|
| 111564 | 208,116 | 114 | 0.0548% |
| 111563 | 175,402 | 29 | 0.0165% |
| 111565 | 107,573 | 46 | 0.0428% |
| 111566 | 45,937 | 13 | 0.0283% |
| 111567 | 18,282 | 5 | 0.0273% |
| 111568 | 10,975 | 3 | 0.0273% |
| 111569 | 5,404 | 2 | 0.0370% |
| 111570 | 3,113 | 0 | 0.0000% |
| 111571 | 2,800 | 2 | 0.0714% |
| 110201 | 167 | 0 | 0.0000% |
| 111284 | 28 | 0 | 0.0000% |
| 111285 | 7 | 0 | 0.0000% |
| 111283 | 6 | 0 | 0.0000% |
| 110349 | 4 | 0 | 0.0000% |
| 110199 | 1 | 0 | 0.0000% |

**CTR by geo_region (Top 15 by volume):**

| Region | Views | Clicks | CTR % |
|--------|-------|--------|-------|
| California | 58,853 | 25 | 0.0425% |
| New York | 37,231 | 16 | 0.0430% |
| Florida | 37,124 | 21 | 0.0566% |
| Texas | 36,205 | 8 | 0.0221% |
| Pennsylvania | 24,949 | 14 | 0.0561% |
| New Jersey | 21,106 | 12 | 0.0569% |
| Ohio | 19,172 | 5 | 0.0261% |
| Virginia | 18,356 | 1 | 0.0054% |
| North Carolina | 15,506 | 5 | 0.0322% |
| Massachusetts | 15,240 | 3 | 0.0197% |
| Michigan | 15,196 | 3 | 0.0197% |
| Maryland | 14,809 | 4 | 0.0270% |
| Georgia | 13,759 | 4 | 0.0291% |
| Illinois | 13,695 | 11 | 0.0803% |
| Tennessee | 12,804 | 6 | 0.0469% |

---

## 3.8 CPC BY CAMPAIGN

| Campaign | Clicks | Total CPC | Avg CPC | Max CPC |
|----------|--------|-----------|---------|---------|
| 471454 | 56 | $280.00 | $5.00 | $5.00 |
| 476568 | 2 | $260.00 | $130.00 | $130.00 |
| 476020 | 2 | $224.00 | $112.00 | $112.00 |
| 471645 | 28 | $140.00 | $5.00 | $5.00 |
| 476989 | 1 | $130.00 | $130.00 | $130.00 |
| 476263 | 1 | $130.00 | $130.00 | $130.00 |
| 477076 | 1 | $130.00 | $130.00 | $130.00 |
| 477052 | 1 | $112.00 | $112.00 | $112.00 |
| 476572 | 1 | $112.00 | $112.00 | $112.00 |
| 476324 | 12 | $93.00 | $7.75 | $7.75 |
| 476471 | 6 | $90.00 | $15.00 | $15.00 |
| 473839 | 2 | $76.00 | $38.00 | $48.00 |
| 475475 | 1 | $75.00 | $75.00 | $75.00 |
| 476473 | 8 | $62.00 | $7.75 | $7.75 |
| 476557 | 2 | $60.00 | $30.00 | $30.00 |

---

## 3.9 HIGH-VALUE CLICKS (CPC >= $50) ANALYSIS

**Summary:**
- Total high-value clicks: 10
- Total revenue from high-value clicks: $1,173.00

**Distribution by geo_region:**

| Region | Count |
|--------|-------|
| Utah | 2 |
| California | 1 |
| Tennessee | 1 |
| Alabama | 1 |
| Nebraska | 1 |

**Distribution by browser_code:**

| Browser | Count |
|---------|-------|
| 14 | 9 |
| 100 | 1 |

**Distribution by internal_adspace_id:**

| Adspace | Count |
|---------|-------|
| 111564 | 5 |
| 111565 | 3 |
| 111569 | 1 |
| 111566 | 1 |

**Distribution by campaign:**

| Campaign | Count |
|----------|-------|
| 476020 | 2 |
| 476568 | 2 |
| 475475 | 1 |
| 476263 | 1 |
| 476572 | 1 |

---

## 3.10 ZERO BID ANALYSIS

**Summary:**
- Zero bids: 13,257 (6.4%)
- Non-zero bids: 193,756
- Zero bids that resulted in views: 0

**Zero bids by month:**

| Month | Count |
|-------|-------|
| 2025-12 | 13,257 |

**Zero bids by internal_adspace_id:**

| Adspace | Count |
|---------|-------|
| 111564 | 4,368 |
| 111565 | 3,651 |
| 111563 | 3,003 |
| 111566 | 1,025 |
| 111567 | 395 |
| 111568 | 302 |
| 111569 | 203 |
| 111571 | 159 |
| 111570 | 151 |

---

## 3.11 FEATURE COMBINATION COVERAGE

**Single Feature Coverage:**
- internal_adspace_id: 9 unique values (0.0% null)
- geo_region_name: 66 unique values (11.4% null)
- browser_code: 9 unique values (0.0% null)

**Two-Feature Combinations:**
- internal_adspace_id + geo_region_name: 525 unique combinations
  - Top 10 combos cover 24.9%
  - Top 50 cover 56.5%
- internal_adspace_id + browser_code: 75 unique combinations
  - Top 10 combos cover 94.9%
  - Top 50 cover 99.9%
- geo_region_name + browser_code: 256 unique combinations
  - Top 10 combos cover 48.8%
  - Top 50 cover 83.8%

**Three-Feature Combination:**
- All 3 features: 1,589 unique combinations
  - Top 10 combos cover 23.7%
  - Top 50 combos cover 53.6%
  - Top 100 combos cover 68.6%

**Combination Frequency Distribution:**

| Frequency | Count |
|-----------|-------|
| ≥ 1 | 1,589 |
| ≥ 2 | 1,261 |
| ≥ 3 | 1,093 |
| ≥ 5 | 882 |
| ≥ 10 | 654 |
| ≥ 50 | 318 |
| ≥ 100 | 218 |
| ≥ 500 | 84 |
| ≥ 1000 | 47 |

---
