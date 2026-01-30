# EDA Batch 2: Deep Dive Analysis

## Execution Command
```bash
python3 EDA_2.py
```

---

## Data Loading Summary

- **Total Records Loaded**: 830,806
- **Processing Focus**: Deep analysis of NPIs, duplicates, payouts, bids, and funnels

---

## 2.1 NPI Validation (Fixed)

### Validation Results by Record Type

| Record Type | Valid NPI Count | Total Count | % Valid |
|-------------|-----------------|-------------|---------|
| View        | 623,316         | 623,545     | 99.96%  |
| bid         | 207,013         | 207,013     | 100.00% |
| link        | 248             | 248         | 98.79%  |

### Invalid NPI Sample
```
Sample invalid NPIs (first 10): [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
```

### Analysis
- **Near-perfect validation**: 99.96%+ of records have valid NPI format
- **Invalid records**: Primarily null/NaN values (232 total across Views and Clicks)
- **Bid records**: 100% valid NPIs

---

## 2.2 View Duplicate Analysis

### Duplicate Summary
- **Total View Records**: 623,545
- **Unique log_txnids**: 577,815
- **Duplicated log_txnids**: 19,325
- **Records Involved in Duplicates**: 65,055 (10.4% of all views)

### Duplicate Count Distribution

| Duplicate Count | Number of log_txnids |
|-----------------|----------------------|
| 2               | 9,202                |
| 3               | 5,480                |
| 4               | 1,505                |
| 5               | 1,051                |
| 6               | 533                  |
| 7               | 422                  |
| 8               | 313                  |
| 9               | 204                  |
| 10              | 153                  |
| 11              | 104                  |

### Sample Duplicate Analysis
**Sample log_txnid**: `a147c179-af7a-413c-8f5d-907af62bf360`
- **Number of Rows**: 35
- **All key fields identical** across duplicate rows:
  - log_dt: 1 unique value (2025-09-19 14:24:13.758000+0000)
  - internal_txn_id: 1 unique value ({c7824a3e-9153-4337-b4e8-13105a0c7edb})
  - internal_adspace_id: 1 unique value (111563)
  - publisher_payout: 1 unique value ({7.50000,7.50000})
  - campaign_code: 1 unique value ({471454,471645})
  - external_userid: 1 unique value (1912409715.0)

### Key Observation
- Duplicates are **exact copies** of the same event, suggesting potential logging/ingestion issues
- All duplicates share identical timestamps and transaction IDs

---

## 2.3 Publisher Payout Analysis

### Overall Payout Distribution (CPM) for Views

| Statistic | Value ($) |
|-----------|-----------|
| Count     | 623,545   |
| Mean      | $7.73     |
| Median    | $7.50     |
| Std Dev   | $1.75     |
| Min       | $0.00     |
| 25th %    | $7.50     |
| 75th %    | $7.50     |
| Max       | $25.00    |

### Top 10 Payout Values

| Payout Value | Record Count | Percentage |
|--------------|--------------|------------|
| $7.50        | 579,538      | 92.94%     |
| $6.00        | 10,629       | 1.70%      |
| $12.50       | 10,347       | 1.66%      |
| $8.00        | 10,281       | 1.65%      |
| $20.00       | 4,792        | 0.77%      |
| $25.00       | 2,538        | 0.41%      |
| $10.00       | 1,997        | 0.32%      |
| $3.00        | 1,436        | 0.23%      |
| $2.25        | 599          | 0.10%      |
| $4.50        | 288          | 0.05%      |

### Average Payout by Month

| Month   | Count   | Mean ($) | Median ($) |
|---------|---------|----------|------------|
| 2025-09 | 106,131 | $7.49    | $7.50      |
| 2025-10 | 290,359 | $7.50    | $7.50      |
| 2025-11 | 128,953 | $7.50    | $7.50      |
| 2025-12 | 65,070  | $7.79    | $7.50      |
| 2026-01 | 33,032  | $11.29   | $8.00      |

### Key Insights
- **Dominant rate**: $7.50 CPM represents 92.94% of all views
- **January spike**: Average payout increased significantly to $11.29 (50% increase)
- **High stability**: September through November maintain consistent $7.50 median

---

## 2.4 Advertiser Spend (CPC) Analysis

### CPC Distribution for Click Records

| Statistic | Value ($) |
|-----------|-----------|
| Count     | 248       |
| Mean      | $12.83    |
| Median    | $7.75     |
| Std Dev   | $22.64    |
| Min       | $0.75     |
| 25th %    | $5.00     |
| 75th %    | $7.75     |
| Max       | $130.00   |

### Top 10 CPC Values

| CPC Value | Click Count | Percentage |
|-----------|-------------|------------|
| $5.00     | 101         | 40.7%      |
| $7.75     | 93          | 37.5%      |
| $30.00    | 13          | 5.2%       |
| $7.00     | 10          | 4.0%       |
| $15.00    | 9           | 3.6%       |
| $130.00   | 5           | 2.0%       |
| $112.00   | 4           | 1.6%       |
| $14.00    | 4           | 1.6%       |
| $17.00    | 3           | 1.2%       |
| $1.55     | 2           | 0.8%       |

### High-Value Clicks
- **High-value clicks (CPC >= $50)**: 10 clicks (4.0% of total)
- **Revenue concentration**: Wide range from $0.75 to $130.00 suggests diverse campaign values

---

## 2.5 Funnel Analysis (Dec 2025 - Jan 2026)

### Records in Dec-Jan Window
**Total**: 305,153 records

| Record Type | Count   |
|-------------|---------|
| bid         | 207,013 |
| View        | 98,102  |
| link        | 38      |

### Funnel Metrics
- **Bids**: 207,013
- **Views**: 98,102
- **Clicks**: 38
- **Implied Win Rate (Views/Bids)**: 47.39%
- **CTR (Clicks/Views)**: 0.0387%

### Key Observations
- **Win rate interpretation**: ~47% of bids resulted in views (won auctions)
- **Low CTR**: 0.0387% indicates approximately 1 click per 2,584 views

---

## 2.6 Num_Ads Impact Analysis

### Bid Requests by num_ads

| Num Ads | Bid Count | Percentage |
|---------|-----------|------------|
| 1       | 57,173    | 27.62%     |
| 2       | 149,840   | 72.38%     |

### View Records by num_ads

| Num Ads | View Count | Percentage |
|---------|------------|------------|
| 1       | 43,661     | 44.51%     |
| 2       | 54,417     | 55.46%     |
| 3       | 14         | 0.01%      |
| 4       | 6          | 0.01%      |
| 6       | 4          | 0.00%      |

### Proportion Analysis
- **1-ad requests**: 57,173 bids → 43,661 views (76.4% implied win rate)
- **2-ad requests**: 149,840 bids → 54,417 views (36.3% implied win rate)

### Analysis
- **2-ad requests dominate bids**: 72.38% of bid requests
- **1-ad requests perform better**: Higher conversion to views
- **Multi-ad capability**: Views can show 3-6 ads (rare but possible)

---

## 2.7 Feature Cardinality Analysis

### Feature Uniqueness in Bid Records

| Feature              | Unique Count | Null % |
|----------------------|--------------|--------|
| browser_code         | 9            | 0.0%   |
| os_code              | 6            | 0.0%   |
| geo_region_name      | 66           | 11.4%  |
| geo_city_name        | 3,333        | 12.7%  |
| domain               | 1            | 0.0%   |
| internal_adspace_id  | 9            | 0.0%   |
| internal_site_id     | 2            | 0.0%   |
| geo_country_code2    | 20           | 0.0%   |
| geo_dma_code         | 1            | 0.0%   |
| geo_postal_code      | 7,027        | 0.0%   |
| carrier_code         | 1            | 0.0%   |
| make_id              | 1            | 0.0%   |
| model_id             | 1            | 0.0%   |

### Key Insights
- **Low cardinality features**: browser_code (9), internal_adspace_id (9), os_code (6)
- **High cardinality features**: geo_postal_code (7,027), geo_city_name (3,333)
- **Single-value features**: domain, geo_dma_code, carrier_code, make_id, model_id (limited variation)

---

## 2.8 Top Values for Key Features

### Browser Code (Top 9)
| Browser Code | Count   | Percentage |
|--------------|---------|------------|
| 14           | 194,300 | 93.9%      |
| 100          | 6,979   | 3.4%       |
| 143          | 2,788   | 1.3%       |
| 146          | 1,912   | 0.9%       |
| 5            | 374     | 0.2%       |
| 144          | 344     | 0.2%       |
| 46           | 191     | 0.1%       |
| 150          | 118     | 0.1%       |
| -1           | 7       | 0.0%       |

### OS Code (Top 6)
| OS Code | Count   | Percentage |
|---------|---------|------------|
| -1      | 137,894 | 66.6%      |
| 8       | 65,718  | 31.7%      |
| 2       | 1,706   | 0.8%       |
| 18      | 1,409   | 0.7%       |
| 1       | 283     | 0.1%       |
| 10      | 3       | 0.0%       |

### Geographic Region (Top 10)
| Region        | Count  | Percentage |
|---------------|--------|------------|
| California    | 28,804 | 13.9%      |
| Texas         | 12,886 | 6.2%       |
| Florida       | 12,584 | 6.1%       |
| New York      | 12,112 | 5.9%       |
| Pennsylvania  | 8,537  | 4.1%       |
| New Jersey    | 7,852  | 3.8%       |
| Virginia      | 6,797  | 3.3%       |
| Ohio          | 5,893  | 2.8%       |
| Michigan      | 5,677  | 2.7%       |
| Tennessee     | 5,466  | 2.6%       |

### Internal Adspace ID (All 9)
| Adspace ID | Count  | Percentage |
|------------|--------|------------|
| 111564     | 69,376 | 33.5%      |
| 111565     | 57,558 | 27.8%      |
| 111563     | 47,231 | 22.8%      |
| 111566     | 15,448 | 7.5%       |
| 111567     | 6,124  | 3.0%       |
| 111568     | 4,211  | 2.0%       |
| 111569     | 2,885  | 1.4%       |
| 111571     | 2,111  | 1.0%       |
| 111570     | 2,069  | 1.0%       |

### Feature Concentration
- **Browser**: Heavily concentrated (94% in code 14)
- **OS**: Two-thirds unknown/missing (code -1)
- **Geography**: California dominates at 13.9%
- **Adspace**: Top 3 adspaces account for 84.1% of bids

---

## 2.9 Campaign Analysis

### Click Campaign Distribution
- **Unique Campaigns**: 64

### Top 10 Campaigns by Clicks

| Campaign ID | Click Count |
|-------------|-------------|
| 471454      | 56          |
| 471645      | 28          |
| 476324      | 12          |
| 474820      | 10          |
| 476473      | 8           |
| 475850      | 7           |
| 476159      | 7           |
| 469165      | 7           |
| 475853      | 6           |
| 476471      | 6           |

### Key Insight
- **Top 2 campaigns**: Account for 33.9% of all clicks (84 out of 248)
- **Long tail**: 64 unique campaigns suggest diverse campaign portfolio

---

## 2.10 Month-on-Month num_ads Distribution

### Bids: num_ads Distribution by Month

| Month   | 1 Ad   | 2 Ads   | % 1-Ad | % 2-Ads |
|---------|--------|---------|--------|---------|
| 2025-12 | 42,550 | 108,609 | 28.15% | 71.85%  |
| 2026-01 | 14,623 | 41,231  | 26.18% | 73.82%  |

### Views: num_ads Distribution by Month

| Month   | 1 Ad    | 2 Ads  | 3 Ads | 4 Ads | 6 Ads | % 1-Ad | % 2-Ads |
|---------|---------|--------|-------|-------|-------|--------|---------|
| 2025-09 | 106,034 | 15     | 80    | 0     | 2     | 99.91% | 0.01%   |
| 2025-10 | 290,305 | 5      | 43    | 0     | 6     | 99.98% | 0.00%   |
| 2025-11 | 128,926 | 8      | 12    | 0     | 7     | 99.98% | 0.01%   |
| 2025-12 | 35,123  | 29,931 | 6     | 6     | 4     | 53.98% | 46.00%  |
| 2026-01 | 8,538   | 24,486 | 8     | 0     | 0     | 25.85% | 74.13%  |

### Clicks: num_ads Distribution by Month

| Month   | 1 Ad |
|---------|------|
| 2025-09 | 49   |
| 2025-10 | 116  |
| 2025-11 | 45   |
| 2025-12 | 21   |
| 2026-01 | 17   |

### Key Observations
- **Major shift in December**: Views transition from 99.9% single-ad to 46% multi-ad
- **Consistent bid behavior**: December and January show stable 2-ad preference (~72-74%)
- **Clicks remain single-ad**: 100% of clicks are on single-ad requests across all months

---

## 2.11 Publisher Payout Distribution

### Cumulative Distribution Statistics

| Percentile | Value ($) |
|------------|-----------|
| Count      | 623,545   |
| Mean       | $7.73     |
| Std Dev    | $1.75     |
| Min        | $0.00     |
| 25%        | $7.50     |
| 50%        | $7.50     |
| 75%        | $7.50     |
| 90%        | $7.50     |
| 95%        | $7.50     |
| 99%        | $20.00    |
| Max        | $25.00    |

### Payout Value Counts (Top 16)

| Payout | Record Count | Percentage |
|--------|--------------|------------|
| $7.50  | 579,538      | 92.94%     |
| $6.00  | 10,629       | 1.70%      |
| $12.50 | 10,347       | 1.66%      |
| $8.00  | 10,281       | 1.65%      |
| $20.00 | 4,792        | 0.77%      |
| $25.00 | 2,538        | 0.41%      |
| $10.00 | 1,997        | 0.32%      |
| $3.00  | 1,436        | 0.23%      |
| $2.25  | 599          | 0.10%      |
| $4.50  | 288          | 0.05%      |
| $15.00 | 250          | 0.04%      |
| $0.00  | 229          | 0.04%      |
| $4.00  | 221          | 0.04%      |
| $21.00 | 166          | 0.03%      |
| $12.00 | 118          | 0.02%      |
| $14.00 | 116          | 0.02%      |

### Monthly Publisher Payout Statistics

| Month   | Count   | Mean ($) | Median ($) | Min ($) | Max ($) | Std Dev ($) |
|---------|---------|----------|------------|---------|---------|-------------|
| 2025-09 | 106,131 | $7.49    | $7.50      | $0.00   | $7.50   | $0.23       |
| 2025-10 | 290,359 | $7.50    | $7.50      | $0.00   | $7.50   | $0.10       |
| 2025-11 | 128,953 | $7.50    | $7.50      | $0.00   | $7.50   | $0.11       |
| 2025-12 | 65,070  | $7.79    | $7.50      | $0.00   | $25.00  | $2.49       |
| 2026-01 | 33,032  | $11.29   | $8.00      | $0.00   | $25.00  | $5.63       |

### Top Payout Values by Month

#### 2025-09 (Top 2)
- $7.50: 106,034 (99.9%)
- $0.00: 97 (0.1%)

#### 2025-10 (Top 2)
- $7.50: 290,304 (100.0%)
- $0.00: 55 (0.0%)

#### 2025-11 (Top 2)
- $7.50: 128,924 (100.0%)
- $0.00: 29 (0.0%)

#### 2025-12 (Top 5)
- $7.50: 54,276 (83.4%)
- $10.00: 1,997 (3.1%)
- $6.00: 1,950 (3.0%)
- $8.00: 1,562 (2.4%)
- $3.00: 1,436 (2.2%)

#### 2026-01 (Top 5)
- $12.50: 9,455 (28.6%)
- $8.00: 8,719 (26.4%)
- $6.00: 8,679 (26.3%)
- $20.00: 4,007 (12.1%)
- $25.00: 2,073 (6.3%)

### Key Insights
- **Extreme concentration**: 92.94% at single price point ($7.50)
- **January diversification**: Significant spread across $6-$25 range
- **Stable early months**: Sept-Nov show 99.9%+ at $7.50

---

## 2.12 Bid Amount Distribution

### Cumulative Bid Amount Statistics

| Percentile | Value ($) |
|------------|-----------|
| Count      | 207,013   |
| Mean       | $9.31     |
| Std Dev    | $16.81    |
| Min        | $0.00     |
| 25%        | $7.50     |
| 50%        | $7.50     |
| 75%        | $7.50     |
| 90%        | $7.50     |
| 95%        | $15.00    |
| 99%        | $130.00   |
| Max        | $130.00   |

### Bid Value Counts (Top 8)

| Bid Value | Record Count | Percentage |
|-----------|--------------|------------|
| $7.50     | 147,739      | 71.37%     |
| $6.75     | 23,900       | 11.55%     |
| $0.00     | 13,257       | 6.40%      |
| $5.00     | 9,934        | 4.80%      |
| $15.00    | 5,171        | 2.50%      |
| $130.00   | 3,858        | 1.86%      |
| $2.70     | 1,828        | 0.88%      |
| $18.00    | 1,326        | 0.64%      |

### Monthly Bid Amount Statistics

| Month   | Count   | Mean ($) | Median ($) | Min ($) | Max ($)  | Std Dev ($) |
|---------|---------|----------|------------|---------|----------|-------------|
| 2025-12 | 151,159 | $10.09   | $7.50      | $0.00   | $130.00  | $19.56      |
| 2026-01 | 55,854  | $7.20    | $6.75      | $2.70   | $18.00   | $2.58       |

### Top Bid Values by Month

#### 2025-12 (Top 5)
- $7.50: 132,015 (87.3%)
- $0.00: 13,257 (8.8%)
- $130.00: 3,858 (2.6%)
- $18.00: 1,322 (0.9%)
- $15.00: 573 (0.4%)

#### 2026-01 (Top 5)
- $6.75: 23,766 (42.6%)
- $7.50: 15,724 (28.2%)
- $5.00: 9,934 (17.8%)
- $15.00: 4,598 (8.2%)
- $2.70: 1,828 (3.3%)

### Non-Zero Bid Analysis

| Metric              | Value         |
|---------------------|---------------|
| Total Bids          | 207,013       |
| Non-Zero Bids       | 193,756 (93.60%) |
| Zero Bids           | 13,257 (6.40%)   |

#### Non-Zero Bid Statistics

| Statistic | Value ($) |
|-----------|-----------|
| Count     | 193,756   |
| Mean      | $9.95     |
| Std Dev   | $17.20    |
| Min       | $2.70     |
| 25%       | $7.50     |
| 50%       | $7.50     |
| 75%       | $7.50     |
| 90%       | $7.50     |
| 95%       | $15.00    |
| 99%       | $130.00   |
| Max       | $130.00   |

### Key Insights
- **$7.50 dominant**: 71.37% of all bids
- **Zero bids present**: 6.40% of bids are $0 (all in December 2025)
- **January shift**: Average bid drops to $7.20, median to $6.75
- **December volatility**: Higher standard deviation ($19.56) due to presence of $130 bids

---
