# EDA Batch 1: Baseline Metrics & Data Quality

## Execution Command
```bash
python3 EDA_1.py
```

---

## Data Loading Summary

### CSV Files Loaded
- **Bids Dataset**: 207,013 records (2 rows skipped due to malformed data)
- **Clicks Dataset**: 248 records (0 rows skipped)
- **Views Dataset**: 623,545 records (15 rows skipped due to malformed data)

### Loading Statistics
- **Total Skipped Rows**: 17
- **Combined Total Records**: 830,806

---

## 1.1 Data Overview

### Dataset Dimensions
- **Total Rows**: 830,806
- **Total Columns**: 71
- **Date Range**: 2025-09-15 00:01:44.614+00 to 2026-01-11 02:10:47.743-05

---

## 1.2 Record Type Distribution

### Absolute Counts
| Record Type | Count   |
|-------------|---------|
| View        | 623,545 |
| bid         | 207,013 |
| link        | 248     |

### Percentage Distribution
| Record Type | Percentage |
|-------------|------------|
| View        | 75.05%     |
| bid         | 24.92%     |
| link        | 0.03%      |

---

## 1.3 Records by Month and Type

### Monthly Distribution Table
| Month   | View   | bid     | link |
|---------|--------|---------|------|
| 2025-09 | 106,131| 0       | 49   |
| 2025-10 | 290,359| 0       | 116  |
| 2025-11 | 128,953| 0       | 45   |
| 2025-12 | 65,070 | 151,159 | 21   |
| 2026-01 | 33,032 | 55,854  | 17   |

### Key Observations
- **Bid data only available**: December 2025 and January 2026
- **View and Click data**: Available across all months (Sept 2025 - Jan 2026)
- **Peak View month**: October 2025 (290,359 records)
- **Peak Bid month**: December 2025 (151,159 records)

---

## 1.4 NPI Validation

### Validation Results
| Record Type | Valid NPI Count | Total Count | % Valid |
|-------------|-----------------|-------------|---------|
| View        | 0               | 623,545     | 0.0%    |
| bid         | 0               | 207,013     | 0.0%    |
| link        | 0               | 248         | 0.0%    |

### Analysis
- **No valid NPIs found** in external_userid field (expected 10-11 digit format)
- This suggests external_userid may be stored in non-standard format or contains non-NPI identifiers

---

## 1.5 Array Field Analysis (internal_txn_id)

### Sample Value
```
{ec69bc8c-916b-4aa8-883f-6850234165a2,e6ebdc8d-9972-4df8-935b-9a3b48de0e94}
```
- **Data Type**: PostgreSQL array stored as string
- **Format**: Curly braces with comma-separated UUIDs

### Number of Ads per Request Distribution

| Record Type | 1 Ad    | 2 Ads   | 3 Ads | 4 Ads | 6 Ads |
|-------------|---------|---------|-------|-------|-------|
| View        | 568,926 | 54,445  | 149   | 6     | 19    |
| bid         | 57,173  | 149,840 | 0     | 0     | 0     |
| link        | 248     | 0       | 0     | 0     | 0     |

### Key Insights
- **Bids**: Predominantly 2-ad requests (72.4% of bids)
- **Views**: Predominantly 1-ad requests (91.2% of views)
- **Clicks**: Exclusively 1-ad requests (100%)
- **Multi-ad capability**: Views show up to 6 ads per request, bids max at 2 ads

---

## 1.6 Unique Identifier Analysis

### Overall Uniqueness
- **Unique log_txnid count**: 726,181
- **Total rows**: 830,806
- **Uniqueness ratio**: 0.8741 (87.41%)

### Uniqueness by Record Type
| Record Type | Unique IDs | Total Rows | Uniqueness Ratio |
|-------------|------------|------------|------------------|
| bid         | 207,004    | 207,013    | 1.0000 (100.00%) |
| link        | 248        | 248        | 1.0000 (100.00%) |
| View        | 577,815    | 623,545    | 0.9267 (92.67%)  |

### Analysis
- **Bids and Clicks**: Perfect uniqueness (each row has unique log_txnid)
- **Views**: 7.33% duplication rate (45,730 duplicate rows)
- **Potential issue**: View records may have duplicate log_txnids requiring investigation

---

## 1.7 Key Columns Null Check

### Null Value Analysis
| Column              | Null Count | Null % |
|---------------------|------------|--------|
| log_txnid           | 0          | 0.00%  |
| internal_txn_id     | 0          | 0.00%  |
| external_userid     | 232        | 0.03%  |
| publisher_payout    | 0          | 0.00%  |
| advertiser_spend    | 0          | 0.00%  |
| browser_code        | 0          | 0.00%  |
| os_code             | 0          | 0.00%  |
| geo_region_name     | 98,879     | 11.90% |
| domain              | 0          | 0.00%  |
| internal_adspace_id | 0          | 0.00%  |
| campaign_code       | 0          | 0.00%  |

### Data Quality Summary
- **Excellent**: 9 out of 11 key columns have <1% null values
- **Attention needed**: geo_region_name has 11.90% nulls (98,879 records missing geographic data)
- **Minor issue**: external_userid has 232 nulls (0.03%)

---

## 1.8 Sample Data by Record Type

### View Record Sample
```json
{
  "log_txnid": "d6f277d2-9a75-4954-a4dc-b5a578c6c264",
  "internal_txn_id": "{39970876-6aad-49b9-9635-e408f79e57cb}",
  "external_userid": 1760482178.0,
  "publisher_payout": "{7.50000}",
  "advertiser_spend": "{0.00000}",
  "browser_code": "14",
  "os_code": "-1",
  "geo_region_name": "North Carolina",
  "campaign_code": "{473969}"
}
```

### Bid Record Sample
```json
{
  "log_txnid": "fdd32ed9-9ce9-432f-a699-09030dcc2b63",
  "internal_txn_id": "{ec69bc8c-916b-4aa8-883f-6850234165a2,e6ebdc8d-9972-4df8-935b-9a3b48de0e94}",
  "external_userid": 1194798132.0,
  "publisher_payout": "{7.50000,7.50000}",
  "advertiser_spend": "{0.00000,0.00000}",
  "browser_code": "14",
  "os_code": "-1",
  "geo_region_name": "Nevada",
  "campaign_code": "{471454,474820}"
}
```

### Click (Link) Record Sample
```json
{
  "log_txnid": "f2ea01f6-4688-45c0-b16e-fdb875b7de9e",
  "internal_txn_id": "{99172298-c2a0-46c9-abec-1df260f1385b}",
  "external_userid": 1952415630.0,
  "publisher_payout": "{0.00000}",
  "advertiser_spend": "{7.75000}",
  "browser_code": "14",
  "os_code": "8",
  "geo_region_name": "New York",
  "campaign_code": "{475453}"
}
```

### Sample Data Observations
- **Array fields**: publisher_payout, advertiser_spend, campaign_code stored as PostgreSQL arrays
- **Array alignment**: Multiple values in arrays correspond to multiple ads (internal_txn_id count)
- **Payout pattern**: Views show $7.50 CPM, Clicks show $0 publisher payout but $7.75 advertiser spend
- **OS codes**: -1 appears to indicate unknown/missing OS information
