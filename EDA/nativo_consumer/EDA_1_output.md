# EDA 1: Nativo Consumer Data Overview

Generated: 2026-01-29 14:11:42

Data file: data_nativo_consumer.csv (reduced 10% sample)

## 1. File Information

- Reduced file: 1.38 GB
- Full file: 13.84 GB
- Rows (reduced): 1,734,971
- Rows (full, estimated): 17,349,710

## 2. Column Analysis

- Total columns: 71

### Column Types
```
log_dt: object
nid: int64
rec_type: object
log_txnid: object
internal_txn_id: object
txn_cnt: int64
traffic_source_id: int64
publisher_id: int64
internal_site_id: int64
internal_adspace_id: int64
zone_code: int64
ip: object
ref_bundle: object
domain: object
subid: float64
matched_cats: object
matched_kw: object
content_cats: object
adx_custom: object
category_codes: object
browser_code: int64
os_code: int64
carrier_code: int64
make_id: int64
model_id: int64
ua: object
click_tracking_id: object
advertiser_id: object
campaign_code: object
banner_code: object
ad_format: object
paytype: object
rule_id: object
advertiser_spend: object
publisher_payout: object
bid_amount: object
data_cost: object
data_revenue: object
list_id: object
total_ads: int64
media_type: object
userid: object
conversion_payout: float64
ae_endpoint_codes: object
ae_response_codes: object
ae_response_bids: object
country_code: object
geo_city_name: object
geo_dma_code: int64
geo_region_name: object
geo_timezone: object
geo_country_code2: object
geo_region_code: object
geo_continent_code: object
geo_latitude: float64
geo_longitude: float64
geo_postal_code: object
geo_country_code3: object
geo_country_name: object
external_userid: float64
external_txn_id: object
external_site_id: object
external_adspace_id: float64
billing_status: object
fraud_status: object
platform_id: int64
source_feed: object
source_filename: object
source_row_number: int64
server_host: object
load_ts: object
```

## 3. Date Range

- Start: 2025-09-15
- End: 2026-01-27
- Range: 134 days

## 4. Record Type Distribution

| rec_type | Count | Percentage |
|----------|-------|------------|
| bid | 1,496,306 | 86.24% |
| View | 237,325 | 13.68% |
| link | 1,340 | 0.08% |

## 5. Record Type Timeline

### First occurrence of each rec_type
- View: 2025-09-15
- link: 2025-09-15
- bid: 2025-12-10

### Monthly rec_type counts
```
rec_type    View      bid  link
log_month                      
2025-09    21995        0    94
2025-10    42776        0   184
2025-11    51776        0   225
2025-12    31153      982   181
2026-01    89625  1495324   656
```

## 6. Feature Cardinality (Potential Segment Features)

| Feature | Unique Values | Top Value | Top % | Nulls % |
|---------|---------------|-----------|-------|---------|
| internal_adspace_id | 1 | -1 | 100.0% | 0.0% |
| domain | 14,626 | cbsnews.com | 2.3% | 0.0% |
| browser_code | 15 | 143 | 61.1% | 0.0% |
| os_code | 9 | -1.0 | 65.2% | 0.0% |
| carrier_code | 4,403 | -1.0 | 22.4% | 0.0% |
| make_id | 66 | -1 | 59.8% | 0.0% |
| model_id | 67 | -1 | 81.3% | 0.0% |
| media_type | 2 | mobile | 69.5% | 0.0% |
| geo_region_name | 65 | New York | 9.5% | 0.3% |
| geo_country_code2 | 26 | US | 100.0% | 0.0% |
| geo_dma_code | 1 | 0 | 100.0% | 0.0% |
| ad_format | 2 | {"13"} | 100.0% | 0.0% |

## 7. Bid and Payout Analysis

### By rec_type

**View** (n=237,325)
- publisher_payout: all null
- bid_amount: all null or zero

**link** (n=1,340)
- publisher_payout: all null
- bid_amount: all null or zero

**bid** (n=1,496,306)
- publisher_payout: all null
- bid_amount: all null or zero

## 8. Win Rate Analysis (Bid Period Only)

- Total bids: 1,496,306
- Total views: 237,325
- Matched (won): 0
- Win rate: 0.0%

## 9. CTR and Conversion Analysis

- Views: 237,325
- Clicks (link): 1,340
- Conversions (lead): 0
- CTR (clicks/views): 0.5646%
- CVR (leads/views): 0.0000%

## 10. Column Size Analysis (for data reduction)

### Estimated column sizes (for 17M rows)

| Column | Type | Est. Size (GB) | Nulls % |
|--------|------|----------------|---------|
| ua | object | 2.74 | 0.7% |
| ref_bundle | object | 2.19 | 0.0% |
| source_filename | object | 1.90 | 0.0% |
| internal_txn_id | object | 1.41 | 0.0% |
| click_tracking_id | object | 1.41 | 0.0% |
| log_txnid | object | 1.35 | 0.0% |
| server_host | object | 1.22 | 0.0% |
| log_dt | object | 1.17 | 0.0% |
| userid | object | 1.16 | 0.1% |
| load_ts | object | 1.14 | 0.0% |
| geo_timezone | object | 1.00 | 0.0% |
| domain | object | 0.98 | 0.0% |
| geo_country_name | object | 0.98 | 0.0% |
| ip | object | 0.98 | 0.0% |
| bid_amount | object | 0.95 | 0.0% |
| publisher_payout | object | 0.95 | 0.0% |
| advertiser_spend | object | 0.95 | 0.0% |
| banner_code | object | 0.93 | 0.0% |
| campaign_code | object | 0.93 | 0.0% |
| advertiser_id | object | 0.92 | 0.0% |

## 11. Preliminary Observations

### Data Period Notes
- [To be filled based on analysis above]

### Columns to Potentially Drop
- [To be filled based on size analysis]

### Feature Candidates
- [To be filled based on cardinality analysis]