# BATCH 5: FINAL FEATURE ANALYSIS

**Data loaded:** 830,806 records

---

## ALL AVAILABLE COLUMNS

**Total columns:** 72

| # | Column Name | # | Column Name |
|---|-------------|---|-------------|
| 1 | ad_format | 37 | geo_postal_code |
| 2 | advertiser_id | 38 | geo_region_code |
| 3 | advertiser_spend | 39 | geo_region_name |
| 4 | adx_custom | 40 | geo_timezone |
| 5 | ae_endpoint_codes | 41 | id |
| 6 | ae_response_bids | 42 | internal_adspace_id |
| 7 | ae_response_codes | 43 | internal_site_id |
| 8 | banner_code | 44 | internal_txn_id |
| 9 | bid_amount | 45 | ip |
| 10 | billing_status | 46 | link_lead_assoc |
| 11 | browser_code | 47 | list_id |
| 12 | campaign_code | 48 | log_dt |
| 13 | carrier_code | 49 | log_txnid |
| 14 | category_codes | 50 | make_id |
| 15 | click_tracking_id | 51 | matched_cats |
| 16 | clickstream_id | 52 | matched_kw |
| 17 | content_cats | 53 | media_type |
| 18 | content_kw | 54 | model_id |
| 19 | conversion_payout | 55 | month |
| 20 | country_code | 56 | nid |
| 21 | data_cost | 57 | os_code |
| 22 | data_revenue | 58 | paytype |
| 23 | domain | 59 | pixel_sync_status |
| 24 | external_adspace_id | 60 | platform_id |
| 25 | external_site_id | 61 | publisher_id |
| 26 | external_txn_id | 62 | publisher_payout |
| 27 | external_userid | 63 | rec_type |
| 28 | fraud_status | 64 | ref_bundle |
| 29 | geo_city_name | 65 | rule_id |
| 30 | geo_continent_code | 66 | subid |
| 31 | geo_country_code2 | 67 | total_ads |
| 32 | geo_country_code3 | 68 | traffic_source_id |
| 33 | geo_country_name | 69 | txn_cnt |
| 34 | geo_dma_code | 70 | ua |
| 35 | geo_latitude | 71 | userid |
| 36 | geo_longitude | 72 | zone_code |

---

## 5.1 REF_BUNDLE (PAGE-LEVEL URL ANALYSIS)

**Sample ref_bundle values:**

1. https://www.drugs.com/interactions-check.php?drug_list=1463-0,4326-19915
2. https://www.drugs.com/interaction/list/?drug_list=1463-0,4326-19915
3. https://www.drugs.com/interactions-check.php?drug_list=1463-0,4326-19915
4. https://www.drugs.com/interactions-check.php?drug_list=1463-0,4326-19915
5. https://www.drugs.com/drug-class/sulfonylureas.html
6. https://www.drugs.com/drug-class/sulfonylureas.html
7. https://www.drugs.com/sfx/fluoxetine-side-effects.html#serious-side-effects
8. https://www.drugs.com/tamoxifen.html
9. https://www.drugs.com/tamoxifen.html
10. https://www.drugs.com/sfx/fluoxetine-side-effects.html#serious-side-effects

**Page Category Summary:**
- Unique page categories: 1,753
- Null page categories: 0 (0.0%)

**Top 20 Page Categories (all data):**

| Page Category | Count | % of Total |
|---------------|-------|-----------|
| imprints.php | 106,950 | 12.87% |
| search.php | 102,111 | 12.29% |
| interaction | 75,233 | 9.06% |
| interactions-check.php | 63,697 | 7.67% |
| dosage | 62,052 | 7.47% |
| drug-interactions | 40,557 | 4.88% |
| sfx | 40,316 | 4.85% |
| mtm | 30,029 | 3.61% |
| medical-answers | 27,253 | 3.28% |
| pro | 26,716 | 3.22% |
| comments | 24,404 | 2.94% |
| condition | 19,246 | 2.32% |
| drug_interactions.html | 15,321 | 1.84% |
| alpha | 15,223 | 1.83% |
| imprints | 14,754 | 1.78% |
| drug-class | 7,728 | 0.93% |
| answers | 6,280 | 0.76% |
| monograph | 4,657 | 0.56% |
| news | 4,345 | 0.52% |
| cg | 3,799 | 0.46% |

**Page Categories by Record Type**

**Bid Records - Top 10 Page Categories:**

| Page Category | Count | % of Bids |
|---------------|-------|-----------|
| search.php | 33,892 | 16.4% |
| imprints.php | 21,863 | 10.6% |
| interactions-check.php | 20,287 | 9.8% |
| dosage | 17,838 | 8.6% |
| drug-interactions | 11,368 | 5.5% |
| sfx | 10,238 | 4.9% |
| mtm | 8,153 | 3.9% |
| medical-answers | 6,592 | 3.2% |
| pro | 6,373 | 3.1% |
| interaction | 6,007 | 2.9% |

**View Records - Top 10 Page Categories:**

| Page Category | Count | % of Views |
|---------------|-------|-----------|
| imprints.php | 85,055 | 13.6% |
| interaction | 69,225 | 11.1% |
| search.php | 68,206 | 10.9% |
| dosage | 44,189 | 7.1% |
| interactions-check.php | 43,402 | 7.0% |
| sfx | 30,059 | 4.8% |
| drug-interactions | 29,176 | 4.7% |
| mtm | 21,863 | 3.5% |
| medical-answers | 20,634 | 3.3% |
| pro | 20,339 | 3.3% |

**Link Records - Top 10 Page Categories:**

| Page Category | Count | % of Links |
|---------------|-------|-----------|
| imprints.php | 32 | 12.9% |
| medical-answers | 27 | 10.9% |
| dosage | 25 | 10.1% |
| sfx | 19 | 7.7% |
| mtm | 13 | 5.2% |
| drug-interactions | 13 | 5.2% |
| search.php | 13 | 5.2% |
| news | 9 | 3.6% |
| interactions-check.php | 8 | 3.2% |
| comments | 7 | 2.8% |

**CTR by Page Category (Views with 50+ records):**

| Page Category | Views | Clicks | CTR % |
|---------------|-------|--------|-------|
| imprints.php | 84,864 | 32 | 0.0377% |
| search.php | 68,036 | 12 | 0.0176% |
| dosage | 44,100 | 22 | 0.0499% |
| interactions-check.php | 43,298 | 8 | 0.0185% |
| sfx | 29,994 | 16 | 0.0533% |
| drug-interactions | 29,141 | 10 | 0.0343% |
| interaction | 24,831 | 1 | 0.0040% |
| mtm | 21,801 | 11 | 0.0505% |
| medical-answers | 20,606 | 22 | 0.1068% |
| pro | 20,279 | 3 | 0.0148% |
| comments | 19,091 | 4 | 0.0210% |
| condition | 14,189 | 5 | 0.0352% |
| drug_interactions.html | 12,694 | 1 | 0.0079% |
| alpha | 11,872 | 1 | 0.0084% |
| imprints | 10,706 | 3 | 0.0280% |
| drug-class | 5,830 | 0 | 0.0000% |
| answers | 5,003 | 1 | 0.0200% |
| monograph | 3,511 | 3 | 0.0854% |
| news | 3,039 | 8 | 0.2632% |
| cg | 2,788 | 4 | 0.1435% |

---

## 5.2 ALL GEO FEATURES COMPARISON

**Geo Feature Cardinality and Nulls:**

| Feature | Unique Values | Null % |
|---------|---------------|--------|
| geo_region_name | 66 | 11.4% |
| geo_city_name | 3,333 | 12.7% |
| geo_postal_code | 7,027 | 0.0% |
| geo_country_code2 | 20 | 0.0% |
| geo_dma_code | 1 | 0.0% |
| geo_metro_code | NOT FOUND | — |

**Win Rate by geo_region_name (Top 15):**

| Region | Bids | Wins | Win Rate |
|--------|------|------|----------|
| California | 27,944 | 11,626 | 41.6% |
| Texas | 12,019 | 3,161 | 26.3% |
| Florida | 11,558 | 3,666 | 31.7% |
| New York | 11,367 | 3,485 | 30.7% |
| Pennsylvania | 8,028 | 2,366 | 29.5% |
| New Jersey | 7,284 | 2,215 | 30.4% |
| Virginia | 6,389 | 1,564 | 24.5% |
| Ohio | 5,429 | 1,396 | 25.7% |
| Michigan | 5,268 | 1,309 | 24.8% |
| Tennessee | 5,174 | 1,487 | 28.7% |
| North Carolina | 4,825 | 1,383 | 28.7% |
| Illinois | 4,591 | 1,211 | 26.4% |
| Maryland | 4,577 | 1,171 | 25.6% |
| Massachusetts | 4,494 | 1,258 | 28.0% |
| Georgia | 4,120 | 1,345 | 32.6% |

**Win Rate by geo_country_code2 (Top 15):**

| Country Code | Bids | Wins | Win Rate |
|--------------|------|------|----------|
| US | 191,304 | 58,159 | 30.4% |
| -1 | 1,825 | 562 | 30.8% |
| DE | 289 | 59 | 20.4% |
| CA | 101 | 43 | 42.6% |
| GB | 60 | 23 | 38.3% |
| BE | 48 | 22 | 45.8% |
| ES | 28 | 1 | 3.6% |
| DK | 18 | 2 | 11.1% |
| FR | 16 | 4 | 25.0% |
| IT | 14 | 7 | 50.0% |
| UA | 10 | 5 | 50.0% |
| BR | 9 | 0 | 0.0% |
| NO | 9 | 0 | 0.0% |
| SA | 9 | 0 | 0.0% |
| RS | 6 | 0 | 0.0% |

---

## 5.3 TIME-BASED FEATURES

**Bids by Hour of Day:**

| Hour | Count |
|------|-------|
| 0 | 2,767 |
| 1 | 1,963 |
| 2 | 1,157 |
| 3 | 1,100 |
| 4 | 723 |
| 5 | 9,404 |
| 6 | 8,420 |
| 7 | 7,079 |
| 8 | 3,862 |
| 9 | 4,484 |
| 10 | 4,923 |
| 11 | 6,052 |
| 12 | 10,124 |
| 13 | 18,140 |
| 14 | 20,125 |
| 15 | 17,966 |
| 16 | 14,545 |
| 17 | 12,417 |
| 18 | 13,441 |
| 19 | 13,100 |
| 20 | 12,214 |
| 21 | 12,800 |
| 22 | 6,558 |
| 23 | 3,649 |

**CTR by Hour of Day:**

| Hour | Views | Clicks | CTR % |
|------|-------|--------|-------|
| 0 | 10,405 | 7 | 0.0673% |
| 1 | 8,008 | 1 | 0.0125% |
| 2 | 5,565 | 4 | 0.0719% |
| 3 | 4,046 | 3 | 0.0741% |
| 4 | 7,243 | 4 | 0.0552% |
| 5 | 12,011 | 5 | 0.0416% |
| 6 | 9,959 | 5 | 0.0502% |
| 7 | 7,074 | 5 | 0.0707% |
| 8 | 4,575 | 2 | 0.0437% |
| 9 | 5,352 | 2 | 0.0374% |
| 10 | 6,695 | 3 | 0.0448% |
| 11 | 9,719 | 1 | 0.0103% |
| 12 | 21,002 | 8 | 0.0381% |
| 13 | 38,951 | 10 | 0.0257% |
| 14 | 50,786 | 24 | 0.0473% |
| 15 | 56,632 | 23 | 0.0406% |
| 16 | 54,270 | 12 | 0.0221% |
| 17 | 48,437 | 15 | 0.0310% |
| 18 | 50,120 | 15 | 0.0299% |
| 19 | 47,846 | 13 | 0.0272% |
| 20 | 42,868 | 20 | 0.0467% |
| 21 | 37,549 | 15 | 0.0399% |
| 22 | 23,913 | 8 | 0.0335% |
| 23 | 14,789 | 9 | 0.0609% |

**Bids by Day of Week:**

| Day | Bids |
|-----|------|
| Monday | 30,785 |
| Tuesday | 38,946 |
| Wednesday | 40,691 |
| Thursday | 33,101 |
| Friday | 39,964 |
| Saturday | 13,309 |
| Sunday | 10,217 |

---

## 5.4 BROWSER AND OS ANALYSIS

**Win Rate by Browser Code:**

| Browser Code | Bids | Win Rate |
|--------------|------|----------|
| 14 | 181,581 | 30.2% |
| 100 | 6,665 | 37.0% |
| 143 | 2,700 | 24.7% |
| 146 | 1,811 | 26.7% |
| 5 | 372 | 43.5% |

**Win Rate by OS Code:**

| OS Code | Bids | Win Rate |
|---------|------|----------|
| -1 | 124,637 | 21.1% |
| 8 | 65,718 | 47.5% |
| 2 | 1,706 | 34.6% |
| 18 | 1,409 | 41.0% |
| 1 | 283 | 54.4% |

---

## 5.5 INTERNAL_SITE_ID ANALYSIS

**Unique internal_site_id values:**

| Site ID | Count |
|---------|-------|
| 211399 | 159,782 |
| 211398 | 47,231 |

**Win Rate by internal_site_id:**

| Site ID | Bids | Win Rate |
|---------|------|----------|
| 211399 | 149,528 | 28.5% |
| 211398 | 44,228 | 36.6% |

---

## 5.6 CAMPAIGN ANALYSIS

**Top 15 Campaigns by Total Revenue:**

| Campaign | Clicks | Total Revenue | Avg CPC | Max CPC |
|----------|--------|---------------|---------|---------|
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

## 5.7 FEATURE SIGNAL SUMMARY

**Feature Signal Analysis (Win Rate Variance):**

| Feature | Unique | Null % | Win Rate Variance | Signal Strength |
|---------|--------|--------|-------------------|-----------------|
| internal_adspace_id | 9 | 0.0% | 40.78 | 93.89 |
| geo_region_name | 65 | 11.3% | 197.63 | 734.24 |
| geo_country_code2 | 19 | 0.0% | 392.12 | 1174.67 |
| browser_code | 9 | 0.0% | 148.78 | 342.58 |
| os_code | 6 | 0.0% | 251.10 | 488.61 |
| internal_site_id | 2 | 0.0% | 32.86 | 36.10 |
| page_category | 1,153 | 0.0% | 682.98 | 4815.69 |

---

## 5.8 RECOMMENDED FEATURE COMBINATIONS

| Combination | Unique Values | ≥50 Observations | Coverage |
|-------------|---------------|-----------------|----------|
| internal_adspace_id | 9 | 9 | 100.0% |
| internal_adspace_id + geo_region_name | 518 | 278 | 86.8% |
| internal_adspace_id + geo_region_name + browser_code | 1,575 | 302 | 83.1% |
| internal_adspace_id + page_category | 4,562 | 207 | 86.8% |
| internal_adspace_id + geo_region_name + page_category | 19,843 | 632 | 52.2% |

---
