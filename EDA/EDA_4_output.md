# BATCH 4: VERIFICATION & FINAL ANALYSIS

**Data loaded:** 830,806 records

---

## 4.1 DATA CLEANING

**Raw Counts:**
- Bids: 207,013
- Views: 623,545
- Clicks: 248

**After Cleaning:**
- Views after deduplication: 577,815
- Bids after removing $0: 193,756
- Views from Oct 1+: 479,516
- Clicks from Oct 1+: 199

---

## 4.2 VERIFIED NUM_ADS WIN RATE (Joined Analysis)

**Summary:**
- Dec-Jan non-zero bids: 193,756
- Dec-Jan deduped views: 92,396
- Joined records: 193,756
- Bids that won (matched to view): 58,887
- Win rate (joined): 30.39%

**Win Rate by Bid num_ads (Properly Joined):**

| Ad Type | Bids | Wins | Win Rate |
|---------|------|------|----------|
| 1-ad bids | 56,895 | 17,042 | 30.0% |
| 2-ad bids | 136,861 | 41,845 | 30.6% |

**Does num_ads change from bid to view?**
- Total won bids: 58,887
- num_ads matches: 58,887 (100.0%)

**Bid num_ads vs View num_ads (won bids only):**

| Bid num_ads | View = 1.0 | View = 2.0 | Total |
|-------------|-----------|-----------|-------|
| 1 | 17,042 | 0 | 17,042 |
| 2 | 0 | 41,845 | 41,845 |
| All | 17,042 | 41,845 | 58,887 |

---

## 4.3 UNMATCHED VIEWS INVESTIGATION

**Summary:**
- Total views (Dec-Jan): 92,396
- Views matching a bid: 58,886
- Views NOT matching any bid: 33,510 (36.3%)

**Unmatched views date range:**
- Min date: 2025-12-01 00:14:08.298000+00:00
- Max date: 2026-01-09 17:09:41.732000+00:00

---

## 4.4 WIN RATE BY BID AMOUNT (Joined)

| Bid Bucket | Bids | Wins | Win Rate |
|-----------|------|------|----------|
| 01_under_5 | 1,828 | 955 | 52.2% |
| 02_5_to_7.5 | 33,834 | 18,858 | 55.7% |
| 03_exactly_7.5 | 147,739 | 35,646 | 24.1% |
| 06_15_to_20 | 6,497 | 2,759 | 42.5% |
| 07_20_plus | 3,858 | 669 | 17.3% |

---

## 4.5 WIN RATE BY SEGMENT (Joined)

**By internal_adspace_id:**

| Adspace | Bids | Wins | Win Rate |
|---------|------|------|----------|
| 111563 | 44,228 | 16,209 | 36.6% |
| 111564 | 65,008 | 22,231 | 34.2% |
| 111565 | 53,907 | 11,954 | 22.2% |
| 111566 | 14,423 | 4,337 | 30.1% |
| 111567 | 5,729 | 1,778 | 31.0% |
| 111568 | 3,909 | 1,019 | 26.1% |
| 111569 | 2,682 | 594 | 22.1% |
| 111570 | 1,918 | 355 | 18.5% |
| 111571 | 1,952 | 410 | 21.0% |

**By geo_region (Top 15):**

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

---

## 4.6 CTR ANALYSIS (Oct 2025 onwards)

**Overall Summary:**
- Unique click txn_ids: 183
- Views (Oct+): 479,516
- Views with clicks: 166
- CTR: 0.0346%

**CTR by Month:**

| Month | Views | Clicks | CTR % |
|-------|-------|--------|-------|
| 2025-10 | 268,075 | 102 | 0.0380% |
| 2025-11 | 119,045 | 41 | 0.0344% |
| 2025-12 | 61,386 | 13 | 0.0212% |
| 2026-01 | 31,010 | 10 | 0.0322% |

**CTR by Adspace:**

| Adspace | Views | Clicks | CTR % |
|---------|-------|--------|-------|
| 111564 | 172,909 | 91 | 0.0526% |
| 111563 | 144,825 | 21 | 0.0145% |
| 111565 | 89,646 | 34 | 0.0379% |
| 111566 | 38,203 | 11 | 0.0288% |
| 111567 | 15,274 | 4 | 0.0262% |
| 111568 | 9,098 | 2 | 0.0220% |
| 111569 | 4,511 | 1 | 0.0222% |
| 111570 | 2,576 | 0 | 0.0000% |
| 111571 | 2,346 | 2 | 0.0853% |
| 110201 | 85 | 0 | 0.0000% |
| 111284 | 25 | 0 | 0.0000% |
| 111285 | 7 | 0 | 0.0000% |
| 111283 | 6 | 0 | 0.0000% |
| 110349 | 4 | 0 | 0.0000% |
| 110199 | 1 | 0 | 0.0000% |

---

## 4.7 EXPECTED VALUE BY SEGMENT

**Overall Avg CPC:** $13.36

**Expected Value by Adspace:**

| Adspace | Win Rate | CTR | E[Revenue] |
|---------|----------|-----|-----------|
| 111563 | 36.6% | 0.0145% | $0.0007 |
| 111564 | 34.2% | 0.0526% | $0.0024 |
| 111565 | 22.2% | 0.0379% | $0.0011 |
| 111566 | 30.1% | 0.0288% | $0.0012 |
| 111567 | 31.0% | 0.0262% | $0.0011 |
| 111568 | 26.1% | 0.0220% | $0.0008 |
| 111569 | 22.1% | 0.0222% | $0.0007 |
| 111570 | 18.5% | 0.0000% | $0.0000 |
| 111571 | 21.0% | 0.0853% | $0.0024 |

---

## 4.8 SUMMARY FOR MODEL TRAINING

**Recommended Training Data:**
- Bids (Dec-Jan, non-zero): 193,756
- Views (Oct+, deduped): 479,516
- Clicks (Oct+): 199

**Feature Summary:**
- internal_adspace_id: 9 unique
- geo_region_name: 65 unique (11.4% null)
- browser_code: 9 unique (94% = code 14)

**Key Metrics:**
- Overall Win Rate: 30.4%
- Overall CTR: 0.0346%
- Avg CPC: $13.36
- Expected Revenue per Bid: $0.0014

---

## SEPTEMBER CLICK QUALITY CHECK

**Click Count Comparison:**
- September clicks: 49
- Oct+ clicks: 199

**CPC Statistics:**

| Period | Mean CPC | Median CPC |
|--------|----------|-----------|
| September | $10.65 | $7.75 |
| Oct+ | $13.36 | $7.00 |

**September Campaigns:**
- Unique campaigns: 20

**Oct+ Campaigns:**
- Unique campaigns: 47

**September clicks by adspace:**

| Adspace | Clicks |
|---------|--------|
| 111564 | 24 |
| 111565 | 12 |
| 111563 | 8 |
| 111566 | 2 |
| 111568 | 1 |
| 111567 | 1 |
| 111569 | 1 |

**Oct+ clicks by adspace:**

| Adspace | Clicks |
|---------|--------|
| 111564 | 106 |
| 111565 | 41 |
| 111563 | 22 |
| 111566 | 15 |
| 111567 | 5 |
| 111571 | 4 |
| 110201 | 3 |
| 111568 | 2 |
| 111569 | 1 |

---
