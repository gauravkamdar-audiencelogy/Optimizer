# RTB Optimizer - Project Notes

## Auction Mechanics

**Auction Type:** First-Price Auction
- Winner pays their bid amount (not second-price)
- For won bids: `bid.publisher_payout == view.publisher_payout`
- Bid shading is important to avoid overpaying

## Data Model

### Record Types

| rec_type | Description | Key Fields |
|----------|-------------|------------|
| `bid` | Bid request we submitted | `publisher_payout` = our bid offer |
| `View` | Won impression (ad shown) | `publisher_payout` = clearing price (same as bid in first-price) |
| `link` | Click on impression | `advertiser_spend` = CPC revenue |

### Win/Loss Determination
- **WON**: Bid has matching view via `internal_txn_id`
- **LOST**: Bid has NO matching view (we bid too low)
- Current win rate: ~30%

### Data Relationships
```
Bid (207K) --[internal_txn_id]--> View (578K) --[internal_txn_id]--> Click (248)
     |                                |
     |  30% match (won)               |  0.04% match (clicked)
     |  70% no match (lost)           |
```

### Key Fields

**publisher_payout:**
- In bids: Our bid amount (what we offered)
- In views: Clearing price (what we paid = same as bid in first-price)
- Format: Postgres array `{7.50000}` or `{7.50000,7.50000}` for multi-slot

**internal_txn_id:**
- Links records together
- Format: Postgres array `{uuid1,uuid2}` for multi-ad slots
- Must check ANY match, not just first

**advertiser_spend:**
- In clicks: CPC (cost per click) - our revenue
- Average: ~$12.82

## Date Ranges

- **Bids**: Dec 10, 2025 onwards (recent data)
- **Views**: Sept 15, 2025 onwards (includes pre-bid period)
- **Clicks**: Sept 15, 2025 onwards

Views before Dec 10 have no corresponding bids but still useful for CTR training.

## Current V3 Bid Formula

```
bid = EV_cpm × (1 - margin) × win_rate_adjustment

Where:
- EV_cpm = CTR × avg_CPC × 1000
- margin = 30% (target)
- win_rate_adjustment = 1 + (target_wr - empirical_wr) × sensitivity
  - Bounded [0.8, 1.2]
  - If winning too much → bid lower
  - If winning too little → bid higher
```

## Data Quality Notes

### Filtering Applied
1. **Bids**: Keep `bid_value > 0` (all bids have offers, no filtering needed)
2. **Views**:
   - Deduplicate by `log_txnid` (removes ~45K duplicates)
   - Filter `clearing_price <= 0` (removes ~229 invalid)
3. **Clicks**: No filtering needed

### Known Issues Fixed
- Click matching now checks ALL `internal_txn_id` values, not just first
- Adaptive binning for CTR calibration metrics (extreme class imbalance)

## Feature Selection

**Anchor feature:** `internal_adspace_id` (always included)

**Auto-excluded (data-driven):**
- `geo_country_code2`: 98.7% US (no variance)
- `domain`: 100% www.drugs.com (single domain)
- `browser_code`: 93.9% Chrome (no variance)
- `hour_of_day`, `day_of_week`: Low signal score

**Selected:** `internal_adspace_id`, `geo_region_name`, `os_code`

## SSP: Drugs.com

Single publisher/SSP for now. Project structure prepared for future multi-SSP support:
- `data_drugs/` - Data for drugs.com
- `EDA_drugs/` - EDA notebooks for drugs.com
- `meta_data_drugs/` - Metadata for drugs.com
