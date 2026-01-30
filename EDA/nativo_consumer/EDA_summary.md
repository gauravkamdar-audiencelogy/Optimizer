# Nativo Consumer EDA Summary

Generated: 2026-01-29

## Data Files

| File | Size | Rows | Notes |
|------|------|------|-------|
| data_nativo_consumer_full.csv | 14GB | ~17.3M | Full dataset |
| data_nativo_consumer.csv | 1.4GB | ~1.7M | 10% sample (breaks relationships) |

**WARNING**: The 10% sample is a random sample that breaks bid-view relationships. Use full file for production analysis.

---

## Key Findings (from EDA_1)

### Date Range
- Start: September 15, 2025
- End: January 27, 2026
- Total days: ~134

### rec_type Distribution
| rec_type | Count (sample) | Percentage | First Seen |
|----------|----------------|------------|------------|
| bid | 1,496,306 | 86.24% | Dec 10, 2025 |
| View | 237,325 | 13.68% | Sep 15, 2025 |
| link | 1,340 | 0.08% | Sep 15, 2025 |
| lead | 0 | 0% | - |

**Note**: `bid` logging started December 10, 2025. Data before this date only has views/clicks.

### Monthly Breakdown
```
Month       View      bid    link
2025-09    21,995       0     94
2025-10    42,776       0    184
2025-11    51,776       0    225
2025-12    31,153     982    181
2026-01    89,625 1,495,324  656
```

### Win Rate (January 2026)
- Total bids: ~1.5M
- Total views: ~90K
- Estimated win rate: **~4.9%** (very low!)
- Compare to drugs.com: ~30%

### CTR
- Views: 237,325
- Clicks (link): 1,340
- CTR: **0.56%**
- Compare to drugs.com: ~0.04%

---

## Feature Analysis (Potential Segment Features)

### Useful Features
| Feature | Unique | Top Value | Top % | Notes |
|---------|--------|-----------|-------|-------|
| domain | 14,626 | cbsnews.com | 2.3% | High cardinality - bucket |
| browser_code | 15 | 143 | 61.1% | ✓ Candidate |
| os_code | 9 | -1.0 | 65.2% | ✓ Candidate |
| media_type | 2 | mobile | 69.5% | ✓ Candidate |
| geo_region_name | 65 | New York | 9.5% | ✓ Candidate |

### Excluded Features
| Feature | Reason |
|---------|--------|
| internal_adspace_id | 100% is -1 |
| geo_country_code2 | 100% is US |
| ad_format | single value |
| make_id | mostly -1 |
| model_id | mostly -1 |
| carrier_code | mostly -1 |

---

## Field Semantics (CRITICAL)

### In `bid` records:
| Field | Meaning |
|-------|---------|
| `bid_amount` | Floor price from SSP |
| `publisher_payout` | Our bid |

### In `view` records:
| Field | Meaning |
|-------|---------|
| `bid_amount` | Campaign bid (IGNORE) |
| `publisher_payout` | Our bid / clearing price |

---

## Heavy Columns (Drop to Reduce Size)

| Column | Est. Size | Notes |
|--------|-----------|-------|
| ua | 2.74 GB | User agent string |
| ref_bundle | 2.19 GB | Referrer bundle |
| source_filename | 1.90 GB | Internal tracking |
| server_host | 1.22 GB | Internal |
| load_ts | 1.14 GB | Load timestamp |

**Estimated reduction**: Dropping these saves ~9GB (64% reduction).

---

## Recommendations

### Data Period
- **Use January 2026+** for bid landscape (has bid records with floor prices)
- **Use all data** for CTR model (views + clicks from Sep 2025)
- Consider discarding Sept-Nov 2025 if patterns differ significantly

### Features for Segmentation
1. `browser_code` (15 values)
2. `os_code` (9 values)
3. `media_type` (2 values)
4. `geo_region_name` (65 values)
5. `domain` - needs bucketing (14K+ values)

### Config Differences from drugs.com
| Setting | drugs.com | nativo_consumer |
|---------|-----------|-----------------|
| `npi_exists` | true | false |
| `floor_available` | false | true |
| Expected win rate | 30% | 4.9% (needs aggressive exploration) |

---

## Known Issues

### Full File Analysis
- EDA_2 script hangs on 14GB file (memory/processing issue)
- Need chunked processing approach
- Alternative: Create reduced file first, then analyze

### Linking Records
- Use `internal_txn_id` to link bid → view → link records
- NOT `log_txnid` (doesn't work)
- PostgreSQL JSON format needs parsing: `{"uuid"}` → `uuid`

---

## Data Volume Recommendation

**Use January 2026 only for initial run:**
- Sept-Nov 2025 has NO bid records (can't build bid landscape)
- Jan 2026 has: 1.5M bids, 90K views, 656 clicks
- This is MORE data than drugs.com had (~200K bids, 262 clicks)
- Reduces file from 17M → ~2.5M rows (85% smaller)

**When to add older data:**
- CTR model needs more clicks (656 may be enough)
- Seasonal pattern analysis

---

## Next Steps

1. ~~Create config file~~ - Done: `optimizer_config_nativo_consumer.yaml`
2. **Create reduced data file**: Run `EDA_nativo_consumer/create_reduced_file.py` OR filter to Jan 2026+
3. **Run optimizer**: Test with volume_first strategy (4.9% WR needs aggressive exploration)

## Scripts Created

| Script | Purpose | Status |
|--------|---------|--------|
| `EDA_1_data_overview.py` | Basic stats on 10% sample | ✓ Ran |
| `EDA_2_full_analysis.py` | Full file analysis | ✗ Hangs on 14GB file |
| `EDA_2_chunked.py` | Memory-efficient chunked version | Ready to run |
| `create_reduced_file.py` | Drop heavy columns, create smaller file | Ready to run |
