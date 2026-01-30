# System Flow Diagram

## End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                         │
│     ┌──────────┐                                                                    ┌──────────┐        │
│     │          │                                                                    │          │        │
│     │    UI    │                                                                    │  BIDDER  │        │
│     │          │                                                                    │          │        │
│     └────┬─────┘                                                                    └────▲─────┘        │
│          │                                                                               │              │
│          │ 1                                                                             │ 9            │
│          │ User triggers                                                                 │ Bids sent    │
│          │ optimizer run                                                                 │ to SSP       │
│          ▼                                                                               │              │
│     ┌──────────┐         2 Config read            ┌───────────────────────────────┐      │              │
│     │          │─────────────────────────────────▶│                               │      │              │
│     │  MYSQL   │                                  │                               │      │              │
│     │          │◀─────────────────────────────────│                               │      │              │
│     └──────────┘         6 Features recorded      │                               │      │              │
│                                                   │         OPTIMIZER             │      │              │
│                            Config read            │                               │      │              │
│                          ─────────────────────────│    ┌─────┐  ┌─────┐  ┌─────┐  │      │              │
│     ┌──────────┐                                  │    │Load │─▶│Proc │─▶│Model│  │      │              │
│     │          │         3 Data extracted         │    │Data │  │ess  │  │s    │  │      │              │
│     │SNOWFLAKE │─────────────────────────────────▶│    └─────┘  └─────┘  └──┬──┘  │      │              │
│     │          │                                  │                         │     │      │              │
│     └──────────┘                                  │                         ▼     │      │              │
│                                                   │  ┌────────────────────────┐   │      │              │
│                                                   │  │    OUTPUT FILES        │   │      │              │
│                                                   │  │ • suggested_bids.csv   │   │      │              │
│                                                   │  │ • domain_mult.csv      │   │      │              │
│                                                   │  │ • npi_mult.csv         │   │      │              │
│                                                   │  │ • metrics.json         │   │      │              │
│                                                   │  └───────────┬────────────┘   │      │              │
│                                                   │              │                │      │              │
│     ┌──────────┐         5 Files uploaded         │              │ 4              │      │              │
│     │          │◀─────────────────────────────────│──────────────┘                │      │              │
│     │    S3    │                                  └───────────────────────────────┘      │              │
│     │          │─────────────────────────────────────────────────────────────────────────┘              │
│     └──────────┘         7 Bidder loads files                                        8 Builds memcache  │
│                                                                                                         │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

┌───────────────────────────────┐ 
│         OPTIMIZER             │ 
│    ┌─────┐  ┌─────┐  ┌─────┐  │
│    │Load │─▶│Proc │─▶│Model│  │
│    │Data │  │ess  │  │s    │  │ 
│    └─────┘  └─────┘  └──┬──┘  │
│                         ▼     │ 
│  ┌────────────────────────┐   │      
│  │    OUTPUT FILES        │   │     
│  │ • suggested_bids.csv   │   │   
│  │ • domain_mult.csv      │   │    
│  │ • npi_mult.csv         │   │    
│  │ • metrics.json         │   │ 
│  └───────────┬────────────┘   │                 
│              │ 4              │
│──────────────┘                │
└───────────────────────────────┘
---

## Numbered Flow Sequence

| Step | From | To | Action |
|------|------|----|--------|
| ① | User | UI | Triggers optimizer run |
| ② | UI | MySQL | Config parameters logged |
| ③ | MySQL | Optimizer | Config read into memory |
| ④ | Snowflake | Optimizer | RTB data extracted (bids, views, clicks) |
| ⑤ | Optimizer | (internal) | Load → Process → Train Models → Generate Files |
| ⑥ | Optimizer | S3 | Output files uploaded after validation |
| ⑦ | Optimizer | MySQL | Selected features & metrics recorded |
| ⑧ | S3 | Bidder | Bidder loads CSV files → builds memcache |
| ⑨ | Bidder | SSP | Bids sent using: `segment_bid × domain_mult × npi_mult` |

---

## Optimizer Internal Pipeline (Step ⑤ Detail)

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  LOAD    │───▶│  JOIN    │───▶│ FEATURE  │───▶│  TRAIN   │───▶│ GENERATE │
│  DATA    │    │ WIN/LOSS │    │ ANALYSIS │    │  MODELS  │    │  OUTPUT  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼
  Bids +          Labels:        Auto-select      Win Rate       suggested_bids.csv
  Views +         won=1          top features     CTR Model      domain_mult.csv
  Clicks          lost=0         from data        Bid Landscape  npi_mult.csv
                                                  Domain/NPI     metrics.json
```

---

## Output Files Summary

| File | Content | Consumer |
|------|---------|----------|
| `suggested_bids_*.csv` | segment → base_bid | Bidder memcache |
| `domain_multipliers_*.csv` | domain → multiplier | Bidder domain cache |
| `npi_multipliers_*.csv` | NPI → multiplier | Bidder NPI cache |
| `selected_features_*.csv` | Features used this run | MySQL logging |
| `metrics_*.json` | Model diagnostics | Analysis / debugging |
| `validation_report_*.json` | Pass/Fail gate | Deployment decision |

---

## Three-Level Bid Calculation (Bidder Runtime)

```
   BID REQUEST arrives
          │
          ▼
   ┌──────────────┐
   │ MEMCACHE     │  segment_bid = $5.00
   │ (segments)   │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ DOMAIN CACHE │  × 1.3 (premium domain)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ NPI CACHE    │  × 2.0 (high-value NPI)
   └──────┬───────┘
          │
          ▼
   FINAL BID = $5.00 × 1.3 × 2.0 = $13.00
```
