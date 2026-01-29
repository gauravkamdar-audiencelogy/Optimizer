# System Flow Diagram

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA SOURCES                                       │
│                                                                                       │
│   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐                  │
│   │  SNOWFLAKE  │          │  NPI FILES  │          │   CONFIG    │                  │
│   │  (RTB Logs) │          │  (Clicks)   │          │   (YAML)    │                  │
│   └──────┬──────┘          └──────┬──────┘          └──────┬──────┘                  │
│          │                        │                        │                          │
└──────────┼────────────────────────┼────────────────────────┼──────────────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                     OPTIMIZER                                         │
│                                                                                       │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│   │    Load     │───▶│   Train     │───▶│  Calculate  │───▶│   Output    │           │
│   │    Data     │    │   Models    │    │    Bids     │    │   Files     │           │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘           │
│                                                                   │                   │
└───────────────────────────────────────────────────────────────────┼───────────────────┘
                                                                    │
           ┌────────────────────────────────────────────────────────┼───────────────┐
           │                                                        │               │
           ▼                        ▼                               ▼               │
┌─────────────────┐      ┌─────────────────┐              ┌─────────────────┐       │
│       S3        │      │      MYSQL      │              │    VALIDATOR    │       │
│   (Storage)     │      │   (Audit Log)   │              │   (Pass/Fail)   │       │
└────────┬────────┘      └─────────────────┘              └─────────────────┘       │
         │                                                                           │
         ▼                                                                           │
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                      BIDDER                                           │
│                                                                                       │
│   Loads: suggested_bids.csv + npi_multipliers.csv + domain_multipliers.csv           │
│   Computes: final_bid = segment_bid × domain_mult × npi_mult                         │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Optimizer Pipeline (8 Steps)

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ 1.Load  │──▶│ 2.Join  │──▶│ 3.Select│──▶│ 4.Train │──▶│ 5.Calc  │──▶│ 6.Build │──▶│ 7.Valid │──▶│ 8.Upload│
│  Data   │   │ Win/Loss│   │ Features│   │ Models  │   │  Bids   │   │ Outputs │   │  Check  │   │   S3    │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
     │              │             │             │             │             │             │             │
     ▼              ▼             ▼             ▼             ▼             ▼             ▼             ▼
  Bids,         Win/Loss      Top 3        Win Rate,      Segment      CSV files     Pass/Fail    Manifest
  Views,        Labels        Features      CTR, Bid      Bids +       for          Gate         Updated
  Clicks                                   Landscape     Multipliers   Bidder
```

---

## Three-Level Bidding

```
                    BID REQUEST
                         │
                         ▼
              ┌─────────────────────┐
              │   SEGMENT LOOKUP    │
              │   (geo, os, etc.)   │
              │   → base_bid: $5    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   DOMAIN LOOKUP     │
              │   premium-site.com  │
              │   → mult: 1.3x      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    NPI LOOKUP       │
              │   (drugs.com only)  │
              │   → mult: 2.0x      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    FINAL BID        │
              │   $5 × 1.3 × 2.0    │
              │   = $13.00          │
              └─────────────────────┘
```

---

## Output Files

| File | Purpose | Consumer |
|------|---------|----------|
| `suggested_bids_*.csv` | segment → bid | Bidder |
| `npi_multipliers_*.csv` | NPI → multiplier | Bidder |
| `domain_multipliers_*.csv` | domain → multiplier | Bidder |
| `metrics_*.json` | Model diagnostics | Analysis |
| `validation_report_*.json` | Pass/Fail status | Deployment |

---

## Key Integrations

| System | Purpose | When |
|--------|---------|------|
| **Snowflake** | Source data (bid/view/click logs) | Export before run |
| **S3** | Store outputs + manifest for bidder | After validation passes |
| **MySQL** | Audit log (run history, metrics) | Start + end of run |

---

## Dataset Differences

| | drugs.com | nativo_consumer |
|---|-----------|-----------------|
| **Targeting** | HCP (healthcare) | Consumer |
| **NPI** | ✅ Yes | ❌ No |
| **Domain** | Single (drugs.com) | 11K+ domains |
| **Floor Prices** | ❌ No | ✅ Yes |
| **Click rec_type** | `click` | `link` |
