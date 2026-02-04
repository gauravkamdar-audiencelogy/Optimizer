
3. Optimizer Logic Flow

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                          OPTIMIZER DECISION FLOW                            │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │   1. LOAD DATA                                                              │
  │      ├── Bids (our offers)                                                  │
  │      ├── Views (won impressions)                                            │
  │      └── Clicks (user engagement)                                           │
  │                                                                             │
  │   2. JOIN → WIN/LOSS LABELS                                                 │
  │      bid + matching view = WON (we bid enough)                              │
  │      bid + no view = LOST (we bid too low)                                  │
  │                                                                             │
  │   3. SELECT FEATURES (data-driven)                                          │
  │      ├── Score each candidate feature                                       │
  │      ├── Exclude low-variance features (e.g., 100% same value)              │
  │      └── Pick top features by signal_score                                  │
  │                                                                             │
  │   4. TRAIN MODELS                                                           │
  │      ├── Win Rate Model: P(win | segment)                                   │
  │      │   └── Empirical + Bayesian shrinkage (main)                          │
  │      │   └── LogReg with isotonic calibration (diagnostics)                 │
  │      ├── CTR Model: P(click | view, segment)                                │
  │      │   └── Used for expected value calculation                            │
  │      ├── Bid Landscape: P(win | bid)                                        │
  │      │   └── Derives exploration multipliers                                │
  │      └── NPI/Domain Models: value multipliers                               │
  │                                                                             │
  │   5. CALCULATE BIDS (per segment)                                           │
  │      ┌─────────────────────────────────────────────────────────────────┐    │
  │      │  IF current_win_rate < target_win_rate:                         │    │
  │      │     → BID UP (we're losing too many auctions)                   │    │
  │      │     adjustment = 1.0 + (gap × up_multiplier)                    │    │
  │      │                                                                 │    │
  │      │  IF current_win_rate > target_win_rate:                         │    │
  │      │     → BID DOWN (we're overpaying)                               │    │
  │      │     adjustment = 1.0 - (gap × down_multiplier)                  │    │
  │      │                                                                 │    │
  │      │  final_segment_bid = base_bid × adjustment                      │    │
  │      │  clipped to [min_bid, max_bid]                                  │    │
  │      └─────────────────────────────────────────────────────────────────┘    │
  │                                                                             │
  │   6. AT BIDDER RUNTIME (not in optimizer)                                   │
  │      final_bid = segment_bid × domain_mult × npi_mult                       │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘
