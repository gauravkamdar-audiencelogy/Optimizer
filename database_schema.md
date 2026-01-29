┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           RTB OPTIMIZER DATABASE SCHEMA                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│                              ┌──────────────────┐                                                       │
│                              │   opt_master     │                                                       │
│                              │──────────────────│                                                       │
│                              │ PK opt_id        │                                                       │
│                              │    opt_code      │  "rtb_optimizer"                                      │
│                              │    opt_name      │  "RTB Optimizer"                                      │
│                              │    status        │                                                       │
│                              └────────┬─────────┘                                                       │
│                                       │                                                                 │
│                       ┌───────────────┼───────────────┐                                                 │
│                       │ 1:N           │           1:N │                                                 │
│                       ▼               │               ▼                                                 │
│  ┌──────────────────────────┐         │         ┌──────────────────────────┐                            │
│  │     opt_versions         │         │         │     opt_entities         │                            │
│  │──────────────────────────│         │         │──────────────────────────│                            │
│  │ PK version_id            │         │         │ PK entity_id             │                            │
│  │ FK opt_id                │         │         │ FK opt_id                │                            │
│  │    version_code          │ "v3"    │         │    entity_code           │ "drugs_com"                │
│  │    version_name          │         │         │    entity_name           │                            │
│  │    release_date          │         │         │    object_id             │ 1334                       │
│  │    status                │         │         │    object_type           │ "ssp"                      │
│  └────────────┬─────────────┘         │         │ FK current_version_id ───┼───────────┐                │
│               │                       │         │ FK active_run_id ────────┼─────────┐ │                │
│               │ 1:N                   │         │    s3_base_path          │         │ │                │
│               ▼                       │         │    status                │         │ │                │
│  ┌──────────────────────────┐         │         └────────────┬─────────────┘         │ │                │
│  │  opt_version_configs     │         │                      │                       │ │                │
│  │──────────────────────────│         │                      │ 1:N                   │ │                │
│  │ PK config_id             │         │                      │                       │ │                │
│  │ FK version_id            │         │         ┌────────────┴────────────┐          │ │                │
│  │    config_key            │         │         │                         │          │ │                │
│  │    config_section        │         │         ▼                         ▼          │ │                │
│  │    display_name          │         │  ┌──────────────────┐  ┌───────────────────┐ │ │                │
│  │    data_type             │         │  │opt_entity_configs│  │    opt_runs       │ │ │                │
│  │    default_value         │         │  │──────────────────│  │───────────────────│◀┘ │                │
│  │    min_value             │         │  │ PK id            │  │ PK run_id         │◀──┘                │
│  │    max_value             │         │  │ FK entity_id     │  │ FK entity_id      │                    │
│  │    is_required           │         │  │    config_key    │  │ FK version_id ────┼──┐                 │
│  │    is_ui_visible         │         │  │    config_value  │  │    run_code       │  │                 │
│  └──────────────────────────┘         │  │    updated_by    │  │    status         │  │                 │
│               ▲                       │  └──────────────────┘  │    triggered_by   │  │                 │
│               │                       │                        │    data_start     │  │                 │
│               │ (version_id FK)       │                        │    data_end       │  │                 │
│               └───────────────────────┼────────────────────────│    config_snapshot│  │                 │
│                                       │                        │    total_segments │  │                 │
│                                       │                        │    error_message  │  │                 │
│                                       │                        └─────────┬─────────┘  │                 │
│                                       │                                  │            │                 │
│                                       │            ┌─────────────────────┼────────────┘                 │
│                                       │            │                     │                              │
│                                       │            │  1:N                │ 1:N                          │
│                                       │            │     ┌───────────────┼─────────────┐                │
│                                       │            │     │               │             │                │
│                                       │            │     ▼               ▼             ▼                │
│                                       │  ┌──────────────────┐ ┌──────────────────┐ ┌───────────────────┐│
│                                       │  │ opt_run_features │ │ opt_run_outputs  │ │opt_run_feat_macros││
│                                       │  │──────────────────│ │──────────────────│ │───────────────────││
│                                       │  │ PK id            │ │ PK id            │ │ PK id             ││
│                                       │  │ FK run_id        │ │ FK run_id        │ │ FK run_id         ││
│                                       │  │    feature_name  │ │    output_type   │ │    macro_combo    ││
│                                       │  │    feature_order │ │    s3_bucket     │ │    macro_format   ││
│                                       │  │    is_anchor     │ │    s3_key        │ └───────────────────┘│
│                                       │  │    signal_score  │ │    row_count     │                      │
│                                       │  └──────────────────┘ └──────────────────┘                      │
│                                       │                                                                 │
│  ┌──────────────────────────┐         │                                                                 │
│  │   opt_feature_macros     │◀────────┘  (Lookup table, FK to opt_master)                               │
│  │──────────────────────────│                                                                           │
│  │ PK id                    │                                                                           │
│  │ FK opt_id                │                                                                           │
│  │    feature_name          │  "internal_adspace_id"                                                    │
│  │    macro_template        │  "[ADX_PLACEMENT_ID]"                                                     │
│  │    macro_format          │  "adx"                                                                    │
│  └──────────────────────────┘                                                                           │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘






                    opt_master
                        │
           ┌────────────┼────────────┐
           │            │            │
           ▼            ▼            ▼
    opt_versions   opt_entities   opt_feature_macros
           │            │
           │    ┌───────┴───────┐
           │    │               │
           │    ▼               ▼
           │  opt_entity    opt_runs ◄─────────┐
           │  _configs          │              │
           │                    │              │
           └────────────────────┤ (version_id) │
                                │              │
                    ┌───────────┼───────────┐  │
                    │           │           │  │
                    ▼           ▼           ▼  │
             opt_run_     opt_run_    opt_run_ │
             features     outputs     feature_ │
                                      macros   │
                                               │
                    opt_entities.active_run_id─┘