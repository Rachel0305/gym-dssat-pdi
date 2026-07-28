# 032_18 LC missing DSSAT-auto daily completion record

## Status

- Target rows: 3.
- Generated rows: 3.
- Failed rows: 0.
- PPO/DQN training: 0.
- Candidate model reselection: none.
- Original DSSAT input files were not modified; rendered per-run templates and snapshots are stored under 032_18.

## Target station-years

| station_code | site | year |
| --- | --- | --- |
| LCA | LC | 2008 |
| LCA | LC | 2009 |
| LCA | LC | 2011 |

## Manifest

| station_code | site | year | scenario | status | details |
| --- | --- | --- | --- | --- | --- |
| LCA | LC | 2008 | dssat_auto | generated_032_18_true_dssat_auto | benchmark_results/032_18_lc_missing_dssat_auto_daily_completion/snapshots/LCA/2008/dssat_auto |
| LCA | LC | 2009 | dssat_auto | generated_032_18_true_dssat_auto | benchmark_results/032_18_lc_missing_dssat_auto_daily_completion/snapshots/LCA/2009/dssat_auto |
| LCA | LC | 2011 | dssat_auto | generated_032_18_true_dssat_auto | benchmark_results/032_18_lc_missing_dssat_auto_daily_completion/snapshots/LCA/2011/dssat_auto |

## Status counts

| status | n |
| --- | --- |
| generated_032_18_true_dssat_auto | 3 |

## Generated summary

| station_code | site | year | scenario | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | max_water_stress | max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LCA | LC | 2008 | dssat_auto | 3080.5914 | 0.0 | 0.0 | 1.12 |  | 0.0 | 0.4804 |
| LCA | LC | 2009 | dssat_auto | 2991.9034 | 0.0 | 0.0 | 1.09 |  | 0.0 | 0.4796 |
| LCA | LC | 2011 | dssat_auto | 2861.1777 | 0.0 | 0.0 | 1.07 |  | 0.0 | 0.4839 |

## Daily row counts

| year | scenario | daily_rows |
| --- | --- | --- |
| 2008 | dssat_auto | 103 |
| 2009 | dssat_auto | 103 |
| 2011 | dssat_auto | 102 |

## Next use

- 032_17 should read `benchmark_results/032_18_lc_missing_dssat_auto_daily_completion/evaluation/032_18_generated_dssat_auto_daily.csv` as a supplemental auto daily source.
- This task only fills missing baseline evidence; it does not alter PPO candidate outputs.
