# 031_34a Reselect 031_33 checkpoints by advisor metrics

## Scope

- No training.
- No DSSAT rerun.
- Re-screen all 031_33 checkpoints using advisor-facing known metrics.
- Known metrics here are yield and PFP_N only; WP_ET is unavailable in 031_33 because ETCP/snapshot was not saved.

## Baseline envelope

site  year  baseline_max_yield baseline_max_yield_scenario  baseline_max_wp_et baseline_max_wp_et_scenario  baseline_max_pfp_n baseline_max_pfp_n_scenario  expert_yield  expert_irrigation  expert_n  expert_wp_et  expert_pfp_n
  FQ  2016              8012.0                  dssat_auto                2.50             recorded_farmer                55.1             recorded_farmer        7940.0              198.8     247.0          2.35          32.1
 HLA  2015              7648.0                  dssat_auto                1.52                  dssat_auto                44.2             recorded_farmer        7603.0              266.1     300.0          1.49          25.3
  LC  2010              8739.0   official_extension_expert                3.13                        null                35.3   official_extension_expert        8739.0              198.8     247.0          3.10          35.3
  YC  2014              9418.0             recorded_farmer                2.64             recorded_farmer                38.0   official_extension_expert        9417.0              228.8     247.0          2.56          38.0

## Advisor-reselected checkpoint per station-seed

station_code  year  seed  checkpoint_step  final_grnwt  baseline_max_yield    gap_yield  total_irrigation  total_n      PFP_N  baseline_max_pfp_n  gap_pfp_n  known_any_metric_win known_winning_metrics              advisor_reselection_reason                                                                                                                                                                                                                                                        action_sequence
         FQA  2016     0            30000  7358.708496              8012.0  -653.291504             132.0    240.0  30.661285                55.1 -24.438715                 False                       no_known_metric_win_highest_known_score                                                                  DAP1 I6/N160; DAP8 I6/N0; DAP9 I0/N80; DAP15 I6/N0; DAP22 I6/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I18/N0; DAP78 I18/N0; DAP85 I18/N0; DAP92 I18/N0
         FQA  2016     1            10000  7311.703491              8012.0  -700.296509             156.0    240.0  30.465431                55.1 -24.634569                 False                       no_known_metric_win_highest_known_score                                                                                                             DAP1 I18/N40; DAP8 I6/N120; DAP15 I24/N40; DAP22 I24/N40; DAP29 I12/N0; DAP36 I12/N0; DAP43 I12/N0; DAP50 I12/N0; DAP57 I12/N0; DAP64 I12/N0; DAP71 I12/N0
         FQA  2016     2            10000  7415.475464              8012.0  -596.524536             156.0    240.0  30.897814                55.1 -24.202186                 False                       no_known_metric_win_highest_known_score                                      DAP1 I6/N0; DAP2 I0/N80; DAP8 I6/N0; DAP9 I0/N120; DAP15 I6/N0; DAP16 I0/N40; DAP22 I6/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I18/N0; DAP50 I18/N0; DAP57 I18/N0; DAP64 I18/N0; DAP71 I18/N0; DAP78 I18/N0; DAP85 I6/N0; DAP92 I6/N0
         HLA  2015     0            75000  6962.310181              7648.0  -685.689819              36.0    240.0  29.009626                44.2 -15.190374                 False                       no_known_metric_win_highest_known_score                                                                                                                                                                                       DAP1 I6/N40; DAP8 I6/N40; DAP15 I6/N40; DAP22 I6/N40; DAP29 I6/N40; DAP36 I6/N40
         HLA  2015     1            20000  6955.821533              7648.0  -692.178467             114.0    240.0  28.982590                44.2 -15.217410                 False                       no_known_metric_win_highest_known_score                                                             DAP1 I18/N160; DAP38 I18/N0; DAP45 I18/N0; DAP47 I0/N40; DAP52 I6/N0; DAP54 I0/N40; DAP59 I6/N0; DAP66 I6/N0; DAP73 I6/N0; DAP80 I6/N0; DAP87 I6/N0; DAP94 I6/N0; DAP101 I6/N0; DAP108 I6/N0; DAP115 I6/N0
         HLA  2015     2            75000  6962.408447              7648.0  -685.591553             108.0    240.0  29.010035                44.2 -15.189965                 False                       no_known_metric_win_highest_known_score DAP1 I6/N120; DAP8 I6/N0; DAP9 I0/N80; DAP15 I6/N0; DAP16 I0/N40; DAP22 I6/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0; DAP99 I6/N0; DAP106 I6/N0; DAP113 I6/N0; DAP120 I6/N0
         LCA  2010     0            50000  5281.353760              8739.0 -3457.646240              78.0     40.0 132.033844                35.3  96.733844                  True                 PFP_N             advisor_known_metric_winner                                                                                     DAP1 I6/N0; DAP2 I0/N40; DAP8 I6/N0; DAP15 I6/N0; DAP22 I6/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0
         LCA  2010     1           100000  7154.531250              8739.0 -1584.468750              12.0     80.0  89.431641                35.3  54.131641                  True                 PFP_N             advisor_known_metric_winner                                                                                                                                                                                                                                               DAP2 I6/N40; DAP8 I6/N40
         LCA  2010     2            10000  8348.992310              8739.0  -390.007690              78.0    240.0  34.787468                35.3  -0.512532                 False                       no_known_metric_win_highest_known_score                                                        DAP1 I6/N40; DAP8 I6/N0; DAP9 I0/N40; DAP15 I6/N0; DAP16 I0/N80; DAP22 I6/N0; DAP23 I0/N80; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0
         YCA  2014     0            20000  8256.719971              9418.0 -1161.280029              18.0    160.0  51.604500                38.0  13.604500                  True                 PFP_N             advisor_known_metric_winner                                                                                                                                                                                                                                 DAP1 I6/N80; DAP8 I6/N40; DAP15 I6/N40
         YCA  2014     1            50000  7624.693604              9418.0 -1793.306396               0.0    120.0  63.539113                38.0  25.539113                  True                 PFP_N             advisor_known_metric_winner                                                                                                                                                                                                                                                           DAP1 I0/N120
         YCA  2014     2           100000  2938.011169              9418.0 -6479.988831              12.0     40.0  73.450279                38.0  35.450279                  True                 PFP_N             advisor_known_metric_winner                                                                                                                                                                                                                                    DAP1 I6/N0; DAP2 I0/N40; DAP8 I6/N0

## Station-level counts

station_code  checkpoint_count  known_metric_winning_checkpoints  yield_winning_checkpoints  pfp_n_winning_checkpoints  selected_seed_count  selected_known_metric_winning_seeds  selected_yield_winning_seeds  selected_pfp_n_winning_seeds
         FQA                18                                 0                          0                          0                    3                                    0                             0                             0
         HLA                18                                 0                          0                          0                    3                                    0                             0                             0
         LCA                18                                 8                          0                          8                    3                                    2                             0                             2
         YCA                18                                17                          4                         17                    3                                    3                             0                             3

## Interpretation

- If a checkpoint wins in this audit, 031_33 already produced a candidate but reward-sum selection may not have selected the most advisor-aligned checkpoint.
- If no checkpoint wins across all checkpoints for a station, the current frozen PPO configuration did not produce a training-year known-metric winner for that station.
- WP_ET must be handled in 031_34 by rerunning frozen selected checkpoints with ETCP/snapshot outputs.


## Guardrailed reselection sensitivity

Because a pure one-metric rule can select low-yield high-PFP_N artifacts, this supplementary audit adds a yield guardrail before accepting a PFP_N win. This better matches the advisor wording: one metric wins while the other metrics remain close.

### Pass counts by yield guardrail

 yield_guardrail_fraction station_code  seed_count  pass_count
                     0.90          FQA           3           0
                     0.90          HLA           3           0
                     0.90          LCA           3           2
                     0.90          YCA           3           3
                     0.95          FQA           3           0
                     0.95          HLA           3           0
                     0.95          LCA           3           2
                     0.95          YCA           3           3
                     0.98          FQA           3           0
                     0.98          HLA           3           0
                     0.98          LCA           3           0
                     0.98          YCA           3           3

### 95% yield-guardrail selected checkpoints

station_code  seed  checkpoint_step  final_grnwt  baseline_max_yield  yield_ratio   gap_yield     PFP_N  gap_pfp_n  known_win_with_yield_guardrail winning_metrics  total_irrigation  total_n
         FQA     0            30000  7358.708496              8012.0     0.918461 -653.291504 30.661285 -24.438715                           False             NaN             132.0    240.0
         FQA     1            10000  7311.703491              8012.0     0.912594 -700.296509 30.465431 -24.634569                           False             NaN             156.0    240.0
         FQA     2            10000  7415.475464              8012.0     0.925546 -596.524536 30.897814 -24.202186                           False             NaN             156.0    240.0
         HLA     0            75000  6962.310181              7648.0     0.910344 -685.689819 29.009626 -15.190374                           False             NaN              36.0    240.0
         HLA     1            20000  6955.821533              7648.0     0.909495 -692.178467 28.982590 -15.217410                           False             NaN             114.0    240.0
         HLA     2            75000  6962.408447              7648.0     0.910357 -685.591553 29.010035 -15.189965                           False             NaN             108.0    240.0
         LCA     0            30000  8348.992310              8739.0     0.955372 -390.007690 41.744962   6.444962                            True           PFP_N              78.0    200.0
         LCA     1            75000  8348.992310              8739.0     0.955372 -390.007690 52.181202  16.881202                            True           PFP_N              78.0    160.0
         LCA     2            10000  8348.992310              8739.0     0.955372 -390.007690 34.787468  -0.512532                           False             NaN              78.0    240.0
         YCA     0            30000  9544.738159              9418.0     1.013457  126.738159 39.769742   1.769742                            True     yield;PFP_N             150.0    240.0
         YCA     1            10000  9243.496704              9418.0     0.981471 -174.503296 46.217484   8.217484                            True           PFP_N              48.0    200.0
         YCA     2            20000  9401.171875              9418.0     0.998213  -16.828125 39.171549   1.171549                            True           PFP_N             120.0    240.0

Interpretation: under a 95% of four-baseline-max yield guardrail, LC2010 has 2/3 passing seeds and YC2014 has 3/3 passing seeds; HLA2015 and FQ2016 still have 0/3 passing seeds.
