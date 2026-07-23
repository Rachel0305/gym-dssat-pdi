# 031_21 SY cross-year frozen transfer of free-timing MaskablePPO

## Scope

- Corrective task after 031_20: this is SY same-station cross-year frozen transfer, not cross-site transfer.
- No new training; load SYA2014 seed0/1/2 models from 031_17/031_18.
- Target years: SYA2012 and SYA2015, chosen because four baseline Summary.OUT metrics already exist from 028_05.
- Reuse four baseline rows only; old 028_05 RL candidate rows are not reused.
- Candidate WP_ET/PFP_N are parsed from DSSAT Summary.OUT snapshot for each new frozen evaluation.

## Model and constraint

- Algorithm: free-timing discrete MaskablePPO.
- Source train site-year: SYA2014.
- Action grid: irrigation [0, 6, 12, 18, 24] mm x nitrogen [0, 40, 80, 120, 160] kg/ha.
- Safety: I cap 160 mm, N cap 250 kg/ha, 7-day min interval, irrigation DAP 1-120, fertilization DAP 1-90.
- Reward used during original training: 0.001 * (0.158*final_yield - 1.1*I - 1.58*N).

## Candidate summary

 year  seed  final_grain_kg_ha  wp_et_kg_m3  pfp_n_kg_kg  irrigation_event_total_mm  nitrogen_event_total_kg_ha  max_water_stress_wspd  max_nitrogen_stress_nstd                                                                                                                                                                                                                                                       action_sequence
 2012     0        9982.597656         2.14         41.6                      108.0                       240.0                    0.0                  0.080320                         DAP1 I6/N120; DAP8 I6/N120; DAP15 I6/N0; DAP22 I6/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0; DAP99 I6/N0; DAP106 I6/N0; DAP113 I6/N0; DAP120 I6/N0
 2015     0       10967.564697         2.26         45.7                      108.0                       240.0                    0.0                  0.139460                         DAP1 I6/N120; DAP8 I6/N120; DAP15 I6/N0; DAP22 I6/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0; DAP99 I6/N0; DAP106 I6/N0; DAP113 I6/N0; DAP120 I6/N0
 2012     1        9981.777954         2.15         41.6                       90.0                       240.0                    0.0                  0.038054               DAP1 I6/N80; DAP8 I6/N0; DAP9 I0/N40; DAP15 I6/N0; DAP16 I0/N40; DAP22 I6/N0; DAP23 I0/N40; DAP29 I6/N0; DAP30 I0/N40; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0; DAP99 I6/N0
 2015     1       10980.637207         2.26         45.8                       96.0                       240.0                    0.0                  0.128589 DAP1 I6/N80; DAP8 I6/N0; DAP9 I0/N40; DAP15 I6/N0; DAP16 I0/N40; DAP22 I6/N0; DAP23 I0/N40; DAP29 I6/N0; DAP30 I0/N40; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0; DAP99 I6/N0; DAP106 I6/N0
 2012     2        9792.156982         2.10         40.8                      156.0                       240.0                    0.0                  0.199993                                                     DAP1 I12/N0; DAP2 I0/N120; DAP24 I12/N0; DAP31 I12/N0; DAP34 I0/N120; DAP38 I12/N0; DAP45 I12/N0; DAP52 I12/N0; DAP59 I12/N0; DAP66 I12/N0; DAP73 I12/N0; DAP80 I12/N0; DAP87 I12/N0; DAP94 I12/N0; DAP101 I12/N0
 2015     2       10518.276367         2.12         43.8                      156.0                       240.0                    0.0                  0.176459                                                      DAP1 I12/N0; DAP2 I0/N120; DAP19 I12/N0; DAP26 I12/N0; DAP33 I12/N0; DAP34 I0/N120; DAP40 I12/N0; DAP47 I12/N0; DAP54 I12/N0; DAP61 I12/N0; DAP68 I12/N0; DAP75 I12/N0; DAP82 I12/N0; DAP89 I12/N0; DAP96 I12/N0

## Gap versus four baselines

 year  seed   gap_yield  gap_wp_et  gap_pfp_n  yield_strict_win  wp_et_strict_win  pfp_n_strict_win  advisor_any_metric_strict_winner winning_metrics
 2012     0 -183.402344      -0.22        0.4             False             False              True                              True           PFP_N
 2015     0   76.564697      -0.07        9.4              True             False              True                              True     yield;PFP_N
 2012     1 -184.222046      -0.21        0.4             False             False              True                              True           PFP_N
 2015     1   89.637207      -0.07        9.5              True             False              True                              True     yield;PFP_N
 2012     2 -373.843018      -0.26       -0.4             False             False             False                             False                
 2015     2 -372.723633      -0.21        7.5             False             False              True                              True           PFP_N

## Four-baseline source

 year                  scenario  final_grain_kg_ha  wp_et_kg_m3  pfp_n_kg_kg  irrigation_event_total_mm  nitrogen_event_total_kg_ha
 2012                      null             7002.0         1.64          NaN                        0.0                         0.0
 2012           recorded_farmer            10166.0         2.36         41.2                        0.0                       293.0
 2012                dssat_auto             7016.0         1.60          NaN                       33.0                         0.0
 2012 official_extension_expert             9609.0         2.21         32.0                      266.1                       300.0
 2015                      null             6601.0         1.51          NaN                        0.0                         0.0
 2015           recorded_farmer             8973.0         2.01         36.3                        0.0                       293.0
 2015                dssat_auto             6868.0         1.52          NaN                       68.7                         0.0
 2015 official_extension_expert            10891.0         2.33         36.3                      266.1                       300.0

## Interpretation boundary

- This task answers only whether SYA2014 free-timing PPO frozen models transfer to SYA2012/SYA2015 under the same station.
- It does not answer cross-site generalization.
- It does not retrain or tune hyperparameters.
- If results are poor, the result is recorded as transfer failure rather than fixed in-place.
