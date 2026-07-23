# 031_32 Four-site free-timing MaskablePPO checkpoint smoke record

## Scope

- Four target stations: HLA2015, FQA2016, LCA2010, YCA2014.
- Seed0 only.
- Short checkpoint smoke only; no scientific success claim.
- Free timing; no expert-DAP windows.
- Reward/action/constraints unchanged from the frozen 031 MaskablePPO line.

## Frozen reward and constraints

Reward: `0.001 * (0.158 * final_yield_at_harvest - 1.1 * irrigation - 1.58 * nitrogen)`.

Action grid: irrigation `[0,6,12,18,24]`; nitrogen `[0,40,80,120,160]`.

Caps/guards: I<=160, N<=250, min interval 7 days, irrigation DAP1-120, nitrogen DAP1-90.

## Checkpoint inventory

station_code  year  seed  checkpoint_step run_status                                                                                                                    model_path                                                     model_sha256
         HLA  2015     0              512         ok benchmark_results/031_32_four_site_free_timing_maskableppo_checkpoint_smoke/models/HLA/HLA_2015_maskableppo_seed0_ckpt512.zip fdbc5c1fb945b61af2e616510d35eb003534104108fbba3e717c8e256788ee9d
         FQA  2016     0              512         ok benchmark_results/031_32_four_site_free_timing_maskableppo_checkpoint_smoke/models/FQA/FQA_2016_maskableppo_seed0_ckpt512.zip e3650e5bc0174e884bc4e718293b18856cc4c9bf781e91a9092e24c5486b306e
         LCA  2010     0              512         ok benchmark_results/031_32_four_site_free_timing_maskableppo_checkpoint_smoke/models/LCA/LCA_2010_maskableppo_seed0_ckpt512.zip 07c4db06f7a65ec6b816f0996e07e4f775f18fbbb9dd46f22980d457263640a5
         YCA  2014     0              512         ok benchmark_results/031_32_four_site_free_timing_maskableppo_checkpoint_smoke/models/YCA/YCA_2014_maskableppo_seed0_ckpt512.zip edc5b5dff012d47f4c68de64f384331a47a904f2c1992024c072b417fcf02469

## Evaluation summary

station_code  year  seed  checkpoint_step run_status  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N  early_dap1_10_irrigation  early_dap1_10_n  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                                                                                                                                                                                                                                                              action_sequence
         HLA  2015     0              512         ok  6787.197266             144.0    240.0    5443.197266 28.279989                      18.0            200.0                          0                   42 DAP1 I6/N160; DAP8 I12/N0; DAP9 I0/N40; DAP15 I12/N0; DAP16 I0/N40; DAP22 I12/N0; DAP29 I12/N0; DAP36 I12/N0; DAP43 I12/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0; DAP99 I6/N0; DAP106 I6/N0; DAP113 I6/N0; DAP120 I6/N0
         FQA  2016     0              512         ok  7071.733398             138.0    240.0    5733.733398 29.465556                      30.0            200.0                          6                    0                                                           DAP1 I6/N160; DAP8 I24/N0; DAP9 I0/N40; DAP15 I24/N0; DAP16 I0/N40; DAP22 I24/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0
         LCA  2010     0              512         ok  8348.992310             156.0    240.0    6992.992310 34.787468                      24.0            160.0                          0                    0                                                                           DAP1 I6/N120; DAP8 I18/N40; DAP15 I12/N0; DAP16 I0/N80; DAP22 I12/N0; DAP29 I12/N0; DAP36 I12/N0; DAP43 I12/N0; DAP50 I12/N0; DAP57 I12/N0; DAP64 I12/N0; DAP71 I12/N0; DAP78 I12/N0; DAP85 I12/N0
         YCA  2014     0              512         ok  9384.971924             138.0    240.0    8046.971924 39.104050                      36.0            200.0                          0                    0                                                         DAP1 I18/N160; DAP8 I18/N40; DAP15 I18/N0; DAP16 I0/N40; DAP22 I18/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0; DAP99 I6/N0

## Pass/fail

- all_four_station_smoke_pass = True

## Interpretation boundary

031_32 only verifies that four target stations can execute the frozen free-timing MaskablePPO pipeline. Long training and all-year transfer require a separate approved task.
