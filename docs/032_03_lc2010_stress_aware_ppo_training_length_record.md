# 032_03 LC2010 stress-aware PPO training-length record

## Scope

- LCA2010 seed0 only.
- MaskablePPO only.
- Same reward/action/constraint design as 032_00.
- Total timesteps: 100,000.
- Checkpoints: 10k, 25k, 50k, 75k, 100k.
- No checkpoint cherry-picking; full trajectory reported.

## Training checkpoint inventory

station_code  year  seed  checkpoint_step run_status                                                                                                                         model_path                                                     model_sha256
         LCA  2010     0            10000         ok  benchmark_results/032_03_lc2010_stress_aware_ppo_training_length/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt10000.zip b84716b00cb6362c0d65e0f2b8a1ae13b71655fce00929bd96e1616896203ce2
         LCA  2010     0            25000         ok  benchmark_results/032_03_lc2010_stress_aware_ppo_training_length/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt25000.zip db6635f08d98aa8f9b03dcf27591832afd5e982279c378367116953341a0c3e5
         LCA  2010     0            50000         ok  benchmark_results/032_03_lc2010_stress_aware_ppo_training_length/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip 29fd9cf5d49f2f65be21db4e6d39ec0dce6b3551b9af1de537e7aa9c0053034e
         LCA  2010     0            75000         ok  benchmark_results/032_03_lc2010_stress_aware_ppo_training_length/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt75000.zip 58fd239dd37bff45ea6d6115e4c5814ce9fdca590e80907295b76b1a2a9fe0fe
         LCA  2010     0           100000         ok benchmark_results/032_03_lc2010_stress_aware_ppo_training_length/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt100000.zip 86e4c0010cd3ab457b0346bbc68dcc46fa266fe6e128e7b46b4cbbded3d9d6a7

## Checkpoint evaluation

 checkpoint_step run_status  final_grnwt  total_irrigation  total_n  reward_stress_aware_sum  stress_relief_bonus_sum_unscaled  early_dap1_10_irrigation  early_dap1_10_n  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap                                                                                               action_sequence
           10000         ok  8273.141479             135.0    240.0                 0.927956                        148.500006                      90.0            160.0                       4              3                     1            1                                                       DAP1 I45/N80; DAP8 I45/N80; DAP15 I30/N80; DAP22 I15/N0
           25000         ok  8325.946655             150.0    240.0                 0.921284                        149.984507                      90.0            200.0                       6              3                     1            2 DAP1 I45/N0; DAP2 I0/N120; DAP8 I45/N80; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0
           50000         ok  8337.896729              45.0    160.0                 1.163588                        148.500006                      45.0            160.0                       1              2                     1            2                                                                        DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
           75000         ok  7831.497803              60.0    120.0                 1.130277                        148.500006                      60.0            120.0                       2              2                     1            1                                                                        DAP1 I45/N80; DAP8 I0/N40; DAP9 I15/N0
          100000         ok  7163.814697              60.0     80.0                 1.087983                        148.500006                      60.0             80.0                       2              2                     1            2                                                            DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0

## First-read branch

- Highest reward checkpoint: 50000.
- Compare against 032_01 `stress_triggered` rule: reward 1.033552, I30/N240, yield 8273.04.
- If no checkpoint approaches lower-water high-reward behavior, longer training alone is not enough.

## Interpretation

- Longer PPO training changes the result materially: the 50k checkpoint reaches the best stress-aware reward in this run (`1.163588`) with much lower resource use (`I45/N160`) than the 5k PPO from 032_00 (`I60/N240`) and the 032_01 stress-triggered rule (`I30/N240`).
- The 50k checkpoint therefore provides a positive signal that the free-timing PPO can learn a lower-input LC2010 strategy under the 032 stress-aware reward, rather than merely reproducing random/full-budget behavior.
- The trajectory is not monotonic. Continuing to 75k and 100k reduces yield substantially (`7831.50` and `7163.81 kg/ha`), so this run does not support using the final 100k checkpoint by default.
- This is still only LCA2010 seed0. It should not be generalized to other seeds, years, or stations before a fixed checkpoint-selection/validation rule is pre-registered.

## Next-step implication

- Do not expand the exact 50k checkpoint result directly to all stations/years.
- Next task should define a small validation protocol for free-timing PPO training length/checkpoint selection, e.g. repeat LC2010 with additional seeds or use a fixed validation-year/checkpoint-selection rule, before producing presentation-level claims.
