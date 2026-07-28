# 032_10 LC multi-year free-timing MaskablePPO smoke record

## Status

- Smoke pass: `True`.
- All train years sampled: `True`.
- All checkpoints saved: `True`.
- All checkpoint-year evaluations complete: `True`.

## Scope

- Station: LCA / LC.
- Training years: 2005, 2006, 2007, 2008, 2009, 2010.
- Algorithm: MaskablePPO only.
- Weather forecast: not included.
- Seed: 0.
- Total timesteps: 12000.
- Checkpoints: 2000, 5000, 10000, 12000.
- Reward/actions/constraints inherited unchanged from 032_00.

## Training checkpoint inventory

station_code                   train_years  seed  checkpoint_step run_status                                                                                                                                  model_path                                                     model_sha256 notes
         LCA 2005,2006,2007,2008,2009,2010     0             2000         ok  benchmark_results/032_10_lc_multiyear_free_timing_ppo_smoke/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt2000.zip 56794c5ecacab734e34147a530845041b151019ea91961b3fb185810062b5c8c      
         LCA 2005,2006,2007,2008,2009,2010     0             5000         ok  benchmark_results/032_10_lc_multiyear_free_timing_ppo_smoke/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt5000.zip 667a5cdad516e4652263d3d436043597abc168021308ac587350efc978291b24      
         LCA 2005,2006,2007,2008,2009,2010     0            10000         ok benchmark_results/032_10_lc_multiyear_free_timing_ppo_smoke/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt10000.zip 20ba41b79edb6674d117201d63a8793231bbcc37a254e4342e3e2fe1980a0918      
         LCA 2005,2006,2007,2008,2009,2010     0            12000         ok benchmark_results/032_10_lc_multiyear_free_timing_ppo_smoke/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt12000.zip 05bdbe488f3461b530d4eb69d44f25e1ee32d834f75c6806a5a90209e42c83e1      

## Training year sampling

 year  episode_count
 2005             20
 2006             16
 2007             18
 2008             22
 2009             25
 2010             19

## Mean performance by checkpoint across LC2005-2010

 checkpoint_step  mean_final_grnwt  mean_total_irrigation  mean_total_n  mean_PFP_N  mean_reward  max_nstres  max_swfac
            2000       9481.892497                  135.0         240.0   39.507885     1.021059    0.025092   0.000000
            5000       9300.322978                  150.0         200.0   46.501615     1.087451    0.268728   0.430268
           10000       9433.478699                  150.0         240.0   39.306161     1.080508    0.012191   0.412890
           12000       9472.841593                  145.0         240.0   39.470173     1.075728    0.030359   0.412890

## Per-year checkpoint evaluation

 year  checkpoint_step run_status  final_grnwt  total_irrigation  total_n     PFP_N  reward_stress_aware_sum  max_swfac  max_nstres  swfac_days_gt_0p05  nstres_days_gt_0p05  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap                                                                                            action_sequence
 2005             2000         ok 12059.154053             135.0    240.0 50.246475                 1.427146   0.000000    0.025092                   0                    0                       5              3                     1            1            DAP1 I15/N120; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0
 2006             2000         ok  8989.793091             135.0    240.0 37.457471                 0.942187   0.000000    0.012191                   0                    0                       5              3                     1            1            DAP1 I15/N120; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0
 2007             2000         ok  7956.701660             135.0    240.0 33.152924                 0.782262   0.000000    0.012191                   0                    0                       5              3                     1            1            DAP1 I15/N120; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0
 2008             2000         ok 10398.839111             135.0    240.0 43.328496                 1.168209   0.000000    0.012191                   0                    0                       5              3                     1            1            DAP1 I15/N120; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0
 2009             2000         ok  9213.829346             135.0    240.0 38.390956                 0.977611   0.000000    0.012191                   0                    0                       5              3                     1            1            DAP1 I15/N120; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0
 2010             2000         ok  8273.037720             135.0    240.0 34.470990                 0.828940   0.000000    0.012191                   0                    0                       5              3                     1            1            DAP1 I15/N120; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0
 2005             5000         ok 11441.096191             150.0    200.0 57.205481                 1.425693   0.000000    0.268728                   0                   18                       5              2                     1            2              DAP1 I30/N0; DAP2 I0/N120; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP22 I30/N0; DAP29 I30/N0
 2006             5000         ok  8881.994629             150.0    200.0 44.409973                 1.021355   0.000000    0.012191                   0                    0                       5              2                     1            2              DAP1 I30/N0; DAP2 I0/N120; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP22 I30/N0; DAP29 I30/N0
 2007             5000         ok  7957.759399             150.0    200.0 39.788797                 0.875326   0.000000    0.012191                   0                    0                       5              2                     1            2              DAP1 I30/N0; DAP2 I0/N120; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP22 I30/N0; DAP29 I30/N0
 2008             5000         ok 10164.819946             150.0    200.0 50.824100                 1.224042   0.430268    0.223706                   2                    9                       5              2                     1            2              DAP1 I30/N0; DAP2 I0/N120; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP22 I30/N0; DAP29 I30/N0
 2009             5000         ok  9134.367676             150.0    200.0 45.671838                 1.061230   0.000000    0.012191                   0                    0                       5              2                     1            2              DAP1 I30/N0; DAP2 I0/N120; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP22 I30/N0; DAP29 I30/N0
 2010             5000         ok  8221.900024             150.0    200.0 41.109500                 0.917060   0.000000    0.012191                   0                    0                       5              2                     1            2              DAP1 I30/N0; DAP2 I0/N120; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP22 I30/N0; DAP29 I30/N0
 2005            10000         ok 11772.905273             150.0    240.0 49.053772                 1.467685   0.000000    0.012191                   0                    0                       5              4                     1            1            DAP1 I45/N40; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I15/N40
 2006            10000         ok  8969.985352             150.0    240.0 37.374939                 0.974739   0.000000    0.012191                   0                    0                       5              3                     1            2 DAP1 I30/N0; DAP2 I0/N80; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I30/N0
 2007            10000         ok  8010.323486             150.0    240.0 33.376348                 0.869931   0.000000    0.012191                   0                    0                       5              4                     1            1            DAP1 I45/N40; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I15/N40
 2008            10000         ok 10465.881348             150.0    240.0 43.607839                 1.257909   0.412890    0.012191                   2                    0                       5              4                     1            1            DAP1 I45/N40; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I15/N40
 2009            10000         ok  9138.184204             150.0    240.0 38.075768                 1.003029   0.000000    0.012191                   0                    0                       5              3                     1            2 DAP1 I30/N0; DAP2 I0/N80; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I30/N0
 2010            10000         ok  8243.592529             150.0    240.0 34.348302                 0.909756   0.000000    0.012191                   0                    0                       5              4                     1            1            DAP1 I45/N40; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I15/N40
 2005            12000         ok 12011.973877             150.0    240.0 50.049891                 1.455958   0.000000    0.030359                   0                    0                       5              3                     1            2 DAP1 I30/N0; DAP2 I0/N80; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I30/N0
 2006            12000         ok  8969.985352             150.0    240.0 37.374939                 0.974739   0.000000    0.012191                   0                    0                       5              3                     1            2 DAP1 I30/N0; DAP2 I0/N80; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I30/N0
 2007            12000         ok  8010.317993             135.0    240.0 33.376325                 0.886430   0.000000    0.012191                   0                    0                       4              4                     1            1             DAP1 I45/N40; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP23 I0/N40
 2008            12000         ok 10481.596680             135.0    240.0 43.673319                 1.276892   0.412890    0.012191                   2                    0                       4              4                     1            1             DAP1 I45/N40; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP23 I0/N40
 2009            12000         ok  9138.184204             150.0    240.0 38.075768                 1.003029   0.000000    0.012191                   0                    0                       5              3                     1            2 DAP1 I30/N0; DAP2 I0/N80; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I30/N0
 2010            12000         ok  8224.991455             150.0    240.0 34.270798                 0.857318   0.000000    0.012191                   0                    0                       5              3                     1            2 DAP1 I30/N0; DAP2 I0/N80; DAP8 I30/N0; DAP9 I0/N80; DAP15 I30/N0; DAP16 I0/N80; DAP22 I30/N0; DAP29 I30/N0

## Interpretation boundary

- This is a smoke test of the no-forecast multi-year training wrapper.
- It is not a final model-selection result.
- It does not evaluate LC2011-2020 or LC2021-2023.
- It does not prove cross-year transfer performance.
- If a follow-up changes reward, forecast inputs, training length, or checkpoint selection, it must be separately pre-registered.
