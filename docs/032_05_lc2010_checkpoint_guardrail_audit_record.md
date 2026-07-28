# 032_05 LC2010 checkpoint-selection guardrail audit record

## Scope

- Pure offline audit of 032_04 checkpoint evaluation results.
- No training, no DSSAT rerun, no reward change.
- Expert yield reference: `8739.0` kg/ha.

## Guardrail pair summary

yield_guardrail nstress_guardrail  selected_seeds  total_seeds  mean_yield  mean_irrigation     mean_n  mean_max_nstres
   85pct_expert     loose_nstress               3            3 8209.718424             80.0 146.666667         0.142437
   85pct_expert  moderate_nstress               3            3 8340.733236            110.0 186.666667         0.048217
   85pct_expert    strict_nstress               3            3 8306.425171            110.0 186.666667         0.012191
   90pct_expert     loose_nstress               3            3 8209.718424             80.0 146.666667         0.142437
   90pct_expert  moderate_nstress               3            3 8340.733236            110.0 186.666667         0.048217
   90pct_expert    strict_nstress               3            3 8306.425171            110.0 186.666667         0.012191
   95pct_expert     loose_nstress               3            3 8340.733236            110.0 186.666667         0.048217
   95pct_expert  moderate_nstress               3            3 8340.733236            110.0 186.666667         0.048217
   95pct_expert    strict_nstress               2            3 8327.659302             90.0 200.000000         0.012191

## Selected checkpoint table

yield_guardrail nstress_guardrail  seed       selection_status  eligible_count  checkpoint_step  final_grnwt  yield_gap_to_expert  total_irrigation  total_n     PFP_N  reward_stress_aware_sum  max_nstres  nstres_days_gt_0p001                                                                                  action_sequence
   95pct_expert    strict_nstress     0               selected               3          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   95pct_expert    strict_nstress     1               selected               2          10000.0  8317.421875          -421.578125             135.0    240.0 34.655924                 0.934953    0.012191                   3.0                                          DAP1 I45/N120; DAP8 I45/N80; DAP16 I45/N0; DAP19 I0/N40
   95pct_expert    strict_nstress     2 no_eligible_checkpoint               0              NaN          NaN                  NaN               NaN      NaN       NaN                      NaN         NaN                   NaN                                                                                              NaN
   95pct_expert  moderate_nstress     0               selected               3          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   95pct_expert  moderate_nstress     1               selected               2          10000.0  8317.421875          -421.578125             135.0    240.0 34.655924                 0.934953    0.012191                   3.0                                          DAP1 I45/N120; DAP8 I45/N80; DAP16 I45/N0; DAP19 I0/N40
   95pct_expert  moderate_nstress     2               selected               1          25000.0  8366.881104          -372.118896             150.0    160.0 52.293007                 1.052667    0.120270                  30.0 DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0; DAP36 I15/N0; DAP63 I0/N80
   95pct_expert     loose_nstress     0               selected               3          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   95pct_expert     loose_nstress     1               selected               2          10000.0  8317.421875          -421.578125             135.0    240.0 34.655924                 0.934953    0.012191                   3.0                                          DAP1 I45/N120; DAP8 I45/N80; DAP16 I45/N0; DAP19 I0/N40
   95pct_expert     loose_nstress     2               selected               1          25000.0  8366.881104          -372.118896             150.0    160.0 52.293007                 1.052667    0.120270                  30.0 DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0; DAP36 I15/N0; DAP63 I0/N80
   90pct_expert    strict_nstress     0               selected               4          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   90pct_expert    strict_nstress     1               selected               2          10000.0  8317.421875          -421.578125             135.0    240.0 34.655924                 0.934953    0.012191                   3.0                                          DAP1 I45/N120; DAP8 I45/N80; DAP16 I45/N0; DAP19 I0/N40
   90pct_expert    strict_nstress     2               selected               1          20000.0  8263.956909          -475.043091             150.0    160.0 51.649731                 1.036405    0.012191                   3.0                            DAP1 I45/N40; DAP8 I30/N80; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0
   90pct_expert  moderate_nstress     0               selected               4          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   90pct_expert  moderate_nstress     1               selected               2          10000.0  8317.421875          -421.578125             135.0    240.0 34.655924                 0.934953    0.012191                   3.0                                          DAP1 I45/N120; DAP8 I45/N80; DAP16 I45/N0; DAP19 I0/N40
   90pct_expert  moderate_nstress     2               selected               2          25000.0  8366.881104          -372.118896             150.0    160.0 52.293007                 1.052667    0.120270                  30.0 DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0; DAP36 I15/N0; DAP63 I0/N80
   90pct_expert     loose_nstress     0               selected               4          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   90pct_expert     loose_nstress     1               selected               4          25000.0  7924.377441          -814.622559              45.0    120.0 66.036479                 1.161452    0.294850                  25.0                                                                                    DAP1 I45/N120
   90pct_expert     loose_nstress     2               selected               2          25000.0  8366.881104          -372.118896             150.0    160.0 52.293007                 1.052667    0.120270                  30.0 DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0; DAP36 I15/N0; DAP63 I0/N80
   85pct_expert    strict_nstress     0               selected               4          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   85pct_expert    strict_nstress     1               selected               2          10000.0  8317.421875          -421.578125             135.0    240.0 34.655924                 0.934953    0.012191                   3.0                                          DAP1 I45/N120; DAP8 I45/N80; DAP16 I45/N0; DAP19 I0/N40
   85pct_expert    strict_nstress     2               selected               1          20000.0  8263.956909          -475.043091             150.0    160.0 51.649731                 1.036405    0.012191                   3.0                            DAP1 I45/N40; DAP8 I30/N80; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0
   85pct_expert  moderate_nstress     0               selected               4          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   85pct_expert  moderate_nstress     1               selected               2          10000.0  8317.421875          -421.578125             135.0    240.0 34.655924                 0.934953    0.012191                   3.0                                          DAP1 I45/N120; DAP8 I45/N80; DAP16 I45/N0; DAP19 I0/N40
   85pct_expert  moderate_nstress     2               selected               2          25000.0  8366.881104          -372.118896             150.0    160.0 52.293007                 1.052667    0.120270                  30.0 DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0; DAP36 I15/N0; DAP63 I0/N80
   85pct_expert     loose_nstress     0               selected               5          50000.0  8337.896729          -401.103271              45.0    160.0 52.111855                 1.163588    0.012191                   3.0                                                           DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120
   85pct_expert     loose_nstress     1               selected               4          25000.0  7924.377441          -814.622559              45.0    120.0 66.036479                 1.161452    0.294850                  25.0                                                                                    DAP1 I45/N120
   85pct_expert     loose_nstress     2               selected               3          25000.0  8366.881104          -372.118896             150.0    160.0 52.293007                 1.052667    0.120270                  30.0 DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0; DAP36 I15/N0; DAP63 I0/N80

## Interpretation

- A guardrail pair is useful only if it selects eligible checkpoints for at least 2/3 seeds.
- Strict nitrogen-stress guardrails test whether high-yield candidates also keep NSTRES near zero.
- Loose guardrails are diagnostic only; they should not be used to claim agronomic reliability if selected policies still have sustained NSTRES.
