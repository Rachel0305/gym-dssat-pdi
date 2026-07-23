# 031_14 DQN nitrogen-cost rerank mechanism audit record

## Scope

- No training.
- No new DSSAT calls.
- Input: 031_12 DSSAT counterfactual variants for SYA2014 seeds 0/1/2.
- Purpose: test whether 031_13 failure is explained by nitrogen penalty still being too weak, or by DQN not learning/using the better full-season ranking.

## Reward used for reranking

```text
R_lit(c_N) = 0.158 * final_grnwt - 1.1 * total_irrigation - c_N * total_n
```

Primary checked nitrogen costs: 0.79 and 1.58. Values 2.37 and 3.16 are diagnostic only.

## Winner by nitrogen cost and seed

 n_cost  seed   winning_variant   lit_score  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N
   0.79     0 stage_spread_N200 1482.185046 10882.183838              72.0    200.0    9810.183838 54.410919
   0.79     1 stage_spread_N200 1396.792793 10954.384766             160.0    200.0    9794.384766 54.771924
   0.79     2 stage_spread_N200 1391.642171 10921.785889             160.0    200.0    9761.785889 54.608929
   1.58     0 stage_spread_N200 1324.185046 10882.183838              72.0    200.0    9810.183838 54.410919
   1.58     1 stage_spread_N200 1238.792793 10954.384766             160.0    200.0    9794.384766 54.771924
   1.58     2 stage_spread_N200 1233.642171 10921.785889             160.0    200.0    9761.785889 54.608929
   2.37     0 stage_spread_N200 1166.185046 10882.183838              72.0    200.0    9810.183838 54.410919
   2.37     1 stage_spread_N200 1080.792793 10954.384766             160.0    200.0    9794.384766 54.771924
   2.37     2 stage_spread_N200 1075.642171 10921.785889             160.0    200.0    9761.785889 54.608929
   3.16     0 stage_spread_N200 1008.185046 10882.183838              72.0    200.0    9810.183838 54.410919
   3.16     1 stage_spread_N200  922.792793 10954.384766             160.0    200.0    9794.384766 54.771924
   3.16     2 stage_spread_N200  917.642171 10921.785889             160.0    200.0    9761.785889 54.608929

## stage_spread_N200 vs original_replay break-even

 seed status  break_even_n_cost  original_yield   n200_yield  yield_delta_n200_minus_original  original_n  n200_n  score_delta_n200_minus_original_at_0p79  score_delta_n200_minus_original_at_1p58
    0     ok           0.233150    10955.965576 10882.183838                       -73.781738       250.0   200.0                                27.842485                                67.342485
    1     ok          -0.099745    10922.819824 10954.384766                        31.564941       250.0   200.0                                44.487261                                83.987261
    2     ok          -0.076184    10897.677002 10921.785889                        24.108887       250.0   200.0                                43.309204                                82.809204

## Primary branch at c_N=1.58

 n_cost  seed   winning_variant   lit_score  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N  stage_spread_N200_rank  stage_spread_N200_score  original_replay_rank  original_replay_score
   1.58     0 stage_spread_N200 1324.185046 10882.183838              72.0    200.0    9810.183838 54.410919                       1              1324.185046                     3            1256.842561
   1.58     1 stage_spread_N200 1238.792793 10954.384766             160.0    200.0    9794.384766 54.771924                       1              1238.792793                     5            1154.805532
   1.58     2 stage_spread_N200 1233.642171 10921.785889             160.0    200.0    9761.785889 54.608929                       1              1233.642171                     3            1150.832967

- stage_spread_N200 outranks original_replay in 3/3 seeds.
- Branch: `B_reward_ranking_already_favors_stage_spread_N200`.

## Aggregate by variant

 n_cost           variant  mean_score   min_score   max_score  mean_rank   mean_yield  mean_irrigation     mean_n  mean_PFP_N
   0.79 stage_spread_N200 1423.540003 1391.642171 1482.185046   1.000000 10919.451497       130.666667 200.000000   54.597257
   0.79 stage_spread_N250 1383.363990 1351.763372 1441.702175   2.666667 10915.172933       130.666667 250.000000   43.660692
   0.79   original_replay 1384.993687 1348.332967 1454.342561   3.000000 10925.487467       130.666667 250.000000   43.701950
   0.79 early_scaled_N200 1368.109879 1304.241090 1438.863883   3.333333 10568.627930       130.666667 200.000001   52.843139
   0.79 stage_spread_N160 1290.957746 1245.917066 1331.453779   5.000000  9880.323283       130.666667 160.000000   61.752021
   0.79 early_scaled_N160 1233.768928 1169.846200 1289.329382   6.000000  9518.368734       130.666667 159.999999   59.489805
   1.58 stage_spread_N200 1265.540003 1233.642171 1324.185046   1.000000 10919.451497       130.666667 200.000000   54.597257
   1.58 early_scaled_N200 1210.109878 1146.241088 1280.863882   2.666667 10568.627930       130.666667 200.000001   52.843139
   1.58 stage_spread_N250 1185.863990 1154.263372 1244.202175   3.333333 10915.172933       130.666667 250.000000   43.660692
   1.58   original_replay 1187.493687 1150.832967 1256.842561   3.666667 10925.487467       130.666667 250.000000   43.701950
   1.58 stage_spread_N160 1164.557746 1119.517066 1205.053779   4.333333  9880.323283       130.666667 160.000000   61.752021
   1.58 early_scaled_N160 1107.368929 1043.446201 1162.929382   6.000000  9518.368734       130.666667 159.999999   59.489805
   2.37 stage_spread_N200 1107.540003 1075.642171 1166.185046   1.000000 10919.451497       130.666667 200.000000   54.597257
   2.37 early_scaled_N200 1052.109877  988.241086 1122.863881   2.333333 10568.627930       130.666667 200.000001   52.843139
   2.37 stage_spread_N160 1038.157746  993.117066 1078.653779   2.666667  9880.323283       130.666667 160.000000   61.752021
   2.37 stage_spread_N250  988.363990  956.763372 1046.702175   4.666667 10915.172933       130.666667 250.000000   43.660692
   2.37   original_replay  989.993687  953.332967 1059.342561   5.000000 10925.487467       130.666667 250.000000   43.701950
   2.37 early_scaled_N160  980.968930  917.046202 1036.529382   5.333333  9518.368734       130.666667 159.999999   59.489805
   3.16 stage_spread_N200  949.540003  917.642171 1008.185046   1.000000 10919.451497       130.666667 200.000000   54.597257
   3.16 stage_spread_N160  911.757746  866.717066  952.253779   2.333333  9880.323283       130.666667 160.000000   61.752021
   3.16 early_scaled_N200  894.109876  830.241084  964.863881   2.666667 10568.627930       130.666667 200.000001   52.843139
   3.16 early_scaled_N160  854.568931  790.646203  910.129382   4.000000  9518.368734       130.666667 159.999999   59.489805
   3.16 stage_spread_N250  790.863990  759.263372  849.202175   5.333333 10915.172933       130.666667 250.000000   43.660692
   3.16   original_replay  792.493687  755.832967  861.842561   5.666667 10925.487467       130.666667 250.000000   43.701950

## Interpretation

- At the doubled nitrogen cost used in 031_13, the already-simulated DSSAT counterfactuals rank the lower-N staged policy above the original learned N250 replay.
- Therefore 031_13 is not cleanly explained by nitrogen penalty still being too weak.
- The more likely issue is that 5k DQN training did not recover this full-season ranking in its learned Q/action policy.
- This points to learning/exploration/value-estimation rather than simply needing a larger N-cost constant.

## Files

- Rerank table: `benchmark_results/031_14_dqn_ncost_rerank_mechanism_audit/evaluation/031_14_variant_rerank_by_ncost.csv`
- Winners: `benchmark_results/031_14_dqn_ncost_rerank_mechanism_audit/evaluation/031_14_winners_by_ncost_seed.csv`
- Break-even table: `benchmark_results/031_14_dqn_ncost_rerank_mechanism_audit/evaluation/031_14_stage_spread_N200_vs_original_breakeven.csv`
- Aggregate table: `benchmark_results/031_14_dqn_ncost_rerank_mechanism_audit/evaluation/031_14_variant_aggregate_by_ncost.csv`
