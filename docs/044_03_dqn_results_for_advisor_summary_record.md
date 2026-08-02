# 044_03 DQN 结果整理：导师汇报用

## 任务性质

本任务只整理已有 DQN 结果，不重新训练，不重新运行 DSSAT。

## 纳入结果

- 040_01：普通 DQN + 外部动作安全修正，不是严格 MaskableDQN。
- 040_02：严格 MaskableDQN，训练和评估时屏蔽非法动作。
- 044_00–044_02：Demo-DQN / DQfD 风格示范经验尝试与 Q 排序诊断。

## 核心结论

1. 普通 DQN 的最好 checkpoint 是按平均产量选出的，但验证年平均产量仍偏低，且动作序列高度固定，早期大量施氮/灌溉特征明显。
2. 严格 MaskableDQN 在 25K 后迅速退化：50K、75K、100K 的平均产量继续下降，并出现少施氮或 no-op 倾向。
3. Demo-DQN / DQfD 路线已确认示范经验进入训练诊断链，但非零 teacher 动作没有被稳定学成 Q 值最高动作；2K smoke 的验证策略为 no-op，产量极低。
4. 这些结果支持把 DQN 作为当前阶段的对照/阴性证据保留，而不是继续无限修 DQN；主线仍应优先回到 PPO 的天气响应性和策略合理性改进。

## 各方法最佳 checkpoint（按验证年平均产量）

| method                 |   checkpoint_step |   validation_years |   mean_final_grnwt |   mean_total_irrigation |   mean_total_n |   mean_PFP_N |   mean_swfac_stress_days_gt_0p05 |   mean_nstres_days_gt_0p05 |
|:-----------------------|------------------:|-------------------:|-------------------:|------------------------:|---------------:|-------------:|---------------------------------:|---------------------------:|
| DemoDQN_DQfD_044_00    |              1000 |                 10 |            1515.3  |                       0 |              0 |     nan      |                            nan   |                        nan |
| 严格MaskableDQN_040_02 |             25000 |                 10 |            6427.41 |                      90 |            240 |      26.7809 |                             17.8 |                          0 |
| 普通DQN_040_01         |             75000 |                 10 |            7721.74 |                     150 |            240 |      32.1739 |                             11   |                          0 |

## Demo-DQN / DQfD Q 排序摘要

| method                |   checkpoint_step |   sample_count |   nonzero_count |   nonzero_teacher_argmax_rate |   mean_q_teacher_minus_noop |   margin_satisfied_rate |
|:----------------------|------------------:|---------------:|----------------:|------------------------------:|----------------------------:|------------------------:|
| DemoDQN在线2K_044_01  |              1000 |           1398 |              67 |                      0        |                 -0.0115312  |                0        |
| DemoDQN在线2K_044_01  |              2000 |           1398 |              67 |                      0        |                 -0.0423258  |                0.379113 |
| DemoOnly预训练_044_02 |                 0 |           1398 |              67 |                      0.567164 |                  0.00145881 |                0        |
| DemoOnly预训练_044_02 |                50 |           1398 |              67 |                      0        |                 -0.00133362 |                0        |
| DemoOnly预训练_044_02 |               200 |           1398 |              67 |                      0        |                 -0.0121879  |                0        |
| DemoOnly预训练_044_02 |               500 |           1398 |              67 |                      0        |                 -0.0370801  |                0.240343 |

## 输出文件

- checkpoint_metrics: `benchmark_results\044_03_dqn_results_for_advisor_summary\tables\044_03_dqn_checkpoint_mean_metrics.csv`
- yearly_best: `benchmark_results\044_03_dqn_results_for_advisor_summary\tables\044_03_dqn_best_checkpoint_yearly_validation.csv`
- q_ranking: `benchmark_results\044_03_dqn_results_for_advisor_summary\tables\044_03_demo_dqn_q_ranking_summary.csv`
- figure: `benchmark_results\044_03_dqn_results_for_advisor_summary\figures\044_03_dqn_checkpoint_metric_overview.png`
- figure: `benchmark_results\044_03_dqn_results_for_advisor_summary\figures\044_03_dqn_best_checkpoint_yearly_yield.png`
- figure: `benchmark_results\044_03_dqn_results_for_advisor_summary\figures\044_03_demo_dqn_q_ranking_diagnostics.png`

## 汇报口径建议

可以说：我们按导师要求补了 DQN/DQfD 路线，并且在同一 lowIC、自由时序、动作约束框架下做了普通 DQN、严格 MaskableDQN 和 Demo-DQN/DQfD 对照。结果显示，当前 DQN 系列没有比 PPO 更可靠：普通 DQN 有早期固定动作倾向，严格 MaskableDQN 随训练加深退化，Demo-DQN/DQfD 虽接入示范经验但仍无法把关键非零动作学成高 Q 值。因此当前阶段把 DQN 作为对照保留，主线继续优化 PPO 的天气响应性更合理。
