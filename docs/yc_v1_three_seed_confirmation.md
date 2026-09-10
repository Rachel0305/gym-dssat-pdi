# YC V1 three-seed confirmation

日期：2026-09-10  
状态：`partial_improvement`  
主结果：Augmented 100K，三种子均值；Original 为同一协议下的 paired comparator。

## A–B. 精确效率回放

已完成 `240` 行精确回放（2 methods × 3 seeds × 4 checkpoints × 10 validation years），失败行数为 `0`。WP_ET/ETCP 来自 DSSAT `Summary.OUT`；定义为 `YPEM × 0.1`（有效时），否则为 `HWAM/(ETCP×10)`，其中 ETCP 单位为 mm。PFP-N 使用 `YPNAM`，仅在 `NICM > 0` 时有效。没有从 daily CSV 推断 WP_ET。

精确回放源代码：`src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py`；回放结果：[yc_v1_exact_efficiency_replay.csv](../results/yc_v1_exact_efficiency_replay.csv)。

## C–H. 结果摘要

seed 0 已复用冻结 Original/221YCA 产物并做同协议回放；seed 1/2 使用独立输出根完成正式 100K 训练。所有正式运行均保留 25K/50K/75K/100K 检查点，验证年份为 2014–2023。下表为 100K 三种子均值；Expert/Farmer 为冻结 `055_02` baseline summary 的 2014–2023 均值。

正式管理行为审计（seed 0 为冻结既有产物，seed 1/2 为本次独立正式运行）：

| Method | Seed | Status | Validation rows | Actions on grid | Diversity signal | next_step_allowed |
|---|---:|---|---:|---|---|---|
| Original | 1 | completed | 10/10 | True | False | False |
| Augmented | 1 | completed | 10/10 | True | 1 | True |
| Original | 2 | completed | 10/10 | True | True | True |
| Augmented | 2 | completed | 10/10 | True | 1 | True |

其中 Original seed1 的最终审计 `next_step_allowed=false`，且 `multiple_nonzero_action_pairs=false`；这被保留为管理策略塌缩/多样性风险，而不是删除该种子或重新训练后再选择结果。

| Method | Yield | Irrigation | N input | PFP-N | WP_ET |
|---|---:|---:|---:|---:|---:|
| Expert | 8200.555 | 211.000 | 245.000 | 33.480 | 2.293 |
| Farmer | 7972.730 | 120.000 | 374.000 | 21.320 | 2.338 |
| Original | 6540.600 | 101.500 | 146.667 | 68.617 | 1.918 |
| Augmented | 7076.300 | 113.500 | 146.667 | 55.957 | 2.046 |

三种子均值与标准差见 [yc_v1_three_seed_summary.csv](../results/yc_v1_three_seed_summary.csv)，逐种子检查点轨迹见 [yc_v1_three_seed_checkpoint_trajectory.csv](../results/yc_v1_three_seed_checkpoint_trajectory.csv)，逐年明细见 [yc_v1_three_seed_by_year.csv](../results/yc_v1_three_seed_by_year.csv)。

四张图：

1. `experiments/222_yc_v1_three_seed_confirmation/figures/figure1_checkpoint_yield_stability.png`
2. `experiments/222_yc_v1_three_seed_confirmation/figures/figure2_100k_multiobjective_comparison.png`
3. `experiments/222_yc_v1_three_seed_confirmation/figures/figure3_100k_seed_robustness.png`
4. `experiments/222_yc_v1_three_seed_confirmation/figures/figure4_100k_by_year_stability.png`

## I–K. 冻结标准判定

Expert mean Yield = `8200.555`；Augmented mean Yield = `7076.300`；Original mean Yield = `6540.600`。

| Criterion | Observed | Threshold | Result |
|---|---:|---:|---|
| 3-seed mean Yield >= 98% expert | 7076.300 | 8036.543 | FAIL |
| Each seed mean Yield >= 95% expert | 6024.500 | 7790.527 | FAIL |
| 3-seed mean irrigation <= 110% expert | 113.500 | 232.100 | PASS |
| 3-seed mean N <= 100% expert | 146.667 | 245.000 | PASS |
| 3-seed mean PFP-N >= 100% expert | 55.957 | 33.480 | PASS |
| 3-seed mean WP_ET >= 95% expert | 2.046 | 2.178 | FAIL |

结论状态为 `partial_improvement`。主要未满足项：3-seed mean Yield >= 98% expert。该结论不宣称 PPO/augmentation 在所有站点、年份或指标上普遍优越；它只适用于本 YC lowIC、固定验证期、固定动作/奖励/网络/训练协议。

## L–N. 复现与未执行项

新增实验目录：`experiments/222_yc_v1_three_seed_confirmation/`。训练期间通过 smoke gate 后才进入 formal；正式运行未改 reward、action evaluation、PPO、network 或其他站点。由于原始模型/daily 输出体量较大，本次 Git commit 只提交脚本、配置、manifest、汇总表、图和文档，不提交大型 checkpoint 与缓存目录；原始输出仍保留在本地 `benchmark_results/222YCA_*`。

未执行：未 push 到远端；未将 `dssat_auto_external_n` 纳入主比较；未改变冻结 criteria；未重新训练 seed 0。
