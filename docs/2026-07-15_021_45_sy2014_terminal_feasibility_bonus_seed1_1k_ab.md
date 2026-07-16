# 021_45 SY2014 终端可行性 bonus：seed1 1K 单变量 A/B 记录

## 设计

Control 直接复用 021_42 online seed1，不重跑。Treatment 仅在终止时、最终产量达到本站点年份官方 expert 阈值 11077 kg/ha 时，在原奖励上增加 1620。bonus 由 `1×120+5×300` 推导，不是拟合值。其余训练、环境和 demo n-step 屏蔽设置不变。

## 实现与单元测试

- 阈值注册表：`configs/expert_yield_thresholds.yaml`；来源：`/workspace/benchmark_results/021_18/021_18_baseline_metrics.csv` 的 `official_extension_expert` 行。
- 11076/11077/11078 边界、非终止步、bonus 推导、N200/N300 保留 500 分差、缺失映射失败保护均通过：**True**。
- 旧奖励实现没有修改；bonus 使用本任务外层 adapter。

## 结果

| arm | checkpoint | yield_kg_ha | irrigation_mm | nitrogen_kg_ha | late_n_after_dap90_kg_ha | expert_efficiency_gate | candidate_reward_recalculated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control_021_42_seed1 | 0 | 11175.0 | 90.0 | 300.0 | 0.0 | True | 5797.0 |
| control_021_42_seed1 | 250 | 11175.0 | 75.0 | 300.0 | 0.0 | True | 5812.0 |
| control_021_42_seed1 | 500 | 11170.0 | 105.0 | 300.0 | 0.0 | True | 5777.0 |
| control_021_42_seed1 | 750 | 11154.0 | 120.0 | 300.0 | 0.0 | True | 5746.0 |
| control_021_42_seed1 | 1000 | 10787.0 | 90.0 | 150.0 | 0.0 | False | 4539.0 |
| treatment_terminal_bonus | 0 | 11175.0 | 90.0 | 300.0 | 0.0 | True | 5797.0 |
| treatment_terminal_bonus | 250 | 11170.0 | 90.0 | 300.0 | 0.0 | True | 5792.0 |
| treatment_terminal_bonus | 500 | 11175.0 | 90.0 | 300.0 | 0.0 | True | 5797.0 |
| treatment_terminal_bonus | 750 | 11154.0 | 120.0 | 300.0 | 0.0 | True | 5746.0 |
| treatment_terminal_bonus | 1000 | 10787.0 | 90.0 | 150.0 | 0.0 | False | 4539.0 |

- Control 通过：3/4，1000-step 失败。
- Treatment 通过：3/4；1000-step 通过：False；最低产量：10787 kg/ha。
- 预注册分支：**C**。终端 bonus 未改善 seed1 保持性，该候选不获支持。

## 硬门槛与 Q 尺度检查

训练中共有 6 个完整季节；其中 4 季实际获得 bonus，2 季未获得，逐季分配与终产量门槛完全一致：True。终产量落在阈值 ±50 kg/ha 内的季节数为 0。产量—累计奖励散点和 online Q 绝对最大值轨迹已保存。由于门槛附近没有样本，本轮不能声称已排除门槛处震荡。

## 边界

这只是 SY2014 online seed1 的 1K 单变量验证。没有启动 seed2 或 5K；没有改 reward 主文件、IC、DSSAT 输入、动作、预算或网络。即使分支 A，也只能支持另立复核任务，不能宣布长期稳定。

第一次启动命令的外层终端等待时间误设为 1 秒，进程在创建输出目录前被终止，没有产生科学数据；随后改用足够等待时间完整重跑。本记录保留该执行失误，不把它计作模型失败。

## 输出

- `benchmark_results/021_45/021_45_unit_tests.json`
- `benchmark_results/021_45/021_45_training_interactions.csv`
- `benchmark_results/021_45/021_45_treatment_update_log.csv`
- `benchmark_results/021_45/021_45_training_episode_terminal_rewards.csv`
- `benchmark_results/021_45/021_45_checkpoint_trajectory_ab.csv`
- `benchmark_results/021_45/021_45_validation.json`
- `benchmark_results/021_45/021_45_summary.json`
- `benchmark_results/021_45/021_45_terminal_bonus_seed1_1k_ab.png/.svg`
- `benchmark_results/021_45/021_45_q_scale_trace.png/.svg`
