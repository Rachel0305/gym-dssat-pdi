# 040_03 SYA lowIC 自由时序算法对照冻结记录

## 结论先说

在相同 SYA lowIC 输入、相同训练/验证年份划分、相同动作空间、相同安全约束、相同奖励函数下，当前三套算法的验证年均产量排序为：

1. **MaskablePPO_040_00**：最佳 checkpoint 75000，验证年均产量 8057.7 kg/ha。
2. **SB3_DQN_040_01**：最佳 checkpoint 75000，验证年均产量 7721.7 kg/ha。
3. **StrictMaskableDQN_040_02**：最佳 checkpoint 25000，验证年均产量 6427.4 kg/ha。

这说明：**严格把 mask 接入 DQN 的探索、贪心、replay 和 Bellman target 后，DQN 仍没有优于 PPO。**  
因此，当前证据支持把 PPO 作为后续主线算法，DQN 作为已完成的公平对照保留。

## 任务边界

- 本任务没有重新训练模型。
- 本任务没有重新运行 DSSAT。
- 本任务只读取 040_00、040_01、040_02 已有 CSV 结果，并统一汇总。
- checkpoint 选择规则固定为：每个算法选择验证年均产量 `mean_final_grnwt` 最高的 checkpoint。

## 最佳 checkpoint 对照

| rank_by_mean_validation_yield | algorithm_label | checkpoint_step | validation_years | mean_final_grnwt | mean_total_irrigation | mean_total_n | mean_PFP_N | mean_swfac_stress_days_gt_0p05 | mean_nstres_days_gt_0p05 | yield_drop_from_best_to_final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MaskablePPO_040_00 | 75000 | 10 | 8057.73 | 150.00 | 240.00 | 33.57 | 10.60 | 0.00 | 859.71 |
| 2 | SB3_DQN_040_01 | 75000 | 10 | 7721.74 | 150.00 | 240.00 | 32.17 | 11.00 | 0.00 | 281.85 |
| 3 | StrictMaskableDQN_040_02 | 25000 | 10 | 6427.41 | 90.00 | 240.00 | 26.78 | 17.80 | 0.00 | 4900.11 |

## 关键解释

### 1. 为什么 040_02 是更严格的 DQN 对照？

040_01 使用 SB3 DQN。SB3 DQN 本身没有原生 MaskableDQN 机制，所以 040_01 只能在确定性评估时使用 masked-greedy 选择合法动作；训练阶段的探索和 Bellman target 并不是严格 mask 的。

040_02 使用项目内实现的 StrictMaskableDQN，mask 进入：

- epsilon 随机探索；
- greedy 动作选择；
- replay buffer 储存；
- next-state Bellman target 的 `max_a Q(s', a)`。

所以 040_02 回答的是：如果把 DQN 的 mask 机制补严格，是否能超过 PPO？当前答案是没有。

### 2. 为什么不继续给 DQN 加训练步数？

StrictMaskableDQN 的最佳 checkpoint 出现在 25000 步；后续 checkpoint 明显退化。  
最终 checkpoint 100000 的验证年均产量为 1527.3 kg/ha，比最佳 checkpoint 低 4900.1 kg/ha。

这更像是 DQN 长训练过程中的策略退化，而不是训练不足。

### 3. 这是否证明 DQN 理论上不行？

不能。这个结论只限于当前实验条件：

- 站点：SYA；
- 输入：lowIC；
- 年份划分：2005–2013 训练，2014–2023 验证；
- 自由时序日尺度环境；
- 当前 stress-aware reward；
- 当前动作空间和安全约束；
- 当前 100K 训练预算。

它能支持的结论是：**在当前公平对照设置下，PPO 比两种 DQN 实现更稳、更好。**

## 输出文件

- checkpoint 汇总：`benchmark_results\040_03_sya_lowIC_algorithm_comparison_freeze\tables\040_03_algorithm_checkpoint_summary.csv`
- 最佳 checkpoint 汇总：`benchmark_results\040_03_sya_lowIC_algorithm_comparison_freeze\tables\040_03_algorithm_best_checkpoint_summary.csv`
- 最佳 checkpoint 年份明细：`benchmark_results\040_03_sya_lowIC_algorithm_comparison_freeze\tables\040_03_algorithm_best_year_detail.csv`

