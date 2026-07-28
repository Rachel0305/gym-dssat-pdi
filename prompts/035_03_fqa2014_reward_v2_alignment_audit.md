# 035_03 FQA2014 自由时序 reward v2 对齐审计

## 背景

034 系列已经修复并验证了 PPO/DQN 外部动作进入 DSSAT 的关键问题：动态 RL 管理必须使用 `IRRIG=L, FERTI=L`，否则 Python 侧动作不会真实写入 DSSAT 管理文件。

035_00 在 FQA2014 linked 自由时序设置下，跑了若干固定规则策略，不训练，只看这个动作空间里是否存在优于 expert 的可行管理方案。结果显示 `water_saving_n160`、`split_moderate_n160`、`critical_i90_n200` 等规则可以在产量或综合收益上优于 official expert。

035_02 在相同 linked 自由时序环境下重新训练 MaskablePPO 50K，并保存 10K/20K/30K/40K/50K checkpoints。结果 5 个 checkpoint 均没有通过 guardrail，主要表现为持续打满或接近打满 I150/N240。

因此当前不继续加训练步数，而是先审计：当前 reward 是否真的把我们希望的策略排在前面。

## 任务目标

本任务是纯离线审计：

1. 不训练模型；
2. 不调用 DSSAT；
3. 只读取 035_00 已有规则策略结果和 034_05/035_00 PPO 回放结果；
4. 计算多套候选 reward / 选择分数；
5. 判断哪些分数能把高产且节水节氮的策略排在前面，哪些分数会偏向 no-op、early dump 或 I150/N240 打满策略。

## 输入

- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/evaluation/035_00_rule_summary.csv`
- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/evaluation/035_00_scenario_comparison.csv`

## 候选分数

至少计算：

1. `logged_reward_sum`：035_00 记录的当前环境累计 reward；
2. `project_simple_profit`：沿用 035_00 代码中的项目现有综合指标，`yield - 1.1 * irrigation - 1.58 * nitrogen`；
3. `old_proxy_y_i_5n`：历史代理分数，`yield - irrigation - 5 * nitrogen`，只作对照，不冒充项目当前 `simple_profit`；
4. `yield_gate_project_profit_1620`：`project_simple_profit + 1620 * 1[yield >= expert_yield]`；
5. `yield_shortfall_project_penalty_x5`：`project_simple_profit - 5 * max(0, expert_yield - yield)`；
6. `strict_resource_gate_profit`：若 `yield >= expert_yield` 且 `irrigation <= expert_irrigation` 且 `nitrogen <= expert_nitrogen` 则通过；否则不通过。此项仅作结果筛选，不作为训练 reward。

## 判据

重点检查：

- no-op 是否被错误排到第一；
- early dump / I150 N240 是否被错误排到前列；
- `water_saving_n160`、`split_moderate_n160`、`critical_i90_n200` 等是否排在 PPO 50K / early dump 前面；
- 候选分数是否与最终指标方向一致。

## 输出

- `benchmark_results/035_03_fqa2014_reward_v2_alignment_audit/evaluation/035_03_candidate_scores.csv`
- `benchmark_results/035_03_fqa2014_reward_v2_alignment_audit/evaluation/035_03_score_rankings.csv`
- `docs/035_03_fqa2014_reward_v2_alignment_audit_record.md`

## 停止线

本任务只做离线排序审计。即使某个候选 reward 看起来更合理，也不在本任务内启动训练。训练必须另开新任务。
