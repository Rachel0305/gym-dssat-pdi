# 035_01 FQA2014 linked 规则策略 reward-指标对齐审计 prompt

## 背景

034_05 证明 linked 修复后 PPO 动作真实进入 DSSAT，但 FQA2014 PPO 50K 结果不优。

035_00 在同一 linked 环境中测试了 8 个固定规则策略，发现：

- 存在比 034_05 PPO 50K 更优的规则策略；
- 甚至存在产量和 WP_ET 超过 official expert 的规则策略；
- 因此 linked 动作空间中并非没有好策略，问题在训练/选择目标。

## 目标

本任务不训练、不跑 DSSAT，只读取 034_05 和 035_00 的已有 CSV，审计当前 reward 与最终汇报指标之间是否一致。

重点回答：

1. 当前 stress-aware reward 排序是否等同于产量、WP_ET、PFP_N、simple_profit 排序；
2. PPO 50K 落点属于哪类策略；
3. 是否存在“reward 高但最终指标不优”或“最终指标好但 reward 排名不高”的错位；
4. 下一轮训练/选 checkpoint 是否需要加入指标 guardrail。

## 输入

- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/evaluation/035_00_rule_summary.csv`
- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/evaluation/035_00_scenario_comparison.csv`
- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_eval_summary.csv`

## 输出

- reward 与各指标排序表；
- Spearman 排序相关；
- 与 expert 的差值；
- 候选选择规则建议；
- 中文实验记录 MD。

## 判读原则

- 若 reward 排序和 final metric 排序明显不一致，不允许继续只用 reward 选 checkpoint。
- 若好策略 reward 较低，下一步应改 reward 或用指标 guardrail 选择 checkpoint。
- 若 PPO 50K 不是 reward 最优也不是指标最优，说明训练本身也没有充分优化当前 reward，需要和 reward 对齐问题分开处理。

