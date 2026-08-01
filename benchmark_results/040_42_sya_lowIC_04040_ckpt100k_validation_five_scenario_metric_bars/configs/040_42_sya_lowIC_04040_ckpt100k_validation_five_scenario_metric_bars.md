# 040_42 SYA lowIC 040_40 checkpoint100K 五情景指标柱状图

## 背景

040_40 在 040_36 的合理措施约束基础上加入 terminal yield guardrail，用于避免 PPO 过度追求资源节约而牺牲产量。

040_40 100K 训练已经完成。初步汇总显示其相对 040_36：

- 平均产量提高；
- 2017 年产量明显提高；
- 灌溉量更接近上限；
- 施氮量提高，PFP_N 有所下降；
- 措施仍无 15mm 小灌、40kg 小肥和 late reserve violation。

## 本任务目的

不训练新模型。固定重放 040_40 checkpoint100K 在 2014–2023 验证年份上的 PPO 动作序列，补齐 Summary.OUT 中的 ETCP / WP_ET，并绘制五情景指标柱状图。

## 输入

- PPO 040_40 checkpoint100K 验证摘要：
  - `benchmark_results/040_40_sya_lowIC_ppo_yield_guardrail_v3/evaluation/040_40_checkpoint_validation_summary.csv`
- SYA lowIC 四情景基线：
  - `benchmark_results/040_21_sya_lowIC_four_baseline_rebuild/evaluation/040_21_baseline_summary.csv`

## 输出

- 五情景指标表；
- PPO fixed replay 日值表；
- 产量、WP_ET、PFP_N 五情景柱状图；
- 灌溉量、施氮量五情景柱状图；
- 中文实验记录。

## 判读边界

本任务只回答 040_40 checkpoint100K 相对于四情景的指标表现，不改变训练、不选择新 checkpoint、不调整 reward。

