# 042_03 SYA lowIC 042_02 checkpoint 策略漂移审计

## 目的

`042_02` 发现：

- 75K checkpoint 表现最好，达到 6/10 验证年份三指标全超四基线最高；
- 100K checkpoint 退化为 N=0，0/10 胜出。

本任务不训练、不跑 DSSAT，只读取 `042_02_rerun100k` 已有验证结果，审计：

1. 75K 与 100K 的逐年动作序列差异；
2. 施氮动作在哪些 checkpoint 消失；
3. 产量、WP_ET、PFP_N 和胁迫指标随 checkpoint 的变化；
4. 是否能支持“后期 PPO fine-tune policy drift”这个判定。

## 输入

- `benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/evaluation/041_03_checkpoint_validation_summary.csv`
- `benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/daily_outputs/SYA/*.csv`
- `benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/evaluation/041_03_training_checkpoint_inventory.csv`

## 边界

- 不训练；
- 不运行 DSSAT；
- 不修改 `042_02` 原始结果；
- 不把 75K 直接指定为正式最终策略。

## 判读

如果 75K 到 100K 之间出现系统性氮动作消失，且对应产量/PFP_N/WP_ET 急剧退化，则记录为：

> 后期 fine-tune policy drift；需要下一步预注册 checkpoint selection 或 early stopping 规则。

如果动作没有系统性变化，则停止，进一步检查指标计算或 DSSAT 输出。
