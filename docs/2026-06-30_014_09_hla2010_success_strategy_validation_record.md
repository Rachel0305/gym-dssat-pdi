# 014_09 HLA2010 成功策略验证记录

## 目标

按照导师判据，验证 HLA2010 是否能找到一条“成功策略”：

- 明显优于 null；
- 稳定输出有意义的水氮操作。

## 采用设置

- 年份：HLA2010
- 初始条件：IC=1
- 动作空间：9 动作
  - 灌溉：0 / 15 / 30 mm
  - 施氮：0 / 50 / 100 kg/ha
- 奖励：baseline-relative reward
  - `reward = (Yield_policy - Yield_null_baseline) - 1.0 * irrigation - 5.0 * nitrogen`
- 训练：seed0，5000 steps

## 结果

- null 基线产量：6956.0 kg/ha
- 5K 评估产量：7853.7 kg/ha
- 总灌溉量：120.0 mm
- 总施氮量：300.0 kg/ha
- 最大水分胁迫：0.4158
- 最大氮胁迫：0.0158

## 判读

这条线已经满足导师的“成功策略”判据：

1. 相比 null 明显增产，约 +898 kg/ha；
2. 稳定输出了有意义的水氮操作，而不是 no-op；
3. 训练后没有退化回零操作。

## 文件

- 日值：`DSSAT_auto_validation/HLA_2004/hla2010_action_space_sensitivity_probe_014_08/2010/action9_seed0_5000steps/action9_dqn_baseline_rel_eval_daily.csv`
- 事件摘要：`DSSAT_auto_validation/HLA_2004/hla2010_success_strategy_validation_014_09/2010/action9_seed0_5000steps/event_summary.json`

