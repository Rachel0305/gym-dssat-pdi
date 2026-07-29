# 040_05 SYA lowIC PPO 2014/2017 额外灌溉反事实审计

## 目的

040_04 显示，SYA lowIC PPO 最佳 checkpoint 75000 在 2014 和 2017 年出现明显水分胁迫失败：

- 2014：水分胁迫天数 27 天，产量比 official expert 低约 5334 kg/ha；
- 2017：水分胁迫天数 31 天，产量比 official expert 低约 5692 kg/ha。

本任务只回答一个机制问题：

> PPO 当前动作是否因为灌水过早/后期水分不足，导致中后期水分胁迫压不住？

## 重要边界

这不是调参，也不是正式训练。

PPO 原策略在 2014/2017 都已经使用约 I150，而当前安全约束的季节灌溉软上限是 I160，单次灌溉最小正档位是 15mm。也就是说，在原约束内无法真正再加入一次正灌溉。

因此本任务的加灌反事实定义为：

- 原 PPO 动作序列保持不变；
- 只在反事实分支中，把季节灌溉软上限临时放宽到 205mm；
- 在水分胁迫首次超过 0.05 的 DAP 加一次 I45/N0；
- 不改变施氮；
- 不改变天气、初始土壤、品种、基线或 reward 公式。

这个分支只用于机制判断：

- 如果加灌明显提高产量/降低水分胁迫，说明 PPO 失败主要来自水量或中后期水分时机不足；
- 如果加灌仍无明显改善，说明问题可能不只是水分限制，还涉及生长早期轨迹、热/辐射、作物状态或 reward/策略表达。

## 固定年份和干预

年份固定为 2014 和 2017。

干预 DAP 由原 PPO daily CSV 预先确定：

- 找到 PPO 原轨迹中第一个 `swfac > 0.05` 的 DAP；
- 在该 DAP 加一次 I45/N0；
- 若该 DAP 不满足最小间隔，则顺延到第一个满足安全间隔的 DAP。

## 输出

- `benchmark_results/040_05_sya_lowIC_ppo_extra_irrigation_counterfactual/tables/040_05_counterfactual_summary.csv`
- `benchmark_results/040_05_sya_lowIC_ppo_extra_irrigation_counterfactual/tables/040_05_counterfactual_daily.csv`
- `docs/040_05_sya_lowIC_ppo_extra_irrigation_counterfactual_record.md`

