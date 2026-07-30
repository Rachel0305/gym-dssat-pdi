# 040_31 SYA lowIC PPO I240 + 水分胁迫惩罚 + 操作事件成本

## 背景

040_28 在 040_26 的 I240 分阶段留水约束基础上加入了水分胁迫过程惩罚，checkpoint 75k 相比 040_26 明显改善，但仍存在两个问题：

1. 指标不够好：平均产量未超过 official expert，节水节氮幅度也有限。
2. 措施不够合理：代表年份图显示 PPO 经常出现“前期两次大灌后，后期 15 mm 小水多次”“40 kg/ha 少量多次施肥”的模式。

这说明 040_28 只惩罚水氮总量，不惩罚“操作次数”。在当前 reward 下，15 mm 灌 6 次和 45 mm 灌 2 次只要总水量相同，资源成本几乎相同，PPO 没有足够理由主动避免频繁小操作。

## 本任务唯一改动

在 040_28 基础上只新增“每次操作固定成本”，用于让 PPO 知道每次灌溉/施肥本身也有管理成本。

保持不变：

- 输入：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 年份划分：训练 2005–2013，验证 2014–2023
- 算法：MaskablePPO
- PPO 超参数：沿用 032/040 系列当前配置
- 灌溉档位：`[0, 15, 30, 45]` mm
- 施氮档位：`[0, 40, 80, 120]` kg/ha
- 最小操作间隔：灌溉 7 天，施肥 7 天
- 季节灌溉上限：240 mm
- 季节施氮上限：250 kg/ha
- DAP90 后禁氮
- DAP≤30 累计灌溉上限 75 mm；DAP≤60 累计灌溉上限 150 mm
- 040_28 水分胁迫过程惩罚：`50 * max(0, SWFAC - 0.05)`

## 新增 reward 项

新增 unscaled 固定操作成本：

```text
irrigation_event_cost_unscaled = 16.5 if actual_irrigation_mm > 0 else 0
nitrogen_event_cost_unscaled   = 63.2 if actual_nitrogen_kg_ha > 0 else 0

event_cost_unscaled = irrigation_event_cost_unscaled + nitrogen_event_cost_unscaled
reward_04031 = reward_04028 - event_cost_unscaled * reward_scale
```

数值来源：

- irrigation event cost = `15 mm * water_cost 1.1 = 16.5`
- nitrogen event cost = `40 kg/ha * nitrogen_cost 1.58 = 63.2`

也就是：一次操作至少要付出“最小非零动作档位”的等价固定成本。这样不是禁止小剂量，而是让小剂量多次操作在 reward 上不再免费。

## 预注册判据

本任务不是要求一次性达到最终成功，而是检查“操作事件成本是否能减少不合理频繁操作，同时不导致产量崩溃”。

优先选择 checkpoint 的条件：

1. 训练/评估全部完成，无输入路径错误。
2. 验证集平均产量不低于 040_28 checkpoint75k 的 95%。
3. 平均灌溉事件数低于 040_28 checkpoint75k。
4. 平均施氮事件数低于 040_28 checkpoint75k。
5. 2017 失败年不能比 040_28 checkpoint75k 进一步明显恶化。

若事件数下降但产量明显崩溃，则判定为“成本过强或训练未成功”，不得现场调成本系数。

## 执行顺序

1. 先跑 2k smoke：确认输入路径、wrapper、daily CSV 字段、事件成本日志都正确。
2. smoke 通过后，跑 100k 正式训练。
3. 正式训练完成后，再绘制代表年份五情景图检查措施。

