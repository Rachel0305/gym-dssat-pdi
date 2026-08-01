# 040_40 SYA lowIC 自由时序 MaskablePPO：保产优先 reward v3

## 背景

040_36 在措施合理性上已经明显改善：粗动作档位避免了 15mm 小灌和 40kg 小肥，DAP90 前保留至少 45mm 灌溉水量，使 PPO 不再把全部水分过早用完。

040_38 的五情景指标柱状图显示，040_36 100K checkpoint 在 2014–2023 验证年份中：

- PFP_N 10/10 年超过四情景最高值；
- 产量仅 2/10 年超过四情景最高值；
- WP_ET 仅 2/10 年超过四情景最高值；
- 2017 年产量缺口最大。

040_39 进一步复查 25K/50K/75K/100K checkpoint，确认已有 checkpoint 没有同时改善产量和资源效率。因此，下一步需要修改 PPO 的训练目标，而不是仅换 checkpoint。

## 本任务目的

在不改变 040_36 措施合理性约束的前提下，新增一个低产 guardrail，使 PPO 从“过度资源节约型”转向“保产前提下节水节氮型”。

## 固定不变项

本任务相对于 040_36 保持不变：

- 输入数据：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 年份划分：2005–2013 训练，2014–2023 验证
- 算法：MaskablePPO
- 训练步数：默认 100000
- checkpoint：25000、50000、75000、100000
- 动作档位：
  - 灌溉 `[0, 30, 45]` mm
  - 施氮 `[0, 80, 120]` kg/ha
- 安全约束：
  - 最小灌溉间隔 7 天
  - 最小施肥间隔 7 天
  - 季节灌溉上限 240 mm
  - 季节施氮上限 250 kg/ha
  - DAP90 后禁氮
  - DAP90 前累计灌溉不得超过 195 mm，给 DAP91+ 至少保留 45 mm
- 原 040_28/040_36 胁迫过程惩罚保留。

## 唯一新增项：terminal yield guardrail

在收获终点追加低产惩罚：

```text
target_yield(year) = 0.98 × official_extension_expert_yield(year)
yield_deficit = max(0, target_yield - final_grnwt)
terminal_penalty_unscaled = yield_coef × yield_deficit
terminal_penalty_scaled = terminal_penalty_unscaled × reward_scale
reward_v3 = reward_04036 - terminal_penalty_scaled
```

其中：

- `yield_coef = 0.158`，沿用原始 reward 的产量权重；
- `reward_scale = 0.001`，沿用现有训练尺度；
- 0.98 不是为了让 PPO 必须完全超过 expert，而是防止明显减产；超过 98% expert 后不再额外给 bonus，避免重新诱导打满水肥。

## 预注册判定

### Smoke 通过条件

2K smoke 必须满足：

- 训练和验证流程跑通；
- daily CSV 中出现 `yield_guardrail_*` 字段；
- 没有 15mm 小灌或 40kg 小肥；
- 没有 late reserve violation；
- 使用 lowIC 输入路径。

### 正式 100K 结果评价

成功不是只看单一指标。正式结果需要同时报告：

- 2014–2023 每年 PPO 与四情景的产量、WP_ET、PFP_N；
- PPO 总灌溉量、总施氮量；
- 水分/氮胁迫；
- 每年操作图。

若 040_40 相比 040_36：

- 平均产量提高；
- 2017 年产量缺口明显缩小；
- PFP_N 不出现大幅退化；
- 措施仍合理；

则进入五情景图和指标汇总图。

若产量提高但回到打满水氮，判为不可接受。

若资源效率保留但产量无改善，说明低产 guardrail 过弱或 PPO 对该 reward 响应不足，需要停止后再设计，不在本任务内现场调系数。

