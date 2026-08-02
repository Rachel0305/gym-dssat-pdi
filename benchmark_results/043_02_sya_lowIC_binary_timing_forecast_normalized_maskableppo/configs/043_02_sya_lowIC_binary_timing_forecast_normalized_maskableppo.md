# 043_02 SYA lowIC binary-timing forecast-normalized MaskablePPO

## 背景

043_01 审计显示，当前较优的 042_15/042_11 binary-timing PPO 对输入扰动响应很弱：对土壤水分、氮胁迫、辐射、温度等扰动，确定性动作几乎不变。这说明当前策略仍偏模板化，不能很好回应导师提出的“不同年份天气不同，策略也应随输入变化”的问题。

## 本任务要验证什么

本任务只验证一个低成本假设：

> 在保留当前低风险 binary-timing 动作空间和 safety mask 的前提下，给 PPO 加入显式天气/完美预报输入，并对全部 observation 做固定物理尺度归一化，是否能改善策略对年份天气差异的响应。

这不是 teacher warm-start，也不是重新调剂量。若效果改善，优先归因为“可观测天气信息 + 归一化 + 已固定的氮胁迫过程惩罚”，而不是 teacher。

## 固定配置

- 站点：SYA。
- 输入数据：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`。
- 年份划分：沿用 032_22/040 系列半分训练-验证设计，2005-2013 训练，2014-2023 验证。
- 算法：MaskablePPO。
- 训练步数：100000。
- checkpoint：25000、50000、75000、100000。
- 动作空间：binary timing。
  - 灌溉：`[0, 45]` mm。
  - 施氮：`[0, 80]` kg/ha。
- 安全约束：沿用 040_36。
  - 最小灌溉间隔 7 天。
  - 最小施氮间隔 7 天。
  - 单季水氮软上限。
  - DAP90 后禁氮。
  - 晚期灌溉保留 mask。

## Observation 改动

基础 25 维 observation 沿用环境输出，并追加 5 个天气/预报变量：

- 当日降雨 `rain_today_mm`
- 当日最低气温 `tmin_today_c`
- 近 7 天累计降雨 `rain_past7_mm`
- 未来 7 天累计降雨 `rain_future7_mm`
- 未来 7 天平均气温 `tmean_future7_c`

这里的“未来 7 天”使用历史天气作为完美天气预报，是一个有意识的信息结构假设，后续论文中必须明确说明。

全部 30 维 observation 使用预先固定的物理尺度归一化，不用验证集结果拟合 scale。

## Reward 改动

继承 042_10/040_36 的主 reward 和 safety 逻辑，并加入 042_02 已使用过的氮胁迫过程惩罚：

```text
penalty = 50 * max(0, nstres_after_step - 0.05) * reward_scale
```

该项目的是让策略不只看最终产量，也对季内氮胁迫积累有过程响应。该系数本轮不扫描、不调参。

## 运行前检查

正式训练前必须通过 observation smoke：

- lowIC 输入目录存在；
- 训练/验证年份为 SYA；
- wrapper 实际输出 30 维 observation；
- 追加天气变量非空；
- 动作空间为 4 个组合动作；
- 不使用 teacher warm-start。

## 停止与解释边界

- 若 observation smoke 不通过，停止，不训练。
- 若 100K 训练完成但指标或响应性仍不佳，不事后硬挑未预注册 checkpoint 作为“最终成功”；后续另写审计/guardrail。
- 本任务结果只能说明“加天气/预报/归一化后的 binary-timing PPO 在 SYA lowIC 半分验证上的表现”，不能直接外推到其他站点。
