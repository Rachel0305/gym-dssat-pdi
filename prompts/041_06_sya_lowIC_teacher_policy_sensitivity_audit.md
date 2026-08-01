# 041_06 SYA lowIC teacher 轨迹差异与 PPO 策略输入敏感性审计

## 目的

041_04 显示 teacher warm-start 可以让 PPO 摆脱 no-op，但 100K fine-tune 后策略趋向统一模板：

- 平均 I 从 240 降到 225；
- 平均 N 从 240 降到 200；
- PFP_N 提高，但多数年份产量和 WP_ET 下降。

041_05 进一步显示，BC init 与 PPO 100K 在 2014–2023 的动作时间高度相似，说明当前策略可能没有充分响应年份间天气/土壤差异。

041_06 的目标是回答两个问题：

1. teacher 轨迹本身是否具有跨年份差异？
2. PPO 策略对输入变量是否敏感，尤其是 DAP、SRAD、TMAX、土壤水、SWFAC、NSTRES、累计灌溉、累计施氮等？

## 不做什么

- 不训练；
- 不重新跑 DSSAT 季节；
- 不改 reward；
- 不改 action mask；
- 不根据结果调参数。

## 输入

- 041_02 teacher 选择表；
- 041_04 BC dataset：包含 teacher replay 得到的 25 维 observation、action mask 和 teacher action；
- 041_04 BC init 模型；
- 041_04 PPO 100K 模型；
- 041_04 日值输出，仅用于动作序列差异对照。

## 方法

### A. teacher 跨年份动作差异

统计 2014–2023 每年的 teacher：

- 总灌溉；
- 总施氮；
- 非零动作次数；
- 非零动作日期；
- action 序列差异。

### B. 观测变量覆盖审计

检查当前 25 维 observation 中是否包含：

- DAP；
- SRAD；
- TMAX；
- RAIN；
- TMIN；
- SW/SWFAC；
- NSTRES；
- 累计灌溉；
- 累计施氮。

注意：如果 RAIN/TMIN 不在 observation 中，不能声称 PPO 直接对当天 RAIN/TMIN 敏感，只能说可能通过 DSSAT 状态变量间接反映。

### C. 模型策略敏感性

在同一批 teacher replay 状态上，分别把输入变量扰动到该变量在数据集中的低分位/高分位，保持 action mask 不变，计算：

- argmax 动作变化率；
- 动作概率分布总变差距离；
- 原 argmax 动作概率变化；
- 非零动作概率变化。

审计对象：

- `BC init`；
- `PPO 100K`。

## 判读

- 如果 teacher 有差异，但 PPO 对输入扰动不敏感：问题在模型训练/策略坍缩。
- 如果 teacher 本身也高度模板化：需要扩展 teacher 搜索空间，而不是继续调 PPO。
- 如果 RAIN/TMIN 不在观测中：这是输入信息结构限制，需要决定是否加入天气预报/天气观测字段。
