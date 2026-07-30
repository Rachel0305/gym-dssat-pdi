# 040_34 SYA lowIC 040_33 checkpoint25k 代表年份五情景日过程图

## 目的

040_33 粗化动作档位后，checkpoint25k 在验证集上没有崩溃，且完全消除了 15mm/40kg 小碎片动作。因此本任务只做可视化审计：绘制 2014、2017、2022 三个代表年份的五情景日过程图，检查管理措施是否比 040_28 更合理。

不训练、不改模型、不改 checkpoint。

## 固定输入

- RL 候选：`040_33_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions`
- checkpoint：25000
- 年份：2014、2017、2022
- 五情景：null、recorded farmer、DSSAT auto、official extension expert、040_33 MaskablePPO candidate

## 重点检查

1. 是否仍出现 15mm 小水或 40kg 小肥：理论上不应出现。
2. 灌溉是否变成少次、较大剂量。
3. 施氮是否变成少次、较大剂量。
4. 2017 的水分胁迫是否仍明显压不住。

