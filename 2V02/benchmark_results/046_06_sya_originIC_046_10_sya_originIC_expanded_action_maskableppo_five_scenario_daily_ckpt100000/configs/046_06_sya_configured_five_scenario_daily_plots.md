# 046_06：SYA 配置化五情景逐日过程图

## 目的

对同一套输入 profile、同一批验证年份，输出五情景的可追溯逐日过程图与合并日值表：Null、Recorded template、DSSAT automatic irrigation + external N rule、Official expert、PPO。

## 图与表

- 每个验证年一张 4×2 图：降雨/气温、土壤水分（SoilWat.OUT 的 SWTD）、WSPD/NSTD、灌溉、施肥、籽粒/生物量、统一累计奖励；
- 一个合并日值 CSV；
- 一个管理事件 CSV 与一个季末汇总 CSV。

## 统一累计奖励（仅用于跨情景展示）

所有情景用相同公式重新计算，不使用 PPO 独有的 stress-relief shaping：

`R = 0.158 × final_grain_yield − 1.1 × seasonal_irrigation − 1.58 × seasonal_nitrogen`

日尺度上先扣当日水氮成本，在收获日加入 `0.158 × final_grain_yield`。这不是训练 reward 的逐项重放，而是消除情景间 reward 定义不一致的报告指标。

## 数据链与硬检查

- Null、Recorded template、Official expert：046_03 快照；
- auto：046_04 快照与外部 N 日值；
- PPO：046_02 所选 checkpoint 的确定性重放快照；重放终值必须与 046_02 保存的对应验证日值终值一致，否则失败，不画混合来源曲线；
- 不训练网络、不更改奖励、不更改 DSSAT 输入。
