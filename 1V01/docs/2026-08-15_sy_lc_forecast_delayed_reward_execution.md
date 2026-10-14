# SY/LC 未来天气七日延迟奖励 2K 执行记录

## 设计

在上一轮紧凑天气输入基础上，只增加一个七日延迟压力缓解奖励：非零灌溉动作发生后等待 7 个模拟步；如果干旱/高温预报下后续 7 天平均 `swfac` 高于动作时的 `swfac`，给 `0.05 × dry_risk × improvement` 的小奖励。原奖励、资源成本、PPO、动作格和 DSSAT 自动灌溉不变。

## 执行结果

- SY 和 LC 均完成 seed=0、2K、1K/2K checkpoint；10/10 验证年份、动作格、窗口和基础传输 gate 均通过。
- SY 延迟奖励总和：0.0000；完成延迟记录 121 条。
- LC 延迟奖励总和：0.01436；完成延迟记录 60 条。
- 说明该奖励在 2K 中几乎没有产生有效训练信号。

## 2K 均值

| 站点 | 原无预报对照产量 | 延迟奖励产量 | 原对照 PFP_N | 延迟奖励 PFP_N |
|---|---:|---:|---:|---:|
| SY | 10004.95 | 10027.90 | 41.687 | 41.783 |
| LC | 9199.56 | 9184.82 | 38.332 | 38.270 |

SY 的小幅上升不能解释为天气响应改善；其动作序列种类从 4 降为 2。LC 轻微下降，动作种类从 8 降为 3。

## forecast-swap 审计

- SY：1381 个决策状态，真实未来天气替换后动作变化率 0%。
- LC：938 个决策状态，真实未来天气替换后动作变化率 0%。
- 两站点所有变体动作均合法。

## 结论

这次延迟奖励没有让 PPO 对天气产生可检测响应，主要原因是奖励触发太少，几乎没有形成可学习的信号。因此不批准 5K/长训练，也不把本轮结果作为性能提升证据。下一次若继续，应先重新设计可持续、可观测的天气信用分配，而不是直接提高训练步数或奖励系数。

## Provenance note

Daily `forecast_mode_056_057` info inherited the shared wrapper label `engineered_perfect_hindcast_lookahead`; the isolated config, forecast calendar audit and actual feature columns define the compact rolling-7-day contract. This label-only discrepancy does not change the trained observation or reward, but should be corrected before any formal replay.
