# 039_01 lowIC 初始土壤水氮减半 DSSAT 胁迫响应 smoke

## 背景

039_00 已确认主线 authoritative templates 在派生目录
`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/`
下通过审计：

- `SH2O` 约为原始值 0.5 倍；
- `SNH4` 约为原始值 0.5 倍；
- `SNO3` 约为原始值 0.5 倍；
- `IC=1` 主线渲染链可用。

本任务只做 DSSAT 前向 smoke，不训练 PPO/DQN。

## 目的

检验“降低初始土壤水分和矿质氮”是否真的传导到 DSSAT 生理过程：

1. lowIC 的 null 情景是否比 original IC 出现更强/更早的水分胁迫或氮胁迫；
2. official expert 管理是否能缓解 lowIC 下的胁迫；
3. lowIC 是否明显影响产量、ET、灌溉/施氮统计。

## 固定设计

- 默认站点年份：`FQA 2014`
- 输入目录：
  - original: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/`
  - lowIC: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/`
- 情景：
  - `null`
  - `official_extension_expert`
- 不运行 RL 训练；
- 不修改原始输入目录；
- 不修改 lowIC 输入目录；
- 不调 reward；
- 不做多站点扩展，除非本 smoke 通过。

## 输出

写入：

`benchmark_results/039_01_lowIC_dssat_stress_smoke/`

包含：

- `tables/039_01_smoke_summary.csv`
- `tables/039_01_smoke_daily.csv`
- `figures/039_01_FQA2014_original_vs_lowIC_stress.png`
- `039_01_result.json`
- `039_01_lowIC_dssat_stress_smoke_record.md`

## 判据

### A 分支：允许进入下一步

满足：

- original 与 lowIC 均成功跑完；
- lowIC-null 的 `max_water_stress` 或 `max_nitrogen_stress` 高于 original-null；
- expert 在 lowIC 下相对 lowIC-null 降低至少一个胁迫指标，或提高产量；
- 生成每日表和图。

### B 分支：lowIC 没有增强胁迫

如果 original 与 lowIC 都跑完，但 lowIC-null 的 WSPD/NSTD 不增强：

- 停止训练；
- 记录为“初始条件减半不足以制造更强胁迫”；
- 后续再讨论是否进一步降低 IC 或检查 DSSAT 胁迫指标定义。

### C 分支：运行失败或输入链异常

如果 DSSAT 跑不完，或渲染输入没有指向预期输入源：

- 停止；
- 不训练；
- 先修输入链或渲染链。

