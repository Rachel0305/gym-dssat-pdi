# 034_00 统一 multisite 输入链与 IC=1 条件下重建五站点四情景基线

## 背景

033_04 已经使用正确输入源：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/
```

完成五站点 half-split MaskablePPO 重跑，并且 033_05 已确认该输入链在可用站点年份中满足：

- 源模板、天气、土壤、品种均来自 `multisite_new_cultivar_inputs_013`；
- 渲染后 treatment 1 启用 `IC=1, MI=1, MF=1`；
- WSTA 与目标年份 WTH 文件一致；
- 不 fallback 到旧 `my_data`。

但旧的四情景基线来自旧输入链或历史拼接结果，不能直接与 033_04 PPO 结果比较。因此本任务先重建四情景基线，作为后续正式比较、作图和汇报的统一尺子。

## 目标

在与 033_04 PPO 完全一致的站点年份范围内，用同一套新输入链和 IC=1 渲染流程，重建以下四情景：

1. `null`：不主动灌溉、不主动施氮；
2. `recorded_farmer_template`：沿用项目已有 recorded farmer 模板复用口径，明确标注为模板复用，不冒充逐年真实观测；
3. `official_extension_expert`：沿用 018_03 的区域专家推荐 DAP 与剂量；
4. `dssat_auto`：使用 DSSAT 自动管理块，不用 RL 外部动作驱动。

## 固定输入与边界

- 站点年份清单：读取 033_04 已锁定的 `033_04_available_weather_half_split_years.csv`；
- 输入源：只允许 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`；
- 渲染：必须经过 `ppo_safe_rendering.py`；
- 必须启用 `IC=1, MI=1, MF=1`；
- 不使用旧 `my_data`；
- 不训练 PPO/DQN；
- 不与旧基线混合；
- 不因某情景失败而静默跳过，失败必须写入 manifest。

## 执行顺序

1. `--mode smoke`：每个站点取一个可用年份，跑四情景，确认 DSSAT 和渲染链路能跑通；
2. 若 smoke 通过，再运行 `--mode full`：覆盖 033_04 所有可用站点年份；
3. 输出 summary、daily、coverage manifest、失败表、实验记录。

## 输出

结果目录：

```text
benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/
```

主要文件：

```text
evaluation/034_00_full_baseline_summary.csv
evaluation/034_00_full_baseline_daily.csv
evaluation/034_00_full_coverage_manifest.csv
evaluation/034_00_full_failures.csv
docs/034_00_multisite_input_ic1_four_baseline_rebuild_record.md
```

## 判读

本任务只回答：

> 在与 033_04 PPO 相同输入链、相同 IC=1 条件下，五站点所有可用年份的四情景基线是否可以统一重建？

不在本任务中评价 PPO 是否优于四情景；正式比较和绘图留给后续任务。
