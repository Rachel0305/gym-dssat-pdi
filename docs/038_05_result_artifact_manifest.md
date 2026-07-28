# 038_05 当前可信小结果包清单

记录时间：2026-07-28

## 目的

本文件说明本次准备上传 GitHub 的“小结果包”包含什么、不包含什么，以及这些结果能如何使用。

这个结果包不是完整 `benchmark_results/` 备份，而是当前阶段最小可复盘证据包：

1. DSSAT 管理事件链审计；
2. FQ 静态四基线重建；
3. FQ PPO 与四基线五情景对照图和日值表。

## 为什么只上传小结果包

完整 `benchmark_results/` 目录体积很大，且包含：

- DSSAT runtime；
- rendered inputs；
- snapshots；
- 大量 `.OUT`；
- 历史失败尝试；
- 旧错误口径结果；
- 模型 checkpoint。

这些不适合直接上传 GitHub。当前阶段更适合上传小体积、可审查、可复用的图表与汇总表。

## 本次包含的结果

### 1. 037_05 DSSAT 管理事件链审计

目录：

```text
benchmark_results/037_05_dssat_management_event_chain_preflight/
```

上传内容：

```text
037_05_result.json
configs/037_05_dssat_management_event_chain_preflight.md
tables/037_05_management_event_chain_audit.csv
tables/037_05_management_event_chain_issues.csv
```

用途：

- 证明后续结果必须先检查 DSSAT 管理事件是否真实进入模拟链条；
- 记录管理事件链审计发现的问题；
- 作为 037 系列可信基线修复的前置证据。

### 2. 037_07 FQ 静态四基线重建

目录：

```text
benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/
```

上传内容：

```text
037_07_FQA_ppo036_vs_fixed_baseline.csv
037_07_full_baseline_summary_partial.csv
037_07_full_FQA_only_management_event_audit.csv
037_07_full_FQA_only_summary.csv
037_07_smoke_baseline_daily.csv
037_07_smoke_baseline_summary.csv
037_07_smoke_baseline_summary_partial.csv
037_07_smoke_coverage_manifest.csv
037_07_smoke_failures.csv
037_07_smoke_management_event_audit.csv
```

用途：

- 保存 FQ 静态四基线重建的核心表；
- 保存 smoke 与 full FQA-only 结果；
- 供后续 FQ 图表和指标比较复核。

注意：

- 不上传 `snapshots/`；
- 不上传 DSSAT `.OUT`；
- 不上传 rendered inputs。

### 3. 037_08 FQ PPO 固定基线验证图与日值表

目录：

```text
benchmark_results/037_08_FQA_validation_ppo_fixed_baseline_figures/
```

上传内容：

```text
figures/
tables/
```

其中 `figures/` 包含：

- FQ2014--FQ2023 的 PPO25K 五情景日过程图；
- FQA validation 的 yield / WP_ET / PFP_N 对四基线柱状图；
- PNG 与 SVG 双格式。

其中 `tables/` 包含：

- 每个年份的五情景日值合并表；
- PPO25K vs fixed four baseline 指标表；
- run manifest。

用途：

- 这是目前最适合给别人快速查看的 FQ 代表性结果包；
- 可用于检查 PPO 措施与天气、土壤水氮胁迫、产量轨迹和累积奖励是否匹配；
- 可用于复核 FQ 的产量、WP_ET、PFP_N 对比。

## 本次不包含的结果

明确不上传：

```text
benchmark_results/037_07_static_level1_four_baseline_rebuild/snapshots/
benchmark_results/037_07_static_level1_four_baseline_rebuild/rendered_inputs/
benchmark_results/037_08_FQA_validation_ppo_fixed_baseline_figures/runtime*/
benchmark_results/037_08_FQA_validation_ppo_fixed_baseline_figures/snapshots/
```

也不上传：

- 模型 zip；
- `.OUT` 中间文件；
- DSSAT 临时运行目录；
- 旧 026--036 大结果目录整体。

## 使用边界

这些结果可以用于：

- 当前 FQ 结果复盘；
- 向其他 AI 或导师展示当前可信小结果；
- 检查图表样式；
- 检查 DSSAT 管理事件链和基线修复结果。

这些结果不应被表述为：

- 五站点全部最终可信结果；
- 所有年份已完成统一重跑；
- 最终论文主结果已经冻结。

目前它们更准确的定位是：

> 在发现 IC/管理链问题后，针对 FQ 站点建立的一套可信基线与 PPO 对照小结果包。

## 推荐 commit message

```text
results037: add trusted FQ baseline and PPO figure artifacts
```
