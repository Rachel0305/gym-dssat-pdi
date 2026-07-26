# 036_02 修正 036_01 的基线与指标后处理

## 背景

036_01 已完成五站点原自由时序 MaskablePPO 正式重跑，但后处理表暴露出三个问题：

1. PPO 结果表没有 `WP_ET_kg_m3` 列，导致 `gap_wp_et_vs_four_max` 全为 `NA`；
2. SYA 在 036_01 使用的四站点统一基线表中不存在，导致 `baseline_rows=0`；
3. FQA2018 出现 PPO `final_grnwt=0`，需要列入异常审计。

## 本任务目标

本任务不训练、不重跑 PPO，仅做结果后处理审计：

- 合并四站点统一基线与 SYA 专用基线；
- 明确 canonical 四情景覆盖情况；
- 对 SYA 的 recorded farmer 模板情景做显式标记，不伪装成原始 recorded farmer；
- 重算 yield 与 PFP_N 相对可用基线最高值的差距；
- 标记 WP_ET 缺失原因；
- 输出异常清单，尤其是 FQA2018 产量为 0。

## 重要边界

- 036_02 不改变 036_01 的模型、checkpoint、reward 或动作约束；
- 036_02 不用 daily CSV 虚构 ET，因此不会假算 PPO 的 WP_ET；
- 若需要补 PPO 的 WP_ET，后续必须另开任务，用已有 checkpoint 做“只评估、不训练”的 DSSAT 轻量重放并保存 Summary/ET。

## 输出

主输出目录：

```text
benchmark_results/036_02_correct_03601_baseline_metric_postprocess
```

预期输出：

- `tables/036_02_corrected_checkpoint_validation_summary.csv`
- `tables/036_02_corrected_by_station_checkpoint.csv`
- `tables/036_02_baseline_coverage_by_station_year.csv`
- `tables/036_02_anomaly_flags.csv`
- `docs/036_02_correct_03601_baseline_metric_postprocess_record.md`
