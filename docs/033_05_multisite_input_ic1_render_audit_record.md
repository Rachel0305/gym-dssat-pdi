# 033_05 multisite 输入源 IC=1 渲染审计记录

## 任务性质

本任务只渲染并解析即将进入 033_04 的五站点 half-split 年份输入，不跑 DSSAT、不训练。

## 结果

- 总通过：97/97
- 因 multisite 包缺少目标年 WTH 而跳过：0

## 按站点通过情况

| station_code | rows | passed |
| --- | --- | --- |
| FQA | 19 | 19 |
| HLA | 20 | 20 |
| LCA | 19 | 19 |
| SYA | 19 | 19 |
| YCA | 20 | 20 |

## 缺天气年份汇总

无记录。

## 输出

- 全量表：`benchmark_results/033_05_multisite_input_ic1_render_audit/033_05_multisite_input_ic1_render_audit.csv`
- 失败表：`benchmark_results/033_05_multisite_input_ic1_render_audit/033_05_multisite_input_ic1_render_audit_failures.csv`
- 缺天气跳过表：`benchmark_results/033_05_multisite_input_ic1_render_audit/033_05_multisite_input_ic1_render_audit_skipped_missing_wth.csv`

## 判定

- 只有有天气数据的年份总通过率为 100% 时，才允许继续进入 033_04 正式训练。
- 缺少 multisite WTH 的年份不得静默训练，必须记录在跳过表或先补齐天气。
- 审计条件包括：`IC=1, MI=1, MF=1`、IC id 有效、WSTA 与目标年 WTH 一致、源模板/天气/土壤/品种均来自 multisite 输入包。
