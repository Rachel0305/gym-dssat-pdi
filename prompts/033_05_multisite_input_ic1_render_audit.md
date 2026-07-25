# 033_05 multisite 输入源 IC=1 渲染审计 prompt

## 目标

在正式重跑 PPO/基线前，逐个检查即将使用的五站点 half-split 年份输入文件，确认不会再出现 `IC=0`。

本任务只渲染和解析输入，不跑 DSSAT、不训练。

## 审计范围

- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`
- 年份集合：`032_21_half_split_years.csv` 中的五站点全部 train/validation 年份
- 渲染函数：`ppo_safe_rendering.safe_render_template`

## 必须全部满足

1. treatment 1 的 `IC=1`
2. treatment 1 的 `MI=1`
3. treatment 1 的 `MF=1`
4. treatment 1 的 `IC` 能在 `*INITIAL CONDITIONS` 中找到对应 id
5. 渲染文件中的所有 WSTA token 均为目标年天气站
6. 渲染目录下存在且只需要目标年 `.WTH`
7. 源模板、土壤、品种、天气均来自 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`

## 输出

```text
benchmark_results/033_05_multisite_input_ic1_render_audit/
docs/033_05_multisite_input_ic1_render_audit_record.md
```

若有任何失败，停止，不允许进入训练。
