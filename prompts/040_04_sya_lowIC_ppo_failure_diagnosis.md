# 040_04 SYA lowIC PPO 验证年失败诊断

## 目的

在 040_03 已经确认 PPO 是当前 SYA lowIC 自由时序框架下最优算法后，本任务回到 PPO 主线，诊断 PPO 在验证年份中仍然失败或表现较弱的年份。

本任务不训练、不重跑 DSSAT，只读取已有结果：

- PPO：`040_00` 最佳 checkpoint 75000 的验证年评估明细；
- 三基线：`039_02` 的 lowIC `null`、`official_extension_expert`、`dssat_auto` 汇总结果。

## 固定规则

1. PPO checkpoint 固定为 040_03 选出的最佳 checkpoint：75000。
2. 比较年份固定为 SYA 验证年份：2014–2023。
3. 基线只使用 lowIC 条件下已经重跑过的三基线，不混用 original IC。
4. 不临时改变 reward、动作约束或 checkpoint。

## 诊断指标

逐年输出：

- PPO 产量、灌溉、施氮、PFP_N、水分胁迫天数、氮胁迫天数；
- official expert、dssat auto、null 的产量、灌溉、施氮、WP_ET、PFP_N、最大水/氮胁迫；
- PPO 相对 official expert 的产量差、节水量、节氮量、PFP_N 差；
- PPO 是否超过三基线最高产量；
- PPO 是否有任一核心指标超过三基线最高值。

## 输出

- `benchmark_results/040_04_sya_lowIC_ppo_failure_diagnosis/tables/040_04_ppo_vs_lowIC_baselines_by_year.csv`
- `benchmark_results/040_04_sya_lowIC_ppo_failure_diagnosis/tables/040_04_ppo_failure_type_summary.csv`
- `docs/040_04_sya_lowIC_ppo_failure_diagnosis_record.md`

