# 036_03 重放 036_01 checkpoint 补算 WP_ET 并审计 FQA2018

## 背景

036_02 确认：

- 036_01 的 PPO 评估表没有 `WP_ET_kg_m3`；
- 这不是模型训练问题，而是评估后处理没有保存 DSSAT Summary/ET；
- SYA 基线已可接入；
- FQA2018 四个 checkpoint 的 `final_grnwt=0`，需要结合 DSSAT Summary/OUT 进一步审计。

## 目标

本任务不训练模型，只用 036_01 已有 checkpoint 做确定性重放：

1. 对 036_01 的 200 条 `station × validation_year × checkpoint` 评估重新运行一次；
2. 从 DSSAT `Summary.OUT` 提取 `ETCP`、`YPEM`、`YPNAM`，补算 PPO 的 `WP_ET_kg_m3` 与 Summary 口径 PFP_N；
3. 只保存 FQA2018 四个异常 checkpoint 的完整 DSSAT snapshot，避免无谓占用空间；
4. 输出带 WP_ET 的完整比较表和站点汇总表。

## 不允许的动作

- 不训练；
- 不改 reward；
- 不改 PPO checkpoint；
- 不改动作约束；
- 不改变 036_01 的模型选择逻辑。

## 输出

主目录：

```text
benchmark_results/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018
```

预期输出：

- `tables/036_03_replay_metrics.csv`
- `tables/036_03_corrected_checkpoint_validation_with_wp_et.csv`
- `tables/036_03_by_station_checkpoint_with_wp_et.csv`
- `tables/036_03_fqa2018_summary_audit.csv`
- `snapshots/FQA/2018/ckpt*/Summary.OUT` 等异常快照；
- `docs/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018_record.md`
