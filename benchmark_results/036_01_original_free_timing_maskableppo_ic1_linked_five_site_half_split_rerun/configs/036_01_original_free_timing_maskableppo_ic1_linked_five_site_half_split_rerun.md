# 036_01 原自由时序 MaskablePPO：IC=1 + linked action 五站点 half-split 正式重跑

## 背景

036_00 已通过正式重跑前审计：

- 原 032_22 配置检查：25/25 通过；
- `IC=1` / `multisite_013` 输入源检查：97/97 通过；
- `IRRIG=L, FERTI=L` linked 强制动作 smoke：5/5 通过。

因此，本任务进入正式主线重跑。

## 本任务目标

在修复后的输入和 DSSAT 动作接口条件下，重新运行原 032_22 自由时序 stress-aware MaskablePPO 五站点 half-split 实验。

本任务不是 reward 调参，不采用 035_04/035_06 的 reward 改动。035_04/035_06 只作为旁支探索记录保留。

## 固定配置

沿用 032_22 原正式配置：

- 算法：MaskablePPO；
- 训练方式：每个站点单独训练一个模型；
- 年份划分：half-split，前半训练、后半验证；
- seed：0；
- 每站点训练步数：100,000；
- checkpoint：25,000 / 50,000 / 75,000 / 100,000；
- 每日环境运行；
- agent 每天观察；
- 灌溉档位：`{0, 15, 30, 45}` mm；
- 施氮档位：`{0, 40, 80, 120}` kg/ha；
- 灌溉最小间隔：7 天；
- 施氮最小间隔：7 天；
- 单季灌溉上限：160 mm；
- 单季施氮上限：250 kg/ha；
- DAP90 后禁氮；
- reward：原 032_22 stress-aware reward。

## 本次只承认的底层修复

相对 032/033 历史结果，本次只承认两个底层修复：

1. 输入来自 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`，并确认 `IC=1`；
2. 动态 RL treatment 使用 `IRRIG=L, FERTI=L`，确保 PPO 动作真实进入 DSSAT。

## 输出

主输出目录：

```text
benchmark_results/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun
```

预期输出文件：

- `evaluation/036_01_training_checkpoint_inventory.csv`
- `evaluation/036_01_checkpoint_validation_summary.csv`
- `evaluation/036_01_validation_summary_by_station_checkpoint.csv`
- `logs/036_01_training_year_reset_counts.csv`
- `docs/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun_record.md`

## 停止线

本任务只运行原配置正式重跑，不现场改动 reward、训练步数、动作约束、seed 数或站点年份划分。
