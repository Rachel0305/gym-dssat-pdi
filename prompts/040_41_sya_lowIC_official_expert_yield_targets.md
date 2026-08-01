# 040_41 SYA lowIC official expert 产量目标表

## 背景

040_40 的 terminal yield guardrail 需要每个训练/验证年份的 official expert 产量作为目标来源。

现有 040_21 四情景基线只覆盖 2014–2023 验证年份，缺少 2005–2013 训练年份。因此 040_40 dry-run 暂停，不能直接训练。

## 本任务目的

只在 SYA lowIC 输入下补齐 2005–2023 年 `official_extension_expert` 情景产量，作为 040_40 reward v3 的目标表。

本任务不是训练任务，不评价 PPO，不修改奖励函数。

## 固定项

- 输入目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 年份：2005–2023
- 情景：仅 `official_extension_expert`
- 原始数据不覆盖。

## 输出

- `benchmark_results/040_41_sya_lowIC_official_expert_yield_targets/evaluation/040_41_official_expert_yield_targets.csv`
- `docs/040_41_sya_lowIC_official_expert_yield_targets_record.md`

## 通过条件

- 19/19 年均成功；
- 输出包含每年 `grain_yield_kg_ha`；
- 源输入为 lowIC；
- 后续 040_40 dry-run 能找到 2005–2023 全部年份目标。

