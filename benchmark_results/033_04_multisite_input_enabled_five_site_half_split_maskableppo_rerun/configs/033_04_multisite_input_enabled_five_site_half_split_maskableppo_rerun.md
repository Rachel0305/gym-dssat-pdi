# 033_04 使用 multisite_new_cultivar_inputs_013 的五站点 half-split MaskablePPO 重跑 prompt

## 背景

用户明确指出：训练输入不应继续使用旧 `my_data/UFGA8201-*.jinja2` 链条，而应使用：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/
```

033_01 只证明旧输入链存在 IC=0 问题；在用户纠正输入源后，旧链结果只能作为历史排查记录，不能作为正式训练输入。

033_05 已完成新输入源的渲染审计：

- 有 multisite WTH 的 81 个站点-年份全部通过；
- 渲染后 treatment 1 均为 `IC=1, MI=1, MF=1`；
- 渲染后 WSTA 与目标年 `.WTH` 文件一致；
- 源模板、天气、土壤、品种均来自 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`；
- 缺少 multisite WTH 的年份必须记录并跳过，不允许静默回退到旧数据。

## 目标

在正确输入包 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013` 下，重跑与 032_22 相同的五站点 half-split stress-aware MaskablePPO 主训练与验证流程。

## 固定设置

- 算法：MaskablePPO；
- 站点：FQA、HLA、LCA、SYA、YCA；
- 年份切分：沿用 `032_21_half_split_years.csv`，但只保留 multisite 包中存在目标 WTH 的年份；
- 每站点训练步数：100,000 timesteps；
- checkpoint：25K、50K、75K、100K；
- seed：0；
- reward、动作空间、约束、训练参数均沿用 032_22；
- 输入源统一为 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`；
- 不使用 `my_data/UFGA8201-*.jinja2`。

## 缺天气年份处理

- 缺少 multisite WTH 的年份不训练、不验证、不 fallback；
- 跳过年份写入：

```text
benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/configs/033_04_skipped_missing_multisite_wth.csv
```

- 当前已知 HLA 仅有 2007、2009、2010、2011 四个可用 WTH，且都在原 half-split 训练段，因此 HLA 本轮可训练但无验证年份。

## 边界

- 不覆盖 032_22 历史结果；
- 不使用旧 IC=0 / 旧输入源四情景基线做正式比较；
- 本轮只产出正确输入源下的 PPO 训练/验证结果；
- 四情景基线需要后续用同样输入源单独重跑，再汇总比较。

## 输出

输出目录：

```text
benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/
```

实验记录：

```text
docs/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun_record.md
```

## 判读

- 本轮结果回答：在正确 multisite 输入源下，同一 PPO 训练框架在五站点 half-split 设置中是否能正常训练和跨年份验证；
- 若 WSPD/NSTD、产量和措施相对旧结果发生明显变化，应首先解释为输入源和 IC 修正导致的正式新结果，不是调参；
- 不在本轮对“是否超过四情景最高值”下最终结论，因为四情景基线也必须用同一输入源重跑。
