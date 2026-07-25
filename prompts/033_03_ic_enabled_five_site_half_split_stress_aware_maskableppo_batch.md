# 033_03 启用 IC 后五站点 half-split stress-aware MaskablePPO 重跑 prompt

## 背景

033_01 证实历史主流程渲染输入中 treatment `IC=0`，导致 `*INITIAL CONDITIONS` 没有被 DSSAT treatment 因子启用。

033_02 已修复并验证 `src/ppo_safe_rendering.py`：未来渲染输入会显式设置 `IC=1, MI=1, MF=1`，且 `IC=1` 能对应到当前模板中的 `*INITIAL CONDITIONS` treatment id。

因此，历史 032_22 五站点 half-split PPO 结果不能作为最终 IC-enabled 结果，需要重跑。

## 目标

在启用 IC 的渲染链条下，重跑与 032_22 相同的五站点 half-split stress-aware MaskablePPO 主训练与验证流程。

## 固定设置

- 算法：MaskablePPO；
- 站点：FQA、HLA、LCA、SYA、YCA；
- 年份切分：沿用 `032_21_half_split_years.csv`；
- 每站点训练步数：100,000 timesteps；
- checkpoint：25K、50K、75K、100K；
- seed：0；
- reward、动作空间、约束、训练参数均沿用 032_22；
- 唯一改变：渲染输入启用 `IC=1`。

## 边界

- 不覆盖 032_22 历史结果；
- 不使用 IC=0 的旧四情景基线做正式比较；
- 本轮只产出 IC-enabled PPO 训练/验证结果；
- 四情景基线需要后续用同样 IC-enabled 渲染链条单独重跑，再汇总比较。

## 输出

输出目录：

```text
benchmark_results/033_03_ic_enabled_five_site_half_split_stress_aware_maskableppo_batch/
```

实验记录：

```text
docs/033_03_ic_enabled_five_site_half_split_stress_aware_maskableppo_batch_record.md
```

## 判读

- 本轮结果只回答：启用 IC 后，同一 PPO 训练框架在五站点 half-split 设置下是否能正常训练和跨年份验证；
- 若 WSPD/NSTD、产量和措施相对 032_22 发生明显变化，应视为 IC 修正导致的正式新结果，而不是调参结果；
- 不在本轮对“是否超过四情景最高值”下最终结论，因为四情景基线也必须重跑。
