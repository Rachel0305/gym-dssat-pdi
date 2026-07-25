# 033_02 启用 IC 后的渲染 smoke prompt

## 背景

033_01 确认源模板和主流程渲染输入中 treatment `IC=0`，导致 `*INITIAL CONDITIONS` 没有通过 treatment 因子启用。

已对 `src/ppo_safe_rendering.py` 做最小补丁：渲染工作副本时显式启用 `IC/MI/MF`，不修改 `my_data/` 原始模板。

## 目标

验证补丁后的渲染函数是否稳定输出：

```text
IC=1, MI=1, MF=1
```

并且 `IC=1` 能对应到 `*INITIAL CONDITIONS` 中存在的 treatment id。

## 设计

从 `experiments/ppo_observed_years/config_ppo_observed_years.yaml` 中，每个站点选第一个 observed year 做渲染 smoke：

- HLA
- SYA
- LCA
- FQA
- YCA

只调用 `safe_render_template` 生成派生输入，不跑 DSSAT，不训练。

## 输出

输出目录：

```text
benchmark_results/033_02_enable_ic_render_smoke/
```

需要生成：

1. `033_02_render_smoke_rows.csv`
2. `docs/033_02_enable_ic_render_smoke_record.md`

## 判定

- 5/5 站点均满足 `IC=1, MI=1, MF=1`；
- 5/5 站点的 `IC=1` 均能对应 `*INITIAL CONDITIONS` treatment id；
- 若不满足，停止，不进入重跑。
