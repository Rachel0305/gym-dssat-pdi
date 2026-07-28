# 035_05 FQA2014 氮边际收益审计

## 背景

035_04 将训练 reward 改为项目现有综合指标口径：

```text
final_GRNWT - 1.1 * irrigation - 1.58 * nitrogen
```

但 FQA2014 MaskablePPO 50K 最好 checkpoint 仍然使用 N240，虽然灌溉从 I150 降到 I105，但施氮明显偏高，PFP_N 很低。

因此本任务不继续训练，而是先用受控 DSSAT 前向模拟审计：在固定灌溉和固定施氮时机下，N80/N120/N160/N200/N240 的边际产量收益是否足以覆盖当前 `1.58 kg yield-equivalent / kg N` 的氮成本。

## 任务目标

回答两个问题：

1. FQA2014 在合理早期分期施氮时，多加 N 是否仍然带来显著产量收益？
2. 当前 `nitrogen_cost = 1.58` 是否足以阻止 N240？

## 固定设置

- 站点：FQA
- 年份：2014
- DSSAT linked 管理：`IRRIG=L, FERTI=L`
- 不训练模型
- 不调算法参数
- 固定灌溉：DAP1、DAP50、DAP65 各 I15，总 I45
- 固定施氮候选窗口：DAP1、DAP30、DAP50、DAP65
- 仅改变总施氮量

## 候选规则

- `i45_n0`
- `i45_n80`
- `i45_n120`
- `i45_n160`
- `i45_n200`
- `i45_n240`

所有动作必须通过 safety layer，且 Python safe total 与 DSSAT Summary total 必须一致。

## 输出

- `benchmark_results/035_05_fqa2014_n_marginal_return_audit/evaluation/035_05_n_marginal_summary.csv`
- `benchmark_results/035_05_fqa2014_n_marginal_return_audit/evaluation/035_05_n_marginal_deltas.csv`
- `docs/035_05_fqa2014_n_marginal_return_audit_record.md`

## 判据

重点看相邻 N 档位：

```text
marginal_yield_per_kgN = delta_yield / delta_N
```

若某一档的边际产量收益明显大于 1.58，则当前项目 simple_profit 仍会偏好继续加氮。

若 N200→N240 的边际收益仍大于 1.58，则 PPO 选择 N240 在该 reward 下不是“训练坏了”，而是 reward 算术上仍然认为 N240 值得。

## 停止线

本任务只做受控前向模拟和边际收益审计。即使发现 nitrogen_cost 偏低，也不在本任务里修改 reward 或启动训练。
