# 012_02 HLA2010 固定 DQN 灌溉动作下的施氮边际收益扫描记录

## 目的

012_01 的 DQN 离散动作试验显示：

- DQN 只用了 30 mm 灌溉，没有像 PPO 一样用满 I120；
- 但 DQN 仍然用了 N150；
- 因此需要判断：在 DQN 学到的灌溉动作下，N150 是否真的有产量边际收益。

本轮不训练，只做确定性前向模拟。

## 设计

固定灌溉：

- DAP30 灌溉 30 mm

扫描施氮：

| 方案 | 施氮安排 |
|---|---|
| N0 | 不施氮 |
| N50 | DAP30 施 50 kg N/ha |
| N100 | DAP30、DAP56 各施 50 kg N/ha |
| N150 | DAP30、DAP56、DAP63 各施 50 kg N/ha |

## 脚本

`src/run_hla2010_dqn_fixed_irrigation_n_scan_012_02.py`

## 执行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_fixed_irrigation_n_scan_012_02.py"
```

## 输出

输出目录：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_fixed_irrigation_n_scan_012_02`

汇总表：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_fixed_irrigation_n_scan_012_02/hla2010_fixed_dqn_irrigation_n_scan_summary.csv`

每个方案都有单独的日值表和 DSSAT/PDI 快照：

- `N0/daily.csv`
- `N50/daily.csv`
- `N100/daily.csv`
- `N150/daily.csv`

## 结果

| 方案 | 灌溉总量 | 施氮总量 | 产量 | 生物量 | 最大水分胁迫 | 最大氮胁迫 |
|---|---:|---:|---:|---:|---:|---:|
| N0 | 30 mm | 0 kg N/ha | 7354 kg/ha | 20143 kg/ha | 0.904 | 0.184 |
| N50 | 30 mm | 50 kg N/ha | 7354 kg/ha | 20139 kg/ha | 0.904 | 0.086 |
| N100 | 30 mm | 100 kg N/ha | 7355 kg/ha | 20144 kg/ha | 0.904 | 0.0158 |
| N150 | 30 mm | 150 kg N/ha | 7355 kg/ha | 20144 kg/ha | 0.904 | 0.0158 |

## 解释

在固定 DQN 灌溉动作（DAP30 灌溉 30 mm）下，N0 到 N150 的产量几乎没有变化，只有约 1 kg/ha 的差异。

施氮的主要作用是降低氮胁迫指标：

- N0 最大氮胁迫：0.184
- N50 最大氮胁迫：0.086
- N100/N150 最大氮胁迫：0.0158

但这种氮胁迫指标改善几乎没有转化为籽粒产量增加。

## 结论

HLA2010 当前情景下，DQN 用满 N150 不具备明显产量收益依据。

这说明：

1. DQN 离散动作确实改变了灌溉行为，使其不再用满 I120；
2. 但“施氮用满 N150”的问题仍然存在；
3. 该问题更可能来自 reward 对氮胁迫/施氮行为的引导，而不是来自真实产量边际收益；
4. 如果继续改算法，必须把“氮素资源效率”明确写进目标或约束，而不能只依赖当前 scalar reward。

## 对下一步的影响

不建议马上扩展 DQN 到 2015 或多 seed。

更合理的下一步是重新定义算法目标，例如：

- 用经济收益 reward，让没有产量收益的施氮受到明确惩罚；
- 或使用 constrained RL，把氮投入作为 cost constraint；
- 或把氮动作从“是否降低氮胁迫”转为“是否带来产量/收益边际改善”的决策。

当前证据支持：问题不只是 PPO 连续动作，也包括水氮联合 reward/目标定义中没有有效区分“降低胁迫指标”和“提高最终产量”。
