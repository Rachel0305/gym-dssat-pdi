# 012_16 HLA2015 文献对齐版 DQN Q-value 诊断记录

## 实验目的

012_15 中，文献对齐版 DQN seed0 5K 学到了：

- 产量约 7652 kg/ha；
- 灌溉 102 mm；
- 施氮 150 kg/ha。

离线复评分显示，按同一套文献式 reward，固定扫描中的 I60/N0 比 DQN 学到的 I102/N150 更高。因此本轮不重新训练，只加载 012_15 已训练模型，检查 Q-value 排序为什么偏向高氮动作。

## 输入

- 模型：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_aligned_dqn_012_15/2015/literature_aligned_seed0_5000steps/models/dqn_literature_aligned_probe.zip
```

- 脚本：

```text
src/run_hla2015_literature_dqn_q_value_diagnostic_012_16.py
```

- prompt：

```text
prompts/012_16_hla2015_literature_dqn_q_value_diagnostic.md
```

- 运行环境：

```text
Docker: b2fd6726c8c1
Python: /opt/gym_dssat_pdi/bin/python
```

## 输出文件

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_q_value_diagnostic_012_16/012_15_seed0_5k_q_values_daily.csv
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_q_value_diagnostic_012_16/012_15_seed0_5k_top8_actions_daily.csv
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_q_value_diagnostic_012_16/012_15_seed0_5k_top8_actions_key_daps.csv
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_q_value_diagnostic_012_16/012_15_seed0_5k_q_value_summary.csv
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_q_value_diagnostic_012_16/012_15_seed0_5k_q_value_summary.json
```

## 主要结果

| 指标 | 数值 |
|---|---:|
| 评估步数 | 158 |
| 灌溉窗口步数 | 63 |
| 施氮窗口步数 | 32 |
| 全季 best-Q 含灌溉动作比例 | 0.582 |
| 全季 best-Q 含氮动作比例 | 0.627 |
| 灌溉窗口 best-Q 含灌溉动作比例 | 0.476 |
| 灌溉窗口 best-Q 含氮动作比例 | 0.587 |
| 施氮窗口 best-Q 含氮动作比例 | 0.500 |
| 最终确定性策略灌溉 | 102 mm |
| 最终确定性策略施氮 | 150 kg/ha |
| 最终产量 | 7651.86 kg/ha |

## 窗口内常见 best-Q 动作

### 灌溉窗口内

| 排名 | best-Q 动作 | 次数 |
|---:|---|---:|
| 1 | I0_N0 | 16 |
| 2 | I0_N160 | 14 |
| 3 | I12_N0 | 9 |
| 4 | I24_N40 | 7 |
| 5 | I12_N40 | 5 |

### 施氮窗口内

| 排名 | best-Q 动作 | 次数 |
|---:|---|---:|
| 1 | I0_N0 | 16 |
| 2 | I24_N40 | 7 |
| 3 | I0_N160 | 4 |
| 4 | I12_N120 | 3 |
| 5 | I6_N40 | 1 |

## 关键 DAP 的 top-1 Q 动作

| DAP | top-1 Q 动作 | 原始灌溉 mm | 原始施氮 kg/ha | Q - Q(no-op) |
|---:|---|---:|---:|---:|
| 27 | I0_N0 | 0 | 0 | 0.000 |
| 35 | I0_N0 | 0 | 0 | 0.000 |
| 46 | I12_N40 | 12 | 40 | 0.340 |
| 53 | I0_N80 | 0 | 80 | 0.905 |
| 60 | I0_N160 | 0 | 160 | 0.335 |
| 67 | I24_N40 | 24 | 40 | 0.699 |
| 74 | I24_N40 | 24 | 40 | 0.729 |
| 82 | I0_N160 | 0 | 160 | 0.360 |
| 89 | I12_N0 | 12 | 0 | 0.531 |
| 96 | I12_N80 | 12 | 80 | 0.755 |

注意：DAP 82、96 已经不在施氮窗口或氮预算已满，因此即使 raw action 含氮，safe action 中氮也会被 wrapper 置零。但这些 Q 排序仍然说明网络对“含氮动作”的价值估计偏高。

## 解释

这次诊断基本坐实了 012_15 的问题：

1. **reward 本身不支持 N150 最优**  
   012_15 的离线复评分显示，按同一套 reward，I60/N0 分数高于 I102/N150。

2. **但 Q 网络把含氮动作排得偏高**  
   全季 best-Q 含氮动作比例达到 62.7%，灌溉窗口里也有 58.7% 的 best-Q 动作含氮。

3. **Q 排序很脆弱**  
   很多关键 DAP 上，best action 与 no-op 的 Q 差距只有 0.3–0.9。对一个最终 reward 约 900–1100 的任务来说，这个差距很小，说明动作排序可能容易受随机初始化、回放样本和 bootstrap 误差影响。

4. **问题更像是信用分配/Q 估计问题，而不是单纯 reward 设计问题**  
   终端产量奖励是稀疏的，水氮动作发生在前中期，收益到收获才体现。普通 DQN 需要通过 bootstrap 把终端收益往前传，这在当前短但高噪声的农业模拟任务里不稳定。

## 结论

012_16 支持以下判断：

> 文献对齐版 DQN 能让 seed0 学会灌溉并提高产量，但没有解决氮动作的 Q-value 高估问题。DQN 最终使用 N150 不是因为 reward 理论上偏好 N150，而是因为 Q 网络在关键状态中把含氮动作排高。

## 下一步建议

不建议继续只跑 seed1/seed2 来赌结果，也不建议继续单独试 Double 或 Dueling。更有针对性的方向是：

1. **n-step return**：让收获期产量奖励更直接地回传到早期水氮动作；
2. **prioritized replay**：提高关键高收益/高差异轨迹的采样概率；
3. **固定扫描/专家策略 warm-start**：用已知较优的 I60/N0 或专家策略给 Q 网络一个初始排序参考；
4. 如果只是为了向导师汇报，当前已经可以说明：参考文献式 DQN 设置能改善“完全不作为”，但还不能稳定解决混合水氮任务中的信用分配问题。
