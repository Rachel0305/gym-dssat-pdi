# 012_18 HLA2015 文献对齐 + n-step DQN Q-value 诊断记录

## 实验目的

012_17 将文献式 DQN 的 `n_steps` 从 1 改为 5 后：

- 产量基本不变；
- 灌溉从 102 mm 降到 96 mm；
- 施氮仍然是 N150。

本轮不重新训练，直接加载 012_17 模型，检查 n-step 是否改善了 Q-value 排序，以及为什么施氮仍然打满。

## 输入

- 模型：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_nstep_012_17/2015/literature_nstep_nstep5_seed0_5000steps/models/dqn_literature_nstep_probe.zip
```

- 脚本：

```text
src/run_hla2015_literature_nstep_q_value_diagnostic_012_18.py
```

- prompt：

```text
prompts/012_18_hla2015_literature_nstep_dqn_q_value_diagnostic.md
```

## 输出

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_nstep_q_value_diagnostic_012_18/012_17_nstep5_seed0_5k_q_values_daily.csv
DSSAT_auto_validation/HLA_2004/hla2015_literature_nstep_q_value_diagnostic_012_18/012_17_nstep5_seed0_5k_top8_actions_daily.csv
DSSAT_auto_validation/HLA_2004/hla2015_literature_nstep_q_value_diagnostic_012_18/012_17_nstep5_seed0_5k_top8_actions_key_daps.csv
DSSAT_auto_validation/HLA_2004/hla2015_literature_nstep_q_value_diagnostic_012_18/012_17_nstep5_seed0_5k_q_value_summary.csv
DSSAT_auto_validation/HLA_2004/hla2015_literature_nstep_q_value_diagnostic_012_18/012_17_nstep5_seed0_5k_q_value_summary.json
```

## 主要结果

| 指标 | 012_16：n=1 | 012_18：n=5 |
|---|---:|---:|
| 全季 best-Q 含灌溉动作比例 | 0.582 | 0.747 |
| 全季 best-Q 含氮动作比例 | 0.627 | 0.627 |
| 灌溉窗口 best-Q 含灌溉动作比例 | 0.476 | 0.889 |
| 灌溉窗口 best-Q 含氮动作比例 | 0.587 | 0.476 |
| 施氮窗口 best-Q 含氮动作比例 | 0.500 | 0.500 |
| 最终灌溉 | 102 mm | 96 mm |
| 最终施氮 | 150 kg/ha | 150 kg/ha |
| 最终产量 | 7651.86 kg/ha | 7652.31 kg/ha |

## n=5 下常见 best-Q 动作

### 灌溉窗口内

| 排名 | best-Q 动作 | 次数 |
|---:|---|---:|
| 1 | I12_N0 | 17 |
| 2 | I6_N0 | 16 |
| 3 | I24_N160 | 13 |
| 4 | I0_N80 | 7 |
| 5 | I6_N40 | 5 |

### 施氮窗口内

| 排名 | best-Q 动作 | 次数 |
|---:|---|---:|
| 1 | I6_N0 | 16 |
| 2 | I24_N160 | 10 |
| 3 | I6_N40 | 4 |
| 4 | I12_N120 | 2 |

## 关键 DAP top-1 Q 动作

| DAP | top-1 Q 动作 | 原始灌溉 mm | 原始施氮 kg/ha | Q - Q(no-op) |
|---:|---|---:|---:|---:|
| 21 | I6_N0 | 6 | 0 | 0.480 |
| 28 | I6_N0 | 6 | 0 | 0.932 |
| 35 | I6_N0 | 6 | 0 | 4.626 |
| 46 | I0_N80 | 0 | 80 | 1.036 |
| 53 | I24_N160 | 24 | 160 | 1.258 |
| 60 | I24_N160 | 24 | 160 | 2.542 |
| 67 | I24_N160 | 24 | 160 | 3.573 |
| 74 | I24_N160 | 24 | 160 | 1.523 |
| 81 | I12_N0 | 12 | 0 | 0.992 |
| 88 | I12_N0 | 12 | 0 | 0.920 |
| 95 | I18_N40 | 18 | 40 | 1.299 |

## 解释

012_18 给出了一个比较干净的判断：

1. **n-step 确实改善了灌溉动作排序**  
   灌溉窗口中 best-Q 含灌溉动作比例从 47.6% 提高到 88.9%。这解释了为什么 012_17 比 012_15 的灌溉策略略微更合理。

2. **n-step 没有解决氮动作高估**  
   全季 best-Q 含氮动作比例仍然是 62.7%，施氮窗口内仍然是 50.0%。尤其 DAP53–74 期间，top-1 Q 动作连续偏向 `I24_N160`。

3. **N150 的来源很明确**  
   012_17 的确定性策略在 DAP60 施 40 kg/ha，DAP67 再施 110 kg/ha，正好把 N150 打满。Q-value 表明这是因为 DAP53–67 附近高氮动作被排在最高。

## 结论

> `n_steps=5` 对灌溉信用分配有帮助，但不足以纠正施氮动作的 Q-value 高估。

因此，“只调 n_steps”这条线可以关闭。它不是完全没用，但不能解决混合水氮任务中最关键的氮策略问题。

## 下一步建议

如果继续算法线，不建议继续单独增加 n_steps 或重复 seed。更值得尝试的是：

1. **prioritized replay**：让高价值的少氮/适量灌溉轨迹更频繁进入更新；
2. **warm-start / imitation**：用固定扫描得到的 I60/N0 轨迹或专家轨迹先预训练 Q 排序；
3. **分阶段/分变量动作结构**：先决定是否操作，再决定水/氮量，减少 25 动作中高氮组合对 Q 排序的干扰。
