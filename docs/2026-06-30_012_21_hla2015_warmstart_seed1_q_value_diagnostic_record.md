# 012_21 HLA2015 warm-start seed1 Q-value 诊断记录

## 实验目的

012_19 warm-start seed0 得到 I66/N0；
012_20 warm-start seed1 得到 I48/N150。

本轮不重新训练，直接加载 012_20 seed1 模型，诊断为什么同样 warm-start 下 seed1 仍然漂回 N150。

## 输入

- 模型：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_warmstart_012_19/2015/literature_warmstart_nstep5_ws8_seed1_5000steps/models/dqn_literature_warmstart_probe.zip
```

- prompt：

```text
prompts/012_21_hla2015_warmstart_seed1_q_value_diagnostic.md
```

- 脚本：

```text
src/run_hla2015_warmstart_seed1_q_value_diagnostic_012_21.py
```

## 输出

```text
DSSAT_auto_validation/HLA_2004/hla2015_warmstart_seed1_q_value_diagnostic_012_21/012_20_seed1_q_values_daily.csv
DSSAT_auto_validation/HLA_2004/hla2015_warmstart_seed1_q_value_diagnostic_012_21/012_20_seed1_top8_actions_daily.csv
DSSAT_auto_validation/HLA_2004/hla2015_warmstart_seed1_q_value_diagnostic_012_21/012_20_seed1_top8_actions_key_daps.csv
DSSAT_auto_validation/HLA_2004/hla2015_warmstart_seed1_q_value_diagnostic_012_21/012_20_seed1_q_value_summary.csv
DSSAT_auto_validation/HLA_2004/hla2015_warmstart_seed1_q_value_diagnostic_012_21/012_20_seed1_q_value_summary.json
```

## 主要结果

| 指标 | 012_19 seed0 | 012_21 seed1 |
|---|---:|---:|
| 全季 best-Q 含灌溉动作比例 | 未单独诊断 | 0.392 |
| 全季 best-Q 含氮动作比例 | 未单独诊断 | 0.791 |
| 灌溉窗口 best-Q 含灌溉动作比例 | 未单独诊断 | 0.190 |
| 灌溉窗口 best-Q 含氮动作比例 | 未单独诊断 | 0.746 |
| 施氮窗口 best-Q 含氮动作比例 | 未单独诊断 | 0.500 |
| 最终灌溉 | 66 mm | 48 mm |
| 最终施氮 | 0 kg/ha | 150 kg/ha |
| 最终产量 | 7653.05 kg/ha | 7653.19 kg/ha |

## seed1 常见 best-Q 动作

### 灌溉窗口内

| 排名 | best-Q 动作 | 次数 |
|---:|---|---:|
| 1 | I0_N120 | 34 |
| 2 | I0_N0 | 16 |
| 3 | I24_N80 | 12 |
| 4 | I0_N160 | 1 |

### 全季

| 排名 | best-Q 动作 | 次数 |
|---:|---|---:|
| 1 | I0_N120 | 53 |
| 2 | I0_N0 | 30 |
| 3 | I24_N80 | 22 |
| 4 | I24_N40 | 19 |
| 5 | I18_N120 | 12 |

## 关键 DAP top-1 Q 动作

| DAP | top-1 Q 动作 | 原始灌溉 mm | 原始施氮 kg/ha | Q - Q(no-op) |
|---:|---|---:|---:|---:|
| 21 | I0_N0 | 0 | 0 | 0.000 |
| 28 | I0_N0 | 0 | 0 | 0.000 |
| 35 | I0_N0 | 0 | 0 | 0.000 |
| 46 | I0_N120 | 0 | 120 | 5.603 |
| 53 | I0_N120 | 0 | 120 | 10.370 |
| 56 | I0_N120 | 0 | 120 | 10.047 |
| 60 | I0_N120 | 0 | 120 | 14.257 |
| 63 | I0_N120 | 0 | 120 | 17.501 |
| 67 | I0_N120 | 0 | 120 | 17.152 |
| 74 | I0_N120 | 0 | 120 | 9.798 |
| 85 | I24_N80 | 24 | 80 | 1.105 |
| 92 | I24_N80 | 24 | 80 | 2.016 |

## 解释

这个诊断非常明确：

1. **seed1 的 Q 网络严重偏向氮动作**  
   全季 best-Q 含氮动作比例 79.1%，灌溉窗口里也有 74.6% 的 best-Q 动作含氮。

2. **漂回 N150 的关键时间段是 DAP46–74**  
   这段时间 top-1 Q 动作连续是 `I0_N120`。到了 DAP56 和 DAP63，确定性策略就执行了 120 + 30 kg/ha，把 N150 打满。

3. **这不是微弱误差**  
   DAP53–67 期间，`I0_N120` 的 Q 值比 no-op 高 10–17 左右。相比 012_16/012_18 那种 0.3–3 的小差距，这里偏向高氮动作更强。

4. **warm-start 被在线训练覆盖**  
   warm-start 目标是 I60/N0，但 seed1 经过 5K 在线 DQN 更新后，Q 排序被重新推向高氮动作。

## 结论

> 012_20 seed1 失败不是因为环境、reward 或 wrapper 不支持少氮策略，而是因为在线 DQN 更新在该 seed 下强烈高估了中期高氮动作，覆盖了 warm-start 的初始排序。

因此，当前一次性 warm-start 不足以稳定解决问题。

## 下一步建议

如果继续算法开发，下一步不应只是再跑 seed 或增加 warm-start epochs，而应改成更系统的方案：

1. **持续 imitation regularization**  
   在 RL 训练过程中持续加入少量 imitation loss，而不是只在训练前 warm-start 一次。

2. **Replay 中混入高分示范轨迹**  
   将 I60/N0 这类高分轨迹持续放入 replay buffer，防止在线样本完全覆盖初始排序。

3. **约束式 action masking / 二阶段动作结构**  
   先判断是否需要施氮，再决定施氮量，减少 `I0_N120/I0_N160` 这类动作直接支配 Q 排序。

如果当前目标是准备导师汇报，可以总结为：

```text
文献式 DQN 能学到灌溉增产；
少氮策略在离线复评分和 seed0 warm-start 中可行；
但当前普通 DQN 在线更新仍存在 seed 敏感性，高氮动作 Q-value 高估会覆盖初始排序；
下一步需要更系统的 imitation/replay 或动作结构约束，而不是继续单纯调 seed、n_steps 或 reward。
```
