# 012_12 HLA2015 DQN Q值排序诊断记录

## 目的

012_11 显示，HLA2015 的失败 seed 并不是训练过程中完全没有探索到灌溉。

seed0 和 seed2 在训练阶段都有大量有效灌溉 episode，但最终 deterministic policy 没有选择灌溉。因此本轮进一步检查最终模型的 Q 值排序：

> 在关键灌溉窗口内，模型到底把哪个 action 估成最高价值？

## 诊断对象

读取已有模型，不重新训练：

- seed0 5K；
- seed1 5K；
- seed2 5K；
- seed0 20K。

## 动作含义

| action | 含义 |
|---:|---|
| 0 | 不操作 |
| 1 | 灌溉 30 mm |
| 2 | 施氮 50 kg/ha |
| 3 | 灌溉 30 mm + 施氮 50 kg/ha |

## 方法

沿 deterministic evaluation 轨迹，在每个 DAP 记录：

- Q(action0)；
- Q(action1)；
- Q(action2)；
- Q(action3)；
- Q 值最高的 action；
- 实际 deterministic action；
- safe_amir / safe_anfer；
- reward；
- grnwt / topwt / swfac / nstres。

重点统计灌溉窗口内：

- 多少比例的状态中，action1 或 action3 是 Q 值最高；
- Q(action1) - Q(action0) 的均值；
- 最终评估中实际执行的灌溉量和 reward。

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_dqn_q_value_diagnostic_012_12.py"
```

绘图命令：

```bash
python src/plot_hla2015_dqn_q_value_diagnostic_012_12.py
```

## 输出文件

- Q值诊断脚本：

```text
src/run_hla2015_dqn_q_value_diagnostic_012_12.py
```

- 绘图脚本：

```text
src/plot_hla2015_dqn_q_value_diagnostic_012_12.py
```

- 所有 DAP 的 Q 值日值表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dqn_q_value_diagnostic_012_12/hla2015_dqn_q_values_all_runs_daily.csv
```

- 关键 DAP 的 Q 值表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dqn_q_value_diagnostic_012_12/hla2015_dqn_q_values_key_daps.csv
```

- Q值汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dqn_q_value_diagnostic_012_12/hla2015_dqn_q_value_summary.csv
```

- Q值诊断图：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dqn_q_value_diagnostic_012_12/figures/hla2015_dqn_q_value_summary.png
```

## 结果

| 模型 | 灌溉窗口内 action1/3 为最高Q比例 | 平均 Q(action1)-Q(action0) | 最终灌溉 mm | 最终施氮 kg/ha | 产量 kg/ha | reward |
|---|---:|---:|---:|---:|---:|---:|
| seed0 5K | 0.000 | -7.403 | 0 | 0 | 6485.8 | 6485.8 |
| seed1 5K | 0.159 | -2.088 | 60 | 0 | 7632.3 | 7572.3 |
| seed2 5K | 0.000 | -0.774 | 0 | 50 | 6522.9 | 6272.9 |
| seed0 20K | 0.079 | -8.439 | 30 | 50 | 7320.6 | 7040.6 |

## 关键解释

### 1. 失败 seed 确实没有把灌溉动作估为最佳

seed0 5K 和 seed2 5K 在灌溉窗口内：

- action1 或 action3 成为最高 Q 的比例都是 0；
- 最终评估灌溉量也都是 0。

这说明它们最终不灌溉，不是 wrapper 拦截导致，而是模型自己的 Q 值排序中灌溉没有赢。

### 2. seed1 的成功来自部分关键窗口中灌溉 Q 值胜出

seed1 5K：

- 灌溉窗口内 action1/3 为最高 Q 的比例是 0.159；
- 最终执行了 60 mm 灌溉；
- 产量和 reward 接近 fixed I60/N0。

所以 seed1 并不是随机评估时碰巧灌溉，而是 Q 网络在部分关键状态下确实把灌溉估成了最佳动作。

### 3. seed0 20K 有改善，但仍不稳定

seed0 20K：

- 灌溉窗口内 action1/3 为最高 Q 的比例从 0 提高到 0.079；
- 最终执行 30 mm 灌溉；
- 产量和 reward 高于 seed0 5K；
- 但仍低于 seed1 5K 和 fixed I60/N0。

这说明延长训练能稍微改善 Q 值排序，但不足以稳定学到合理策略。

### 4. 平均 Q(action1)-Q(action0) 仍为负

所有模型中，灌溉窗口内平均 Q(action1)-Q(action0) 都是负值。

这说明即使 seed1 成功，灌溉动作也不是在所有灌溉窗口状态都被广泛高估，而是在少数关键状态胜出。

这也解释了为什么该任务不稳定：

> 灌溉收益是时机敏感的，错误时机灌溉没有收益甚至扣成本；DQN 必须学到少数关键 DAP 的动作价值，而不是简单地认为“灌溉总是好”。

## 当前结论

012_12 把问题进一步定位到 Q 值排序层面：

> 失败 seed 在训练中见过灌溉，但最终 Q 网络没有稳定地把关键窗口的灌溉动作估成最高价值。

因此，当前问题不是：

- DSSAT/gym 数据传输错误；
- reward 完全错；
- 模型从未探索灌溉；
- wrapper 把灌溉全部拦掉。

更准确的问题是：

> 当前 DQN 在这个稀疏、时机敏感的水氮管理任务中，Q 值学习不够稳定；少数关键灌溉动作的长期收益没有被稳定传播到最终策略。

## 下一步建议

现在可以停止继续单纯加 seed 或加步数。

如果继续算法线，建议优先尝试 DQN 稳定性改进，而不是继续调 reward：

1. Double DQN：
   - 减少 Q 值过估/错估；
   - 适合当前“动作排序不稳定”的问题。

2. Dueling DQN：
   - 分离状态价值 V(s) 和动作优势 A(s,a)；
   - 可能更适合多数时间不操作、少数关键时间操作的作物管理任务。

3. Prioritized replay：
   - 让高 TD-error 的关键灌溉转移更频繁被学习；
   - 可能有助于稀疏关键动作。

4. n-step return：
   - 改善灌溉动作到后期产量收益之间的信用分配。

## 给导师汇报的简洁表述

可以这样说：

> 我们进一步检查了 DQN 的 Q 值排序。失败 seed 训练中并非没有尝试灌溉，但最终 Q 网络没有把关键窗口的灌溉动作估为最高价值；成功 seed 则在部分关键窗口把灌溉动作排到最高并获得接近 fixed I60/N0 的结果。这说明问题集中在 DQN 的值函数学习稳定性和长期收益信用分配上，下一步应考虑 Double/Dueling/Prioritized replay 或 n-step return，而不是继续盲目调 reward 或堆训练步数。

