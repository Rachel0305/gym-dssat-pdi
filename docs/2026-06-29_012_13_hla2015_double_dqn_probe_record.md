# 012_13 HLA2015 Double DQN 最小算法改进验证记录

## 目的

012_12 的 Q 值诊断显示，HLA2015 DQN 失败 seed 的核心问题是：

> 最终 Q 网络没有稳定地把关键窗口内的灌溉动作估成最高价值。

因此本轮尝试一个最小算法改进：Double DQN。

本实验只改 TD target，不改 reward、不改窗口、不改动作空间、不改训练步数、不改 site-packages。

## Double DQN 修改内容

普通 DQN target：

```text
target = r + gamma * max_a Q_target(s', a)
```

Double DQN target：

```text
a* = argmax_a Q_online(s', a)
target = r + gamma * Q_target(s', a*)
```

也就是：

- 当前 q_net 负责选择下一步动作；
- target_q_net 负责评估这个动作；
- 目标是减少 Q 值过估/错估造成的动作排序不稳定。

## 实现方式

新增项目内子类：

```text
src/custom_double_dqn.py
```

不修改：

```text
site-packages/stable_baselines3/
```

运行脚本：

```text
src/run_hla2015_double_dqn_economic_012_13.py
```

Q值 probe：

```text
src/run_hla2015_double_dqn_q_probe_012_13.py
```

绘图脚本：

```text
src/plot_hla2015_double_dqn_012_13.py
```

## 实验设置

- 年份：HLA2015
- 算法：CustomDoubleDQN
- seed：0
- 训练步数：5000
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

- 其他 DQN 参数与 012_03/012_07/012_09 保持一致。

## 运行命令

smoke test：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_double_dqn_economic_012_13.py --year 2015 --timesteps 200 --seed 0 --water-cost 1.0 --nitrogen-cost 5.0 --label smoke_double_dqn"
```

正式 5K：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_double_dqn_economic_012_13.py --year 2015 --timesteps 5000 --seed 0 --water-cost 1.0 --nitrogen-cost 5.0 --label double_dqn_medium_N_cost"
```

Q值 probe：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_double_dqn_q_probe_012_13.py"
```

绘图：

```bash
python src/plot_hla2015_double_dqn_012_13.py
```

## 输出文件

- 正式运行目录：

```text
DSSAT_auto_validation/HLA_2004/hla2015_double_dqn_economic_012_13/2015/double_dqn_medium_N_cost_seed0_5000steps/
```

- 汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_double_dqn_economic_012_13/summary/hla2015_double_dqn_seed0_5k_summary.csv
```

- 汇总图：

```text
DSSAT_auto_validation/HLA_2004/hla2015_double_dqn_economic_012_13/figures/hla2015_double_dqn_seed0_5k_summary.png
```

- Double DQN Q值 probe：

```text
DSSAT_auto_validation/HLA_2004/hla2015_double_dqn_economic_012_13/q_probe/double_dqn_seed0_5k_q_summary.csv
```

## 结果

| 情景 | 算法 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | economic reward |
|---|---|---:|---:|---:|---:|
| DQN seed0 5K | DQN | 6486 | 0 | 0 | 6485.77 |
| DQN seed1 5K | DQN | 7632 | 60 | 0 | 7572.28 |
| DQN seed0 20K | DQN | 7321 | 30 | 50 | 7040.59 |
| DoubleDQN seed0 5K | CustomDoubleDQN | 6486 | 0 | 0 | 6485.77 |
| fixed I60/N0 | fixed | 7625 | 60 | 0 | 7564.80 |

Double DQN seed0 5K 有效动作：

```text
无有效灌溉；
无有效施氮。
```

## Q值 probe 结果

| 指标 | DoubleDQN seed0 5K |
|---|---:|
| 灌溉窗口内 action1/3 为最高 Q 的比例 | 0.000 |
| 平均 Q(action1)-Q(action0) | -8.575 |
| 最终灌溉 | 0 mm |
| 最终施氮 | 0 kg/ha |
| 最终产量 | 6485.77 kg/ha |
| 最终 reward | 6485.77 |

## 关键结论

Double DQN 单独没有救回 HLA2015 seed0 5K。

它的表现与普通 DQN seed0 5K 基本相同：

- 都不灌溉；
- 都不施氮；
- 都等同 null；
- Q值排序中，灌溉动作仍然没有在灌溉窗口中成为最高 Q。

## 解释

这说明 HLA2015 的失败可能不是普通 DQN 中最典型的“max target 过估”问题。

Double DQN 只改变 target 的动作选择/评估分离，但没有改变：

- 状态价值和动作优势的表示方式；
- 关键少数动作的采样权重；
- 长期回报传播长度；
- 多数时间 no-op、少数时间操作的结构性不平衡。

因此，Double DQN 单独不足并不意外。

## 当前判断

012_13 是一个负结果，但有用：

> 单独把 DQN target 改成 Double DQN，不能解决 HLA2015 seed0 的关键灌溉动作 Q 值排序问题。

因此下一步不建议继续只围绕 Double DQN 加 seed 或加步数。

## 下一步建议

更有针对性的下一个单变量改进是 Dueling DQN。

理由：

- 作物管理任务多数天都是 no-op；
- 只有少数关键窗口动作真正重要；
- Dueling DQN 把状态价值 V(s) 和动作优势 A(s,a) 分开，可能更适合“当前状态整体好不好”和“此时是否值得灌溉”分离学习。

如果继续算法线，建议：

> 012_14 尝试 Dueling DQN seed0 5K，其他设置全部不变。

