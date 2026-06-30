# 012_14 HLA2015 Dueling DQN 最小算法改进验证记录

## 目的

012_13 中，Double DQN seed0 5K 没有救回 HLA2015，结果仍然等同 null。

本轮测试另一个单变量算法改进：Dueling DQN。

## Dueling DQN 的含义

普通 DQN 直接学习：

```text
Q(s,a)
```

Dueling DQN 将其拆成：

```text
Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

其中：

- `V(s)`：当前状态本身的价值；
- `A(s,a)`：某个动作相对于其他动作的优势。

直观理解：

> 先判断“这个状态整体有没有价值”，再判断“在这个状态下，某个动作是不是比其他动作更好”。

这理论上适合本项目这种多数时间不操作、少数关键窗口才操作的作物管理任务。

## 实现方式

新增项目内文件：

```text
src/custom_dueling_dqn.py
```

其中包含：

- `DuelingQNetwork`
- `CustomDuelingDQNPolicy`

不修改：

```text
site-packages/stable_baselines3/
```

运行脚本：

```text
src/run_hla2015_dueling_dqn_economic_012_14.py
```

绘图脚本：

```text
src/plot_hla2015_dueling_dqn_012_14.py
```

## 实验设置

- 年份：HLA2015
- 算法：DQN + CustomDuelingDQNPolicy
- seed：0
- 训练步数：5000
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

- 其他参数与 012_03 / 012_07 / 012_09 保持一致。

## 运行命令

smoke test：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_dueling_dqn_economic_012_14.py --year 2015 --timesteps 200 --seed 0 --water-cost 1.0 --nitrogen-cost 5.0 --label smoke_dueling_dqn"
```

正式 5K：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_dueling_dqn_economic_012_14.py --year 2015 --timesteps 5000 --seed 0 --water-cost 1.0 --nitrogen-cost 5.0 --label dueling_dqn_medium_N_cost"
```

绘图：

```bash
python src/plot_hla2015_dueling_dqn_012_14.py
```

## 调试记录

第一次 smoke test 报错：

```text
RuntimeError: index 3 is out of bounds for dimension 1 with size 1
```

原因是自定义 dueling Q 网络中错误使用了：

```python
get_action_dim(self.action_space)
```

对于 `Discrete(4)`，该函数返回动作维度 1，而不是动作数量 4。

已修复为：

```python
action_dim = int(self.action_space.n)
```

修复后 smoke test 通过。

## 输出文件

- 正式运行目录：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dueling_dqn_economic_012_14/2015/dueling_dqn_medium_N_cost_seed0_5000steps/
```

- 汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dueling_dqn_economic_012_14/summary/hla2015_dueling_dqn_seed0_5k_summary.csv
```

- 汇总图：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dueling_dqn_economic_012_14/figures/hla2015_dueling_dqn_seed0_5k_summary.png
```

## 结果

| 情景 | 算法 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | economic reward |
|---|---|---:|---:|---:|---:|
| DQN seed0 5K | DQN | 6486 | 0 | 0 | 6485.77 |
| DQN seed1 5K | DQN | 7632 | 60 | 0 | 7572.28 |
| DQN seed0 20K | DQN | 7321 | 30 | 50 | 7040.59 |
| DoubleDQN seed0 5K | CustomDoubleDQN | 6486 | 0 | 0 | 6485.77 |
| DuelingDQN seed0 5K | CustomDuelingDQN | 6526 | 0 | 150 | 5775.81 |
| fixed I60/N0 | fixed | 7625 | 60 | 0 | 7564.80 |

## Dueling DQN 有效动作

| DAP | 灌溉 mm | 施氮 kg/ha |
|---:|---:|---:|
| 56 | 0 | 50 |
| 63 | 0 | 50 |
| 70 | 0 | 50 |

总量：

- 灌溉：0 mm；
- 施氮：150 kg/ha。

## 关键结论

Dueling DQN 单独没有改善 HLA2015 seed0。

它不但没有学到灌溉，反而学成了：

> N150 / I0

这在当前 economic reward 和固定反事实下是不合理的，因为 HLA2015 的主要收益来自灌溉，而不是施氮。

## 与 Double DQN 对比

Double DQN：

- I0/N0；
- 等同 null；
- reward 6485.77。

Dueling DQN：

- I0/N150；
- 产量几乎仍接近 null；
- 因施氮成本，reward 降到 5775.81。

所以 Dueling 单独比 Double DQN 还差。

## 当前判断

012_13 和 012_14 连续两个单变量算法改进都没有解决问题：

- Double DQN 单独无效；
- Dueling DQN 单独无效，且偏向错误施氮。

这说明问题可能不是单一 DQN 组件能直接解决的，至少不能指望“换一个 DQN 变体”马上让结果稳定。

## 下一步建议

目前不建议立刻继续尝试第三个算法组件，例如 PER 或 n-step。

更稳妥的下一步是暂停算法扩展，整理当前证据链：

1. PPO mixed task 不稳定；
2. economic reward + DQN 能在 HLA2010 和 HLA2015 seed1 学到合理行为；
3. HLA2015 固定反事实证明 I60/N0 是合理方向；
4. DQN 5K seed 方差大；
5. 20K 有改善但不足；
6. action coverage 说明不是完全没探索；
7. Q值诊断说明是动作价值排序不稳定；
8. Double 和 Dueling 单独改进无效。

这条证据链已经足以向导师说明：

> 当前问题不是简单调 reward 或换 PPO/DQN 就能解决，而是作物水氮管理任务中稀疏、时机敏感、长期回报信用分配导致的 RL 稳定性问题。

如果继续算法线，下一步应该由导师确认是否值得进一步投入：

- PER；
- n-step return；
- Dueling + Double 组合；
- 阶段动作空间；
- imitation/warm-start；
- 或者把这部分定位为方法探索而非主线结果。

