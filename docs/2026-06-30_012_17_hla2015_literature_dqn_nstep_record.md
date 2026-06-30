# 012_17 HLA2015 文献对齐版 DQN n-step return 探针记录

## 实验目的

012_15 使用文献式终端产量奖励后，DQN seed0 能学到灌溉并增产，但仍然把施氮打满 N150。

012_16 的 Q-value 诊断显示，reward 离线复评分并不偏好 N150，问题更像是 Q 网络对含氮动作长期价值估计偏高。

本轮尝试最小算法改动：在 012_15 基础上只把 DQN 的 `n_steps` 从 1 改为 5，测试多步回报是否能改善终端产量奖励的信用分配。

## 单变量设置

与 012_15 相比，本轮只改：

```text
n_steps = 1 -> 5
```

保持不变：

- HLA2015；
- IC=1；
- 文献式 reward：

```text
非终止步：R_t = -0.79 * N_t - 1.10 * W_t
终止步：R_T = 0.158 * GRNWT_final - 0.79 * N_T - 1.10 * W_T
```

- 25 个水氮离散动作；
- I120/N150 总预算；
- 灌溉窗口、施氮窗口、7 天最小操作间隔；
- seed0；
- 5000 timesteps；
- 3 层 256 网络；
- learning_rate=1e-5；
- batch_size=1024。

## 输入和脚本

- prompt：

```text
prompts/012_17_hla2015_literature_dqn_nstep_probe.md
```

- 脚本：

```text
src/run_hla2015_literature_dqn_nstep_012_17.py
```

- 运行环境：

```text
Docker: b2fd6726c8c1
Python: /opt/gym_dssat_pdi/bin/python
```

## smoke test

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_literature_dqn_nstep_012_17.py --year 2015 --timesteps 200 --seed 0 --n-steps 5 --label smoke_literature_nstep"
```

结果目录：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_nstep_012_17/2015/smoke_literature_nstep_nstep5_seed0_200steps
```

smoke test 正常完成。

## 5K seed0 结果

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_literature_dqn_nstep_012_17.py --year 2015 --timesteps 5000 --seed 0 --n-steps 5 --label literature_nstep"
```

结果目录：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_nstep_012_17/2015/literature_nstep_nstep5_seed0_5000steps
```

| 指标 | 012_15 DQN n=1 | 012_17 DQN n=5 |
|---|---:|---:|
| 产量 kg/ha | 7651.86 | 7652.31 |
| 生物量 kg/ha | 19050.05 | 19060.52 |
| 总灌溉 mm | 102 | 96 |
| 总施氮 kg/ha | 150 | 150 |
| 累计 reward | 978.29 | 984.96 |
| 投入成本项 | 230.70 | 224.10 |
| 最大水分胁迫 | 0.000 | 0.000 |
| 最大氮胁迫 | 0.0145 | 0.0145 |

## 012_17 管理动作

| DAP | 灌溉 mm | 施氮 kg/ha | 说明 |
|---:|---:|---:|---|
| 21 | 6 | 0 | 早期灌溉 |
| 28 | 6 | 0 | 早期灌溉 |
| 35 | 6 | 0 | 早期灌溉 |
| 53 | 6 | 0 | 灌溉 |
| 60 | 6 | 40 | 灌溉 + 施氮 |
| 67 | 0 | 110 | 剩余氮预算几乎打满 |
| 74 | 24 | 0 | 灌溉 |
| 81 | 12 | 0 | 灌溉 |
| 88 | 12 | 0 | 灌溉 |
| 95 | 18 | 0 | 灌溉 |

## 判断

### 有改善的地方

1. n-step return 没有导致崩溃或 OOM；
2. 相比 012_15，灌溉从 102 mm 降到 96 mm；
3. 产量基本不变；
4. reward 从 978.29 提高到 984.96。

### 没解决的地方

最关键的问题没有解决：

```text
施氮仍然是 N150。
```

而 012_08/012_15 复评分已经显示，当前 reward 下 I60/N0 更优。因此 n-step return 虽然让灌溉略微节省，但仍然没有纠正高氮动作的 Q 排序。

## 结论

012_17 说明：

> 只把 DQN 从 1-step return 改成 5-step return，不足以解决 HLA2015 mixed water-nitrogen task 中的氮动作高估问题。

它有轻微正向效果，但不足以让策略接近 I60/N0。

## 下一步建议

不建议继续盲目扩大 n_steps 或继续堆 seed。更有信息量的下一步是：

1. 对 012_17 再做一次 Q-value 诊断，确认 N150 是否仍然由含氮动作 Q 高估导致；
2. 如果确认，则可以关闭“n-step 单独解决问题”这条线；
3. 后续更值得尝试的是：
   - prioritized replay；
   - warm-start / imitation；
   - 把固定扫描中 I60/N0 这类高分轨迹用于预训练或监督初始化。
