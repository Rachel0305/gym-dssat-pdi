# 012_20 HLA2015 文献式 DQN warm-start seed1 稳定性验证记录

## 实验目的

012_19 中，HLA2015 seed0 使用：

- 文献式 terminal reward；
- 25 个水氮离散动作；
- `n_steps=5`；
- I60/N0 启发式 warm-start；

得到了 I66/N0、高产、高 reward 的结果。

本轮只把 seed0 改为 seed1，检查 warm-start 是否稳定。

## 单变量设置

与 012_19 完全一致，唯一变化：

```text
seed: 0 -> 1
```

其他保持：

- HLA2015；
- IC=1；
- 文献式 terminal reward；
- 25 个水氮离散动作；
- I120/N150 总预算；
- 操作窗口和 7 天间隔；
- `n_steps=5`；
- warm-start epochs=8；
- 5000 timesteps。

## 执行

prompt：

```text
prompts/012_20_hla2015_literature_dqn_warmstart_seed1.md
```

脚本：

```text
src/run_hla2015_literature_dqn_warmstart_012_19.py
```

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_literature_dqn_warmstart_012_19.py --year 2015 --timesteps 5000 --seed 1 --n-steps 5 --warmstart-epochs 8 --label literature_warmstart"
```

结果目录：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_warmstart_012_19/2015/literature_warmstart_nstep5_ws8_seed1_5000steps
```

## 主要结果

| 指标 | 012_19 seed0 | 012_20 seed1 |
|---|---:|---:|
| 产量 kg/ha | 7653.05 | 7653.19 |
| 生物量 kg/ha | 19076.28 | 19081.28 |
| 总灌溉 mm | 66 | 48 |
| 总施氮 kg/ha | 0 | 150 |
| 累计 reward | 1136.58 | 1037.90 |
| 投入成本项 | 72.60 | 171.30 |
| 最大水分胁迫 | 0.000 | 0.000 |
| 最大氮胁迫 | 0.0800 | 0.0145 |

## seed1 管理动作

| DAP | 灌溉 mm | 施氮 kg/ha | 说明 |
|---:|---:|---:|---|
| 56 | 0 | 120 | 大量施氮 |
| 63 | 0 | 30 | 补满 N150 |
| 85 | 24 | 0 | 灌溉 |
| 92 | 24 | 0 | 灌溉 |

总计：

```text
I48 / N150
```

## 判断

012_20 没有通过稳定性验证。

虽然 seed1 的产量仍然高，但是它又回到了 N150。相比 seed0：

- 产量几乎一样；
- 但 seed1 多用了 150 kg/ha 氮；
- reward 低了约 98.7；
- 管理策略明显不是 012_19 想要的少氮路线。

## 解释

这说明当前 warm-start 只证明了：

> 合理初始排序可以在某些 seed 下阻止 DQN 漂回高氮动作。

但它还没有证明：

> 这个方法能稳定跨 seed 解决高氮动作 Q-value 高估。

seed1 的结果说明，在线 DQN 训练仍然可能覆盖 warm-start 初始化，并重新把高氮动作估高。

## 结论

当前算法线状态：

1. 普通文献式 DQN：能学灌溉，但 N150；
2. n-step：改善灌溉，但 N150；
3. warm-start seed0：成功变成 I66/N0；
4. warm-start seed1：失败，回到 I48/N150。

因此：

> warm-start 是一个有效线索，但当前实现还不是稳定算法方案。

## 下一步建议

不建议继续盲目加 seed。下一步更有信息量的是：

1. 对 012_20 seed1 做 Q-value 诊断，确认 warm-start 后为什么仍然漂回高氮；
2. 或者直接进入更系统的 imitation / replay 方案，而不是只做一次 supervised warm-start；
3. 如果要省算力用于汇报，可以把当前算法线总结为：

```text
DQN 在文献式奖励下能识别灌溉增产价值；
少氮策略在离线复评分和 seed0 warm-start 中成立；
但当前在线 DQN 训练仍存在 seed 敏感性，高氮动作 Q-value 高估没有稳定解决。
```
