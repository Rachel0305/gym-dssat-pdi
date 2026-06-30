# 012_19 HLA2015 文献式 DQN warm-start / imitation 探针记录

## 实验目的

012_15–012_18 表明：

- 文献式 DQN 能学会灌溉，但容易把氮打满；
- reward 离线复评分并不偏好 N150；
- Q-value 诊断显示问题来自含氮动作的 Q 高估；
- n-step return 改善灌溉排序，但没有解决施氮高估。

本轮测试 warm-start / imitation：先用一个 I60/N0 启发式轨迹给 Q 网络一个合理初始排序，再进行正常 DQN 训练，观察策略是否仍然漂回 N150。

## 实验设置

与 012_17 基本一致：

- HLA2015；
- IC=1；
- 文献式 terminal reward；
- 25 个水氮离散动作；
- I120/N150 总预算；
- 操作窗口和 7 天间隔；
- `n_steps=5`；
- seed0；
- 5000 timesteps。

新增：

- warm-start 数据集：8 条启发式 I60/N0 轨迹；
- warm-start 目标动作：
  - DAP 45, 52, 73, 80, 87 附近选择 `I12_N0`；
  - 其他日期 no-op；
  - 目标总量约 I60/N0；
- 使用监督式 cross entropy 预训练 Q 网络 8 epochs；
- 然后进入正常 DQN 训练。

## 输入与脚本

- prompt：

```text
prompts/012_19_hla2015_literature_dqn_warmstart_probe.md
```

- 脚本：

```text
src/run_hla2015_literature_dqn_warmstart_012_19.py
```

- 运行环境：

```text
Docker: b2fd6726c8c1
Python: /opt/gym_dssat_pdi/bin/python
```

## smoke test

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_literature_dqn_warmstart_012_19.py --year 2015 --timesteps 200 --seed 0 --n-steps 5 --warmstart-epochs 2 --label smoke_literature_warmstart"
```

结果：

- warm-start 数据集：1264 条；
- smoke 正常完成；
- 没有 OOM 或报错。

## 5K seed0 正式结果

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_literature_dqn_warmstart_012_19.py --year 2015 --timesteps 5000 --seed 0 --n-steps 5 --warmstart-epochs 8 --label literature_warmstart"
```

结果目录：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_warmstart_012_19/2015/literature_warmstart_nstep5_ws8_seed0_5000steps
```

## 主要结果

| 指标 | 012_17 n-step | 012_19 n-step + warm-start |
|---|---:|---:|
| 产量 kg/ha | 7652.31 | 7653.05 |
| 生物量 kg/ha | 19060.52 | 19076.28 |
| 总灌溉 mm | 96 | 66 |
| 总施氮 kg/ha | 150 | 0 |
| 累计 reward | 984.96 | 1136.58 |
| 投入成本项 | 224.10 | 72.60 |
| 最大水分胁迫 | 0.000 | 0.000 |
| 最大氮胁迫 | 0.0145 | 0.0800 |

## 012_19 管理动作

| DAP | 灌溉 mm | 施氮 kg/ha | 说明 |
|---:|---:|---:|---|
| 21 | 6 | 0 | 早期小灌溉 |
| 28 | 6 | 0 | 早期小灌溉 |
| 35 | 6 | 0 | 早期小灌溉 |
| 76 | 12 | 0 | 中后期灌溉 |
| 83 | 12 | 0 | 中后期灌溉 |
| 90 | 24 | 0 | 中后期灌溉 |

总计：

```text
I66 / N0
```

## 解释

这是目前算法线中第一个比较清楚的正结果：

1. **warm-start 阻止了策略漂回 N150**  
   012_17 在相同 reward 和 n-step 设置下仍然 N150；012_19 加入 I60/N0 初始排序后，最终为 N0。

2. **产量没有下降**  
   012_19 产量 7653.05 kg/ha，与 012_17 的 7652.31 kg/ha 基本一致，甚至略高。

3. **reward 明显提高**  
   因为省掉了 150 kg/ha 氮肥，累计 reward 从 984.96 提高到 1136.58。

4. **这支持之前的诊断**  
   之前离线复评分认为 I60/N0 更优；012_19 说明如果 Q 网络初始排序合理，DQN 确实可以收敛到接近这一类少氮适量灌溉策略。

## 结论

> HLA2015 中，DQN 使用 N150 并不是环境或 reward 必然要求，而是 Q 初始排序/在线学习过程容易把高氮动作价值估高。warm-start 可以显著改善这一点，使策略变为 I66/N0，并保持同等产量和更高 reward。

## 局限

1. 当前只跑了 seed0；
2. warm-start 轨迹是启发式 I60/N0，不是严格最优调度；
3. 这还不能证明算法稳定，只能证明“合理初始排序有帮助”；
4. 下一步必须做 seed1，否则不能说方法稳定。

## 下一步建议

建议继续做：

```text
012_20 HLA2015 literature DQN n-step + warm-start seed1 5K
```

判断：

- 如果 seed1 也得到少氮、适量灌溉、高产，则 warm-start 线值得继续；
- 如果 seed1 又漂回 N150，则说明 warm-start 对 seed 敏感，需要更系统的 imitation / replay 设计。
