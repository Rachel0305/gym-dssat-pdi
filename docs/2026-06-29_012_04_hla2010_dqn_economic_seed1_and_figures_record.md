# 012_04 HLA2010 经济 reward DQN seed1 稳定性验证与四情景图

## 目的

012_03 中，HLA2010 economic reward DQN seed0 达到：

- 产量约 7854 kg/ha；
- 灌溉 120 mm；
- 施氮 50 kg N/ha；
- 相比 windowed PPO 用氮从 150 降到 50，但产量基本不变。

本轮验证：

> 这个“高产 + 少氮”的结果是否在 seed1 下也成立？

同时生成四情景过程图：

- null
- 专家策略平移
- DSSAT auto irrigation + auto-N attempt
- economic DQN

以及 economic DQN seed0/seed1 稳定性图。

## 训练设置

算法：DQN  
年份：HLA2010  
训练步数：5000  
reward：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

动作空间：

| action | 含义 |
|---:|---|
| 0 | 不操作 |
| 1 | 灌溉 30 mm |
| 2 | 施氮 50 kg N/ha |
| 3 | 灌溉 30 mm + 施氮 50 kg N/ha |

约束：

- I ≤ 120 mm
- N ≤ 150 kg N/ha
- 灌溉窗口：DAP 20-35、45-65、70-95
- 施氮窗口：DAP 25-40、55-70
- 最小操作间隔：7 天

## 执行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2010 --timesteps 5000 --seed 1 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

画图：

```bash
python src/plot_hla2010_dqn_economic_012_04.py
```

## 输出

seed1 case：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/2010/medium_N_cost_seed1_5000steps`

图和汇总：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_04`

主要图：

- `hla2010_four_scenario_with_economic_dqn_seed0_process.png`
- `hla2010_economic_dqn_seed0_seed1_process.png`

主要表：

- `hla2010_four_scenario_with_economic_dqn_seed0_daily.csv`
- `hla2010_economic_dqn_seed0_seed1_daily.csv`
- `hla2010_economic_dqn_four_scenario_and_seed_summary.csv`

## seed0 与 seed1 结果

| seed | 产量 | 灌溉 | 施氮 | 最大水分胁迫 | 最大氮胁迫 | economic reward |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7854 kg/ha | 120 mm | 50 kg N/ha | 0.416 | 0.0158 | 7483.7 |
| 1 | 7854 kg/ha | 120 mm | 100 kg N/ha | 0.416 | 0.0158 | 7233.7 |

## 管理事件

seed0：

- DAP56：施氮 50 kg N/ha
- DAP63、71、78、85：灌溉，累计 120 mm

seed1：

- DAP35、46、53、93：灌溉，累计 120 mm
- DAP60、67：施氮，累计 100 kg N/ha

## 解释

seed1 也达到与 seed0 相同的产量和胁迫控制水平，说明“economic reward DQN 可以在 HLA2010 上达到高产”不是 seed0 偶然。

但 seed0 和 seed1 的施氮量不同：

- seed0 用 N50；
- seed1 用 N100；
- 二者产量相同。

这说明：

1. “少于 N150 仍能高产”这个结论在两个 seed 中成立；
2. 但具体最省氮策略还没有完全稳定；
3. 在当前 reward 系数下，seed0 的 economic reward 更高，因为它产量相同但少用 50 kg N/ha。

## 四情景解释

HLA2010 中，economic DQN seed0 与 DSSAT auto / windowed PPO 达到接近相同的产量，但只用 50 kg N/ha。

这比之前的 windowed PPO 更有解释力：

- windowed PPO：高产，但 N150；
- economic DQN seed0：同样高产，但 N50。

## 当前结论

经济收益型 reward + DQN 离散动作，是目前最值得继续推进的方向。

它初步解决了此前最关键的问题：

> 算法不再因为降低氮胁迫指标而无条件用满 N150。

## 结论边界

当前仍不能说该方法已经最终成功，因为：

- 只验证了 HLA2010；
- 只验证了两个 seed；
- water_cost=1.0、nitrogen_cost=5.0 仍是诊断性折算系数；
- 还没有验证 HLA2015；
- 还没有做价格/成本敏感性。

## 下一步建议

下一步可以推进到 HLA2015：

- 先跑 economic DQN HLA2015 seed0 5K；
- 如果仍能高产且少氮，再跑 seed1；
- 如果 HLA2015 失败，则回头做成本敏感性或年份差异诊断。

不建议现在直接大规模多年训练。
