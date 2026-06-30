# 012_01 HLA2010 DQN 离散动作初步试验记录

## 目的

上一阶段 PPO 在加入预算和生育期操作窗口后，已经不再出现 DAP2 直接打满水氮的问题，但仍然倾向于在窗口内用满 I120/N150，尤其氮肥使用效率不理想。

本轮不再继续调 PPO 参数，而是新开一条 DQN 离散动作试验线，检查：

> 如果把连续动作 PPO 换成离散动作 DQN，策略是否还会用满水氮预算？

## 试验设计

只做 HLA2010 seed0，先做低成本 probe，不扩展多年和多 seed。

保持不变：

- HLA2010 corrected cultivar
- IC=1 输入基础
- PDI/gym-DSSAT 4.8.0 容器环境
- official gym-DSSAT reward，按当前方式转成 scalar reward
- 水氮预算：I120/N150
- 操作窗口：
  - 灌溉：DAP 20-35、45-65、70-95
  - 施氮：DAP 25-40、55-70
- 最小操作间隔：7 天

改变：

- PPO 连续动作改为 DQN 离散动作。

## 离散动作表

| action | 含义 |
|---:|---|
| 0 | 不操作 |
| 1 | 灌溉 30 mm |
| 2 | 施氮 50 kg N/ha |
| 3 | 灌溉 30 mm + 施氮 50 kg N/ha |

## 脚本

`src/run_hla2010_dqn_discrete_action_probe_012_01.py`

## 执行命令

smoke test：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_discrete_action_probe_012_01.py --year 2010 --timesteps 200 --seed 0"
```

5K probe：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_discrete_action_probe_012_01.py --year 2010 --timesteps 5000 --seed 0"
```

## 输出位置

smoke：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_discrete_action_probe_012_01/2010/dqn_discrete_seed0_200steps`

5K：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_discrete_action_probe_012_01/2010/dqn_discrete_seed0_5000steps`

## 5K 结果

| 指标 | DQN 5K seed0 |
|---|---:|
| 产量 | 7373 kg/ha |
| 生物量 | 20160 kg/ha |
| 灌溉总量 | 30 mm |
| 施氮总量 | 150 kg N/ha |
| 灌溉事件 | 1 次 |
| 施肥事件 | 3 次 |
| 最大水分胁迫 | 0.915 |
| 最大氮胁迫 | 0.0158 |
| eval 中 action=0 次数 | 63 |
| eval 中有效安全动作次数 | 3 |

关键动作：

- DAP30：灌溉 30 mm + 施氮 50 kg N/ha
- DAP56：施氮 50 kg N/ha
- DAP63：施氮 50 kg N/ha

## 与 windowed PPO 的初步比较

| 方法 | 产量 | 灌溉 | 施氮 | 解释 |
|---|---:|---:|---:|---|
| windowed PPO seed0 | 7854 kg/ha | 120 mm | 150 kg N/ha | 用满水氮，产量高，水分胁迫控制好 |
| DQN discrete seed0 | 7373 kg/ha | 30 mm | 150 kg N/ha | 明显少用水，但仍用满氮，产量低于 PPO |

## 解释

DQN 离散动作确实改变了水分管理行为：它没有像 PPO 一样用满 I120，而是只灌了 30 mm。这说明“连续动作 PPO 容易在窗口内用满灌溉预算”的问题，可能与动作空间/算法结构有关。

但 DQN 仍然用满 N150，说明“氮肥用满”不是简单换成 DQN 就能解决的。这个问题可能来自 reward、氮边际收益、状态信息或当前情景本身。

当前 DQN 5K 只是 probe，不应直接作为最终算法结论。它的价值是证明：

1. DQN 路线可以跑通；
2. 离散动作能明显改变水分使用行为；
3. 氮素过量使用问题仍然存在，需要单独诊断。

## 下一步建议

不要马上扩展 2015 或多 seed。建议先做一个低成本确定性对照：

- 固定 DQN 的灌溉动作，只改变施氮量 0/50/100/150；
- 判断 HLA2010 在当前情景下 N150 是否真的有边际收益；
- 如果 N150 增产明显，说明 DQN 施氮有合理性；
- 如果 N0/N50 与 N150 产量接近，说明 reward/算法仍然没有学会省氮。
