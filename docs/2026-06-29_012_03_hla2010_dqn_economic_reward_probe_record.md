# 012_03 HLA2010 DQN 经济收益型 reward 初步试验记录

## 目的

012_02 已经确认：在固定 DQN 灌溉动作下，N0 到 N150 几乎没有产量差异。因此，DQN 在 012_01 中用满 N150 并不是因为 N150 有明显产量收益，而更可能是当前 reward/目标函数没有惩罚“无产量收益的施氮”。

本轮尝试经济收益型 reward：

```text
R_t = ΔGRNWT_t - water_cost × I_t - nitrogen_cost × N_t
```

其中：

- `ΔGRNWT_t`：当前步籽粒重正增量；
- `I_t`：当前步实际灌溉量；
- `N_t`：当前步实际施氮量；
- `water_cost = 1.0`；
- `nitrogen_cost = 5.0`。

成本单位暂时使用“籽粒产量等价量”，用于算法诊断，不代表最终经济参数。

## 动作空间

| action | 含义 |
|---:|---|
| 0 | 不操作 |
| 1 | 灌溉 30 mm |
| 2 | 施氮 50 kg N/ha |
| 3 | 灌溉 30 mm + 施氮 50 kg N/ha |

## 约束

- 灌溉预算：I120
- 施氮预算：N150
- 灌溉窗口：DAP 20-35、45-65、70-95
- 施氮窗口：DAP 25-40、55-70
- 最小操作间隔：7 天

## 脚本

`src/run_hla2010_dqn_economic_reward_probe_012_03.py`

## 执行命令

smoke：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2010 --timesteps 200 --seed 0 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

5K：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2010 --timesteps 5000 --seed 0 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

## 输出

输出目录：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03`

5K case：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/2010/medium_N_cost_seed0_5000steps`

对比汇总：

`DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/hla2010_dqn_economic_compare_summary.csv`

## 结果

| 方法 | reward/目标 | 产量 | 灌溉 | 施氮 | 最大水分胁迫 | 最大氮胁迫 | 说明 |
|---|---|---:|---:|---:|---:|---:|---|
| windowed PPO 5K | official reward scalar | 约 7854 kg/ha | 120 mm | 150 kg N/ha | 约 0.423 | 约 0.0158 | 用满水氮 |
| DQN 012_01 5K | official reward scalar | 7373 kg/ha | 30 mm | 150 kg N/ha | 0.915 | 0.0158 | 少用水，但仍用满氮 |
| DQN 012_03 200 steps | economic reward | 7532 kg/ha | 30 mm | 50 kg N/ha | 0.905 | 0.0586 | smoke 阶段已开始少施氮 |
| DQN 012_03 5K | economic reward | 7854 kg/ha | 120 mm | 50 kg N/ha | 0.416 | 0.0158 | 产量达到 PPO 水平，但只用 N50 |

## 012_03 5K 有效管理事件

经济 reward 下，DQN 最终使用：

- 灌溉：120 mm
- 施氮：50 kg N/ha
- 产量：7854 kg/ha

主要有效事件：

- DAP56：施氮 50 kg N/ha
- DAP63、DAP71、DAP78、DAP85 附近：灌溉累计达到 120 mm

说明：评估过程中 DQN 后期仍会输出一些原始施氮动作，但由于预算和窗口安全层限制，实际 `safe_anfer` 为 0；真正进入 DSSAT 的施氮总量只有 50 kg N/ha。

## 关键解释

这次结果非常重要：

1. 经济 reward 后，DQN 不再用满 N150，而是只用 N50；
2. 在只用 N50 的情况下，产量仍达到约 7854 kg/ha，基本等于 windowed PPO；
3. 这与 012_02 的 N 扫描一致：HLA2010 当前情景下，额外 N100/N150 几乎没有产量收益；
4. 因此，原来 PPO/DQN 用满氮，不是作物产量真正需要，而是目标函数没有正确惩罚无效施氮。

## 当前结论

经济收益型 reward 是目前最有希望的方向。

它解决了一个核心问题：

> 算法开始区分“降低氮胁迫指标”和“真正提高产量/收益”。

在 HLA2010 seed0 上，DQN + economic reward 实现了：

- 与 windowed PPO 接近/相同的产量；
- 更低的施氮量；
- 水分胁迫控制接近 DSSAT auto/windowed PPO；
- 氮肥使用效率显著提高。

## 结论边界

当前还不能说：

> DQN economic reward 已经稳定优于 PPO。

因为目前只有：

- HLA2010；
- seed0；
- 一个成本组合：water_cost=1.0，nitrogen_cost=5.0。

还需要继续检查：

- HLA2010 seed1 是否稳定；
- HLA2015 是否也能少施氮；
- 不同氮成本是否改变策略；
- 该经济 reward 是否有真实价格依据。

## 下一步建议

建议下一步做 012_04：

先不扩多年，先做 HLA2010 的 seed1。

如果 seed1 也能维持：

- 产量接近 7854；
- 施氮明显低于 N150；
- 管理时点不病态；

再扩展到 HLA2015。

不建议现在立刻做大量价格敏感性，因为当前首要问题是确认该结果不是 seed0 偶然。
