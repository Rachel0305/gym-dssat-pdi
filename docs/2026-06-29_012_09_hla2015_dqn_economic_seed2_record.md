# 012_09 HLA2015 economic reward DQN seed2 稳定性验证记录

## 目的

012_07 中，HLA2015 economic DQN seed1 学到了接近 fixed I60/N0 的策略：

- 灌溉 60 mm；
- 施氮 0 kg/ha；
- 产量约 7632 kg/ha；
- economic reward 约 7572。

012_08 固定 N0 水量扫描确认，fixed I60/N0 是当前 economic reward 下的固定水量近优/最优方案。

本轮继续运行 HLA2015 economic DQN seed2，判断 seed1 的成功是否可重复。

## 运行设置

- 年份：HLA2015
- 算法：DQN
- seed：2
- 训练步数：5000
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

- 灌溉预算：120 mm
- 追加氮预算：150 kg/ha
- 灌溉窗口：DAP 20–35、45–65、70–95
- 施氮窗口：DAP 25–40、55–70

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2015 --timesteps 5000 --seed 2 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

## 输出文件

- seed2 输出目录：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/2015/medium_N_cost_seed2_5000steps/
```

- 三 seed 对比图：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_07_hla2015_seed_comparison/hla2015_dqn_seed0_seed1_process.png
```

- 三 seed 与固定 N0 水量扫描最终汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_09_hla2015_three_seed_final/hla2015_dqn_three_seed_vs_fixed_n0_water_scan_summary.csv
```

## 结果

| 来源 | 情景 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | economic reward | 最大水分胁迫 | 最大氮胁迫 |
|---|---|---:|---:|---:|---:|---:|---:|
| DQN | seed0 | 6486 | 0 | 0 | 6485.77 | 1.000 | 0.093 |
| DQN | seed1 | 7632 | 60 | 0 | 7572.28 | 0.161 | 0.099 |
| DQN | seed2 | 6523 | 0 | 50 | 6272.87 | 1.000 | 0.020 |
| fixed N0 scan | I0/N0 | 6486 | 0 | 0 | 6485.77 | 1.000 | 0.093 |
| fixed N0 scan | I30/N0 | 7017 | 30 | 0 | 6987.07 | 0.896 | 0.115 |
| fixed N0 scan | I60/N0 | 7625 | 60 | 0 | 7564.80 | 0.195 | 0.101 |
| fixed N0 scan | I90/N0 | 7645 | 90 | 0 | 7555.12 | 0.000 | 0.057 |
| fixed N0 scan | I120/N0 | 7645 | 120 | 0 | 7525.12 | 0.000 | 0.059 |

## seed2 有效动作

seed2 只执行了一次施氮：

| DAP | 灌溉 mm | 施氮 kg/ha |
|---:|---:|---:|
| 27 | 0 | 50 |

总量：

- 灌溉：0 mm；
- 施氮：50 kg/ha。

## 关键观察

1. seed2 没有学到灌溉。
   - 最大水分胁迫仍为 1.0；
   - 产量只有 6523 kg/ha，接近 null；
   - 与 fixed I60/N0 相比，少了约 1100 kg/ha。

2. seed2 学到的施氮动作是经济上不合理的。
   - 012_08 显示 HLA2015 的主要收益来自灌溉，而不是施氮；
   - seed2 施了 50 kg/ha 氮，但没有明显增产；
   - economic reward 低于 null。

3. HLA2015 的 DQN 5K seed 稳定性没有通过。
   - seed0：失败，null；
   - seed1：成功，接近 fixed I60/N0；
   - seed2：失败，施氮但不灌溉。

## 当前结论

HLA2015 的结论应当分成两层：

### 固定反事实层面

HLA2015 在当前 economic reward 下的合理管理方向很清楚：

> 少量灌溉、零施氮。固定扫描中 I60/N0 的 economic reward 最高。

### DQN 学习层面

DQN 具有学到该方向的能力，但 5K 训练不稳定：

> 3 个 seed 中只有 seed1 学到了接近固定最优的 I60/N0，seed0 和 seed2 均失败。

## 不能得出的结论

当前不能说：

- DQN economic reward 已经稳定解决 HLA2015；
- DQN 已经跨 seed 稳定优于专家或 DSSAT auto；
- 5K 训练已经足够。

## 可以得出的结论

当前可以说：

- economic reward 的目标方向在 HLA2015 上是可解释的；
- 固定反事实显示最优方向近似 I60/N0；
- DQN seed1 能学到这个方向；
- 但 DQN 5K 的 seed 方差很大，稳定性不足。

## 下一步建议

不建议继续加 seed 到 seed3、seed4，因为目前已经能判断 5K seed 稳定性不足。

更有意义的下一步是：

1. HLA2015 seed0 或 seed2 延长到 20K，判断失败 seed 是否能通过更长训练学到灌溉；
2. 或者调整 DQN 探索策略，例如延长 exploration_fraction、提高 replay buffer 中有效动作覆盖；
3. 或者改为先用固定反事实最优策略作为 imitation/warm-start 参考，再训练 RL。

如果目标是给导师汇报，当前最稳妥的表述是：

> DQN + economic reward 在 HLA2010 和 HLA2015 都出现了可解释的少氮/节水高产行为，但在 HLA2015 的 5K 训练下存在明显 seed 不稳定性，需要进一步改善训练稳定性。

