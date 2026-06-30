# 012_07 HLA2015 economic reward DQN seed1 稳定性验证记录

## 目的

012_05 中，HLA2015 economic reward DQN seed0 跑 5000 步后退化为 null，没有有效灌溉或施肥。

012_06 固定动作反事实显示，HLA2015 并不是没有管理增益：

- 固定 I120/N0 能把产量从约 6486 kg/ha 提高到约 7643 kg/ha；
- 施氮 N50–N150 几乎不增加产量，只降低氮胁迫并降低 economic reward；
- 因此 seed0 的失败更像是没有学到灌溉动作。

本轮运行 HLA2015 economic DQN seed1，检查这种失败是否只是 seed0 偶然现象。

## 运行设置

- 年份：HLA2015
- 算法：DQN
- seed：1
- 训练步数：5000
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

- 动作空间：
  - 0：不操作
  - 1：灌溉 30 mm
  - 2：施氮 50 kg/ha
  - 3：灌溉 30 mm + 施氮 50 kg/ha
- 灌溉预算：120 mm
- 追加氮预算：150 kg/ha
- 灌溉窗口：DAP 20–35、45–65、70–95
- 施氮窗口：DAP 25–40、55–70

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2015 --timesteps 5000 --seed 1 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

## 输出文件

- seed1 输出目录：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/2015/medium_N_cost_seed1_5000steps/
```

- 对比绘图脚本：

```text
src/plot_hla2015_dqn_economic_seed_comparison_012_07.py
```

- seed0/seed1 过程图：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_07_hla2015_seed_comparison/hla2015_dqn_seed0_seed1_process.png
```

- DQN 与固定反事实汇总图：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_07_hla2015_seed_comparison/hla2015_dqn_seed0_seed1_vs_fixed_summary.png
```

- 汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_07_hla2015_seed_comparison/hla2015_dqn_seed0_seed1_vs_fixed_summary.csv
```

## 结果

| 情景 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | economic reward | 最大水分胁迫 | 最大氮胁迫 |
|---|---:|---:|---:|---:|---:|---:|
| DQN seed0 | 6486 | 0 | 0 | 6485.77 | 1.000 | 0.093 |
| DQN seed1 | 7632 | 60 | 0 | 7572.28 | 0.161 | 0.099 |
| fixed I0/N0 | 6486 | 0 | 0 | 6485.77 | 1.000 | 0.093 |
| fixed I120/N0 | 7643 | 120 | 0 | 7523.27 | 0.000 | 0.056 |
| fixed I120/N50 | 7643 | 120 | 50 | 7272.90 | 0.000 | 0.015 |
| fixed I120/N100 | 7643 | 120 | 100 | 7022.89 | 0.000 | 0.015 |
| fixed I120/N150 | 7643 | 120 | 150 | 6772.89 | 0.000 | 0.015 |

## 有效动作

seed1 只执行了两次灌溉：

| DAP | 灌溉 mm | 施氮 kg/ha |
|---:|---:|---:|
| 46 | 30 | 0 |
| 53 | 30 | 0 |

总量：

- 灌溉：60 mm；
- 施氮：0 kg/ha。

## 关键结论

1. HLA2015 economic DQN 存在明显 seed 敏感性。
   - seed0 完全不操作，等同 null；
   - seed1 学到两次灌溉，产量接近固定 I120/N0。

2. seed1 的结果很有价值。
   - 它没有施氮，符合 012_06 中“施氮没有产量边际收益”的反事实结果；
   - 它只用 60 mm 水，却达到约 7632 kg/ha，接近 fixed I120/N0 的 7643 kg/ha；
   - 因为少用 60 mm 水，seed1 的 economic reward 甚至高于 fixed I120/N0。

3. 不能简单说 seed1 “没打满水所以不好”。
   - 在当前 economic reward 下，少水高产可能更优；
   - 但 012_06 之前只扫了 I0 和 I120，没有扫 I30/I60/I90；
   - 因此需要补一个固定 N0 水量扫描，确认 I60/N0 是否真的是更优或近优方案。

## 当前结论边界

可以说：

> HLA2015 中，DQN seed1 学到了有经济意义的灌溉策略：以 60 mm 水、0 氮达到接近 I120/N0 的产量，并获得更高 economic reward。

还不能说：

> DQN 已经稳定解决 HLA2015。

因为 seed0 和 seed1 差异很大，seed 稳定性仍未通过。

## 下一步

补充 HLA2015 N0 固定水量扫描：

- I0/N0
- I30/N0
- I60/N0
- I90/N0
- I120/N0

目的：判断 seed1 的 I60/N0 是否接近固定反事实最优。

