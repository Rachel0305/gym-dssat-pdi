# 012_06 HLA2015 固定动作反事实检查记录

## 目的

012_05 中，HLA2015 economic reward DQN seed0 没有产生有效灌溉或施肥，结果几乎等同于 null。

为了判断这是“DQN 没学到动作”，还是“2015 年在当前经济奖励下本来就不值得操作”，本实验不训练模型，而是人为固定执行几组水氮动作，检查这些动作本身是否能增产、是否能提高 economic reward。

## 实验性质

这是确定性反事实评估，不是强化学习训练。

所有情景使用同一套：

- HLA2015 输入；
- IC=1 初始条件；
- 当前更新后的品种参数；
- DQN economic reward 的窗口和预算约束；
- PDI/gym-DSSAT 运行环境。

## 固定动作方案

为了避免 wrapper 中“任意两次操作至少间隔 7 天”的限制导致动作被拦截，灌溉和施肥日程调整为：

- 灌溉：DAP 28、49、78、92，每次 30 mm，总量 120 mm；
- 施氮：DAP 35、60、67，每次 50 kg/ha，最多总量 150 kg/ha。

扫描情景：

| 情景 | 灌溉 | 施氮 |
|---|---:|---:|
| fixed_I0_N0 | 0 mm | 0 kg/ha |
| fixed_I120_N0 | 120 mm | 0 kg/ha |
| fixed_I120_N50 | 120 mm | 50 kg/ha |
| fixed_I120_N100 | 120 mm | 100 kg/ha |
| fixed_I120_N150 | 120 mm | 150 kg/ha |

奖励函数仍为：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_fixed_action_counterfactual_012_06.py"
```

绘图命令：

```bash
python src/plot_hla2015_fixed_counterfactual_012_06.py
```

## 输出文件

- 运行脚本：

```text
src/run_hla2015_fixed_action_counterfactual_012_06.py
```

- 绘图脚本：

```text
src/plot_hla2015_fixed_counterfactual_012_06.py
```

- 日值数据：

```text
DSSAT_auto_validation/HLA_2004/hla2015_fixed_action_counterfactual_012_06/hla2015_fixed_action_counterfactual_daily.csv
```

- 汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_fixed_action_counterfactual_012_06/hla2015_fixed_action_counterfactual_summary.csv
```

- 过程图：

```text
DSSAT_auto_validation/HLA_2004/hla2015_fixed_action_counterfactual_012_06/figures/hla2015_fixed_action_counterfactual_process.png
```

- 汇总图：

```text
DSSAT_auto_validation/HLA_2004/hla2015_fixed_action_counterfactual_012_06/figures/hla2015_fixed_action_counterfactual_summary.png
```

## 结果

| 情景 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | economic reward | 最大水分胁迫 | 最大氮胁迫 |
|---|---:|---:|---:|---:|---:|---:|
| fixed_I0_N0 | 6486 | 0 | 0 | 6485.77 | 1.000 | 0.093 |
| fixed_I120_N0 | 7643 | 120 | 0 | 7523.27 | 0.000 | 0.056 |
| fixed_I120_N50 | 7643 | 120 | 50 | 7272.90 | 0.000 | 0.015 |
| fixed_I120_N100 | 7643 | 120 | 100 | 7022.89 | 0.000 | 0.015 |
| fixed_I120_N150 | 7643 | 120 | 150 | 6772.89 | 0.000 | 0.015 |

## 关键结论

1. HLA2015 在当前输入下确实存在明显灌溉增产空间。
   - I0/N0 产量约 6486 kg/ha；
   - I120/N0 产量约 7643 kg/ha；
   - 增产约 1157 kg/ha。

2. HLA2015 在这套初始条件和固定灌溉方案下，施氮没有带来额外产量增益。
   - I120/N0 到 I120/N150 的产量几乎相同；
   - 施氮主要降低氮胁迫指数，但没有转化为籽粒产量提升。

3. 在当前 economic reward 下，I120/N0 的经济奖励最高。
   - 这说明如果模型能学到理想动作，它应该至少学会灌溉；
   - 但不应该额外施氮，除非奖励目标明确要求降低氮胁迫。

4. 因此，012_05 中 DQN seed0 完全不操作并不是因为 2015 年没有管理增益。
   - 更准确地说：2015 年有灌溉增益，但 DQN 没有学到灌溉动作；
   - 施氮不增产，所以 DQN 不施氮本身是合理的。

## 对 DQN 线的影响

HLA2010 中，DQN economic reward 能学到高产少氮；

HLA2015 中，固定反事实表明高产最优方向接近 I120/N0，但 DQN seed0 没有学到灌溉。

所以目前问题不再是“economic reward 是否完全错误”，而更像是：

> 当前 DQN 在不同年份下的探索和学习稳定性不足。2015 年存在可学的灌溉增益，但 5K seed0 没有学到。

## 下一步建议

有两个低成本方向：

1. 先跑 HLA2015 economic DQN seed1。
   - 如果 seed1 学到 I120/N0 或接近 I120/N0，说明 2015 是 seed 敏感；
   - 如果 seed1 也退化为 null，说明 5K DQN 在 2015 学习不足或探索不足。

2. 或者先把 HLA2015 DQN 训练步数延长到 20K，但只跑 seed0。
   - 这个能回答“是不是 5K 不够”；
   - 但比 seed1 更耗时。

当前更推荐先跑 seed1，因为它成本更低，也更符合稳定性验证逻辑。

