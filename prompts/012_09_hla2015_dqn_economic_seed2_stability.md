# 012_09 HLA2015 economic reward DQN seed2 稳定性验证 prompt

## 背景

012_07 中，HLA2015 economic reward DQN 出现明显 seed 差异：

- seed0：不操作，等同 null；
- seed1：灌溉 60 mm、施氮 0 kg/ha，产量约 7632 kg/ha。

012_08 固定 N0 水量扫描显示：

- I60/N0 的 economic reward 最高；
- I90/N0 和 I120/N0 产量略高，但因为多用水，economic reward 反而更低；
- 因此 seed1 的 I60/N0 行为不是坏结果，而是接近当前 economic reward 下的固定水量最优。

现在需要再跑 seed2，判断 seed1 是否只是偶然成功，还是 HLA2015 在 DQN economic reward 下具有可重复趋势。

## 实验设置

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

## 判定标准

优先判断 seed2 是否接近以下三类结果：

1. 接近 seed1 / fixed I60/N0：
   - 灌溉约 60 mm；
   - 施氮约 0；
   - 产量约 7625–7635 kg/ha；
   - economic reward 约 7560–7575。

2. 接近 fixed I90/I120/N0：
   - 灌溉 90–120 mm；
   - 施氮约 0；
   - 产量约 7645 kg/ha；
   - economic reward 略低于 I60/N0，但仍是合理策略。

3. 退化为 null：
   - 灌溉 0；
   - 施氮 0；
   - 产量约 6486 kg/ha；
   - 说明 5K DQN seed 稳定性不足。

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2015 --timesteps 5000 --seed 2 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

