# 012_10 HLA2015 economic reward DQN seed0 20K 训练不足诊断 prompt

## 背景

012_07–012_09 已经确认 HLA2015 的 5K DQN 存在明显 seed 方差：

- seed0：不操作，等同 null；
- seed1：灌溉 60 mm、施氮 0 kg/ha，接近 fixed I60/N0；
- seed2：只施氮 50 kg/ha、不灌溉，低于 null 的 economic reward。

012_08 固定 N0 水量扫描显示：

- fixed I60/N0 的 economic reward 最高；
- 因此 HLA2015 的合理方向是“适量灌溉、零施氮”。

现在需要判断 seed0 的失败是否只是 5K 训练不足。如果把 seed0 从头训练到 20K 后能学到灌溉，说明主要问题是训练步数不够；如果仍然不学灌溉，说明仅仅加步数可能不够，需要改探索策略或算法结构。

## 实验设置

- 年份：HLA2015
- 算法：DQN
- seed：0
- 训练步数：20000
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

- 灌溉预算：120 mm
- 追加氮预算：150 kg/ha
- 灌溉窗口：DAP 20–35、45–65、70–95
- 施氮窗口：DAP 25–40、55–70

## 判定标准

对比三类参考：

1. seed0 5K：
   - I0/N0；
   - 产量约 6486 kg/ha；
   - economic reward 约 6486。

2. seed1 5K：
   - I60/N0；
   - 产量约 7632 kg/ha；
   - economic reward 约 7572。

3. fixed I60/N0：
   - 产量约 7625 kg/ha；
   - economic reward 约 7565。

如果 seed0 20K 接近 I60/N0，说明 5K 对 seed0 不够。

如果 seed0 20K 仍然接近 null 或施氮不灌溉，说明当前 DQN 训练稳定性问题不能靠简单延长到 20K 解决。

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2015 --timesteps 20000 --seed 0 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

