# 012_07 HLA2015 economic reward DQN seed1 稳定性验证 prompt

## 背景

012_05 中，HLA2015 economic reward DQN seed0 跑 5000 步后没有产生任何有效管理动作，结果几乎等同于 null。

012_06 随后做了固定动作反事实扫描，发现 HLA2015 并不是没有管理增益：

- I0/N0 产量约 6486 kg/ha；
- I120/N0 产量约 7643 kg/ha；
- 灌溉能显著增产；
- 施氮从 N0 到 N150 几乎不增加产量，只降低氮胁迫指数并增加经济成本；
- 在当前 economic reward 下，固定方案 I120/N0 的 reward 最高。

因此，012_05 seed0 的失败更像是 DQN 没有学到灌溉动作，而不是 2015 年没有可优化空间。

## 本轮目标

运行 HLA2015 economic reward DQN seed1，训练 5000 步，检查 seed1 是否能学到接近固定反事实最优方向的策略。

## 实验设置

- 年份：HLA2015
- 算法：DQN
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
- 训练步数：5000
- seed：1

## 判定标准

优先看三个指标：

1. 是否学到有效灌溉。
   - 参考固定反事实：I120/N0 产量最高且 economic reward 最高；
   - 如果 seed1 灌溉接近 120 mm，说明它学到了 2015 年的主要收益来源。

2. 是否避免无收益施氮。
   - 固定反事实显示 N0、N50、N100、N150 产量几乎相同；
   - 在当前 economic reward 下，额外施氮会降低 reward；
   - 因此理想策略应接近 N0。

3. 产量是否接近固定反事实 I120/N0。
   - 固定反事实 I120/N0 产量约 7643 kg/ha；
   - 若 DQN seed1 接近该值，说明 seed0 的 null 结果可能是 seed 敏感或探索失败。

## 当前不能做的事

- 不改 reward；
- 不改窗口；
- 不改动作空间；
- 不延长到 20K；
- 不开始 2015 seed2 或其他年份；
- 不把这轮结果直接宣布为最终成功。

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2015 --timesteps 5000 --seed 1 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

## 预期输出

- seed1 训练/评估目录；
- seed1 event_summary.json；
- seed0/seed1/fixed counterfactual 对比表；
- HLA2015 economic DQN seed0/seed1 过程对比图；
- 中文实验记录 MD。

