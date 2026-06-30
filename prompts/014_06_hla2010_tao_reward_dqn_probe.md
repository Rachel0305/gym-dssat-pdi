# 014_06 HLA2010 Tao et al. 文献奖励函数 DQN 探针

## 背景

014_03/014_04/014_05 发现：

- HLA2010 当前 economic DQN 5K 学成 no-op；
- 不是动作链路坏，而是当前奖励 `ΔGRNWT - water_cost × I - nitrogen_cost × N` 下，施氮轨迹净收益低于 no-op；
- 因此需要测试更文献化的奖励结构，而不是给 HLA 单独调一个站点专属奖励。

## 文献依据

Tao et al. 2023, “Optimizing Crop Management with Reinforcement Learning and Imitation Learning”, IJCAI.

该文使用 DQN 优化玉米水氮管理，奖励函数为：

```text
if harvest:
    r_t = w1 * Y - w2 * N_t - w3 * W_t - w4 * N_leach,t
else:
    r_t = - w2 * N_t - w3 * W_t - w4 * N_leach,t
```

其中 RF1 economic profit 权重为：

```text
w1 = 0.158
w2 = 0.79
w3 = 1.1
w4 = 0
```

本轮先不加入硝态氮淋失项，使用 RF1 简化版。

## 本轮目标

只在 HLA2010 上做低成本探针：

1. 先 200 step smoke；
2. 检查动作是否进入 DSSAT；
3. 检查是否仍然 no-op；
4. 如果 smoke 合理，再决定是否 5K。

## 方法

- 年份：HLA2010
- 输入：IC=1，新品种参数，沿用 014_03 的 linked 管理生成逻辑
- 算法：DQN
- 动作：沿用 YC/FQ linked DQN 四动作
  - 0：不操作
  - 1：灌溉 30 mm
  - 2：施氮 100 kg/ha
  - 3：灌溉 30 mm + 施氮 100 kg/ha
- 预算：
  - I ≤ 120 mm
  - N ≤ 300 kg/ha
  - 单次 I≤30 mm，N≤100 kg/ha
  - 最小操作间隔 7 天
- 窗口：
  - 先用 free_daily：DAP 1–120 水氮都允许

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_tao_reward_dqn_probe_014_06/`
- `dqn_tao_eval_daily.csv`
- `event_summary.json`
- `014_06_hla2010_tao_reward_summary.csv`
- `docs/2026-06-30_014_06_hla2010_tao_reward_dqn_probe_record.md`

## 注意

- 不直接跑三站点。
- 不覆盖旧结果。
- 先 smoke，避免浪费算力。
