# 014_05 HLA2010 奖励目标诊断记录

## 目的

诊断 HLA2010 DQN 5K 退化为 no-op，是否是当前经济奖励函数下的理性结果。

当前奖励口径：

```text
reward = ΔGRNWT - 1.0 × irrigation - 5.0 × nitrogen
```

null 参考产量：6956.00 kg/ha。

## 重评分结果

| case | final_yield | irrigation | nitrogen | yield_gain_vs_null | water_cost_term | nitrogen_cost_term | current_net_objective | net_gain_vs_null_under_current_reward | break_even_n_cost_given_water_cost_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dssat_auto | 7854.00 | 190.40 | 0.00 | 898.00 | 190.40 | 0.00 | 7663.60 | 707.60 |  |
| dqn_free_daily_5000step | 6956.45 | 0.00 | 0.00 | 0.45 | 0.00 | 0.00 | 6956.45 | 0.45 |  |
| baseline_null | 6956.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6956.00 | 0.00 |  |
| baseline_expert_2007_shifted | 7679.00 | 30.00 | 165.00 | 723.00 | 30.00 | 825.00 | 6824.00 | -132.00 | 4.20 |
| dqn_free_daily_200step_smoke | 7853.67 | 120.00 | 200.00 | 897.67 | 120.00 | 1000.00 | 6733.67 | -222.33 | 3.89 |
| dqn_agronomic_window_200step_smoke | 7853.67 | 120.00 | 300.00 | 897.67 | 120.00 | 1500.00 | 6233.67 | -722.33 | 2.59 |

## 关键结论

- 在当前 water_cost=1、nitrogen_cost=5 的口径下，最高 current_net_objective 是 `baseline_dssat_auto`，数值为 7663.60。
- HLA2010 的 5K DQN 选择 no-op，并不是动作链路故障；从当前经济奖励看，no-op 确实比 DQN smoke 的施氮轨迹更划算。
- 200 step smoke 的 `free_daily` 轨迹虽然产量高约 897 kg/ha，但用了 120 mm 水和 200 kg/ha 氮；在 N cost=5 下，氮成本为 1000，已经吃掉全部增产收益。
- 要让 `free_daily 200step` 这种 I120/N200 轨迹相对 no-op 不亏，在 water_cost=1 下，氮成本需要低于约 3.89 kg grain/kg N。
- 因此 HLA 当前问题首先是奖励目标定义问题，不是继续加 seed 或继续训练能解决的问题。

## 对下一步的含义

- 如果论文目标是经济净收益最大化，那么 HLA2010 no-op 是合理结果，不能强行说 DQN 失败。
- 如果论文目标是约束内产量最大化或节水节氮下保产，那么当前 `ΔGRNWT - 水氮成本` 奖励不适合 HLA，需要统一修改所有站点的奖励口径，而不能只给 HLA 特调。
- 下一步应先和导师确定主目标：经济净收益、约束内产量最大化，还是多目标折中。

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_reward_objective_diagnosis_014_05/014_05_hla2010_reward_rescore_summary.csv`
