# 017_09 SY2014 DQN 资源使用与四情景过程图记录

## 目的

复用 017_08 已有结果，不重新训练，检查 SY2014 最佳 DQN checkpoint 的过程、资源使用和是否具有节水节氮空间。

## 输入

- 来源目录：`DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08`
- DQN 情景：`transfer_SY2014_ckpt15000`，在本记录中重命名为 `dqn`。
- 四情景：null、recorded、DSSAT auto、DQN。

## 统一 reward proxy

为了让四个情景可以按同一目标函数比较，本阶段重新计算 proxy reward：

```text
daily_proxy_reward = -1.0 * irrigation_mm - 5.0 * fertilizer_kg_ha
terminal_bonus = max(0, final_GWAD - null_GWAD)
```

注意：这是 DQN 目标函数的统一代理值，不是 DSSAT 原生 reward。

## 汇总结果

| scenario_label | final_gwad | final_cwad | irrigation_total_mm | fertilizer_total_kg_ha | max_water_stress | max_nitrogen_stress | final_cumulative_proxy_reward | yield_diff_vs_recorded | irrigation_saving_vs_recorded | fertilizer_saving_vs_recorded | resource_success_vs_recorded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Null | 2769.00 | 5648.00 | 0.00 | 0.00 | 0.00 | 0.44 | 0.00 | -6824.00 | 0.00 | 293.00 | False |
| Recorded expert | 9593.00 | 18302.00 | 0.00 | 293.00 | 0.81 | 0.01 | -6136.00 | 0.00 | 0.00 | 0.00 | False |
| DSSAT auto | 2724.00 | 5715.00 | 33.40 | 0.00 | 0.00 | 0.44 | -33.40 | -6869.00 | -33.40 | 293.00 | False |
| DQN ckpt15000 | 11216.00 | 20250.00 | 120.00 | 300.00 | 0.00 | 0.01 | 6827.00 | 1623.00 | -120.00 | -7.00 | False |

## 判断

- DQN 相对 recorded 的产量变化：1623 kg/ha。
- DQN 相对 recorded 的灌溉节省：-120.0 mm（负值表示用水更多）。
- DQN 相对 recorded 的施氮节省：-7.0 kg/ha（负值表示施氮更多）。
- 因此，当前 SY2014 DQN 可以表述为“高产型成功”：产量超过 recorded 和 DSSAT auto。
- 但不能表述为“节水节氮型成功”：它使用满 I120/N300 预算，且相对 recorded 多用水、略多施氮。
- 下一步若要追求节水节氮，需要做水氮成本/预算敏感性或候选 checkpoint 筛选，而不是直接继续增加训练步数。

## 输出

- 四情景日值表：`DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/017_09_sy2014_four_scenario_daily.csv`
- 四情景事件表：`DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/017_09_sy2014_four_scenario_events.csv`
- 四情景汇总表：`DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/017_09_sy2014_four_scenario_summary.csv`
- 过程图：`DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/figures/017_09_sy2014_four_scenario_process.png`
