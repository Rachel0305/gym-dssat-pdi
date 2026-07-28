# 035_05 FQA2014 氮边际收益审计记录

## 结论先说

- 固定 I45 与施氮窗口后，project_simple_profit 最高规则是 `i45_n80`。
- N200→N240 边际产量收益为 0.025 kg grain/kg N。
- 当前 nitrogen_cost = 1.58 kg yield-equivalent/kg N；若某档边际收益高于该值，当前 reward 会认为继续加氮仍然划算。
- 本任务不训练模型，只做受控 DSSAT 前向模拟。

## 固定规则结果

| rule | interface_pass | grain_yield_kg_ha | summary_irrigation_mm | summary_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | project_simple_profit | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i45_n0 | True | 7864.2047 | 45.0 | 0.0 | 2.21 |  | 7814.7047 | 0 | 24 | DAP1 I15/N0; DAP50 I15/N0; DAP65 I15/N0 |
| i45_n80 | True | 8617.4969 | 45.0 | 80.0 | 2.42 | 107.7 | 8441.5969 | 0 | 0 | DAP1 I15/N40; DAP30 I0/N40; DAP50 I15/N0; DAP65 I15/N0 |
| i45_n120 | True | 8617.3785 | 45.0 | 120.0 | 2.42 | 71.8 | 8378.2785 | 0 | 0 | DAP1 I15/N40; DAP30 I0/N40; DAP50 I15/N40; DAP65 I15/N0 |
| i45_n160 | True | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | 8315.0785 | 0 | 0 | DAP1 I15/N40; DAP30 I0/N40; DAP50 I15/N40; DAP65 I15/N40 |
| i45_n200 | True | 8617.3547 | 45.0 | 200.0 | 2.42 | 43.1 | 8251.8547 | 0 | 0 | DAP1 I15/N40; DAP30 I0/N40; DAP50 I15/N80; DAP65 I15/N40 |
| i45_n240 | True | 8618.3618 | 45.0 | 240.0 | 2.43 | 35.9 | 8189.6618 | 0 | 0 | DAP1 I15/N80; DAP30 I0/N40; DAP50 I15/N80; DAP65 I15/N40 |

## 相邻施氮档位边际收益

| from_rule | to_rule | delta_n_kg_ha | delta_yield_kg_ha | marginal_yield_per_kgN | delta_project_simple_profit | current_n_cost | marginal_exceeds_current_n_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| i45_n0 | i45_n80 | 80.0 | 753.2922 | 9.4162 | 626.8922 | 1.58 | True |
| i45_n80 | i45_n120 | 40.0 | -0.1184 | -0.003 | -63.3184 | 1.58 | False |
| i45_n120 | i45_n160 | 40.0 | 0.0 | 0.0 | -63.2 | 1.58 | False |
| i45_n160 | i45_n200 | 40.0 | -0.0238 | -0.0006 | -63.2238 | 1.58 | False |
| i45_n200 | i45_n240 | 40.0 | 1.0071 | 0.0252 | -62.1929 | 1.58 | False |

## 解释边界

- 这是 FQA2014、固定 I45、固定早期分期施氮窗口下的局部审计。
- 它不能直接证明所有站点年份都需要同样的 nitrogen_cost。
- 它可以用于判断 035_04 中 PPO 偏向高氮是否与当前 reward 算术一致。

耗时：20.4 秒
