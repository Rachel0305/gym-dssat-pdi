# E3：E2 双分支结构的零天气信号对照

## 目的

E3 保留 E2 的 25+12 双分支观测结构、96 维 latent、动作网格、奖励函数、安全约束和 PPO 设置；唯一改变是把 12 个天气分支输入在每个决策时刻固定为 0。该实验用于判断 E2 的变化是否确实依赖天气信号，而不是仅来自网络结构变化。

## 执行与门禁

- 站点：SYA / originIC；训练年：2005–2013；验证年：2014–2023。
- 2K smoke：通过。10 个验证年份完整，动作均在 16 格网内，存在 DAP1 之后的正动作，天气列存在且全为 0。
- 5K seed0：完成，checkpoint 为 2000 和 5000；正式 5K 的动作审计门禁通过。
- 未修改奖励函数和动作评价机制；未覆盖 E2 或 no-forecast 原始结果。

## 5K checkpoint=5000 结果

| 情景 | 平均产量 (kg/ha) | 平均 PFP-N (kg/kg) | 平均灌溉 (mm) | 平均施氮 (kg/ha) |
|---|---:|---:|---:|---:|
| E3 零天气分支 | 9952.58 | 41.4691 | 240 | 240 |
| E2 天气分支 | 10070.19 | 41.9591 | 240 | 240 |
| raw no-forecast | 10009.38 | 41.7057 | 240 | 240 |

相对同 seed 的 E2，E3 平均产量低 117.61 kg/ha（−1.17%），PFP-N 低 0.4900；10 年逐年比较中，E3 仅 4/10 年在产量和 PFP-N 上胜出。相对 raw no-forecast，E3 平均产量低 56.79 kg/ha（−0.57%），PFP-N 低 0.2366；产量和 PFP-N 均为 3/10 年胜出。

E3 从 2K 到 5K 的平均产量由 9988.16 降至 9952.58 kg/ha，PFP-N 由 41.6173 降至 41.4691；这是轻微回落，不是明显的训练坍缩。行为上，E3 在 5K 平均把 240 kg/ha 氮肥全部放在 DAP1–30，灌溉时段均值为 75/75/45/45 mm（DAP1–30/31–60/61–90/91+），说明零天气分支仍能产生多样动作，但没有形成优于 raw no-forecast 的管理策略。

## WP_ET replay

对 2014–2023 的 10 个冻结轨迹完成了与 E2/no-forecast 相同口径的 DSSAT ETCP replay，10/10 年成功，且与原 daily CSV 的产量、灌溉、施氮和动作序列闭合。E3 平均 WP_ET 为 **2.009 kg/m³**，加权 WP_ET 为 **2.008 kg/m³**；E2 seed0 为 2.115/2.115，raw no-forecast seed0 为 2.102/2.099。E3 相对 E2 seed0 逐年 0/10 胜出（1 年并列），相对 raw no-forecast 仅 1/10 胜出。

## 结论

E3 说明：天气信号对 E2 的表现有正贡献，但零天气双分支在产量、PFP-N 和 WP_ET 上都没有超过 raw no-forecast。当前不建议扩展 E3 到更多 seed；下一项干预应针对天气特征的冗余/噪声，而不是继续扩大网络或训练步数。

## 主要文件

- Prompt：`prompts/2026-08-16_sya_E3_dual_branch_noforecast_control_5k.md`
- 配置：`configs/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0.json`
- Runner：`src/run_152E3_sya_dual_branch_noforecast_control.py`
- 5K 结果：`benchmark_results/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0/152E3_5k_result.json`
- 5K 验证汇总：`benchmark_results/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0/evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- 动作/天气审计：`benchmark_results/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0/audits/152E3_formal_action_forecast_audit.csv`
- WP_ET replay：`benchmark_results/152E3N_wp_et_5k_replay/2026-08-16_sya_E3_wp_et_5k_replay.md`
