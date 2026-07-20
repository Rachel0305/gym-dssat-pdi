# 027_05 DQN/PPO 决策合理性初审

## 证据边界

- 历史 DQN 五情景和日值证据来自原始 DSSAT `Weather.OUT`、`SoilWat.OUT`、`PlantGro.OUT`、`MgmtEvent.OUT` 和 `Summary.OUT`，本轮未训练、未重新运行 DSSAT。
- 五个 DQN 候选均保持 `historical_candidate_provisional`：旧分段训练协议存在探索率日程压缩问题；这些图用于审查候选决策，不恢复其正式科学结论地位。
- SY2014/HLA2010 MaskablePPO 已复用冻结终值与六阶段动作，并用原 checkpoint 各做一次确定性冻结复评估，保存了完整日尺度 DSSAT 状态快照。模型哈希、num_timesteps、动作序列和终值全部复现；`training_steps=0`，未调用 `learn()`，未重新选择 checkpoint。
- 五情景公共累计奖励仅用于同口径对照：`max(0, final_yield-null_yield)-I-5N`，不是各基线原始训练奖励。

## 历史 DQN 候选

| 站点年份 | DQN 措施 | 终值证据 | 初步判断 |
|---|---|---|---|
| HLA2010 | 灌溉 DAP 78/85/92/109/116，各 15 mm；N=0 | Y=7854 kg/ha，I=75 mm，N=0，WP_ET=1.69 | 灌溉主要落在后期水分胁迫窗口，方向可解释；但 DAP109/116 偏晚且完全不施氮高度依赖初始土壤氮，只能作为历史候选。HLA DOY153 的源气象 Tmax=154.3°C 是已标记输入异常。 |
| YC2014 | 灌溉 DAP 1/9/43/52/59/66/73；施氮 DAP 1/9/86/93/100 | Y=9418 kg/ha，I=120 mm，N=250 kg/ha，WP_ET=2.53，PFP_N=37.7 | 产量与 recorded/expert 基本持平，投入比 expert 更省；但 DAP86/93/100 的晚期施氮是明确的农学风险点，不能只凭终值称合理。 |
| FQ2016 | 灌溉 DAP 83/90，各 30 mm；N=0 | Y=7995 kg/ha，I=60 mm，N=0，WP_ET=2.47 | 与图中的晚期水分胁迫相对应，灌溉时点具有可解释性；产量仅比 auto 低 17 kg/ha。零氮结果依赖初始条件，PFP_N 不可定义，需谨慎表述。 |
| LC2010 | 灌溉 DAP 1/11 各 15 mm、DAP82/89 各 30 mm；N=0 | Y=8739 kg/ha，I=90 mm，N=0，WP_ET=3.06 | 早季和后期补水与 null 的胁迫窗口大体一致，且比 auto/expert 少水；但这是 seed0 5K smoke 候选，不是跨 seed 正式结论，零氮同样依赖初始土壤氮。 |
| SY2014（历史 IC=0） | 施氮 DAP34/41/48/55，共 300 kg/ha；灌溉 DAP34/48/62/69/76/83/90/97，共 120 mm | Y=11216 kg/ha，I=120 mm，N=300 kg/ha，WP_ET=2.30，PFP_N=37.4 | 氮集中在中前期、未出现晚期施氮，时序比 YC 更可解释；但投入仍高，且这是旧 IC=0 证据，不能与当前 IC=2 PPO 结果混用。 |

## 当前 MaskablePPO 候选

| 站点年份 | 冻结策略措施 | 五情景终值证据 | 初步判断 |
|---|---|---|---|
| SY2014 seed0 ckpt120 | DAP50/65 各 I15+N100；DAP85/110 各 I15；合计 I60/N200 | Y=11204.88 kg/ha，WP_ET=2.31，PFP_N=56.0；相对 official expert 增产约146、少水约206 mm、少氮100 kg/ha | 当前最清晰的合理候选。N 集中在 DAP50/65，没有晚期追氮；四次小水灌溉把 WSPD 基本维持为 0，仅季末短暂升至 0.159，NSTD 季末最高 0.136，远低于 null 的 0.617。DAP110 灌溉偏晚但只有15 mm，结合其后土壤水分继续下降和季末轻微水胁迫，可解释为防止灌浆后期缺水；是否为最小必要水量仍需反事实删水验证，不能仅凭轨迹证明。 |
| HLA2010 seed0 ckpt180 | DAP1 N50；DAP65/85/110 各 I30；合计 I90/N50 | Y=7853.67 kg/ha，WP_ET=1.69，PFP_N=157.1；产量基本等于 auto/expert | 冻结日值支持其资源效率优势：候选 NSTD 最高0.016，与 recorded/expert 相当；WSPD最高0.414，主要是早期一次短暂胁迫，DAP65/85/110灌溉后未出现 null 那样的季末0.919重胁迫。DAP110灌溉具有预防季末缺水的过程证据，但候选仍比 recorded 多用60 mm水、WP_ET低0.01，故应称“接近 recorded、明显优于 auto/expert 的资源折中”，而不是全面优于所有基线。 |

## 当前结论

1. 不能把所有候选统称为“决策合理”。SY2014 PPO 的时序、胁迫和终值证据最完整；HLA2010 PPO 是可解释的资源折中；YC2014 DQN 的 DAP86/93/100 晚期施氮最值得警惕。
2. DQN 图表已足以指出具体可疑 DAP，但因旧探索率协议问题只能用于历史候选审查。
3. PPO 日值缺口已经补齐，不需要为绘图重新训练。两个冻结候选均通过模型、动作和终值复现检查；本轮 225/225 项一致性检查通过。
4. 图中五情景公共累计奖励是统一反事实审阅指标，不等于各算法原始训练奖励；决策合理性必须同时看事件时点、胁迫轨迹和终值，不能只看奖励曲线。

## 对应文件

- `benchmark_results/027_05/027_05_five_scenario_summary.csv`
- `benchmark_results/027_05/027_05_daily_values.csv`
- `benchmark_results/027_05/027_05_ppo_five_scenario_summary.csv`
- `benchmark_results/027_05/027_05_ppo_daily_snapshot_summary.csv`
- `benchmark_results/027_05/027_05_ppo_daily_values.csv`
- `benchmark_results/027_05/027_05_ppo_selected_stage_actions.csv`
- `benchmark_results/027_05/figures/`
