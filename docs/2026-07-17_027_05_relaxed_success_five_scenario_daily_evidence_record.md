# 027_05 松弛成功标准下五情景与日值证据整理记录

## 结论先行

本轮完成五个历史 DQN 代表站点年份，以及 SY2014/HLA2010 两个当前阶段型 MaskablePPO 候选的五情景原始 DSSAT 输出重建、统一日值表、终值表和 PNG/SVG 图。没有训练；PPO 仅使用冻结模型各做一次确定性 DSSAT 复评估，共 2 个季节。

这些 DQN 结果必须保留为 `historical_candidate_provisional`：021_05 已确认旧分段训练协议压缩探索率日程。本轮图表用于核对候选策略的决策过程，不恢复其正式科学结论地位。

SY/HLA 的阶段型 PPO 已有模型、五情景终值与六阶段动作。本轮 Docker 恢复后以冻结 checkpoint 做确定性复评估，并在容器退出前保存完整 DSSAT snapshot；模型哈希、num_timesteps、六阶段动作序列和终值指标均与已发布证据精确一致。没有调用 `learn()`，没有重新选择 checkpoint。

## 数据口径

- 五情景：null、recorded farmer、DSSAT auto、official extension expert、RL candidate。
- 雨量和气温直接解析各情景 `Weather.OUT`，修复旧 expert CSV 中雨量全 0 的显示问题。
- 土壤水分解析 `SoilWat.OUT: SWTD`；水/氮胁迫解析 `PlantGro.OUT: WSPD/NSTD`，两者均以 0 表示无胁迫。
- 灌溉和施氮事件解析 `MgmtEvent.OUT`；若同一输出文件包含重复运行块，按事件键去重。
- 累积奖励统一使用反事实同口径：`max(0, final_yield-null_yield) - irrigation - 5*nitrogen`。它不是 null/recorded/auto/expert 的原生 reward。
- PFP_N 在 DSSAT Summary 的施氮量为 0 时保持 NA。
- PPO 冻结评估的环境终值保留浮点精度；PlantGro.OUT 的 GWAD/CWAD 仅保存整数 kg/ha。因此二者复核采用由源文件精度决定的 0.5 kg/ha 容差，水氮投入与效率指标仍按 1e-9 核对。终值图使用已发布浮点终值，日值图使用 PlantGro.OUT 原始整数轨迹，二者不混填。
- HLA2010 原始 `T2.xls`、WTH 和 `Weather.OUT` 在 DOY153 均记录 `TMAX=154.3 C`。该源数据异常未被猜测性修正：CSV 保留原值并以 `temperature_source_qc` 标记，正式折线图仅隐藏该异常点并附注。

## 审计统计

- manifest rows: 7
- five-scenario summary rows: 35
- PPO endpoint rows reused: 10
- consistency checks passed: 225/225
- unresolved evidence gaps: 0
- PPO frozen deterministic reevaluation: 2 DSSAT seasons; training steps = 0; `learn()` was not called.

## 失败与修复记录

1. 第一次构建未去除 DSSAT 输出中的重复追加运行块，导致管理总量被重复累加；结果保存在 `benchmark_results/027_05_failed_attempt_1_repeated_appended_dssat_blocks`，未用于结论。
2. 一次宿主命令仅因客户端等待时间过短而超时，没有产生科学结果。
3. 初版通用 Summary 解析器错配 WP/PFP 行；结果保存在 `benchmark_results/027_05_failed_attempt_3_summary_parser_mismatch`，随后改用项目已验证的 019_10 解析器。
4. 视觉复核发现 HLA2010 DOY153 的源数据 Tmax=154.3 C。未修改原始 WTH；旧图保存在 `benchmark_results/027_05_failed_attempt_4_weather_source_anomaly_unflagged`，正式 CSV 保留原值并标 QC，图中隐藏异常点。
5. PPO 基线 CSV 的字面场景 `null` 首次被 pandas 当成 NA，五情景完整性检查主动失败；结果保存在 `benchmark_results/027_05_failed_attempt_5_null_label_parsed_as_na`，修复为 `keep_default_na=False`。
6. 初次把 PPO 日值终点与已发布浮点终值按 1e-9 直接比较时有4项失败。核查确认 `PlantGro.OUT` 的 GWAD/CWAD 只保留整数 kg/ha，而 Gym 冻结评估保留浮点值；该版保存在 `benchmark_results/027_05_failed_attempt_6_endpoint_source_precision_mismatch`。正式检查按源文件最高有效精度采用0.5 kg/ha容差，水氮投入与效率仍按1e-9核对，225/225通过。

## 仍未完成事项

1. 本轮指定的成功候选证据整理已经完成，没有未解决的日值证据缺口。
2. YC/FQ/LC 尚无正式阶段型 PPO 成功模型；本轮没有把 smoke 或未授权训练冒充成功 PPO，也没有为补图而启动新训练。
3. 历史 DQN 候选仍受 021_05 探索率日程问题影响，只能用于过程诊断，不能恢复为正式科学结论。

## 输出

- `benchmark_results/027_05/027_05_evidence_manifest.csv`
- `benchmark_results/027_05/027_05_gap_matrix.csv`
- `benchmark_results/027_05/027_05_five_scenario_summary.csv`
- `benchmark_results/027_05/027_05_daily_values.csv`
- `benchmark_results/027_05/027_05_dqn_five_scenario_summary.csv`
- `benchmark_results/027_05/027_05_dqn_daily_values.csv`
- `benchmark_results/027_05/027_05_ppo_five_scenario_summary.csv`
- `benchmark_results/027_05/027_05_ppo_daily_snapshot_summary.csv`
- `benchmark_results/027_05/027_05_ppo_daily_values.csv`
- `benchmark_results/027_05/027_05_ppo_selected_stage_actions.csv`
- `benchmark_results/027_05_ppo_frozen_daily_completion/027_05_result.json`
- `benchmark_results/027_05/027_05_data_consistency_checks.csv`
- `benchmark_results/027_05/figures/*.png` and `*.svg`
