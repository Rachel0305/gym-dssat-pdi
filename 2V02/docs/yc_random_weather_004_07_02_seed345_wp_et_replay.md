# YC Random-weather PPO seed 3/4/5 ETCP 回放与对照

## 设计与证据口径

- 固定 004_05 的 100K checkpoint，逐模型应用于 2014–2023 observed-weather 验证年；没有逐年换 seed、重训或修改 PPO/reward/action/DSSAT。
- 先以 seed 5 / 2019 作为完整 season smoke。它通过 yield（误差不超过 1 kg/ha）、管理事件与 episode / Summary.OUT 闭合、有效 ETCP、步数和 observed 输入 SHA-256 一致性门槛后，才回放其余 seed-year。
- ETCP、WP_ET、PFP_N 来自匹配最终 yield 的 Summary.OUT 行；不由日尺度表或消耗量反推 ETCP。WP_ET 使用项目既有 summary_metrics 口径。
- 四基准数据取 055_03 season summary；共同 reward 采用正式 baseline unified_reward。PPO common reward 单列计算为 `0.158 × yield − 1.1 × irrigation − 1.58 × N`；PPO canonical reward 单独保留，不和 common reward 混比。
- 全部 2014–2023 已经参与 seed 筛选/比较，是探索性证据，不是独立确认集。seed 5/3/4 的相互筛选仍有选择偏差。
- 所有 episode 使用 CPU 串行回放；进程树 RSS 上限 4096 MiB。峰值 445.9 MiB。
- 首次 smoke 的 DSSAT episode 已完成，但校验器最初查错了新渲染目录；保留原 failed manifest 后，以现存快照、输入哈希、Summary.OUT 和 MgmtEvent.OUT 作事后核验通过，没有重跑 smoke。正式 30 行表全部写出后，原汇总脚本又在派生 Pareto 表处遇到变量名错误；本报告由独立 finalize 阶段从已审计表生成，两个失败 manifest 与日志均保留。

## 三个 seed 的十年摘要

| Seed | 产量 kg/ha | 灌溉 mm | N kg/ha | ETCP mm | WP_ET kg/m³ | PFP_N | common reward | canonical reward | 管理事件/季 | 非零动作日/季 | 五情景非支配年数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 6901.9 | 75.0 | 100.0 | 335.8 | 2.033 | 72.68 | 850.00 | 0.5723 | 4.5 | 3.5 | 10/10 |
| 4 | 8202.9 | 214.5 | 240.0 | 368.6 | 2.224 | 34.17 | 680.91 | 0.7491 | 17.3 | 16.3 | 10/10 |
| 5 | 8148.7 | 165.0 | 240.0 | 362.3 | 2.249 | 33.96 | 726.79 | 0.7712 | 9.0 | 5.0 | 10/10 |

## Seed 5 逐年结果

| 年份 | Yield | 灌溉 | N | ETCP | WP_ET | PFP_N | common reward | 五情景非支配 | 对四基准最高产量差值 | 管理事件 | 非零动作日 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 2014 | 9253.2 | 165.0 | 240.0 | 362.5 | 2.550 | 38.60 | 901.30 | 是 | -264.8 | 9 | 5 |
| 2015 | 9088.7 | 165.0 | 240.0 | 348.8 | 2.610 | 37.90 | 875.31 | 是 | -0.3 | 9 | 5 |
| 2016 | 6496.6 | 165.0 | 240.0 | 334.1 | 1.940 | 27.10 | 465.77 | 是 | -11.4 | 9 | 5 |
| 2017 | 7827.1 | 165.0 | 240.0 | 338.4 | 2.310 | 32.60 | 675.99 | 是 | +7.1 | 9 | 5 |
| 2018 | 8323.5 | 165.0 | 240.0 | 357.5 | 2.330 | 34.70 | 754.41 | 是 | -4.5 | 9 | 5 |
| 2019 | 7225.0 | 165.0 | 240.0 | 344.8 | 2.100 | 30.10 | 580.85 | 是 | -229.0 | 9 | 5 |
| 2020 | 6516.7 | 165.0 | 240.0 | 379.1 | 1.720 | 27.20 | 468.94 | 是 | -1.3 | 9 | 5 |
| 2021 | 8625.5 | 165.0 | 240.0 | 364.9 | 2.360 | 35.90 | 802.13 | 是 | -18.5 | 9 | 5 |
| 2022 | 8543.3 | 165.0 | 240.0 | 386.4 | 2.210 | 35.60 | 789.15 | 是 | -20.7 | 9 | 5 |
| 2023 | 9587.2 | 165.0 | 240.0 | 406.2 | 2.360 | 39.90 | 954.07 | 是 | +19.2 | 9 | 5 |

## 解释

- seed 5 达到四基准年度最高产量的年份：2017, 2023；低于年度最高基准的年份：2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022。
- seed 5 与 seed 4 的均产差为 -54.2 kg/ha、均灌溉差为 -49.5 mm、均 N 差为 +0.0 kg/ha。
- seed 5 与 seed 3 的均产差为 +1246.8 kg/ha、均灌溉差为 +90.0 mm、均 N 差为 +140.0 kg/ha。
- seed 5 在 10/10 年位于 yield↑、灌溉↓、N↓ 的五情景部分 Pareto 前沿；其 common reward 在与四基准的 40 个 seed-year 配对中胜过或持平 38/40 个配对。
- seed 5 的 WP_ET 在 22/40 个 PPO-基准配对中不低于对应基准，PFP_N 在 18/40 个配对中不低于对应基准。效率指标仍需结合产量与资源投入解释。
- canonical reward 属 PPO 自身训练奖励，不等同于 common reward；跨 seed 的 canonical mean 只作描述，不能据此覆盖农业指标权衡。
- 动作画像保存在逐年 CSV 的 action_signature、management_event_count、management_action_days_nonzero、irrigation_action_days、nitrogen_action_days 字段；事件以 DSSAT MgmtEvent.OUT 为准，动作天数沿用正式 step-level evaluation。

## 数据与可复核文件

- 回放汇总：`results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/replay_episode_summary.csv`（30 行，seed 5/3/4 × 2014–2023）。
- 与四基准逐年配对：`results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/seed345_validation_wp_et_comparison.csv`（120 行）。
- 快照目录：`results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/snapshots`；每个 seed-year 包含 Summary.OUT、PlantGro.OUT、SoilWat.OUT、Weather.OUT、MgmtEvent.OUT。
- 逐文件 SHA-256 与输入一致性：`results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/provenance.json`。
- 运行脚本：`src/run_004_07_02_yc_seed345_etcp_replay.py`；提示文件：`prompt_02/004_07_02_yc_seed5_focus_wp_et_seed3_seed4_reference_replay.md`。

## 可重复性摘要

- 结果行数：30；唯一 seed-year：30；快照数：30。
- 最终 provenance：`{"baseline_csv_sha256": "475F379FA259B860BBBA94480082F8B7B13400273E468D07EAD3B86D2D45056B", "baseline_scenarios": ["dssat_auto_external_n", "null", "official_extension_expert", "recorded_farmer_template"], "frozen_episode_csv_sha256": {"3": "3CEB83DBFFC09B61A09F52D0A0DFCE685110886281AF1030B769629070AE4514", "4": "E5BFC767C5EB3F60F0359FFADB890C4933DFCC7115584437347FD2EC130DB25D", "5": "3C1E5B1F83DDFAFD843A61BA609B82FE5B7AF847E45EA2D84888D1362F61C37F"}, "frozen_step_csv_sha256": {"3": "ED6E77CE847AC8375DA0CC8863EA02052D17EEFD96E6E37D1D7A20D0C71A11C3", "4": "5CEE1194D451282BFC199A23D4BDB40FDA57D93B9B5484EAC874D5E4D11F1DAB", "5": "25B45176C95A2873612D657A7409E2C0F4638F898D196AE51BF19BDAA0CE687A"}, "model_sha256": {"3": "78D3CF89E0FB0B0100E06583842A7FADBEE862F7C06AA896524A54D85CA6CDA9", "4": "6B7895714C6011E9359A2A0F6E656135AE8B671AE35D86C126A63458CA605CF7", "5": "814A5C73404D2B046927BA0ECA42DCE7F8F1D804B550A226C96B61CFDA97E2BF"}, "original_formal_manifest_status": "failed_after_30_episodes_due_to_postprocessing_NameError; retained", "original_smoke_manifest_status": "failed_due_to_validator_path_bug; retained", "pairwise_row_count": 120, "peak_rss_mb": 445.88671875, "prompt_sha256": "91085832D72D95A81D0C26A99791853C35B7C96849515404EAE47B47D617F577", "recovery_note": "Simulation completed once; post-run validation and finalization were repaired without repeating any DSSAT episode.", "result_row_count": 30, "rss_limit_mb": 4096.0, "script_sha256": "2D2294920DCF4F31B5FDC1B9EADF096C50FC1FA6606D4EAE7C1D33CE85B9D840", "snapshot_file_count": 150, "source_summary_metric_function": "src/build_relaxed_success_five_scenario_daily_evidence_027_05.py::summary_metrics"}`
- 未提交、未推送。
