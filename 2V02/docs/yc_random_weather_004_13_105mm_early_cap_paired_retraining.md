# YC 004_13：105 mm early-cap 配对重训练实验

## 实验边界

- Treatment 为 DAP≤30 累计灌溉 cap=105 mm，seed 0–7 各从头训练 100K requested steps；Control 为 004_05 canonical 75 mm seed 0–7 原模型/正式结果，不重训。
- 除 early cap 外，WGEN schedule、observation、reward、动作网格、PPO、mid cap=150 mm、late/pre-DAP90=195 mm、I/N 7-day 间隔、季节 I240/N250、daily limits、DSSAT/runtime、evaluation protocol 均冻结。未修改旧配置、wrapper 或 004_05 结果；无 commit/push。
- 执行记录：attempt_01–04 的 smoke checkpoint、评估产物和失败日志均保留，均未启动正式训练。attempt_01/02 暴露 episode-key join 错误；attempt_02 隔离诊断确认 I/N action trace 闭合。attempt_03 的输入哈希 gate 发现 seeds 0–2 的 control 输入归档在 004_03；attempt_04 又确认 004_05 episode 汇总表未保留 held-out runtime weather hash，但其逐日 trace 有可重算且 reconciled 的 hash。attempt_05 改用 004_05 step-level weather hash 作权威对照，并按 seed manifest 回溯 control inputs/checkpoints。相关日志见同级 attempt_01_smoke.log 至 attempt_04_smoke.log。
- 正式运行审计：8 个 treatment 模型均从头完成 100080 actual steps；评估过程中的多次后处理 gate 错误均保留在原日志和 evaluation 子目录，修复后从既有 checkpoint 继续、未重训。DSSAT MgmtEvent.OUT 会累积多个 rollout block，最终提取与 Summary.OUT yield 匹配的 block；PFP_N 在 N=0 时按定义为 N/A。seed 3–7 的 observed inputs 使用 004_05/evaluation/runs 归档；W3–W7 历史 held-out hash 缺失时，仅用 W0–W2 对相同 WGEN weather seed 的共同 reconciled hash 作参照。旧正式 CSV 的投入列使用 total_irrigation/total_fertilizer。最终 treatment metrics 位于 evaluation_final（seed0–2）和 evaluation_final_retry_03（seed3–7）；control exact replay 位于 control_replay_final。
- Smoke：True；训练完成 seeds=[0, 1, 2, 3, 4, 5, 6, 7]；峰值 RSS=931.9 MiB，硬上限 2048 MiB。004_05 control evaluation replay exact episodes=240 / 240。
- 75 mm control 仍以 004_05 正式 CSV/action trace 为权威。只在逐 episode yield/I/N/reward、动作 trace 和 runtime weather hash 均复现时，才将 evaluation-only Summary.OUT 补出的 WP_ET/PFP_N 作为可比值；不匹配项的 control WP_ET/PFP_N 保留为空。

## 配对总体结果

“Pareto 支配”按 seed×评估域的平均三目标定义：Yield 不低、灌溉和 N 不高且至少一项严格更优。它不是事后加权总分；canonical/common reward 作为单独指标列示。下表的 ± 为 8 个 seed 间 SD。

| Domain | Group | Yield kg/ha | Irrigation mm | N kg/ha | WP_ET | PFP_N | Common reward | Canonical reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Held-out WGEN 1081–1100 | control_75 | 6747.2 ± 846.1 | 89.2 | 121.8 | 2.044 | 80.57 | 775.513 | 0.829 |
| Observed 2014–2023 | control_75 | 6328.5 ± 1559.7 | 88.7 | 121.0 | 1.866 | 71.08 | 711.168 | 0.457 |
| Held-out WGEN 1081–1100 | treatment_105 | 6241.6 ± 1042.2 | 96.8 | 83.5 | 1.888 | 95.37 | 747.715 | 0.800 |
| Observed 2014–2023 | treatment_105 | 5578.9 ± 1991.3 | 96.2 | 81.0 | 1.643 | 77.91 | 647.683 | 0.395 |

## 8 个 paired seeds

- Observed：105 mm 支配 75 mm：0/8；被 75 mm 支配：0/8；三目标互不支配/折中：8/8；同时增产且降低 N：0/8。
- Held-out WGEN：105 mm 支配 75 mm：0/8；被 75 mm 支配：1/8；三目标互不支配/折中：7/8；同时增产且降低 N：0/8。
- 单 seed、逐域的 delta 和 Pareto 标记见 paired_comparison.csv；不得仅凭均值或单个最好 seed 宣称胜出。
- 逐 episode 配对：Observed episode Pareto=1/80、增产且降 N=1/80；Held-out WGEN episode Pareto=3/160、增产且降 N=1/160。
- Observed 年份级 treatment/control 对照和共同 held-out WGEN seed 对照见 paired_episode_comparison.csv，可追溯到每个 paired PPO seed。

## 策略 archetype

- Observed very-low-input seed 数：75 mm=3/8，105 mm=5/8。
- Observed yield-collapse seed 数：75 mm=3/8，105 mm=5/8。
- Observed fixed N=240 seed 数：75 mm=2/8，105 mm=2/8。
- Held-out WGEN very-low-input seed 数：75 mm=3/8，105 mm=5/8。
- Held-out WGEN yield-collapse seed 数：75 mm=0/8，105 mm=0/8。
- Held-out WGEN fixed N=240 seed 数：75 mm=2/8，105 mm=2/8。

## 结论

- 105 mm 未显示整体改善：Observed 平均 Δ（105−75）为 Yield **−749.6 kg/ha**、灌溉 **+7.5 mm**、N **−40.0 kg/ha**、WP_ET **−0.224**、common reward **−63.485**、canonical reward **−0.062**；Held-out WGEN 对应为 **−505.6 kg/ha**、**+7.6 mm**、**−38.3 kg/ha**、**−0.157**、**−27.797**、**−0.030**。两域方向一致：省 N 但平均减产、略增灌溉、WP_ET/reward 下降。
- Paired seed 层面：yield 提高 Observed **3/8**、WGEN **2/8**；N 降低两域均 **4/8**；灌溉降低两域均 **3/8**；没有 seed 同时增产且减 N。三目标下 105 mm 支配数为 Observed **0/8**、WGEN **0/8**；WGEN 中 75 mm 支配 105 mm **1/8**。
- PFP_N 的有效 seed 组均值差为 Observed **+6.83**、WGEN **+14.81**，但两组有效 seed 组成不完全相同；仅看 **5/8** 个两侧均有效的 paired seeds，差值均值分别为 Observed **+0.78**、WGEN **+7.68**。总表按有效 seed 的等权均值；单 seed 内先对 N>0 episode 求均值。N=0 时 PFP_N 为 N/A；有效 episode 行数由 75 mm 的 **70/80、140/160** 降为 105 mm 的 **51/80、107/160**，有效 seed 均值数由 75 mm **7/8** 降为 105 mm **6/8**。N=0 episode 增至 Observed **29/80**（75 mm=10/80）、WGEN **53/160**（75 mm=20/160），故不可把 PFP_N 上升单独解读为综合效率胜出。
- 结构上，Observed very-low-input 与 yield-collapse 均由 75 mm 的 **3/8** 增至 105 mm 的 **5/8**；fixed N=240 均为 **2/8**，没有减少。当前证据更支持 105 mm 加剧低投入/减产倾向，而非稳定改善综合策略。
- 这是固定训练预算下的单次 8-seed mask treatment 比较。2014–2023 与 held-out WGEN 是 004_05 的既定 evaluation domains，不等于新增独立外部验证；实验也不自动批准将 105 mm 升级为 canonical mask。

## 可复现证据

- Smoke gate: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/smoke/smoke_manifest.json
- 运行总 manifest: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/formal_manifest.json
- Episode-level 数据: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/summaries/episode_metrics_all.csv
- 105 mm final evaluation: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/evaluation_final/treatment_105（seed 0–2），results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/evaluation_final_retry_03/treatment_105（seed 3–7）
- 75 mm exact replay: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/control_replay_final/control_75
- Seed-domain 汇总: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/summaries/seed_domain_summary.csv
- Paired comparison: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/summaries/paired_comparison.csv
- Paired episode/year-weather comparison: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/summaries/paired_episode_comparison.csv
- Archetypes: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/summaries/seed_archetypes.csv
- Checkpoint/config/input provenance: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/provenance

## 图表

- results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/figures/figure_seed_paired_metric_deltas.png
- results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/figures/figure_seed_paired_pareto_tradeoff.png
- results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/figures/figure_overall_domain_summary.png

未提交、未推送。
