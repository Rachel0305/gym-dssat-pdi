# YC 75 mm PPO 八个 seed 五情景图表记录

## 目的与范围

为 004_05 中已训练完成的 Random-weather PPO seeds 0–7 分别制作同规格结果图。每一套图都将单个固定 seed 与同一批四个正式基准情景逐年对照，覆盖 observed weather 2014–2023；没有逐年切换 PPO seed，也没有把不同 seed 拼成一个策略。

本任务只读取现有正式结果，没有训练 PPO、没有运行 DSSAT evaluation/replay，也没有修改任何 canonical 配置或旧结果。

## 样式来源

| 图表元素 | 来源 |
|---|---|
| 五情景柱图与颜色顺序 | `benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca_five_scenario_metrics.png` |
| 原始绘图参数和布局 | `src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py` 的 `draw_grouped`、`COLORS`、`STYLES` |
| seed 专属图的前序版本 | `results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/best_seed_seed3/` 与 `best_seed_seed5/` |

沿用原系列的五情景顺序、灰/红/金/紫/绿配色、按验证年分组的柱图、浅网格和图例样式。管理时序图沿用 seed5 版本的 DAP 横轴及灌溉/施氮不同 marker 语言。由于八个 seed 没有齐全的精确 DSSAT 日过程快照，本次不制作会造成数据覆盖不一致的 4×2 日过程图；所有 seed 都有同一套逐年指标和动作图。

## 数据与校核

- PPO 逐年指标：`results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv`，随机天气训练的 8 seeds × 10 observed 验证年，共 80 个 episode。
- PPO 措施时间与剂量：`results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv`，按行流式读取，仅提取 80 个 observed seed-year episode 的正灌溉/施氮动作。
- 四个基准的逐年指标及动作：`benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/` 下的 `055_03_yca_five_scenario_season_summary.csv` 和 `055_03_yca_five_scenario_management_events.csv`。
- 精确 PPO `WP_ET`：`results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/replay_episode_summary.csv` 中 seed 3、4、5 的 `Summary.OUT`/ETCP 回放；共 30 个 seed-year，与 004_05 episode 的产量、水、氮闭合。
- seed 0、1、2、6、7 没有精确 ETCP 回放，PPO `WP_ET` 明确为 N/A；没有从其他 seed 或总量推算。
- PPO `PFP_N = yield / total N`；N=0 时记为 N/A。四基准采用 055_03 正式表中值。
- 跨情景 reward 图使用同一共同公式 `0.158 × yield − 1.1 × irrigation − 1.58 × N`。PPO canonical reward 使用单独面板展示，不与基准 common reward 混为同一标度。
- PPO 管理动作 trace 与 episode 汇总的逐年灌溉量、施氮量及灌溉/施氮事件数全部闭合：80/80 episode；水氮总量最大绝对差均为 0。
- 指标表覆盖 8 seeds × 10 years × 5 scenarios，共 400 行。

## 每个 seed 的图

每个 `seed_0` 至 `seed_7` 子目录包含相同六张 PNG：

1. `seedN_five_scenario_metrics.png`：逐年产量、WP_ET、PFP_N。
2. `seedN_five_scenario_management.png`：逐年灌溉总量、施氮总量。
3. `seedN_five_scenario_reward.png`：共同 reward 五情景柱图，以及单独的 PPO canonical reward 曲线。
4. `seedN_five_scenario_yearly_trends.png`：产量、共同 reward、灌溉、施氮、PFP_N 的五情景逐年曲线。
5. `seedN_five_scenario_action_calendar.png`：五情景逐年灌溉/施氮 DAP 时间分布。
6. `seedN_management_action_doses_by_year.png`：该 PPO seed 每年实际灌溉和施氮 DAP、剂量。

## Seed 均值速览

以下均值仅用于快速定位各 seed 的取舍，不能代替逐年图，也不据此宣布唯一赢家。`WP_ET` 仅对有精确回放的 seed 提供。

| Seed | Yield kg/ha | Irrigation mm | N kg/ha | WP_ET kg/m³ | PFP_N | Common reward | Canonical reward | 平均管理动作日/季 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5348 | 45.0 | 40 | N/A | 133.7 | 732.3 | 0.331 | 1.0 |
| 1 | 6304 | 45.0 | 120 | N/A | 52.5 | 756.9 | 0.351 | 1.0 |
| 2 | 6822 | 75.0 | 188 | N/A | 36.7 | 698.3 | 0.423 | 2.7 |
| 3 | 6902 | 75.0 | 100 | 2.033 | 72.7 | 850.0 | 0.572 | 3.5 |
| 4 | 8203 | 214.5 | 240 | 2.224 | 34.2 | 680.9 | 0.749 | 16.3 |
| 5 | 8149 | 165.0 | 240 | 2.249 | 34.0 | 726.8 | 0.771 | 5.0 |
| 6 | 3552 | 45.0 | 0 | N/A | N/A | 511.8 | 0.128 | 1.0 |
| 7 | 5348 | 45.0 | 40 | N/A | 133.7 | 732.3 | 0.331 | 1.0 |

挑选 seed 属于对已看过的验证年结果进行事后筛选，因此这里的图表适合候选筛查，不构成独立确认性泛化证据。需要精确 WP_ET 的 seed 0、1、2、6、7 时，应单独审批并规划 evaluation-only ETCP replay；本任务没有触发该计算。

## 产物

- 图表及 CSV：`results/yc_random_weather_ppo/004_15_75mm_all_seed_five_scenario_figures/`
- 可复现脚本：`src/plot_004_15_yc_75mm_all_seed_five_scenario.py`
- 运行清单：`results/yc_random_weather_ppo/004_15_75mm_all_seed_five_scenario_figures/manifest.json`
- 每 seed 汇总：`results/yc_random_weather_ppo/004_15_75mm_all_seed_five_scenario_figures/seed_screening_summary.csv`
- 动作闭合：`results/yc_random_weather_ppo/004_15_75mm_all_seed_five_scenario_figures/ppo_action_trace_closure.csv`
