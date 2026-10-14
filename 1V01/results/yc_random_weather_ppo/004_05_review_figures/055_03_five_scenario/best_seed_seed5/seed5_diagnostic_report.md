# YC seed 5 五情景逐年结果图

## 样式复用
沿用 `best_seed_seed3` 的五情景顺序、灰/红/金/紫/绿配色、线型、分组柱状图、年度指标折线、管理事件图、管理画像及 4×2 逐年过程布局。seed3 原文件没有被覆盖。seed5 有精确 ETCP 回放，因此 WP_ET 面板展示真实数值，不沿用 seed3 的缺失标注。

- 样式参考图：`results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/best_seed_seed3/seed3_metrics_vs_four_baselines.png`
- 逐年过程参考图：`results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/best_seed_seed3/figures/004_05_seed3_yca2014_five_scenario_daily.png`
- 参考脚本：`results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/plot_004_05_seed3_diagnostic.py`

## 数据口径
- seed5 十年 replay 和精确 ETCP/WP_ET/PFP_N：`results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/replay_episode_summary.csv`及其 seed5 年度 DSSAT snapshots。
- 四基准年度表：`benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_season_summary.csv`；逐日表：`benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_daily.csv`。
- 所有情景的对比 reward 使用 `0.158×yield − 1.1×irrigation − 1.58×nitrogen`，不与 canonical reward 混算。无 PPO 训练或新增 DSSAT 运行。

## 图上结果摘要
- seed5 十年均值：产量 8148.7 kg/ha；灌溉 165.0 mm；施氮 240.0 kg N/ha；WP_ET 2.249 kg/m³；PFP_N 33.96 kg grain/kg N；common reward 726.8。
- 产量超过当年四基准最高值的年份：2017, 2023（2/10）。五情景年度冠军数：产量 2/10、WP_ET 0/10、PFP_N 8/10、common reward 8/10。
- 五情景最低投入年份数：灌溉 0/10，氮 0/10。管理画像均值：灌溉 165.0 mm/年、施氮 240.0 kg N/ha/年、施用日 5.0 天/年、灌溉/施氮事件 9.0 次/年。
- 这些是选定单一 checkpoint/seed 对 2014–2023 已观察验证年的结果，不构成独立年份上的确认性泛化结论。

## 文件
- `figures/004_05_seed5_five_scenario_metrics.png`：逐年产量、WP_ET、PFP_N 五情景柱状图。
- `figures/004_05_seed5_five_scenario_management.png`：逐年灌溉、施氮五情景柱状图。
- `figures/004_05_seed5_five_scenario_reward.png`：逐年 common reward 五情景柱状图。
- `figures/004_05_seed5_yca{year}_five_scenario_daily.png`：十个验证年的五情景逐日过程图。
- 根目录另有 `seed5_metrics_vs_four_baselines.png`、`seed5_management_actions_by_year.png`、`seed5_yearly_management_profile.png`。
