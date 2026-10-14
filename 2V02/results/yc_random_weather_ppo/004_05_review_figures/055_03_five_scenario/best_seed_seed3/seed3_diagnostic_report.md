# 004_05 Random-weather PPO：seed 3 单 seed 诊断

## 选择规则与边界

本图组展示 004_05 的 Random-weather PPO seed 3，并与 055_03 原图中的四个基准对比。选择规则是：分别计算每个 PPO seed 在 `observed_weather` 与 `heldout_wgen` 的正式 canonical reward 均值，再对两个评估域等权平均；seed 3 得分最高（0.763）。该选择使用了评估结果，是**事后描述性筛选**，会产生选择偏差，不能作为独立验证或多 seed 总体胜出的证据。原实验 8-seed 的配对成功门槛仍为 0/8。

seed 3 在 observed 2014–2023 的平均 canonical reward 为 0.572；平均产量 6901.9 kg/ha，灌溉 75.0 mm，施氮 100.0 kg N/ha。seed 3 的 055_03 可比统一奖励均值为 850.0，产量只高于四基准中的 DSSAT auto + external N，低于官方专家与 recorded template，因此按五组排第三。两种 reward 不能混为一谈：seed 选择用 004_05 canonical reward；跨基准图中 common reward 使用 `0.158 × yield - 1.1 × irrigation - 1.58 × nitrogen`。前者含 stress relief 和 SWFAC guardrail 等分量，后者明确奖励节省投入，所以 reward 排名高不意味着产量最高。

seed 3 的平均 `PFP_N` 为 72.67 kg grain/kg N，按逐季 yield/N 计算。`WP_ET` 未绘制：004_05 seed 3 没有与评估逐季结果对应的 ETCP replay，不能从生物量、灌溉或其他指标推算替代。

## 图与数据

- 完整复刻版结果图位于 `figures/`，共 13 张：`004_05_seed3_yca2014_five_scenario_daily.png` 至 `...yca2023...` 十张逐年日过程图，以及 `004_05_seed3_five_scenario_metrics.png`、`...management.png`、`...reward.png` 三张跨年汇总图。它们采用 055_03 原套图的 16:13 日图画布、四行两列布局、五情景颜色/线型、分面、legend 与分组柱图结构。
- `figures/004_05_seed3_five_scenario_metrics.png`：逐年产量、WP_ET、PFP_N；PPO seed 3 的 WP_ET 空缺并明确标注原因。
- `figures/004_05_seed3_five_scenario_management.png` 与 `...reward.png`：逐年水氮投入及 055_03 common reward，与原系列汇总图对应。
- `figures/004_05_seed3_yca{year}_five_scenario_daily.png`：逐年天气、水分状态/胁迫、氮胁迫、实际措施、grain/biomass 和 common reward 过程对比。
- 额外的单 seed 快速诊断图仍保留在本目录根部：`seed3_metrics_vs_four_baselines.png`、`seed3_yearly_management_profile.png`、`seed3_management_actions_by_year.png`。
- `seed3_yearly_metrics_vs_four_baselines.csv`、`seed3_observed_management_events.csv`：逐年指标和真实动作明细。
- `seed_selection_equal_domain_reward_ranking.csv`、`seed3_diagnostic_manifest.json`：选择排序、数据来源与方法清单。

逐年指标源：`results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv`；动作与 seed 3 日状态源：`results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv`；四基准逐日/季节源：`benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_daily.csv` 与 `...season_summary.csv`。只读已有正式结果，没有重训或重跑 DSSAT。

日过程图中 seed 3 的 SWFAC、NSTRES、施用日期/剂量和 common reward 来自 seed 3 trace。004_05 没有保存 seed 3 的逐日 soil-water、grain/biomass 状态，因此这些面板只显示四基准，并在图内标注 PPO 状态缺失；seed 3 的终产量仍按正式 episode 汇总参加逐年产量图。

## 管理画像与解释

十年有 35 个正动作日记录，其中 20 次灌溉、25 次施氮（10 天水氮同日施用）；每年总灌溉固定为 75 mm，施氮为 80 或 120 kg/ha。主要动作时点重复出现在 DAP 0、8、9、16 附近，3–4 个正动作日/季。与 004_05 八个 seed 的 observed 平均 3.94 个正动作日/季相比，seed 3 略低；均值受 seed 4 的 16.3 日/季高频策略拉高，而 seed 0、1、6、7 都只有 1 日/季。把这种差异压成均值线，会得到不存在于任何单一 seed 的“平均策略”。

但 seed 3 的动作日期与灌溉量在 10 年间高度固定，且 N 仅在两档间变化。较低投入与较高 common reward 并未带来最高产量：其均产仍为五组第三，只有 1/10 年产量超过当年最好的单一基准。这是一个相对高 reward、低投入的代表 seed，不是综合表现胜出的 PPO，也不能代表 Random-weather PPO 的总体表现。
