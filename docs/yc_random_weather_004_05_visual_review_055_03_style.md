# YC 004_05 按 055_03 系列重制图实验记录

> **对照范围更正：**此版仅比较两种 PPO。用户指定的是 Random-weather PPO 与四个既有基准情景的五情景图；最终结果请见[五情景报告](yc_random_weather_004_05_five_scenario_055_03_review.md)。本文件保留为此前双 PPO 图记录。

## 样式来源确认

| 用途 | 最终采用的旧文件 |
|---|---|
| 逐年日过程图 | `benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca2014_five_scenario_daily.png`，并核对同目录 2015-2023 图 |
| 三层指标柱图 | 同目录 `055_03_yca_five_scenario_metrics.png` |
| 两层管理柱图 | 同目录 `055_03_yca_five_scenario_management.png` |
| 原生成脚本 | `src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py` 的 `plot_year`、`draw_grouped`、`COLORS`、`STYLES` |
| 原文档 | `docs/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_five_scenario_figures_ckpt100000_record.md` |

```yaml
style_source_daily: benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca2014_five_scenario_daily.png
style_source_metrics: benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca_five_scenario_metrics.png
style_source_management: benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca_five_scenario_management.png
style_source_script: src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py
style_source_palette: "Historical PPO #2A9D55 (旧 PPO 绿); Random-weather PPO #7E63B6 (旧紫); negative difference #C44E52; rain #80A9C7"
```

旧系列是每年 4×2 面板、16×13 英寸、220 dpi 的日过程图，另有 16 英寸宽、每层 4 英寸高、200 dpi 的逐年分组柱图。新图直接沿用这些布局、白底浅网格、细线/虚线、圆点事件 stem、图例位置和字体层级。不是此前误用的 `222_yc_v1_three_seed_confirmation` 风格。旧系列仅交付 PNG，本次也只导出 PNG。

## 数据与转换

只读取 004_05 正式的 [episode 表](../results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv) 和 [step 表](../results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv)，筛选 `observed_weather` 的 2014-2023：两种训练方式 × 八个配对 seed × 十年，共 160 个 episode、16130 个 step。绘图脚本为 [plot_004_05_055_03_style.py](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/plot_004_05_055_03_style.py)。它逐 episode 检查 step 的奖励、灌溉、施氮求和与正式 episode 值一致；可用的 grain 终点也与正式 yield 完全一致。没有重训、重评估、DSSAT replay 或模型改动。

旧图的五情景基线来自 055_02/055_01 和另一份 PPO 检查点；不是 004_05 配对实验的共同评估合同。本次只画 Historical PPO 与 Random-weather PPO，不能把旧 Null、Recorded template、DSSAT auto、Official expert 柱子拼进来。绿色沿用旧 PPO 色，紫色借用旧调色板作为另一种 PPO，图例重新明确命名。

日图保留旧 4×2 的八类内容，但使用本轮已有变量：天气、土壤水分、水/氮压力、灌溉/施氮事件、grain/biomass、累积奖励。`swfac` 和 `nstres` 是本轮 trace 状态，不伪装成旧图的 `WSPD`/`NSTD`；土壤水分是 `post_step_state_json.sw` 的顶层体积含水率，不伪装成旧图 `SWTD` 毫米。天气取共同 observed-weather trace，已核对完整 seed 0-2 的日天气逐项一致。日过程 x 轴采用 `timestep-5`：正日数与 DAP 一致，负日数表示播前 4 日，避免五个 DAP=0 步重叠。事件 stem 是八 seed 每日平均施用量，不代表单一 seed 的事件剂量。累积奖励由正式 `instant_reward` 累加，不使用旧图的另一套 `unified_reward` 公式。

**缺失限制：** seed 0-2 两组有完整天气、`grnwt`、`topwt` 和 `post_step_state_json`；seed 3-7 两组缺这些日级字段。因此日图的土壤水分和 grain/biomass 面板仅用 seed 0-2，标题明确标注；水/氮压力、事件和累积奖励使用全部八 seed。逐年指标/管理柱图始终使用八 seed 的正式 episode 表，不能把日图的三 seed 轨迹误读成八 seed 平均。旧图 `WP_ET` 需要准确 ET/ETCP 依据，本次未画。第三层指标改为 pooled `PFP_N = 八 seed 总 yield / 八 seed 总 N`，不是逐 seed PFP_N 平均；个别 N=0 episode 没有被单独赋予有限 PFP_N。

## 新图

全部输出位于 [055_03_style](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/)。

| 图 | 文件 | 数据及可比范围 |
|---|---|---|
| 十年日过程（逐年策略） | `004_05_yca2014_two_ppo_daily.png` 至 `004_05_yca2023_two_ppo_daily.png` | step 表；上述面板按可用 seed 数分别标注 |
| 指标对比 | [004_05_two_ppo_metrics.png](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_two_ppo_metrics.png) | episode 表；yield、reward 为八 seed 均值 ± seed SD，PFP_N 为 pooled ratio |
| 管理资源 | [004_05_two_ppo_management.png](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_two_ppo_management.png) | episode 表；逐年灌溉、施氮八 seed 均值 ± seed SD |
| 施用次数 | [004_05_two_ppo_management_events.png](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_two_ppo_management_events.png) | episode 表；逐年灌溉/施氮次数均值，延续旧管理图两层布局 |
| 逐年配对差值 | [004_05_two_ppo_paired_by_year.png](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_two_ppo_paired_by_year.png) | 同 seed、同年 Random - Historical；柱为八配对均值、点为八 seed 差值，零线突出 |
| seed 稳健性 | [004_05_two_ppo_seed_robustness.png](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_two_ppo_seed_robustness.png) | episode 表；每个 seed 跨十年均值，沿用旧逐年分组柱布局 |

校核表：[逐年组均值](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_observed_year_summary.csv)、[80 对 seed-year 差值](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_paired_seed_year.csv)、[逐年配对摘要](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_paired_by_year.csv)、[逐 seed 均值](../results/yc_random_weather_ppo/004_05_review_figures/055_03_style/004_05_observed_seed_means.csv)。

## 从图直接可见

- 八 seed 组均值的 reward 和 yield：Random-weather 只在 **2014、2018** 高于 Historical；其余 **2015-2017、2019-2023** 更低。2014 的 reward 正差值约 `+0.039`、yield `+392` kg/ha；2018 约 `+0.015`、`+97` kg/ha。2014 reward 正差值仅 5/8 配对，yield 仅 3/8；2018 两者均为 5/8。年度均值微弱转正，不等于稳定胜出。
- 十年八 seed 平均：Historical / Random-weather 的 reward 为 `0.5239 / 0.4571`，yield 为 `6575.7 / 6328.5` kg/ha，灌溉为 `100.1 / 88.7` mm，施氮为 `110 / 121` kg/ha。Random-weather 整体倾向**少灌溉、少灌溉事件、略多施氮、低 yield 和 reward**；个体 seed 方向差别显著，不能描述为统一的策略迁移。
- 配对 seed 的 observed 十年平均 reward：Random-weather 局部胜出的为 **seed 1、4、5**；其他 seed 不胜出。seed 4/5 更高的 yield 同时伴随更高灌溉和 N；不能按单一 reward 柱称为资源综合胜利。004_05 预定的跨 observed/held-out 联合成功条件为 **0/8**，本图没有“这轮整体胜出”的视觉证据。
- 日图允许查看天气和策略时间位置，但 seed 0-2 的 grain/biomass 轨迹不能用来代替八 seed 的最终均值。对策略类型的正式标签仍以 [policy_archetype_assignment.csv](../results/yc_random_weather_ppo/004_05/policy_archetype_assignment.csv) 为准，不能由图形颜色重新命名。

## 保留与未完成

此前按 222 系列绘制的 [旧版目录](../results/yc_random_weather_ppo/004_05_review_figures/) 和原报告仅作历史尝试，已被本版替代，不删改原始结果。未补做 seed 3-7 的完整土壤水分/产量日轨迹、精确 WP_ET/NUE，也未混入旧五情景；这些都不能从现有 004_05 正式文件无歧义得出。没有 Git push。
