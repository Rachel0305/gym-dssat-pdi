# YC 004_05 weather augmentation 视觉审查

> **样式与对照范围已更正：**本次有效结果是新 Random-weather PPO 对 Null、Recorded template、DSSAT auto、Official expert 四基准的五情景图，详见[五情景对照报告](yc_random_weather_004_05_five_scenario_055_03_review.md)。下文与[双 PPO 报告](yc_random_weather_004_05_visual_review_055_03_style.md)保留为先前尝试记录。修改前副本保存在 `backups/`。

## 样式来源

| 作用 | 旧图或脚本 |
|---|---|
| 逐年稳定性 | [figure4_100k_by_year_stability.png](../experiments/222_yc_v1_three_seed_confirmation/figures/figure4_100k_by_year_stability.png) |
| seed 稳健性 | [figure3_100k_seed_robustness.png](../experiments/222_yc_v1_three_seed_confirmation/figures/figure3_100k_seed_robustness.png) |
| 管理资源对比 | [figure2_100k_multiobjective_comparison.png](../experiments/222_yc_v1_three_seed_confirmation/figures/figure2_100k_multiobjective_comparison.png) |
| 单栏线图 | [figure1_checkpoint_yield_stability.png](../experiments/222_yc_v1_three_seed_confirmation/figures/figure1_checkpoint_yield_stability.png) |
| 原绘图代码 | [compile_confirmation.py](../experiments/222_yc_v1_three_seed_confirmation/scripts/compile_confirmation.py) |
| 原文档引用 | [yc_v1_three_seed_confirmation.md](../docs/yc_v1_three_seed_confirmation.md) |

```yaml
style_source_figure_by_year: experiments/222_yc_v1_three_seed_confirmation/figures/figure4_100k_by_year_stability.png
style_source_figure_seed: experiments/222_yc_v1_three_seed_confirmation/figures/figure3_100k_seed_robustness.png
style_source_figure_management: experiments/222_yc_v1_three_seed_confirmation/figures/figure2_100k_multiobjective_comparison.png
style_source_script: experiments/222_yc_v1_three_seed_confirmation/scripts/compile_confirmation.py
style_source_palette: "Historical #3366CC; Random-weather #D55E00; reference #777777"
```

直接复用了旧脚本的 `font.size=9`、`axes.titlesize=10`、白底、180 dpi、蓝橙配色、圆点、网格透明度 0.2、无边框 legend、`6.8 × 4.2` 单栏线图、`15 × 3.7` seed 并列柱图及 `12 × 6.6` 多面板图。旧套图只有 PNG，故按旧交付格式输出 PNG。`figureA` 和 `figureD` 继承旧 by-year 图；`figureC` 继承旧 seed robustness；`figureB` 沿用旧线图并增加清晰的零线。

## 数据与必要调整

全部绘图只读 004_05 的正式 episode 表、paired-seed 表和 archetype 表：[all_evaluation_episode_level_0_7.csv](../results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv)、[paired_seed_performance_comparison.csv](../results/yc_random_weather_ppo/004_05/paired_seed_performance_comparison.csv)、[policy_archetype_assignment.csv](../results/yc_random_weather_ppo/004_05/policy_archetype_assignment.csv)。训练、评估和 probe QC 均为通过；逐年对照严格筛选 `observed_weather`，共 2 组 × 8 seed × 10 年 = 160 行，每一配对只有一行。held-out WGEN 为 320 行，天气种子 1081–1100。episode 均值与 paired-seed 正式表逐项核对一致。

旧 `figure4` 为五指标 `2 × 3` 布局；本轮核心指标为 Yield、irrigation、N 和 reward，因此 `figureA`/`figureB` 用 `2 × 2` 汇总布局，另输出四张单指标图。为展示八个 seed，`figureA`/`figureD` 在组均值粗线下增加同色浅细单 seed 线；`figureB` 复用旧逐年图的均值线，配对明细另存 CSV，以便零线附近的小差值可见。旧 `figure3` 的五栏改为四栏。`figureD` 保留旧 `2 × 3` 结构：四个逐年管理面板、一个 archetype 频数面板，末格留白。图中 Historical 与 Random-weather 分别对应旧图的 Original 蓝与 Augmented 橙；旧图的 Expert/Farmer 不属于这轮配对实验，故未加入。N 为零的 episode 有 10 行，本轮未计算 PFP_N；没有可直接复用的 ETCP 精确回放，因此未绘制 WP_ET/NUE。

## 图与数据源

| 图                                                                                                                                             | 内容                       | 数据                                                                                           |
|:----------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------|:---------------------------------------------------------------------------------------------|
| [figureA_by_year_yield.png](../results/yc_random_weather_ppo/004_05_review_figures/figureA_by_year_yield.png)                                 | 逐年指标与 seed 线             | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureA_by_year_irrigation.png](../results/yc_random_weather_ppo/004_05_review_figures/figureA_by_year_irrigation.png)                       | 逐年指标与 seed 线             | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureA_by_year_nitrogen.png](../results/yc_random_weather_ppo/004_05_review_figures/figureA_by_year_nitrogen.png)                           | 逐年指标与 seed 线             | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureA_by_year_reward.png](../results/yc_random_weather_ppo/004_05_review_figures/figureA_by_year_reward.png)                               | 逐年指标与 seed 线             | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureA_by_year_stability.png](../results/yc_random_weather_ppo/004_05_review_figures/figureA_by_year_stability.png)                         | 逐年指标与 seed 线             | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureB_by_year_yield_difference.png](../results/yc_random_weather_ppo/004_05_review_figures/figureB_by_year_yield_difference.png)           | 逐年配对差值均值与零线              | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureB_by_year_irrigation_difference.png](../results/yc_random_weather_ppo/004_05_review_figures/figureB_by_year_irrigation_difference.png) | 逐年配对差值均值与零线              | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureB_by_year_nitrogen_difference.png](../results/yc_random_weather_ppo/004_05_review_figures/figureB_by_year_nitrogen_difference.png)     | 逐年配对差值均值与零线              | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureB_by_year_reward_difference.png](../results/yc_random_weather_ppo/004_05_review_figures/figureB_by_year_reward_difference.png)         | 逐年配对差值均值与零线              | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureB_by_year_paired_difference.png](../results/yc_random_weather_ppo/004_05_review_figures/figureB_by_year_paired_difference.png)         | 逐年配对差值均值与零线              | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureC_seed_robustness.png](../results/yc_random_weather_ppo/004_05_review_figures/figureC_seed_robustness.png)                             | 8 个配对 seed 的四指标柱图        | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather                                 |
| [figureC_seed_robustness_heldout_wgen.png](../results/yc_random_weather_ppo/004_05_review_figures/figureC_seed_robustness_heldout_wgen.png)   | 8 个配对 seed 的四指标柱图        | 004_05 all_evaluation_episode_level_0_7.csv；heldout_wgen                                     |
| [figureD_management_strategy_profile.png](../results/yc_random_weather_ppo/004_05_review_figures/figureD_management_strategy_profile.png)     | 逐年管理次数、资源量与 archetype 频数 | 004_05 all_evaluation_episode_level_0_7.csv；observed_weather；policy_archetype_assignment.csv |

汇总数据：[observed_by_year_group_summary.csv](../results/yc_random_weather_ppo/004_05_review_figures/observed_by_year_group_summary.csv)、[observed_by_year_paired_difference.csv](../results/yc_random_weather_ppo/004_05_review_figures/observed_by_year_paired_difference.csv)、[observed_seed_management_summary.csv](../results/yc_random_weather_ppo/004_05_review_figures/observed_seed_management_summary.csv)；80 条原始配对差值为 [observed_seed_year_paired_difference.csv](../results/yc_random_weather_ppo/004_05_review_figures/observed_seed_year_paired_difference.csv)。差值均为同一 seed、同一年 `Random-weather - Historical`，`figureB` 画八个差值的均值；各年正差值 seed 数和标准差可从汇总表核查。`figureC_seed_robustness.png` 为 observed 2014–2023，`figureC_seed_robustness_heldout_wgen.png` 为 held-out WGEN 1081–1100。

## 逐年直接观察

正的 reward/yield 差值表示 random-weather 的该指标更高；负的 irrigation/N 差值仅表示投入更少，不单独构成综合胜出。

|   Year |   Reward Δ | Reward + pairs   |   Yield Δ | Yield + pairs   |   Irrigation Δ (mm) |   N Δ (kg/ha) |
|-------:|-----------:|:-----------------|----------:|:----------------|--------------------:|--------------:|
|   2014 |      0.039 | 5/8              |     392.2 | 3/8             |               -11.2 |            15 |
|   2015 |     -0.149 | 3/8              |    -407.6 | 4/8             |               -13.1 |            15 |
|   2016 |     -0.106 | 2/8              |    -316.6 | 3/8             |               -11.2 |            15 |
|   2017 |     -0.065 | 2/8              |    -389   | 3/8             |               -11.2 |            10 |
|   2018 |      0.015 | 5/8              |      96.9 | 5/8             |                -7.5 |            15 |
|   2019 |     -0.071 | 4/8              |    -104.9 | 3/8             |               -11.2 |             5 |
|   2020 |     -0.068 | 2/8              |    -343.4 | 4/8             |               -13.1 |            15 |
|   2021 |     -0.069 | 3/8              |    -429.6 | 4/8             |               -13.1 |            10 |
|   2022 |     -0.064 | 3/8              |    -435   | 4/8             |               -11.2 |             5 |
|   2023 |     -0.129 | 3/8              |    -535   | 4/8             |               -11.2 |             5 |

按八 seed 均值，random-weather 的 reward 更高年份：**2014, 2018**；更低年份：**2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023**。Yield 更高年份：**2014, 2018**；更低年份：**2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023**。每年 `Reward + pairs` 和 `Yield + pairs` 显示八个同 seed 配对中有多少个差值为正；年份均值为正不表示八个 seed 一致获益。

## Seed 与管理行为

Observed 2014–2023 的八 seed 平均：Historical reward `0.5239`、yield `6575.7` kg/ha、灌溉 `100.1` mm、施氮 `110.0` kg/ha；Random-weather 分别为 `0.4571`、`6328.5`、`88.7`、`121.0`。管理次数均值为 Historical 灌溉 `4.55`、施氮 `2.38`，Random-weather 灌溉 `3.04`、施氮 `2.27`。总体表现为较少灌溉及灌溉次数、较多施氮、较低 yield/reward；各 seed 的资源方向并不相同。

|   Seed | Historical archetype   | Random-weather archetype   |   Irrigation Δ (mm) |   N Δ (kg/ha) |   I events Δ |   N events Δ |
|-------:|:-----------------------|:---------------------------|--------------------:|--------------:|-------------:|-------------:|
|      0 | HIGH_INPUT             | VERY_LOW_INPUT             |              -169.5 |          -200 |        -11.3 |         -5   |
|      1 | VERY_LOW_INPUT         | AMBIGUOUS                  |                 0   |            80 |          0   |          0   |
|      2 | MODERATE_MIXED         | AMBIGUOUS                  |               -27   |           108 |         -2.8 |          1.7 |
|      3 | OTHER_NEW              | VERY_LOW_INPUT             |              -139.5 |            60 |        -10.3 |          1.5 |
|      4 | VERY_LOW_INPUT         | OTHER_NEW                  |               139.5 |           160 |         10.3 |          3   |
|      5 | OTHER_NEW              | OTHER_NEW                  |               105   |            80 |          2   |          2   |
|      6 | OTHER_NEW              | AMBIGUOUS                  |                 0   |          -160 |          0   |         -4   |
|      7 | AMBIGUOUS              | VERY_LOW_INPUT             |                 0   |           -40 |          0   |          0   |

Archetype 按 004_05 固定 probe 分类，不依 reward 重新贴标签。组频数和配对转换参见原 [archetype_frequency_by_regime.csv](../results/yc_random_weather_ppo/004_05/archetype_frequency_by_regime.csv) 与 [paired_seed_archetype_transition.csv](../results/yc_random_weather_ppo/004_05/paired_seed_archetype_transition.csv)。图中 `VERY_LOW_INPUT`、`OTHER_NEW` 等标签就是该正式分类；八个配对中 7/8 更换标签，但主转换方向仅占 14.3%，所以不把它解释为统一策略迁移。

## 是否胜出

Observed mean reward 的局部正差值 seed：**1, 4, 5**；held-out WGEN mean reward 的局部正差值 seed：**0, 3**；两个域均为正的 seed：**无**。Held-out WGEN 的八 seed 均值 reward 为 Historical `0.8769`、Random-weather `0.8291`，yield 为 `7026.2` 与 `6747.2` kg/ha。004_05 预设的跨域资源/产量/reward 联合成功条件为 **0/8**；图中没有这一轮整体稳定胜出的视觉证据。Archetype 频率描述为 `POSSIBLE_SHIFT`，与性能成功分开解读。

Observed 2014–2023 为独立比较期，不宣称 pristine final test set。以上图是已完成正式评估的可视化，没有新训练、DSSAT 运行或策略变更。
