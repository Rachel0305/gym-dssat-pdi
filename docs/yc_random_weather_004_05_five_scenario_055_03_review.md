# YC 004_05 新 PPO 与四基准五情景对照

## 样式来源

| 用途 | 复用来源 |
|---|---|
| 每年 4×2 日过程 | `benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca2014_five_scenario_daily.png` 及同目录 2015–2023 图 |
| 三层指标图 | 同目录 `055_03_yca_five_scenario_metrics.png` |
| 两层管理图 | 同目录 `055_03_yca_five_scenario_management.png` |
| 旧绘图脚本 | `src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py` |

```yaml
style_source_daily: benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca2014_five_scenario_daily.png
style_source_metrics: benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca_five_scenario_metrics.png
style_source_management: benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca_five_scenario_management.png
style_source_script: src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py
style_source_palette: "Null #555555; Recorded template #C44E52; DSSAT auto #D8A305; Official expert #7E63B6; PPO #2A9D55"
```

本轮按上述旧图的画布比例、标题、字体层级、颜色、线型、事件 stem、legend、白底浅网格及柱图布局重制。唯一 PPO 情景现在是 004_05 的 Random-weather PPO，不再画旧 055_03 PPO。因为新 PPO 有 8 个训练 seed，年柱展示 mean ± seed SD，日级状态线显示 seed 平均；旧基准每年仍是正式单条结果。

## 数据与可比性

四个基准来自 055_03 的正式 daily/season summary：Null、Recorded template、DSSAT auto + external N、Official expert。新策略使用 004_05 `RANDOM_WEATHER_WGEN` 训练组的 `observed_weather` episode，共 8 seeds × 10 年 = 80 个 episode。两批结果均为 YCA/YC、2014–2023、`lowIC` 输入档案。新旧 weather trace 按 DOY 核对：降水逐年完全一致；温度最大绝对差为 0.042°C（2023），与 WTH 输出舍入量级相符。基准并非 004_05 同一批次重跑，结论以已存档的正式输出为基础。

来源文件：[055_03 五情景日表](../benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_daily.csv)、[055_03 season summary](../benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_season_summary.csv)、[004_05 episodes](../results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv)、[004_05 step traces](../results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv)。生成脚本：[plot_004_05_055_03_five_scenario.py](../results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/plot_004_05_055_03_five_scenario.py)。step 级 reward、灌溉和 N 的合计逐 episode 与正式表核对一致。

日过程图的天气与四个基准取自同年共同 weather；新 PPO 的灌溉/施氮事件、`SWFAC`、`NSTRES` 和统一累积奖励取八 seed 均值。PPO 的 `grnwt`、`topwt`、土壤水分状态只在 seed 0–2 有完整 trace，因此图中仅这三个 seed 用于 grain/biomass 与土壤水分；土壤水分是 `post_step_state_json.sw` 顶层体积含水率，放在右轴，与基准 `SWTD`（全剖面 mm）分轴显示。WSPD 与 SWFAC 定义不同，图上用独立坐标轴并保留原变量名；不把它们解释成相同数值尺度。旧基准 `NSTD` 与 PPO `NSTRES` 依原名标出。

指标图保留旧版 Yield、WP_ET、PFP_N 三层结构。新 PPO 没有对应的精确 ETCP replay，所以 WP_ET 不画 PPO 数值，并在面板显式标注 unavailable。新 PPO 的 PFP_N 是各 seed 的 `yield/N` 均值，只对 N>0 的 seed 计算；每年有效 seed 数在 season summary 表的 `pfp_seed_count` 列中。另增一张逐年统一奖励图，所有情景统一按 055_03 原式 `0.158×yield - 1.1×irrigation - 1.58×nitrogen` 计算；它是跨情景对照分数，不是 004_05 PPO 的 canonical reward。

## 结果图

所有图和汇总数据位于 [055_03_five_scenario](../results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/)：

| 文件 | 内容 |
|---|---|
| `004_05_yca2014_five_scenario_daily.png` 至 `004_05_yca2023_five_scenario_daily.png` | 10 张逐年五情景日过程图，沿用 055_03 的 4×2 布局 |
| [004_05_five_scenario_metrics.png](../results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/004_05_five_scenario_metrics.png) | 逐年 yield、WP_ET、PFP_N 对比 |
| [004_05_five_scenario_management.png](../results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/004_05_five_scenario_management.png) | 逐年灌溉量、施氮量对比 |
| [004_05_five_scenario_reward.png](../results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/004_05_five_scenario_reward.png) | 逐年统一公式 reward 对比 |
| [004_05_five_scenario_season_summary.csv](../results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/004_05_five_scenario_season_summary.csv) | 50 个 year × scenario 汇总行 |

## 直接结论

- 新 PPO 八 seed 的 2014–2023 平均 yield 为 **6328.5 kg/ha**；四基准均值分别为 Null 3290.8、DSSAT auto 4417.1、Official expert 8200.6、Recorded template 7972.8 kg/ha。新 PPO 在 **0/10 年**超过当年四基准中的最高 yield；逐年一般高于 Null 和 DSSAT auto，低于 Recorded template 与 Official expert。
- 新 PPO 的平均灌溉为 **88.7 mm**、N 为 **121 kg/ha**。与四基准相比，投入低于 Recorded template、Official expert 和 DSSAT auto 的灌溉量，但高于 Null；施氮高于 Null/Auto，低于 farmer/expert。呈现中等投入画像，不是最低投入。
- 依 055_03 统一公式计算，新 PPO 的平均统一 reward 为 **711.2**，四基准的多年均值为 Null 520.0、Auto 536.8、Farmer 536.8、Expert 676.7。逐年 reward 高于当年最佳基准 **7/10 年**（2015–2018、2020–2022）；2014、2019、2023 未超过当年最佳。该共同公式突出产量与投入权衡，不能替代 canonical PPO reward 或单独证明综合胜出。
- 可计算的 PFP_N 平均值，新 PPO 为 **71.1**，farmer 为 21.3、expert 为 33.5；Null/Auto 因施氮为零而为 N/A。较高 PFP_N 与低于 farmer/expert 的产量同时出现，应作为氮效率信息看待，不能替代 yield 与灌溉/N 的联合判断。

综合看，新 PPO 显示**资源投入较 farmer/expert 低、产量也较低**的权衡；统一 reward 在多数年份领先，yield 没有任何一年超过当年四基准最高值，且 WP_ET 缺少 PPO 精确 ETCP 数据。因此这组图支持“部分指标/年份有优势”，不支持无条件宣称整体胜出。

没有新增训练或 DSSAT replay。PNG 已做尺寸和非空像素抽查；输出与本任务的本地 Git commit 状态见任务汇报。没有 push。
