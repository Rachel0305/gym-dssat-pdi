# 037_08 FQA 验证十年 PPO 五情景图与指标汇总记录

## 任务目的

为导师汇报整理 FQ/FQA 验证年份 2014–2023 的现有 PPO 候选结果，包括：

1. 每年一张五情景日过程对照图；
2. 产量、WP_ET、PFP_N 三个指标相对四基线最高值的汇总图；
3. 导出 year-level 指标表，便于复核。

## 数据来源

- PPO 候选：
  - 来源：`036_04_select_checkpoint_and_plot_03601_03603_summary`
  - 站点：FQA
  - checkpoint：`036_01`, seed0, `checkpoint_step=25000`
  - 说明：这是 036_04 预先选定的 FQA checkpoint，本任务没有重新挑选 checkpoint。

- 四基线：
  - 来源：`037_07_static_level1_four_baseline_rebuild`
  - 使用 FQA-only 修正后 summary：
    - `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_full_FQA_only_summary.csv`
  - 情景：
    - `null`
    - `recorded_farmer_template`
    - `official_extension_expert`
    - `dssat_auto`

## 执行脚本

- Prompt：
  - `prompts/037_08_FQA_validation_ppo_fixed_baseline_figures.md`
- 脚本：
  - `src/build_fqa_validation_ppo_fixed_baseline_figures_037_08.py`
- 命令：

```powershell
python src/build_fqa_validation_ppo_fixed_baseline_figures_037_08.py
```

## 输出文件

### 年度日过程图

输出目录：

```text
benchmark_results/037_08_FQA_validation_ppo_fixed_baseline_figures/figures/
```

年度图：

- `037_08_FQ2014_ppo25k_five_scenario_daily.png`
- `037_08_FQ2015_ppo25k_five_scenario_daily.png`
- `037_08_FQ2016_ppo25k_five_scenario_daily.png`
- `037_08_FQ2017_ppo25k_five_scenario_daily.png`
- `037_08_FQ2018_ppo25k_five_scenario_daily.png`
- `037_08_FQ2019_ppo25k_five_scenario_daily.png`
- `037_08_FQ2020_ppo25k_five_scenario_daily.png`
- `037_08_FQ2021_ppo25k_five_scenario_daily.png`
- `037_08_FQ2022_ppo25k_five_scenario_daily.png`
- `037_08_FQ2023_ppo25k_five_scenario_daily.png`

同时导出了同名 `.svg`。

### 指标汇总图

- `037_08_FQA_validation_yield_vs_four_baselines.png`
- `037_08_FQA_validation_wp_et_vs_four_baselines.png`
- `037_08_FQA_validation_pfp_n_vs_four_baselines.png`

同时导出了同名 `.svg`。

### 表格

- `benchmark_results/037_08_FQA_validation_ppo_fixed_baseline_figures/tables/037_08_FQA_validation_ppo25k_vs_fixed_four_baseline_metrics.csv`
- 每年合并后的五情景 daily CSV：
  - `benchmark_results/037_08_FQA_validation_ppo_fixed_baseline_figures/tables/037_08_FQ{year}_five_scenario_daily_merged.csv`
- manifest：
  - `benchmark_results/037_08_FQA_validation_ppo_fixed_baseline_figures/tables/037_08_run_manifest.json`

## 重要口径说明

1. 本任务只绘图和汇总，不训练、不重放 DSSAT。
2. 四基线使用 037_07 修正后的静态 level=1 结果。
3. PPO daily CSV 没有保存 `SWTD` 土壤水列，因此年度图中的 Soil water 面板只画四基线，不画 PPO。PPO 的天气、胁迫、措施、产量、生物量、累计统一奖励均正常绘制。
4. 累计奖励图使用统一公式：

```text
common_reward_step = ΔGRNWT - irrigation - 5 × nitrogen
```

该图只用于过程对照，不代表 PPO 训练时的完整 reward。

5. 读取 CSV 时必须使用 `keep_default_na=False`，否则字符串 `null` 会被 pandas 误读为缺失值。脚本已处理。
6. DSSAT `.OUT` 文件中同一日表可能含多个 RUN 段，脚本按 `YEAR/DOY/DAP` 去重；管理事件按 `DAP/irrigation/nitrogen` 去重。

## 结果总览

以“PPO vs 四个基线情景中的最高值”为标准：

- 10 年全部纳入时：
  - 产量超过：0/10
  - WP_ET 超过：4/10
  - PFP_N 超过：0/10
- 排除 FQ2018 零产异常年后：
  - 产量超过：0/9
  - WP_ET 超过：4/9
  - PFP_N 超过：0/9

相对 official expert：

- 正常 9 年平均节水：174 mm
- 正常 9 年平均节氮：85 kg/ha

## 年份级结果

| 年份 | PPO产量 | 四基线最高产量 | 产量差 | PPO WP_ET | 四基线最高WP_ET | WP_ET差 | PPO PFP_N | 四基线最高PFP_N | PFP_N差 | 相对expert节水 | 相对expert节氮 | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2014 | 8300.73 | 8656.55 | -355.82 | 2.34 | 2.36 | -0.02 | 51.9 | 57.5 | -5.6 | 184 | 85 |  |
| 2015 | 8005.62 | 8086.64 | -81.02 | 2.45 | 2.48 | -0.03 | 50.0 | 55.1 | -5.1 | 184 | 85 |  |
| 2016 | 7803.56 | 8012.42 | -208.86 | 2.56 | 2.51 | +0.05 | 48.8 | 55.1 | -6.3 | 154 | 85 | WP_ET超过 |
| 2017 | 7704.48 | 7705.50 | -1.02 | 2.29 | 2.30 | -0.01 | 48.2 | 53.5 | -5.3 | 184 | 85 | 产量几乎持平 |
| 2018 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | NA | NA | NA | 113 | 52 | 零产异常年 |
| 2019 | 7859.62 | 8810.97 | -951.35 | 2.74 | 2.67 | +0.07 | 49.1 | 55.7 | -6.6 | 184 | 85 | WP_ET超过 |
| 2020 | 8963.34 | 8968.44 | -5.10 | 2.52 | 2.51 | +0.01 | 56.0 | 62.2 | -6.2 | 184 | 85 | 产量几乎持平，WP_ET超过 |
| 2021 | 7463.09 | 7479.31 | -16.21 | 2.40 | 2.43 | -0.03 | 46.6 | 51.8 | -5.2 | 184 | 85 | 产量接近 |
| 2022 | 6687.14 | 6822.54 | -135.41 | 2.07 | 2.07 | 0.00 | 41.8 | 46.1 | -4.3 | 154 | 85 | WP_ET持平 |
| 2023 | 9099.09 | 9267.49 | -168.40 | 2.65 | 2.58 | +0.07 | 56.9 | 63.2 | -6.3 | 154 | 85 | WP_ET超过 |

## 汇报时建议表述

这批图说明：在修正后的 FQ 四基线口径下，现有 036_04 PPO 25k 候选表现为“明显节水节氮、产量多数接近但不超过四基线最高值、WP_ET 在部分年份超过”。它适合作为阶段性验证图，但不应表述为“FQ 验证十年全面优于四基线”。FQ2018 为零产异常年，应单独标注，不做正常农学解释。
