# 037_07 FQ 站点静态 level-1 四基线结果说明

## 当前状态

用户要求先停止全站点重建，只查看一个站点结果。因此 `037_07 --mode full` 在跑到 HLA2014 时被手动终止。本文件只整理已经完整跑完的 FQ/FQA 站点，不把 HLA 部分结果作为正式输出。

## 修复内容

034_00 的静态基线错误来自 DSSAT `.MZX` 管理表的行首编号理解错误：

- 旧写法把 `@I IDATE` / `@F FDATE` 行首编号写成事件序号 `1,2,3...`；
- DSSAT 实际将该列解释为 management level；
- treatment 只选中 level 1，所以旧结果经常只执行第一条灌溉/施肥事件；
- 037_07 修正为：同一情景所有静态灌溉/施肥应用行均使用 level 1。

该修复使 recorded farmer template 中超过 RL 动作通道上限的大事件也能通过静态 DSSAT 管理表进入 DSSAT，不再需要被 linked-action 通道裁剪。

## FQ 覆盖

- 站点：FQ / FQA
- 年份：2005–2023，共 19 年
- 情景：null、recorded_farmer_template、official_extension_expert、dssat_auto
- 成功运行：76 / 76
- 管理事件链路审计：76 / 76 ok

输出文件：

- `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_full_FQA_only_summary.csv`
- `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_full_FQA_only_management_event_audit.csv`

## FQ 四情景均值预览

| scenario | n | mean_yield | mean_irrigation | mean_nitrogen | mean_wp_et | mean_pfp_n | max_wstress | max_nstress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dssat_auto | 19 | 7244.464 | 25.474 | 0.000 | 2.231 |  | 0.000 | 0.454 |
| null | 19 | 7077.918 | 0.000 | 0.000 | 2.246 |  | 0.710 | 0.454 |
| official_extension_expert | 19 | 7335.237 | 217.368 | 243.263 | 2.155 | 31.600 | 0.000 | 0.012 |
| recorded_farmer_template | 19 | 7271.636 | 75.000 | 144.000 | 2.227 | 53.306 | 0.561 | 0.012 |

## 逐年产量预览 kg/ha

| year | dssat_auto | null | official_extension_expert | recorded_farmer_template |
| --- | ---: | ---: | ---: | ---: |
| 2005 | 7972.7 | 7972.7 | 7895.3 | 7961.8 |
| 2006 | 7249.5 | 7249.5 | 7173.5 | 7075.1 |
| 2007 | 8101.3 | 8101.3 | 7919.4 | 8059.0 |
| 2008 | 7891.8 | 6949.9 | 8191.4 | 7880.4 |
| 2009 | 7835.5 | 7834.9 | 7755.2 | 7833.5 |
| 2010 | 6636.3 | 6635.7 | 6630.6 | 6634.8 |
| 2011 | 6600.7 | 6628.5 | 6951.5 | 6663.6 |
| 2012 | 6243.6 | 6243.6 | 6430.7 | 6228.0 |
| 2013 | 7331.8 | 7331.8 | 7301.8 | 7807.2 |
| 2014 | 7953.6 | 7953.6 | 8656.5 | 8273.1 |
| 2015 | 8086.6 | 8086.6 | 8019.6 | 7932.6 |
| 2016 | 8012.4 | 7234.5 | 7922.2 | 7932.3 |
| 2017 | 7705.5 | 7694.0 | 7640.3 | 7704.5 |
| 2018 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2019 | 8458.3 | 7013.7 | 8811.0 | 8026.8 |
| 2020 | 8434.7 | 8434.7 | 8968.4 | 8956.4 |
| 2021 | 7477.1 | 7479.3 | 7322.5 | 7462.0 |
| 2022 | 6822.5 | 6822.5 | 6512.1 | 6631.6 |
| 2023 | 8830.9 | 8813.5 | 9267.5 | 9098.3 |

## 初步解释边界

1. FQ 的 corrected static baseline 链路已经可信：recorded/expert 的多次事件能进入 `DSSAT48.INP`、`MgmtEvent.OUT` 和 `Summary.OUT`。
2. FQ2018 四情景产量均为 0，需要单独检查，不应作为正常年份解释。
3. 本结果只是四基线，不包含 PPO/DQN 候选策略。
4. 全站点 full 任务已被中断，除 FQ 外的部分输出不能作为正式完整站点结果使用。

