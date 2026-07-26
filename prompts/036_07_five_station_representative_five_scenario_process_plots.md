# 036_07 五站点代表年份五情景过程图

## 目的

在不重新训练、不重新运行 DSSAT 的前提下，基于当前已冻结的 036_01 PPO 结果和 034_00 统一 IC=1 四基线 daily，为五个站点各生成一张五情景过程对照图，用于阶段性备份和措施合理性人工审查。

## 数据来源

- PPO 候选策略：`benchmark_results/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun/daily_outputs/`
- 当前 036_04 选定 checkpoint 与年份级比较表：`benchmark_results/036_04_select_checkpoint_and_plot_03601_03603_summary/tables/036_04_selected_year_level_comparison.csv`
- 四情景基线 daily：`benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_full_baseline_daily.csv`

## 代表年份选择规则

每个站点只选一张代表图。选择规则预先固定：

1. 使用 036_04 已经选定的站点 checkpoint 结果；
2. 排除 PPO 终产量为 0 的异常年份；
3. 优先选择至少一个指标超过四情景最高值的年份；
4. 在候选年份中，优先选择正向指标数量更多的年份，指标包括：
   - 产量高于四情景最高值；
   - WP_ET 高于四情景最高值；
   - PFP_N 高于四情景最高值；
5. 若仍并列，选择 PPO 终产量更高的年份；
6. 若某站点没有满足第 3 条的年份，则选择非零产量中终产量最高的年份，并在记录中标注。

## 图件内容

每张图包含：

1. 降雨、Tmax、Tmin；
2. 累计灌溉；
3. 水分胁迫指数；
4. 氮胁迫指数；
5. 灌溉事件；
6. 施氮事件；
7. 籽粒产量和生物量轨迹；
8. 统一累计奖励。

累计奖励统一重算为：

```text
common_step_reward = ΔGRNWT - 1.1 × irrigation - 1.58 × nitrogen
```

不使用各实验源文件里可能不同尺度的原始 reward。

## 边界

- 不训练；
- 不重跑 DSSAT；
- 不修改已有结果；
- 只读取现有 daily CSV 并作图；
- 图用于审查和备份，不直接证明 PPO 措施完全合理。

