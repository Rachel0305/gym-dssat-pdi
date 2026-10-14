# YC 004_05：全部 Random-weather PPO seed 的逐年固定模型对照

## 评估设计

对 8 个已训练 seed 分别固定模型，各自使用已有正式 episode 覆盖 2014–2023 十个验证年；每个 seed 独立与同年四基准比较。没有逐年换 seed，也没有拼接不同 seed 的年度最佳结果。没有训练、DSSAT replay 或天气生成。

seed 3 是此前按 observed/heldout canonical reward 等权平均事后选出的，不应因此忽略其余 seed；但现在再按十年验证数据比较并挑候选仍带有选择偏差，所以本报告只能用于探索性筛查，不能当作未见数据上的确认结果。

## 数据质量与口径

- PPO observed episodes：**80 / 80**，完整包含 seed 0–7 × 2014–2023，每个 seed-year 唯一。
- YC 四基准：**40 / 40**，每年四个情景完整且唯一。
- PPO `WP_ET`：全部缺失，未做推算；`PFP_N` 在 N>0 时按 yield/N 计算，N=0 记 N/A。
- PPO common reward 按 `0.158*yield - 1.1*irrigation - 1.58*N` 计算；基准使用正式 `unified_reward`。基准已舍入列重新计算的最大差值为 **0.440**。PPO canonical reward 与其分列。
- 每个 seed-year 的 Pareto 是与四基准组成的五情景部分前沿，使用 yield 最大、水和 N 最小三项独立指标。common reward 单独排名；它是水、N、yield 的派生式，不重复充当 Pareto 维度。

## 八个固定模型汇总

| Seed | 年均产量 kg/ha | 灌溉 mm | N kg/ha | PFP_N | common reward | Yield 第一 | PFP_N 第一 | Reward 第一 | 部分前沿年数 | 超过年度最高基准产量 | 对年度最高基准平均差 | 动作日/季 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5348 | 45.0 | 40.0 | 133.7 | 732.3 | 0/10 | 10/10 | 6/10 | 9/10 | 0/10 | -2853 | 1.0 |
| 1 | 6304 | 45.0 | 120.0 | 52.5 | 756.9 | 3/10 | 8/10 | 6/10 | 10/10 | 3/10 | -1897 | 1.0 |
| 2 | 6822 | 75.0 | 188.0 | 36.7 | 698.3 | 2/10 | 8/10 | 8/10 | 9/10 | 2/10 | -1379 | 2.7 |
| 3 | 6902 | 75.0 | 100.0 | 72.7 | 850.0 | 1/10 | 9/10 | 8/10 | 10/10 | 1/10 | -1299 | 3.5 |
| 4 | 8203 | 214.5 | 240.0 | 34.2 | 680.9 | 3/10 | 10/10 | 4/10 | 10/10 | 3/10 | +2 | 16.3 |
| 5 | 8149 | 165.0 | 240.0 | 34.0 | 726.8 | 2/10 | 8/10 | 8/10 | 10/10 | 2/10 | -52 | 5.0 |
| 6 | 3552 | 45.0 | 0.0 | nan | 511.8 | 0/10 | 0/10 | 1/10 | 9/10 | 0/10 | -4649 | 1.0 |
| 7 | 5348 | 45.0 | 40.0 | 133.7 | 732.3 | 0/10 | 10/10 | 6/10 | 9/10 | 0/10 | -2853 | 1.0 |

## 结果解读

- 八个 seed 的侧重点不同，不存在可以只凭一个均值说“全面最好”的模型。按当前表格的边际摘要，年均产量最高的是 seed **4**；年均 common reward 最高的是 seed **3**；年均 `PFP_N` 最高的是 seed **0, 7**；平均灌溉最低的是 seed **6**（其 N 均值 0.0 kg/ha）。这些是不同优化维度，不构成统一冠军。
- seed 3 的位置：均产 6902 kg/ha，灌溉 75.0 mm，施氮 100.0 kg/ha，`PFP_N` 72.7，common reward 850.0；yield 第一 1/10、common reward 第一 8/10、部分前沿 10/10。它是有吸引力的低投入候选，但不能代表其他 seed。
- 表内“超过年度最高基准产量”与“平均产量最高”是两种不同证据；逐年差值保存在 `seed_year_comparison.csv`。不能按年挑 seed 后拼成一个实际 PPO 模型。
- 原始配对实验的预设成功判据仍为 **0/8**（`results/yc_random_weather_ppo/004_05/multi_seed_decision.json`）；在相同验证结果中筛出表现较好的 seed，不能证明 weather augmentation 对 PPO 模型族整体有效。
- `PFP_N` 与用氮量并列阅读；没有 PPO `WP_ET` 就无法完成水分效率评价。`common_reward` 与 canonical PPO reward 不是一个标度。

## 建议

不启动新训练。先从 `seed_profile_summary.csv` 和 `seed_year_comparison.csv` 选出少量**不同权衡类型**的固定候选（高产、低投入、高 PFP_N），再做精确、evaluation-only `Summary.OUT/ETCP` replay 补齐这些候选十年的 `WP_ET`。根据完整逐年指标重新判断 Pareto 支配关系；如果要做确认性结论，另留未参与 seed 选择的新天气/年份验证。

脚本：`src/run_004_07_01_yc_all_seed_fixed_model_assessment.py`

输出：
- `results/yc_random_weather_ppo/004_05_review_figures/all_seed_fixed_model_assessment/seed_year_comparison.csv`（80 行）
- `results/yc_random_weather_ppo/004_05_review_figures/all_seed_fixed_model_assessment/five_scenario_long.csv`（120 行扩展候选池）
- `results/yc_random_weather_ppo/004_05_review_figures/all_seed_fixed_model_assessment/seed_profile_summary.csv`（8 行）

运行命令：`python src/run_004_07_01_yc_all_seed_fixed_model_assessment.py`。未 commit、未 push。
