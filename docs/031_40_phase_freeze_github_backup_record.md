# 031_40 阶段性冻结与 GitHub 备份记录

## 冻结内容

本次冻结的是 031 系列自由时序强化学习阶段性成果，核心结论是：

- 当前展示结果来自自由时序离散 MaskablePPO，而不是固定 expert DAP 的旧阶段型 PPO。
- 每个站点用一个训练年训练 3 seed，再冻结候选模型迁移到该站点其他可用年份。
- 五站点汇总结果见 `031_38`。
- 代表五情景日过程图见 `031_39`。

## 当前训练框架

- 算法：MaskablePPO。
- 动作空间：
  - 灌溉：0、6、12、18、24 mm。
  - 施氮：0、40、80、120、160 kg/ha。
- 主要安全约束：
  - 单日灌溉上限 24 mm。
  - 单日施氮上限 160 kg/ha。
  - 季节灌溉软上限 160 mm。
  - 季节施氮软上限 250 kg/ha。
  - 灌溉/施氮最小间隔 7 天。
  - 灌溉 DAP 1-120。
  - 施氮 DAP 1-90。
- PPO 参数：
  - learning_rate = 0.0003
  - gamma = 1.0
  - gae_lambda = 1.0
  - n_steps = 144
  - batch_size = 144
  - n_epochs = 5
  - ent_coef = 0.01
  - clip_range = 0.2
  - net_arch = [64, 64]
- 最大训练步数：100000。
- checkpoint：10000、20000、30000、50000、75000、100000。

## 当前五站点汇总结果

`031_38` 输出：

- `benchmark_results/031_38_five_station_ppo_water_n_saving_summary/tables/031_38_station_water_n_saving_summary.csv`
- `benchmark_results/031_38_five_station_ppo_water_n_saving_summary/figures/031_38_overall_ppo_water_n_saving_vs_official.png`

结果解释：

- 指标胜出按照四情景 envelope 判断，即 PPO 的产量、WP_ET 或 PFP_N 是否超过 null / DSSAT auto / official expert / recorded farmer template 中相应指标最高值。
- 节水节氮量相对 official expert 计算。原因是 null 和 DSSAT auto 可能使用 0 水/0 氮，不适合作为投入节约基线。

## 代表图

`031_39` 输出 LCA2010 当前自由时序 PPO 五情景日过程图：

- `benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/figures/027_05_lc2010_maskableppo_five_scenario_daily.png`
- `benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/031_39_lca2010_current_free_timing_ppo_five_scenario_daily.csv`

该图无新训练、无新 DSSAT 运行，只读取已有 snapshot 与 CSV。

## GitHub 备份边界

本次计划纳入 Git 的内容：

- prompt / docs / config / src 中 031 系列关键文件。
- 031_26 / 031_33 模型 zip。
- 031_27 / 031_34 daily outputs 和 evaluation tables。
- 031_29 / 031_30 / 031_35 / 031_36 / 031_37 / 031_38 / 031_39 的核心 evaluation、tables、figures。

不纳入 Git 的内容：

- DSSAT `runs/` snapshot。
- runtime train/eval/input 临时目录。
- 大量中间 input 拷贝。
- 与本阶段无关的 proposal material 修改。

## 后续优化方向

当前不是“最终最优参数”，仍有优化空间，尤其是 FQA 和 HLA。

建议下一步不要逐年手调，而是预注册一个小规模稳健性/参数敏感性任务：

1. 固定开发集和测试集，避免逐年后验调参。
2. 只扫描少数关键参数，例如：
   - nitrogen_cost / water_cost 比例；
   - season irrigation / nitrogen soft limit；
   - entropy coefficient；
   - checkpoint selection rule。
3. 成功标准仍用四情景 envelope：
   - 产量、WP_ET、PFP_N 中至少一项超过四情景最高值；
   - 同时记录节水节氮是否相对 official expert 改善。
4. 如果导师允许“站点级参数校准”，可以每个站点一套固定参数；不建议每个年份单独调参。

