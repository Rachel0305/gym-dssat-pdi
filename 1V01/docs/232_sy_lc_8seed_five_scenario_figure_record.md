# 232 SY/LC 新增七种子五情景结果图与表格

## 范围

- 站点：SY 和 LC，各自新增 seed1–7；冻结 seed0 只用于八种子汇总。
- 模型：各 seed 的 100K checkpoint；验证年均为 2014–2023。
- 训练结果、冻结 seed0 和既有基线目录均未修改。
- 新版图表根目录：`results/sy_lc_8seed_extension_231/five_scenario_figures_v2`。

## 与冻结结果一致的口径

- SY 冻结参考：`benchmark_results/046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000`。
- LC 冻结参考：`benchmark_results/053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000`。
- 每套保留 Null、recorded template、DSSAT auto + external N、official expert、PPO 五情景。四个非 PPO 情景从对应站点的冻结报告读取；仅新增 PPO 使用各自 100K 模型逐年重放。
- PPO 每年重放终值由原 `046_06`/`053_03` 方法与已保存的验证终值核对；从重放快照的 `Summary.OUT` 提取 ETCP、水氮投入及 `WP_ET`、`PFP_N`。无实际施氮时 `PFP_N` 保持缺失。
- 五情景累计奖励统一采用 `0.158×产量 − 1.1×灌溉 − 1.58×施氮`，仅作报告比较，不等于 PPO 训练奖励。

## 交付

- `SY/seed_01` 至 `SY/seed_07`、`LC/seed_01` 至 `LC/seed_07`：每套 10 张逐年五情景过程图、1 张指标图、1 张管理图，共 12 张；并包含逐日、管理事件、季节指标、相对四基线上限差值表及 10 个 PPO 重放快照。
- 共 14 套、168 张 PNG；14 份 `audit.json` 均记录模型出处、冻结基线来源、行为审计及十年快照。
- 每站 `eight_seed_annual_ppo_metrics.csv` 含 seed0–7 的 80 条逐年 PPO 记录；`eight_seed_ten_year_means.csv` 含 8 条十年均值。
- 顺序生成日志：`results/sy_lc_8seed_extension_231/figure_generation_events.jsonl` 和 `figure_generation_logs/`。日志末尾 `complete`，种子包数为 14。

## 审计与失败尝试

- 首次 SY seed1 的逐日重放通过，但冻结基线的无施氮 `PFP_N` 在柱状图绘制时被读取为空字符串，旧函数尝试转为浮点数而失败。该部分输出保留于 `five_scenario_figures/SY/seed_01`，不用于交付。
- `five_scenario_figures_v2` 将指标栏明确转为数值，缺失值保持为缺失；SY seed1、LC seed1 完整生成后，再逐个生成其余 12 套。没有覆盖先前失败现场。
- 14 套均核对 `figure_count=12`、`summary_rows=50`、PPO 十年行数为 10。SY seed2、6、7 和 LC seed1 的动作多样性行为门未通过，但模型、逐年验证和动作传递技术检查通过，结果按原样保留并纳入八种子表。

## 复现入口

- 单套：`python src/plot_sy_lc_seed_extension_231.py --site SY --seed 1`（LC 同理；现有输出会拒绝覆盖）。
- 顺序编排与八种子汇总：`python src/orchestrate_sy_lc_figures_231.py`（现有输出会被核对并跳过，已存在汇总表会拒绝覆盖）。
