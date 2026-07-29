# 039_02 originalIC vs lowIC 三情景 DSSAT 基线响应审计

## 目的

在不训练任何强化学习模型的前提下，比较原始初始土壤条件与手动降低后的 lowIC 条件下，三个 DSSAT/管理情景的表现：

1. `null`
2. `official_extension_expert`
3. `dssat_auto`

本轮回答的问题是：降低初始土壤水分和矿质氮后，DSSAT 是否会产生更明显的水分/氮素胁迫；已有管理方案和 DSSAT auto 是否能缓解这些胁迫。

## 输入边界

- 原始输入目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`
- lowIC 输入目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 年份清单：复用 `benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/configs/033_04_available_weather_half_split_years.csv`
- 只使用有天气数据且已在 033_04 年份清单中的站点年份。

## 执行边界

- 不训练 PPO/DQN。
- 不修改原始输入文件。
- 不修改 lowIC 输入文件。
- 每个站点年分别运行 originalIC 和 lowIC。
- 每个 IC 条件下运行三情景：`null`、`official_extension_expert`、`dssat_auto`。
- `official_extension_expert` 通过静态 `.MZX` 管理表执行。
- `dssat_auto` 通过 DSSAT 自动管理块执行。
- 所有 DSSAT 原始输出保存到独立 snapshot 目录。

## 输出

- 总汇总表：产量、生物量、灌溉量、施氮量、ET、WP_ET、PFP_N、最大 WSPD、最大 NSTD。
- originalIC 与 lowIC 差值表。
- 每站点一张 lowIC 响应总览图。
- 运行 manifest、失败记录、render check。
- 中文实验记录 Markdown。

## 判读规则

- 若 lowIC-null 相比 original-null 的最大 WSPD 或 NSTD 增大，说明降低初始条件确实增强了胁迫信号。
- 若 lowIC-expert 或 lowIC-auto 相比 lowIC-null 产量提高、WSPD/NSTD 降低，说明管理措施对低初始条件有缓解作用。
- 若 lowIC 下 expert/auto 仍出现强胁迫或明显减产，标记为 `lowIC_too_severe_candidate`，后续不能直接无脑用于 PPO 主训练。

## 停止条件

- smoke 模式若出现系统性失败，先停下修复，不进入 full。
- full 模式只做前向基线审计，不据此直接修改 PPO 奖励或训练参数。
