# 014_01 封丘站全年份筛选与 DQN 方法迁移

## 目的

在不扩大无效训练的前提下，先筛选封丘站 2000–2023 年中具有水氮优化空间的年份，再把禹城站已经跑通的 linked DQN 离散动作方法迁移到封丘站，生成两年四情景对比图和日值表格。

## 背景

- 禹城站已经出现两个可写的 DQN 结果：
  - YC2008：agronomic window DQN 在较少水氮投入下达到较好产量。
  - YC2014：free daily DQN 跨 seed 稳定高产，但用满 I120/N300。
- 封丘站之前只看过少数年份，FQ2007/FQ2010 优化空间不明显，不能直接代表全部年份。
- 封丘站输入包中有 `CNFQ0001.WTH`–`CNFQ2301.WTH` 多年天气，但 MZX 只有 2007、2008、2010 三套实测管理。

## 输入口径

- 输入目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ`
- 基础 MZX：`CNFQ0801.MZX`
- 天气：`CNFQyy01.WTH`
- 土壤：`SOIL.SOL`
- 品种：`MZCER048.CUL`
- 初始条件：保持 MZX 中已有 IC=1 剖面。
- 专家策略迁移：以 FQ2008 treatment 2 作为完整专家管理模板，将日期前缀 `08xxx` 平移为目标年份 `yyxxx`，并把天气站改为目标年份 `CNFQyy01`。

## 情景

筛选阶段：

1. `null`：无灌溉、无施肥。
2. `recorded_shifted`：FQ2008 记录管理平移到目标年份。
3. `dssat_auto`：DSSAT 原生自动灌溉和自动施肥。

DQN 阶段：

4. `dqn_linked_free_daily`：复用 YC linked DQN 方法，自由日窗口，预算 I120/N300，单次 I30/N100，最小间隔 7 天。
5. `dqn_linked_agronomic_window`：复用 YC linked DQN 方法，农艺窗口，预算 I120/N300，单次 I30/N100，最小间隔 7 天。

最终四情景图优先展示：

- `null`
- `recorded_shifted`
- `dssat_auto`
- 该年份表现较好的 DQN 情景

## 年份筛选标准

优先选择满足以下条件的两年：

1. recorded 或 dssat_auto 明显高于 null，说明存在管理响应。
2. null 有一定水分或氮胁迫，说明 RL 有可干预空间。
3. 作物能正常完成生长季，产量和生物量不出现明显异常。
4. 两年尽量覆盖不同限制类型：一个偏水分限制，一个偏氮素/综合限制；若无法区分，则选择管理增产幅度最大的两年。

## 执行约束

- 必须使用 Docker 容器内环境：`b2fd6726c8c1`，Python：`/opt/gym_dssat_pdi/bin/python`。
- 不进行 PPO 训练。
- DQN 只在筛出的两年上跑 5K，先跑 seed0；如时间允许再跑 seed1。
- 所有中间 CSV、图、脚本、日志、试验记录均保存。
- 不覆盖旧实验结果。

## 输出

- 脚本：`src/run_fq_all_year_screen_and_dqn_transfer_014_01.py`
- 输出目录：`DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/`
- 筛选汇总表：`014_01_fq_all_year_screening_summary.csv`
- 两年四情景日值表：`014_01_fq_selected_four_scenario_daily.csv`
- 两年四情景事件表：`014_01_fq_selected_four_scenario_events.csv`
- 图：
  - `figures/fq_<year>_four_scenario_process.png`
- 试验记录：
  - `docs/2026-06-30_014_01_fq_all_year_screen_and_dqn_transfer_record.md`

## 预期判断

- 如果封丘站仍然没有明显管理响应，则记录为“该站点当前参数与年份下不适合作为 RL 主展示站点”，不继续强行调参。
- 如果筛出年份并 DQN 优于或接近专家/auto，同时投入更少，则可作为封丘站补充结果。
