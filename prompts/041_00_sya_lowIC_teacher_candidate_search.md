# 041_00 SYA lowIC 优质 teacher 候选轨迹搜索

## 背景

040_40 的 SYA lowIC 自由时序 MaskablePPO 在验证年份上已经有一定指标优势，但 040_53 发现其确定性策略对已解析的 `SW`、`NSTRES`、`SRAD` 单点扰动没有动作响应，灌溉时机几乎固定。因此下一步不能继续直接用 originIC 时期的确定性搜索结果作为 teacher。

本任务的目的不是训练 PPO，也不是证明“优化空间是否存在”这个旧问题，而是在当前 lowIC 输入条件和当前 PPO 可执行动作边界下，重新生成一批 DSSAT 验证过的优质候选轨迹，供后续 imitation warm-start / PPO fine-tune 使用。

## 输入条件

- 站点：SYA
- 输入数据根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 年份：默认 SYA 验证年份 2014–2023
- 基线比较来源：`benchmark_results/040_42_sya_lowIC_04040_ckpt100k_validation_five_scenario_metric_bars/tables/040_42_sya_lowIC_04040_ckpt100k_five_scenario_metric_summary.csv`
- 只使用四个外部基线作阈值比较：
  - `null`
  - `recorded_farmer_template`
  - `dssat_auto`
  - `official_extension_expert`

## 候选动作边界

必须与 040_40 可执行动作边界一致，不允许 teacher 轨迹使用 PPO 不能执行的动作。

- 灌溉档位：0、30、45 mm
- 施氮档位：0、80、120 kg/ha
- 季节灌溉上限：240 mm
- 季节施氮上限：250 kg/ha
- 灌溉最小间隔：7 天
- 施氮最小间隔：7 天
- 灌溉允许 DAP：1–120
- 施氮允许 DAP：1–90
- DAP ≤ 90 时累计灌溉不得超过 195 mm，保留至少一次 45 mm 后期灌溉机会

## 预注册候选集合

为节省 DSSAT 调用，本任务只跑一个小型、可解释的固定网格，不做无限穷举。候选由预定义水分 schedule × 预定义施氮 schedule 组成。

候选 schedule 的设计原则：

1. 包含当前 PPO 近似固定灌溉模式；
2. 包含更均匀的中后期灌溉模式；
3. 包含较节水模式；
4. 施氮总量覆盖 0、160、200、240 kg/ha；
5. 所有候选均满足 040_40 动作边界。

## 输出

输出到：

`benchmark_results/041_00_sya_lowIC_teacher_candidate_search/`

主要文件：

- `tables/041_00_candidate_summary.csv`
- `tables/041_00_teacher_candidates.csv`
- `tables/041_00_requested_actions.csv`
- `tables/041_00_candidate_grid.csv`
- `041_00_result.json`
- `docs/041_00_sya_lowIC_teacher_candidate_search_record.md`

每个候选保存：

- 年份
- 候选名称
- 灌溉 schedule
- 施氮 schedule
- 产量
- biomass
- ETCP
- WP_ET
- PFP_N
- 实际灌溉量
- 实际施氮量
- 最大水分胁迫
- 最大氮胁迫
- 相对四基线最高值的 gap
- 是否可作为 teacher

## teacher 判定

一条候选轨迹若满足以下任一条件，可进入 teacher 候选表：

1. `any_metric_win_vs_four_max = True`，即产量、WP_ET、PFP_N 中至少一项超过同年四基线最高值；
2. 或者产量不低于 official expert 的 98%，且相对 official expert 同时节水或节氮。

注意：进入 teacher 候选不等于最终采用；041_01 之前仍需人工/脚本复核管理措施合理性。

## 停止线

- 如果 lowIC 输入根目录不存在，停止；
- 如果四基线表缺失，停止；
- 如果 dry-run 显示候选数量异常大，停止；
- 如果候选没有任何 teacher，停止，不进入 imitation；
- 不根据结果现场扩展候选网格。

