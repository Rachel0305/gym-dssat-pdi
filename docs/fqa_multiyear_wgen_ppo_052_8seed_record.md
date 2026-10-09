# FQ WGEN PPO 100K 八种子实验记录

日期：2026-10-09
最终状态：`PASS_8SEED_TRAINING_EVALUATION_FIGURES`

## 目的与实验边界

依照用户明确要求，完成 FQA 的 PPO seed 0–7 八种子 100K 训练和 2014–2023 固定历史天气验证，并生成与 HL 055_03 结果目录同结构的五情景对照图。正式对照使用各 seed 精确 100,000 步 checkpoint；seed0 复用 051 的已完成结果，seed1–7 从头独立训练。

保持 FQA originIC、冻结 PPO / reward / 原始观测 / 16 离散动作及安全掩码、WGEN 运行合同和四个冻结基线。训练年份为 2005–2013，WGEN seed 池为 1001–1080，调度 RNG 为 64003/64004；留出 seed 1081–1100 未进入训练。验证固定为 FQA 2014–2023、确定性 policy、evaluation seed 0。未改写 051 或 055_03 原有结果。

八个 PPO seed 的 episode 调度、weather_seed、runtime `_rseed1` 和逐 episode 天气 SHA-256 序列逐条相同。因此，seed 间差异主要衡量 PPO 初始化与优化随机性，不能当成八套独立天气样本。

## 训练结果与资源

seed0 指向 `results/fqa_multiyear_wgen_ppo_051/attempt_01/`；seed1–7 的训练产物位于 `results/fqa_multiyear_wgen_ppo_052_8seed/seed_XX/attempt_01/`。seed1 的 432 步 smoke 已通过后，正式训练按 seed1→7 顺序运行。

| PPO seed | 训练来源 | 精确 checkpoint | 实际步数 | 天气日数 / episode 归档 | 峰值进程树 RSS (MiB) | 运行时间 (s) |
|---:|---|---:|---:|---|---:|---:|
| 0 | 复用 051 | 100,000 | 100,080 | 100,080 / 956 | 1,431.64 | 2,831.17 |
| 1 | 本次正式训练 | 100,000 | 100,080 | 100,080 / 956 | 1,431.95 | 2,844.13 |
| 2 | 本次正式训练 | 100,000 | 100,080 | 100,080 / 956 | 1,433.11 | 2,853.38 |
| 3 | 本次正式训练 | 100,000 | 100,080 | 100,080 / 956 | 1,433.12 | 2,861.32 |
| 4 | 本次正式训练 | 100,000 | 100,080 | 100,080 / 956 | 1,433.70 | 2,896.19 |
| 5 | 本次正式训练 | 100,000 | 100,080 | 100,080 / 956 | 1,433.43 | 2,835.45 |
| 6 | 本次正式训练 | 100,000 | 100,080 | 100,080 / 956 | 1,433.77 | 2,909.24 |
| 7 | 本次正式训练 | 100,000 | 100,080 | 100,080 / 956 | 1,433.20 | 2,928.54 |

每个 seed 均保存 25K、50K、75K、100K checkpoint，并另存实际 100,080 步最终模型。每个 seed 有 955 个完整 episode 和 1 个已保留的结束时部分 episode；总步数与逐日天气归档日数闭合。每个 seed 归档 553 个不同天气哈希，八条逐 episode 天气哈希序列完全一致。所有八个 `audit_gate.json` 均为 `PASS_100K_ARCHIVE_ONLY`，峰值 RSS 均低于 1,536 MiB。

## 验证、指标口径与完整性

使用精确 100K checkpoint 对每个 seed 分别进行 10 个验证季回放，单模型、单 season 顺序运行。每季保存 DSSAT 快照和 `Summary.OUT`，并记录 evaluation metadata。逐年 PPO 天气与同年份冻结 Null 基线按日期核对；八个 seed 的 2014–2023 对照均通过。`WP_ET` 使用确切 `Summary.OUT` 的 ETCP；`PFP_N` 使用相同 Summary.OUT 口径。未获得作物氮吸收量，因此不推断 NUE。

每个 seed 有 50 行五情景年度指标、6 个 CSV 表、1 个管理事件/动作审计 JSON 和 23 张 PNG；80 个验证年度快照均完整。每个 seed 图目录：

`results/fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ/best_seed_seed0/figures/` 至 `best_seed_seed7/figures/`

跨 seed 汇总位于 `results/fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ/cohort_8seed_summary/`，含 400 行逐年五情景总表、80 行 PPO seed-year 表、每 seed 十年均值、年度 mean/SD、配对差值、README 和 2 张 cohort 图。完整性检查确认 8/8 训练审计通过、8×10 快照完整、8×23 seed 图齐全、cohort 汇总齐全。

### 2014–2023 十年均值：PPO seed 间离散度与冻结基线

下表 PPO 的 mean ± SD 是八个 seed 各自十年均值之间的均值与样本标准差；基线为冻结 051_03 同十年均值。所有差值均为描述性比较，不作显著性推断。

| 指标 | PPO 8 seed mean ± SD | PPO seed 范围 | Null | Recorded template | DSSAT auto + external N | Official expert | PPO 高于各基线的 seed 数 |
|---|---:|---:|---:|---:|---:|---:|---|
| 产量 (kg/ha) | 7,234.3 ± 86.6 | 7,118.4–7,350.2 | 6,953.4 | 7,201.7 | 7,178.2 | 7,311.9 | 8/8、3/8、7/8、3/8 |
| WP_ET (kg/m³) | 2.140 ± 0.082 | 1.996–2.200 | 2.145 | 2.145 | 2.125 | 2.079 | 5/8、5/8、5/8、6/8 |
| PFP_N (kg grain/kg N) | 79.05 ± 55.25 | 33.87–197.74 | 不适用 | 55.58 | 不适用 | 33.16 | 不适用、5/8、不适用、8/8 |
| 灌溉 (mm) | 110.6 ± 90.6 | 45–222 | 0.0 | 75.0 | 34.7 | 212.9 | 8/8、3/8、8/8、3/8 |
| 施氮 (kg/ha) | 145.0 ± 82.6 | 40–240 | 0.0 | 144.0 | 0.0 | 241.7 | 8/8、3/8、8/8、0/8 |

PFP_N 对 N=0 的基线没有定义值，不作虚构比较。seed 4、6、7 的平均管理投入为约 219–222 mm 灌溉和 240 kg N/ha；其他 seed 平均灌溉为 45 mm、施氮为 40–120 kg/ha。这种分化造成较大的跨 seed 管理投入和 PFP_N 离散度。

| PPO seed | 产量 (kg/ha) | WP_ET (kg/m³) | PFP_N (kg grain/kg N) | 灌溉 (mm) | 施氮 (kg/ha) |
|---:|---:|---:|---:|---:|---:|
| 0 | 7,188.5 | 2.200 | 99.83 | 45 | 80 |
| 1 | 7,188.0 | 2.200 | 66.56 | 45 | 120 |
| 2 | 7,188.5 | 2.200 | 99.83 | 45 | 80 |
| 3 | 7,118.4 | 2.179 | 197.74 | 45 | 40 |
| 4 | 7,350.2 | 2.096 | 34.01 | 222 | 240 |
| 5 | 7,188.0 | 2.200 | 66.56 | 45 | 120 |
| 6 | 7,314.9 | 1.996 | 33.87 | 219 | 240 |
| 7 | 7,337.5 | 2.049 | 33.97 | 219 | 240 |

## 结果解释

八 seed 平均产量高于 Null 和 DSSAT auto + external N，略高于 recorded farmer template，但低于 official expert；相对 expert 只有 3/8 seed 的十年均值更高。平均 WP_ET 与 Null / recorded 几乎相同（低约 0.005 kg/m³），PPO 整组不能据此称为稳定水分效率提升。灌溉、施氮以及 PFP_N 显示明显 seed 依赖：有些 seed 投入低而有些接近 expert 投入，因此目前不支持“八个 seed 都稳定获益”的结论。

FQ 2018 的冻结基线四个情景及 PPO 八个 seed 的 grain yield 全部为 0；按任务约定保留原始 WTH 中已记录的 Tmin 符号异常，不修正、不剔除该年。它影响十年绝对均值，但同一验证年对照口径一致。后续若解释 2018，应沿用该数据边界说明。

总体判断：训练和复现证据完整，八 seed cohort 的效果呈现“产量相对部分基线改善、相对专家未稳定占优，水氮管理策略对 PPO seed 敏感”。这是有效的 100K 八 seed 结果，但不是全面或稳健的管理优势证明。

## 失败尝试、修复与复用

1. seed0 首次评估发现 `build_seed_frames` 未接收 `config/env_config`；失败发生于年度回放开始前，日志保存在 `seed_00_eval_attempt_01_config_nameerror.log`。
2. 传参修复后，seed0 2014 回放快照已保存；日期核对遇到基线文本日期与 DSSAT trace datetime 类型不同。统一基线日期 dtype 后重跑并复用已保存快照，未重复运行该季 DSSAT。原错误日志保存在 `seed_00_eval_attempt_02_date_merge.log`。
3. 首次 cohort 绘图时，合并表包含每个 seed 重复的冻结基线行，折线图年份索引重复。核验四个基线在所有 seed 表逐年完全一致后，聚合前按 scenario/year 去重；原日志保存在 `cohort_summary_attempt_01_duplicate_baseline.log`，修复后聚合通过。

所有失败证据均保留；没有重训 seed0，也没有重跑/改写冻结基线。新增脚本修复前副本位于 `backups/`。

## 关键产物

- Prompt：`prompt_02/052_fqa_wgen_ppo_100k_8seed_validation_figures.md`
- 训练 runner / archive auditor：`results/fqa_multiyear_wgen_ppo_052_8seed/run_seed_100k.py`、`audit_seed.py`
- 评估 / 汇总脚本：`results/fqa_multiyear_wgen_ppo_052_8seed/evaluate_100k.py`、`summarize_cohort.py`
- seed0 复用来源和精确 checkpoint SHA 指针：`results/fqa_multiyear_wgen_ppo_052_8seed/seed_00_reuse.json`
- 训练和 WGEN 天气逐 episode 存档：seed0 在 `results/fqa_multiyear_wgen_ppo_051/attempt_01/`，seed1–7 在各自 `results/fqa_multiyear_wgen_ppo_052_8seed/seed_XX/attempt_01/`
- 验证 snapshots、Summary.OUT、daily trace：`results/fqa_multiyear_wgen_ppo_052_8seed/validation/`
- 8×23 seed 图和 cohort 图表：`results/fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ/`
- 最终门槛：`results/fqa_multiyear_wgen_ppo_052_8seed/final_gate.json`
- 代码、报告、表图、训练天气、模型及验证 Summary.OUT 文件 SHA-256 清单：`results/fqa_multiyear_wgen_ppo_052_8seed/github_sha256_manifest.csv`，由 `generate_sha256_manifest.py` 生成；天气项沿用通过 archive audit 的 episode manifest 哈希。
- 2026-10-09 冻结记录：`docs/fqa_wgen_ppo_100k_8seed_freeze_2026-10-09.md` 和 `results/fqa_multiyear_wgen_ppo_052_8seed/freeze_backup_2026-10-09.json`。

冻结备份纳入全部 40 个小型模型 ZIP，以及 80 个验证季的 `Summary.OUT` 和对应 metadata。完整 DSSAT runtime snapshots 与临时 cache 继续保留在本地，不纳入版本控制。逐 episode 天气 SHA-256 保存在各自 `episode_manifest.csv`。
