# YC corrected CLI 与完整 WGEN seed pilot 记录

## 1. 任务范围

本轮仅完成 YC 单站点的 corrected `CNYC.CLI` 正式重生成，以及 101a、101b、102–105 六次串行 WGEN pilot。没有运行 PPO，没有修改其他站点、奖励、动作空间、观测或湿日阈值。

## 2. 正式 CLI 重生成

使用 `scripts/build_dssat_cli.py` 和已修复的 `I6 + 14*(1X,F5.0)` formatter，从冻结 2004–2013 天气重新生成正式文件。正式输出为 `results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI`，SHA256 为 `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`。

静态 schema 检查通过：12 个月的 monthly averages 与 WGEN 参数行齐全，每月 WGEN 行 90 字符、15 列；没有非有限值或格式错误。正式文件与 `003_06_05_01` parsefix 候选逐字节一致，所有 WGEN 参数值及序列化均一致，无未解释差异。正式状态：`CORRECTED_CLI_READY`。

CLI 元数据记载站点 CNYC，纬度 36.83、经度 116.57、海拔 22 m；拟合输入为 2004-01-01 至 2013-12-31 的 3653 行。拟合没有使用 validation 数据。湿日定义仍为 `RAIN > 0.0 mm`。

## 3. CLI 与输入 provenance

| 对象 | 路径 | SHA256 / 核验 |
|---|---|---|
| 冻结天气 | `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv` | `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`；3653 行，日期范围正确 |
| 正式 corrected CLI | `results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI` | `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929` |
| parsefix 对照 | `results/yc_wgen_cli_pilot/003_06_05_01/candidate/CNYC_parsefix_01.CLI` | 与正式 CLI 字节一致 |
| 历史错误 CLI | `results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI` | `5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0`；未修改 |

生成日志、字段元数据、schema 检查和正式 CLI 对照记录均保存在本任务的 `final/` 与 `validation/` 目录。formatter 回归测试已通过。

## 4. Seed 设计与接口

六次运行按顺序串行：`seed_101_run_a=101`、`seed_101_run_b=101`、`seed_102=102`、`seed_103=103`、`seed_104=104`、`seed_105=105`。全部使用 YC 2008 treatment 1、同一 FileX/cultivar/soil、`WTHER=W`、`WSTA=CNYC0801`、DSSAT 4.8.0.024 和相同 Gym-DSSAT wrapper 设置。

本轮 `ppo_seed=NOT_APPLICABLE`。Gym 构造时 `seed=None`；pilot 将 `weather_seed` 明确写入 DSSAT PDI 的 `rseed1_`，并在六份 runtime 配置中逐一确认实际值吻合。PDI 配置归一化后除每次启动动态分配的 ZeroMQ 本地端口外一致；runtime FileX 六份字节哈希一致。不存在 PPO/WGEN seed 混用。

## 5. 六次 runtime 与 same-seed reproducibility

六次 DSSAT runtime 均返回 `PASS`，WGEN daily state 均成功捕获。101a 与 101b 各有 120 行，`DATE/DOY/RAIN/SRAD/TMAX/TMIN` 完全相同，CSV SHA256 均为 `D9CD9926AE4382BA24783E498A928ED25F5EA92F1C60D1B3C7034448AAE5C29D`；首差异为无（`first_difference_if_any=null`）。

日期由 pilot 的 2008-06-01 起始日和每日观测序号生成，并与每次 `INFO.OUT` 的 CSM 起始日期及 ENDRUN 日期交叉核对。101–105 所有日期连续、DOY 与日期一致；这个窗口是 DSSAT 单季仿真期，不能解读为年度天气序列。

## 6. Different-seed diversity

| weather seed | 捕获行数 | 序列 SHA256 前 12 位 | 与其他 seed 的序列 |
|---:|---:|---|---|
| 101 | 120 | `D9CD9926AE43` | 与 102–105 均不同 |
| 102 | 117 | `240C34531DC8` | 不同 |
| 103 | 117 | `DC5C2EAA31F9` | 不同 |
| 104 | 116 | `7E6D92914D38` | 不同 |
| 105 | 118 | `94E881EBFFFF` | 不同 |

五个不同 seed 得到五条不同序列，10 组 pairwise 比较全部非相同。逐天气变量的不同日期数、共同日期数、独有日期及首差异均写入 `seed_diversity_check.json`。不同运行行数为 116–120；差异反映 DSSAT 各随机天气下的作物仿真终止日不同。逐变量比较仅在共同日期上计数，长度/日期差另行报告。

## 7. 天气物理 QC 与训练期 sanity reference

六条序列均通过有限值、日期解析、连续日期、无重复日、DOY 对齐、`RAIN >= 0`、`SRAD >= 0`、`TMAX >= TMIN`、screening 极端值及非全零降雨检查。QC 使用的极端值边界只作筛查，不代表经本地校准的极值模型。

| Seed | 累计降雨 (mm) | 雨日 | 日雨量最大值 (mm) | 最长无雨日 | 均温 Tmax (°C) | 均温 Tmin (°C) | 均值 SRAD (MJ/m²/d) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 507.73 | 36 | 56.64 | 18 | 30.67 | 20.15 | 16.44 |
| 102 | 509.51 | 35 | 62.71 | 21 | 30.61 | 19.98 | 16.94 |
| 103 | 466.44 | 37 | 73.90 | 12 | 31.32 | 20.74 | 16.01 |
| 104 | 290.10 | 22 | 51.05 | 19 | 32.11 | 21.08 | 16.45 |
| 105 | 483.08 | 30 | 97.32 | 12 | 30.87 | 20.34 | 17.03 |

以上均为 **simulation-period statistics**，每个 seed 的具体 Tmin 最小值、Tmax 最大值、SRAD 最大值等见 `weather_summary_by_seed.csv`。月均比较仅以冻结 2004–2013 训练天气为描述性 sanity reference；未用验证天气拟合，也未据此调整参数。

## 8. Runtime warnings 与未决事项

没有 WGEN/CLI lookup/parse/seed-interface runtime error，未再出现 `WGENIN 5010`。六次 `WARNING.OUT` 均包含相同类型的 DSSAT 提示：PHOTO 方法从 L 改为 C；FileX 未提供纬度、经度和海拔；`CYCRDin`、`CXCRDin`、`CELEVin` 未成功传递并在后续读取时置零。`INFO.OUT` 还报告 soil `STONES`、`ADCOEF` 使用默认值。它们没有阻止 WGEN 天气生成或本轮门禁，但尚未证明对后续 crop output 无影响。

下一步 crop-output smoke 应确认这些元数据/soil/cultivar 输入提示对作物生长、产量与水氮状态的影响；必要时先修复 YC isolated copy 并保留来源，不改变本轮 WGEN 拟合阈值。

## 9. 最终判定与 readiness

`WGEN_SEED_PILOT_PASS`：正式 corrected CLI 与冻结天气 hash/schema 均核验通过；六次运行通过；101 重复运行完全重现；102–105 与 101 组成的五个不同 seed 全部不同；物理 QC 通过；seed 接口分离明确；无 validation fitting、无 PPO 训练、无其他站点修改。该结论只代表 YC 的 CLI/WGEN pilot 通过，不代表 crop yield smoke 已通过，也不授权开展 PPO 长训。

推荐下一阶段：`003_06_06_yc_wgen_weather_qc_and_dssat_smoke`，重点进行更完整气候分布 QC，并在处理上述 runtime warnings 后运行单季 DSSAT crop-output smoke。

## 10. 测试

执行命令：

```text
python -m pytest tests/test_build_dssat_cli.py tests/test_yc_wgen_seed_pilot.py tests/test_yc_wgen_final_seed_analysis.py -q
```

结果：**27 passed, 0 failed**（0.27 s）。

执行中的工具错误均在对应阶段修复后继续：seed 编排最初读取了不存在的 `schema_pass` 字段，在启动 DSSAT 前退出；改读 schema 实际的 `passed` 字段后再开始六次运行。分析脚本首次启动时未能从 results 子目录导入 repo runner，改为自包含 WTHER 解析后重算 QC（没有重跑 DSSAT）。PPT finalizer 首轮不接受 Han script fallback 字段，移除该声明并重新导出，最终结构和版面校验通过。

## 11. 结果与文件

- 汇总：`results/yc_wgen_cli_pilot/003_06_05_02/pilot_summary.json`
- 重现性：`results/yc_wgen_cli_pilot/003_06_05_02/reproducibility_check.json`
- seed 多样性：`results/yc_wgen_cli_pilot/003_06_05_02/seed_diversity_check.json`
- 天气摘要：`results/yc_wgen_cli_pilot/003_06_05_02/weather_summary_by_seed.csv`
- 物理 QC：`results/yc_wgen_cli_pilot/003_06_05_02/physical_sanity_check.json`
- runtime warnings 与固定输入审计：`results/yc_wgen_cli_pilot/003_06_05_02/validation/`
- 六份逐日天气及逐次 runtime evidence：`results/yc_wgen_cli_pilot/003_06_05_02/generated_weather/` 与 `runtime/`
- 本轮执行 prompt 已复制至 `prompts/003_06_05_02_finalize_cli_and_resume_seed_pilot.md`，与原文件 SHA256 相同。
- 新增本轮专用的 CLI/pilot 编排与分析脚本，以及分析单测；没有更改既有训练或 runner 源码。

## 12. Git 状态

提交前检查发现工作区内存在大量与本任务无关的既有修改和未跟踪文件，均保持原样。本次仅暂存本任务 prompt、报告、结果证据、专用脚本、单测和 PPT；不会清理或覆盖其他文件。

本地提交信息：`test: finalize YC CLI and complete WGEN seed pilot`。GitHub backup pending explicit user approval；本轮不执行 `git push`。
