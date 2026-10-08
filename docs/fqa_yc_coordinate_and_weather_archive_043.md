# 043 YC 坐标对照与 FQA 随机天气小规模归档

## 结论

按 [043 计划](../prompt_02/043_fqa_yc_coordinate_control_and_archived_weather_pilot.md)完成 **1 条 YC 对照 + 5 条 FQA 不同 seed + 1 条 FQA 同 seed 新进程重复**，均为 2007 年处理 1、WGEN `WTHER=W`、零动作、无 PPO。7 条运行都正常结束；实际 runtime daily state 的完整逐日 `RAIN/SRAD/TMAX/TMIN`、规范化 CSV、SHA-256、输入清单和日志快照均分别归档。逐文件复读检查通过，详见 [realization_manifest.csv](../results/fqa_archived_weather_pilot_043/realization_manifest.csv)与[机器门禁](../results/fqa_archived_weather_pilot_043/final_gate.json)。

**YC 与 FQA 在本次原生坐标症状相同：** 两者的 `DSSAT48.INP` FIELD 均为经度 `-999`、纬度 `-99`、海拔 `-99`，两者 `WARNING.OUT` 均有读取纬度、经度、海拔的 `IPFLD` 警告。由此可确认 042 的 FQA 警告不是 YC 对照路线中未出现的新症状；仍不能证明它对 WGEN 或作物结果没有功能影响。坐标影响状态为 `UNRESOLVED`，不能把“共同出现”写成“坐标正确”。

FQA 五个预先固定 weather seeds `1001–1005` 的实际天气哈希 **5/5 不同**；seed `1001` 的新进程重复与首次逐日 CSV **字节完全相同**。这证明小样本不同 seed 有天气多样性，且固定 2007/处理 1/CLI/新进程上下文下 seed 1001 可复现；不保证缓存进程或不同 episode 上下文下只凭 seed 能唯一复现。全部种子的实际逐日天气都已保存，不存在仅留哈希或 seed 的归档缺口。

## FQA 小样本描述

| weather seed | 归档日数 | 单季降雨 mm | 雨日 | 最大日雨 mm | 逐日天气 SHA-256 前 12 位 |
|---:|---:|---:|---:|---:|---|
| 1001 | 107 | 553.0 | 20 | 72.3 | `2AAEB1954154` |
| 1002 | 114 | 580.9 | 33 | 91.9 | `F148BFAC293B` |
| 1003 | 114 | 348.7 | 24 | 80.9 | `99F91EAC52D` |
| 1004 | 112 | 261.1 | 19 | 47.8 | `6924429AA6AC` |
| 1005 | 108 | 369.1 | 24 | 78.7 | `C6D6C1E9B939` |

五份归档日期连续、四变量有限、RAIN/SRAD 非负、TMAX ≥ TMIN，运行时 CLI 与冻结输入哈希一致，PDI `rseed1_` 与请求一致；峰值进程树 RSS 约 126–127 MB。各季长度不同，表内单季雨量不可当作相同固定时间窗的气候分布检验。**完整 80/20 池、全年/生长季气候分布 QC 和 PPO 均未执行。**

## 归档链与下一步

[capture_one.py](../results/fqa_archived_weather_pilot_043/capture_one.py)在每次隔离运行中捕获并保存实际逐日天气；[evaluate_archive.py](../results/fqa_archived_weather_pilot_043/evaluate_archive.py)复读验证 7 条归档的行数、日期、物理基本条件、输入哈希、运行时 CLI 和 seed。后续训练入口需按[逐 episode 归档合同](../results/fqa_archived_weather_pilot_043/per_episode_archive_contract.md)把捕获与落盘嵌入训练 episode；本轮状态是 `CONTRACT_DEFINED_NOT_INTEGRATED`，不能把 043 零动作 Pilot 误称为 PPO 归档已落实。

下一阶段先在新 FQA PPO smoke 入口实现并验证**实际训练 episode** 的天气归档，再做有限训练与留出 seed 的天气质量检查。坐标警告作为 YC/FQA 共同运行时限制记录，并在正式结果前明确其对功能解释的边界；现有证据不支持单凭 043 宣布正式天气池或 PPO 放行。HLA 也必须执行同一逐 realization 归档合同，不能仅复用 seed 列表。
