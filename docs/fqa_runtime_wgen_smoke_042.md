# 042 FQA WGEN 单季运行与逐日天气归档

## 结论

按 [042 任务](../prompt_02/042_fqa_runtime_wgen_weather_archive_smoke.md)和 YC 已完成路线，封丘隔离 FileX 在处理 1 启用 `WTHER=W`，运行时读取 041 的 `CNFQ.CLI` 并注入 `rseed1_=101`。2007 年单季两次新进程各完成 **117** 日；两份实际运行时生成的完整逐日 `RAIN/SRAD/TMAX/TMIN` 已归档，规范化字节 SHA-256 均为 `0B2DA6DEC0CFC2DA4F4FB7EEA47AFE7CB8D2CC858B573E30F31B94CCD48E6309`，逐日文件字节完全一致。状态为 `runtime_weather_capture=PASS`、`same_seed_daily_reproducibility=PASS_THIS_CONTEXT`。

**原生 FIELD 坐标检查失败。** 两次 `DSSAT48.INP` 的 FIELD 经纬高程均为缺失哨兵 `-999/-99/-99`，`WARNING.OUT` 有 `IPFLD` 读取纬度、经度、海拔失败。虽能正常结束并生成天气，不能据此证明生成器和作物计算使用了正确的封丘空间坐标；`ready_for_formal_weather_pool=false`、`ready_for_ppo=false`。机器门禁见 [final_gate.json](../results/fqa_runtime_wgen_smoke_042/final_gate.json)。HLA 032/037 曾见类似现象，但不以“其他站也出现”来证明 FQA 没有影响。

YC 既有随机天气训练的 [任务模板](../results/yc_random_weather_ppo/004_03/runtime_templates/RANDOM_WEATHER_WGEN_ppo_seed_0_2007/YCA_2007_rseed1_1066.jinja2) 的 FIELD 行也写 `-99`；这说明 FQA 与 YC 在**输入形式**上相同。当前没有该 YC episode 的原生 `DSSAT48.INP` 与 `WARNING.OUT`，因此尚不能证明两站运行时坐标处理完全相同，也不能借 YC 已完成训练来消除这个警告。

## 运行证据

| 项目 | 第一次 | 同 seed 新进程重复 |
|---|---:|---:|
| 处理 / 年份 / weather seed | 1 / 2007 / 101 | 相同 |
| FileX `WTHER` / WSTA | `W` / `CNFQ0701` | 相同 |
| 运行目录 CLI 与 041 原件哈希 | 一致 | 一致 |
| PDI `rseed1_` | 101，已核对 | 相同 |
| 逐日四变量归档 | 117/117 天 | 117/117 天 |
| 规范化天气哈希 | `0B2DA6…48E6309` | 完全相同 |
| 基础物理筛查 | PASS | PASS |
| 正常结束 | 是 | 是 |
| 峰值进程树 RSS | 约 126 MB | 见逐次结果 |
| 原生 FIELD 坐标 | 缺失 | 缺失 |

第一次[逐日天气](../results/fqa_runtime_wgen_smoke_042/runtime/attempt_01/runtime_weather_daily.csv)与第二次[逐日天气](../results/fqa_runtime_wgen_smoke_042/runtime/attempt_02_same_seed/runtime_weather_daily.csv)都作为独立原件保存；相应 [result.json](../results/fqa_runtime_wgen_smoke_042/runtime/attempt_01/result.json)、[运行快照](../results/fqa_runtime_wgen_smoke_042/runtime/attempt_01/runtime_snapshot)和 [输入清单](../results/fqa_runtime_wgen_smoke_042/input_provenance.json)记录 CLI、FileX、土壤、品种、年份、处理、seed、资源与日志。单季生成雨量约 **795.33 mm**、30 个雨日、最大日雨量约 **161.47 mm**；这些是 seed 101 的生成序列描述，不是 FQA 气候分布通过判定。本轮没有运行 PPO。

## YC 路线与后续门禁

复用了 YC 的任务副本 `WTHER=W`、PDI `rseed1_` 注入、从实际 daily state 提取天气和哈希的路线；没有把 CLI 输入或 seed 标签冒称为实际生成天气。当前两次相同只证明**2007 处理 1、CLI 固定、seed 101、各自新进程**这一上下文的复现；未来训练 episode 若在不同上下文生成不同天气，必须分别保存。

此后 FQA 和 HLA 均采用同一硬门槛：每个实际用于训练或留出的 WGEN realization 保存完整逐日天气 CSV、规范化哈希、CLI/FileX 哈希、年份、weather seed、进程与 episode 上下文；先建立可追踪归档，再开始正式训练。下一步先解决或明确隔离原生 FIELD 坐标影响，然后做少量不同 seed 的**已归档**天气质量检查，不能直接由本 smoke 扩成 80/20 天气池或 8-seed PPO。
