# 021_16 SY2014 施氮时点确定性反事实对照记录

## 1. 背景与问题

021_15 发现，SY2014 两个 10K checkpoint 虽然季节总投入同为 I120/N300，但高产 checkpoint（11046 kg/ha）主要在吐丝前施氮，低产 checkpoint（8120 kg/ha）则将 300 kg/ha 氮全部安排在吐丝后，其中 200 kg/ha 位于灌浆开始后、灌浆期内。

该发现仍来自两个不同 checkpoint 的观察性比较。021_16 使用确定性 DSSAT 前向模拟，在总量和输入完全一致时交换管理时点，以分离施氮时点和灌溉时点的影响。

## 2. 严格控制条件

- 站点年份：SY2014。
- 输入：直接复用 021_14 固化的 IC=2 输入与环境参数。
- 模型：容器 `b2fd6726c8c1` 内 PDI/DSSAT 4.8.0。
- Python：`/opt/gym_dssat_pdi/bin/python`。
- 所有方案总投入均为 I120/N300。
- 不训练、不修改 reward、IC、动作空间、观测空间和 DSSAT 输入。
- 不经过预算 wrapper；直接回放 021_15 记录的实际管理事件，以排除动作裁剪与动作混叠。
- 物候期：吐丝 DAP89，灌浆开始 DAP99，灌浆结束 DAP136，成熟 DAP139。

## 3. 两套时序

### 3.1 早施氮时序 N-early

- DAP69：100 kg/ha
- DAP76：50 kg/ha
- DAP83：100 kg/ha
- DAP90：50 kg/ha

### 3.2 晚施氮时序 N-late

- DAP94：100 kg/ha
- DAP102：100 kg/ha
- DAP109：100 kg/ha

因此，N-late 的 300 kg/ha 全部在吐丝后，其中 DAP102 和 DAP109 合计 200 kg/ha 位于灌浆开始后。这里不能写成“灌浆结束后施氮”。

两套灌溉时序均为 8 次、每次 15 mm。I-early 位于 DAP1/8/15/22/29/36/43/50；I-late 仅将最后一次从 DAP50 移至 DAP55。

## 4. Fidelity smoke

先只运行原高产组合 I-early + N-early：

- 确定性回放产量：11046 kg/ha。
- 原 checkpoint：11046.28125 kg/ha。
- 绝对误差：0.28125 kg/ha。
- 实际 `MgmtEvent.OUT`：I120/N300，所有计划事件均执行。

因此，输入、回放与产量解析链路通过 fidelity 检查，随后才运行另外三组。

## 5. 2×2 结果

| 灌溉时序 | 施氮时序 | GWAD (kg/ha) | CWAD (kg/ha) | 实际灌溉 | 实际施氮 |
|---|---|---:|---:|---:|---:|
| I-early | N-early | 11046 | 19752 | 120 mm | 300 kg/ha |
| I-early | N-late | 8127 | 14508 | 120 mm | 300 kg/ha |
| I-late | N-early | 11026 | 19664 | 120 mm | 300 kg/ha |
| I-late | N-late | 8120 | 14461 | 120 mm | 300 kg/ha |

原低产组合 I-late + N-late 的确定性回放为 8120 kg/ha，与原 checkpoint 8120.24609 kg/ha 的误差仅 0.24609 kg/ha。

## 6. 因素对照

- 固定 I-early，只将 N-early 换成 N-late：减产 2919 kg/ha。
- 固定 I-late，只将 N-early 换成 N-late：减产 2906 kg/ha。
- 固定 N-early，只将 I-early 换成 I-late：减产 20 kg/ha。
- 固定 N-late，只将 I-early 换成 I-late：减产 7 kg/ha。
- 两套灌溉背景平均后，早施氮相对晚施氮增产 2912.5 kg/ha。
- 两套施氮背景平均后，I-late 相对 I-early 仅减产 13.5 kg/ha。

## 7. 结论

在本次 SY2014 IC=2、I120/N300 的受控 DSSAT 模拟中，两个 10K checkpoint 约 2.93 t/ha 的产量差异几乎全部由施氮时点解释，而不是由两套灌溉时点的细微差异解释。

这是受控确定性前向对照，因此比 021_15 的观察性比较更强：相同输入、相同水氮总量、相同灌溉时序下，仅交换施氮时点即可使产量下降约 2.91 t/ha；反向交换可恢复产量。

## 8. 结论边界

本实验能够说明：

- DSSAT 模型对施氮时点具有强烈且可重复的响应；
- 低产 checkpoint 的晚施氮是真实的农学低效时序，不是产量解析误差；
- DQN 在训练过程中学到的时序策略质量发生了实质变化。

本实验不能单独说明：

- 部分可观测性一定是唯一根因；
- 只加入剩余预算就一定能解决问题；
- reward、target network 或探索率已经被排除；
- 应立即修改冻结观测空间。

下一步若提出 Markov 修复，应明确讨论需加入的信息不仅可能包括剩余水氮预算和距上次操作天数，还可能包括 DAP/物候阶段或距关键生育期的相对时间。该改动属于环境定义变更，需要单独立项并由导师确认。

## 9. 输出文件

- Prompt：`prompts/021_16_sy2014_nitrogen_timing_deterministic_counterfactual.md`
- 执行脚本：`src/run_sy2014_n_timing_counterfactual_021_16.py`
- 分析脚本：`src/analyze_sy2014_n_timing_counterfactual_021_16.py`
- 汇总表：`benchmark_results/021_16/021_16_summary.csv`
- 因素效应：`benchmark_results/021_16/021_16_factorial_effects.csv`
- 汇总 JSON：`benchmark_results/021_16/021_16_summary.json`
- 每组日值、动作、`MgmtEvent.OUT` 解析和完整 PDI 临时输出：`benchmark_results/021_16/<scenario>/`

## 10. 失败与修正记录

- 查询 021_15 物候文件时最初使用了不存在的文件名 `021_15_phenology_summary.csv`，命令失败；随后改为正确文件 `021_15_dssat_phenology_by_checkpoint.csv`，未影响模拟或数据。
- 首次汇总 `MgmtEvent.OUT` 时，施氮量格式 `100.` 未被只接受 `100.0` 的正则表达式识别，导致核验表误报实际施氮为 0；原始事件文件和模拟不受影响。已备份分析脚本、放宽尾随小数点解析并重新核验。
- 无训练失败、无 DSSAT 失败、无 OOM。
- 未执行 Git commit 或 push。
