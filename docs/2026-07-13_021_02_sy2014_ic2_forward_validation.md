# 021_02 SY2014 IC=2 输入固化与前向验证记录

## 1. 背景与目的

021_01 审计发现：站点配置期望 SY2014 使用 IC=2，但当时保存的 MZX treatment 2 实际为 IC=0，导致历史训练结果与当前配置说明不一致。用户随后恢复了 SY2014 的完整 IC=2 剖面。本实验只验证输入来源与三种非 DQN 前向情景，不训练 DQN，不修改 reward、IC、土壤、天气或品种参数。

## 2. 执行约束

- 容器：`b2fd6726c8c1`
- Python：`/opt/gym_dssat_pdi/bin/python`
- 站点年份：SY2014，treatment 2
- 情景：null、recorded、DSSAT auto
- 顺序执行，先 20 步 smoke，再跑完整生长季；未并行，未启动 DQN。
- 所有运行使用新目录 `DSSAT_auto_validation/sy2014_ic2_forward_validation_021_02/`，未覆盖历史结果。

## 3. 输入固化与审计

源文件：

`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX`

SHA-256：

`20b071bc49549cbf561be3aae81caa355aa0582564db524e17d68ab6a274418a`

确认内容：

- treatment 2 的 IC 指针为 2；
- IC level 2 包含 10、20、30、40、60、100 cm 六层；
- 六层 SH2O 为 0.37、0.32、0.31、0.32、0.31、0.35；
- 六层 SNH4 为 2.30、2.30、2.05、2.05、1.90、1.75；
- 六层 SNO3 为 1.30、1.30、1.25、1.25、1.00、0.75；
- smoke 以及三个完整情景的 PDI 运行时 `fileX.MZX` 均再次通过 IC=2 审计。

## 4. 情景定义

- **null**：关闭灌溉和施肥，清零 treatment 2 的报告管理行。
- **recorded**：保留 MZX 中 treatment 2 的实测/记录管理。
- **DSSAT auto**：仅把灌溉和施肥管理方式切换为 DSSAT 自动管理。

三个情景的天气、土壤、品种、种植设置和 IC=2 完全一致。

## 5. 结果

| 情景 | HWAM/GWAD (kg/ha) | CWAM (kg/ha) | 灌溉 (mm) | 施氮 (kg N/ha) | max WSPD | max NSTD | 最终 DAP | 完整结束 | 运行时 IC=2 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| null | 5408 | 10603 | 0 | 0 | 0.000 | 0.617 | 139 | 是 | 是 |
| recorded | 9613 | 18194 | 0 | 293 | 0.961 | 0.012 | 139 | 是 | 是 |
| DSSAT auto | 5498 | 10688 | 66 | 0 | 0.000 | 0.564 | 139 | 是 | 是 |

说明：WSPD/NSTD 在这里按 `PlantGro.OUT` 原始诊断列保存，表中只报告最大值；本实验的目标是输入和链路验证，不对胁迫机理作新的因果解释。

## 6. 与历史候选扫描的复核

016_02 的 `wa1p0_ns0p5` 候选结果为：

- GRNWT = 9620.21 kg/ha；
- TOPWT/CWAD = 18201.95 kg/ha；
- max SWFAC = 0.96137；
- max NSTRES = 0.01219。

本次 recorded 结果为 9613、18194、0.961、0.012，分别只相差约 7–8 kg/ha 或四舍五入误差，说明当前恢复的 IC=2 正是历史扫描所指向的 `wa1p0_ns0p5` 水氮剖面。

同时，这也澄清了一个口径问题：016_02 的“zero action”是强化学习代理不追加动作，但仍保留 MZX recorded 管理；它对应本次 recorded，而不是真正关闭全部管理的 null。因此不能用历史约 9620 kg/ha 与本次 null 5408 kg/ha 直接比较并声称模型不一致。

## 7. 执行中发现并修复的问题

1. 首次汇总使用通用空格分隔解析 `Summary.OUT`。由于标识字段存在空白，列发生错位，曾产生 null `IRCM=281`、`CWAM=0` 等明显错误值。
2. 修复后，HWAM/CWAM 改由 `PlantGro.OUT` 最终 GWAD/CWAD 获取，灌溉与施氮改由去重后的 `MgmtEvent.OUT` 获取。
3. CSV 中的字符串 `null` 曾被 pandas 默认识别成缺失值，导致图中 null 曲线丢失；已通过 `keep_default_na=False` 修复。
4. 首次绘图期间本机命令终端暂时无响应；三个 DSSAT 运行已经完成且结果未受影响。恢复后使用无界面 `Agg` 后端完成绘图。

## 8. 结论

### 已确认

- SY2014 当前输入的 IC=2 指针、六层剖面、源文件哈希和 PDI 运行时输入一致。
- null、recorded、DSSAT auto 三个情景均能完成完整生长季。
- recorded 结果复现了历史 `wa1p0_ns0p5` 候选量级。
- SY 的“输入 provenance 阻塞”已经解除。

### 尚未确认

- 本实验没有训练 DQN，因此没有证明 SY2014 的 IC=2 DQN 能稳定优化。
- 历史 017_08 使用 IC=0 的 checkpoint 和结果仍只能作为历史证据，不能改称 IC=2 结果。
- 是否把 SY 纳入正式多 seed DQN，应在五站点优先级确定后另开任务，并先 smoke test。

## 9. 输出文件

- Prompt：`prompts/021_02_sy2014_ic2_forward_validation.md`
- 脚本：`src/run_sy2014_ic2_forward_validation_021_02.py`
- 汇总：`DSSAT_auto_validation/sy2014_ic2_forward_validation_021_02/021_02_sy2014_ic2_forward_summary.csv`
- 日值：`DSSAT_auto_validation/sy2014_ic2_forward_validation_021_02/021_02_sy2014_ic2_forward_daily.csv`
- 管理事件：`DSSAT_auto_validation/sy2014_ic2_forward_validation_021_02/021_02_sy2014_ic2_forward_events.csv`
- 图：`DSSAT_auto_validation/sy2014_ic2_forward_validation_021_02/021_02_sy2014_ic2_forward_validation.png`
- 每个情景的输入、日志、PDI 快照和运行时 IC 审计均位于对应 `runs/` 子目录。

## 10. 对当前五站点主线的影响

本实验只改变 SY 的证据状态：从“输入来源不明确”更新为“IC=2 前向链路已确认、正式 DQN 训练待完成”。它不改变以下既有判断：HLA 尚非所有基准下全面胜出；YC/FQ 当前冻结配置未全面胜出且简单成本系数扫描不敏感；LC 主要问题是跨 seed 资源策略不稳定；是否增加淋洗惩罚继续暂缓。

