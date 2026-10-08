# FQA WGEN 天气与 PPO 前置门槛归档索引（041–046）

本索引对应 FQ/FQA 单站天气路线。**截至 046，天气输入审计通过；正式 100K 或多 PPO seed 训练尚未启动。**所有步骤保留原有非空 D222 降雨值；D222 空白→0 mm 是明确的建模假设，不等同于观测到零雨。训练和评估中生成的随机天气必须保存其**实际使用的逐日序列**，不能仅凭 weather seed 复现。

| 步骤 | 问题与动作 | 最终门槛 | 主要记录 |
|---|---|---|---|
| [041](../prompt_02/041_fqa_d222_weather_resume_static_cli.md) | D222 降雨、缺失温度/辐射补值、2005–2013 日天气与 `CNFQ.CLI` 冻结 | 静态天气/CLI PASS | [记录](fqa_weather_resume_041.md)、[gate](../results/fqa_weather_resume_041/final_gate.json) |
| [042](../prompt_02/042_fqa_runtime_wgen_weather_archive_smoke.md) | FQA WGEN 运行时、同 seed 两次隔离重跑与实际日天气存档 | 运行时归档 PASS | [记录](fqa_runtime_wgen_smoke_042.md)、[gate](../results/fqa_runtime_wgen_smoke_042/final_gate.json) |
| [043](../prompt_02/043_fqa_yc_coordinate_control_and_archived_weather_pilot.md) | YC 坐标 warning 对照、FQA 五 seed 小样本与逐 episode 归档合同 | 五 seed 归档 PASS；气候分布未评估 | [记录](fqa_yc_coordinate_and_weather_archive_043.md)、[gate](../results/fqa_archived_weather_pilot_043/final_gate.json)、[归档合同](../results/fqa_archived_weather_pilot_043/per_episode_archive_contract.md) |
| [044](../prompt_02/044_fqa_wgen_ppo_2k_episode_archive_smoke.md) | FQA 冻结 16 动作 PPO 单 seed、固定 2007 年 2K WGEN smoke；每个完成/未完成 episode 均保存实际天气 | `PASS_SMOKE_ONLY`；2,016 步与 20 份、2,016 天归档闭合 | [记录](fqa_wgen_ppo_smoke_044.md)、[gate](../results/fqa_wgen_ppo_smoke_044/attempt_05/final_gate.json) |
| [045](../prompt_02/045_fqa_wgen_multiyear_heldout_archive_gate.md) | 九个训练年份与两个留出 seed 的独立进程归档接口检查 | `PASS_ARCHIVE_GATE_ONLY`；11 条、1,155 天 | [记录](fqa_wgen_multiyear_heldout_gate_045.md)、[gate](../results/fqa_wgen_multiyear_heldout_gate_045/final_gate.json) |
| [046](../prompt_02/046_fqa_wgen_full_pool_climate_qc.md) | 完整训练 80 / 留出 20 天气轨迹、同窗口气候分布筛查 | `PASS_INPUT_QC_ONLY`；100 条、10,460 天、97 天可比窗口 | [记录](fqa_wgen_full_pool_qc_046.md)、[gate](../results/fqa_wgen_full_pool_qc_046/final_gate.json) |

## 冻结输入与可追溯性

- [041 的输入/输出 SHA-256](../results/fqa_weather_resume_041/final_gate.json)标识原始 D222 降雨 Excel、`my_data/T2.xls`、`my_data/D32.xls`、NASA 候选补值与 `CNFQ.CLI`。原始 Excel 保留在项目本地，不进入本次 GitHub 包；派生的 [拟合日天气](../results/fqa_weather_resume_041/fitting_weather.csv)、[逐日来源](../results/fqa_weather_resume_041/daily_provenance.csv)和 CLI 进入包。
- [045 冻结调度](../results/fqa_wgen_multiyear_heldout_gate_045/full_schedule_plan.json)规定训练 weather seed 1001–1080、留出 1081–1100，PPO seed 与 weather seed 分离。[046 训练池 manifest](../results/fqa_wgen_full_pool_qc_046/train/episode_manifest.csv)和[留出池 manifest](../results/fqa_wgen_full_pool_qc_046/heldout/episode_manifest.csv)逐 episode 指向实际日天气、SHA-256、运行时证据与资源记录。
- 044 使用的唯一归档 checkpoint 为 [FQA seed 0 2K smoke 模型](../results/fqa_wgen_ppo_smoke_044/attempt_05/models/fqa_ppo_seed0_2k.zip)，仅用于 045–046 的确定性动作驱动。它不是正式训练结果。模型 SHA-256 由 [044 run_result](../results/fqa_wgen_ppo_smoke_044/attempt_05/run_result.json)记录。
- [GitHub 包文件清单](fqa_wgen_041_046_file_manifest.csv)记录各纳入文件的字节数和 SHA-256；在项目根目录运行 `python scripts/verify_fqa_wgen_archive_041_046.py` 可逐文件核对。清单自身不自引用。

## 失败尝试与解释边界

044 首次尝试发现 `_get_state` 每训练步会触发两次且两次天气状态并非总一致，归档门槛因此停止；之后改为 DSSAT `step` 返回时每步捕获一次。044 的第四次尝试发现 PDI YAML 在缓存进程中保留 bootstrap seed，后续 episode 要核对 reset 后的实际 `_rseed1`。失败记录保留在对应结果目录，见 [044 完整记录](fqa_wgen_ppo_smoke_044.md)。046 未删去任何生成轨迹；训练 seed 1068 在 2009-08-04 有一日生成降雨 438.96 mm，超过拟合期共同窗口最大日雨量 240 mm，但低于预设物理上界 500 mm/day。它是尾部提示，不改变预设门槛。

046 的气候检查为 6 月 11 日至 9 月 15 日共 97 天的**描述性**筛查，不证明全年分布或极端降雨真实发生概率。YC/FQA 共同见到的 native FIELD 坐标 warning 仍没有功能影响结论。045–046 的候选天气文件不能代替未来正式 PPO 运行中的真实天气归档；正式 100K/多 seed 训练、政策收益和水氮绩效均未由本归档完成或证实。

## GitHub 包范围

纳入 prompt、中文记录、构建/运行/审计脚本、冻结方案、门槛结果、精简来源与失败证据、041 派生日天气和 CLI、042–046 的实际日天气/manifest/运行时证明，以及 044 的单个 177 KB checkpoint。排除原始 Excel、重复 checkpoint、DSSAT 临时目录、渲染输入、缓存和与 FQA 041–046 无关的工作区文件。按清单复读的是这份研究记录；原始 Excel 若需异地恢复，应另按其原始数据管理方案保存。
