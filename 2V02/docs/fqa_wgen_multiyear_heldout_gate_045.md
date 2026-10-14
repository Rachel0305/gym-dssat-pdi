# 045 FQA WGEN 多年份与留出天气归档门槛

按 [045 prompt](../prompt_02/045_fqa_wgen_multiyear_heldout_archive_gate.md)执行。使用 [044 的 2K checkpoint](../results/fqa_wgen_ppo_smoke_044/attempt_05/models/fqa_ppo_seed0_2k.zip)做确定性推断，未执行 PPO 学习，也未更改 FQA `051_00` 的动作、奖励、掩码或输入。

## 结果

[最终门槛](../results/fqa_wgen_multiyear_heldout_gate_045/final_gate.json)：**PASS_ARCHIVE_GATE_ONLY**。训练池 2005–2013 九个年份各跑一个 WGEN episode，累计 951 天；在独立进程中以留出 seed 1081、1100 跑两个 2007 年 episode，累计 204 天。共 11 个完整 episode、1,155 天实际天气，11 份归档天气哈希各不相同。逐份复读 CSV 行数/字段与 SHA-256，并核对运行时 FileX `WTHER=W`、CLI、PDI YAML 的 bootstrap seed 和每次 reset 后的 `_rseed1`；全部通过物理筛查。最高进程树 RSS 428.48 MB，低于 1,536 MB 限额。

[完整 80/20 调度方案](../results/fqa_wgen_multiyear_heldout_gate_045/full_schedule_plan.json)使用训练年份调度 seed 64003、训练天气调度 seed 64004；训练天气池为 1001–1080，留出池为 1081–1100，静态核查全部互斥。本次实际执行的是训练方案前九个 episode，以及留出方案的 1081、1100 两个 episode。训练与留出各自的[日天气与证据目录](../results/fqa_wgen_multiyear_heldout_gate_045)分别保存 manifest、CSV、运行时证据、资源记录和失败追踪入口。

## 范围边界

本次证明九个 FQA 年份及两个留出 seed 的运行、实际天气归档和调度分离能够闭合。未生成完整 80/20 个实现，也未检验其气候分布；不能据此放行正式 100K 或多 seed 训练。使用 044 checkpoint 的推断回报和产量不用于科学比较。native FIELD 坐标 warning 仍与 YC/FQA 既有记录一致，功能影响未由本门槛证明。
