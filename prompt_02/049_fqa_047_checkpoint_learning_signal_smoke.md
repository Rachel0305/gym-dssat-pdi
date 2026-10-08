# 049 FQA PPO 5K/10K 同轨迹 checkpoint 学习信号 smoke

## 目标

用 047 同一次 PPO seed 0 训练保存的 5K 与 10K checkpoint，在 048 已跑过的两个 2007 留出 WGEN seed（1081、1100）上进行确定性配对推断。检查策略行为是否发生变化，并复核产量、wrapper 累计水氮、episode reward 与动作分布。此为低成本诊断 smoke，不作统计性效果结论。

## 冻结范围与安全门槛

1. 只读取 047 的 `checkpoint_5000.zip`、`checkpoint_10000.zip`、冻结配置和 048 的计划/归档。不得训练、微调、覆盖既有结果或启动 100K。
2. 两 checkpoint 分别在新进程、隔离目录运行；同一站点 FQA、历史年 2007、留出天气 seed 1081/1100，环境配置与 047 的 PPO/reward/action/safety 合同相同。确定性推断，单进程 CPU 线程 1、缓存年度环境最多 1 个、RSS <1.5 GB、每 checkpoint 壁钟 <5 分钟。
3. 每步保存选择的离散动作、wrapper 安全动作、grnwt/topwt/SWFAC/NSTRES（存在时），每 episode 继续保存实际 WGEN 天气、哈希和运行时 FileX/CLI/PDI seed 证据。5K/10K 同 seed 天气 SHA 必须相同才可比较。
4. 比较动作类别数、动作频数、正灌溉/施氮天数、wrapper 累计量、末态产量和 episode reward。逐日天气、动作轨迹、checkpoint 哈希、合同、资源及结果闭合后，门槛标记 `PASS_CHECKPOINT_DIAGNOSTIC_ONLY`。
5. 如果策略行为和结果几乎不变，可以说这两个 checkpoint 在这两个样本上没有显示明显学习变化；不能据此断言没有学习。如果行为变化但收成与资源权衡不一致，也只记录现象。不得由此自动扩大到 20 seed 或 100K。

## 交付

运行入口、逐日轨迹、实际天气与运行时证据、对照表、门槛及中文记录保存在 `results/fqa_047_checkpoint_learning_signal_049/` 和 `docs/`。明确注明仅两个天气实现，且水氮来自 wrapper 累计量、未以 Summary.OUT 闭合。
