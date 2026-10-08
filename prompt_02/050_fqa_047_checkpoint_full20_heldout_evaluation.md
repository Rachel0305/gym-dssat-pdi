# 050 FQA 047 PPO 5K/10K checkpoint 全部留出天气配对评估

## 目标

扩展 049 的两例 checkpoint smoke，对 047 同一次 PPO seed 0 训练保存的 5K 与 10K checkpoint，在 2007 年预留的完整 WGEN 留出 seed 1081–1100 上分别进行确定性推断。逐 seed 配对比较产量、wrapper 水氮、PFP_N、episode reward 和动作分布，判断 5K→10K 是否呈现一致的策略变化或性能方向。不是正式政策对照，也不启动训练。

## 冻结合同与资源限制

1. 仅使用 047 `checkpoint_5000.zip` 与 `checkpoint_10000.zip`，PPO seed 0、FQA 2007、留出 WGEN seed 1081–1100；每个 checkpoint 使用独立进程和目录。沿用 047 的 PPO/reward/action/safety 合同和 048/049 的环境构建。不得改模型、再训练、使用训练池 seed 或覆盖既有结果。
2. 每步记录离散动作及 wrapper 安全动作、grnwt/topwt/SWFAC/NSTRES（环境提供时）。每 episode 保存逐日实际天气、SHA-256、行数/日期/物理筛查、运行时 FileX `WTHER=W`、CLI、PDI `_rseed1` 和资源。两 checkpoint 同 seed 天气哈希必须一致，且 seed 1081/1100 哈希需与 048/049 已有归档一致。
3. 每进程 CPU 线程 1、年度环境最多缓存 1 个、进程树 RSS <1.5 GB、壁钟 <10 分钟。失败立即保存证据并停止该 checkpoint；不得放宽资源门槛后自动重试。
4. 计算每 seed 配对差值，以及 20 个 seed 的均值、中位数、标准差、方向计数；汇总正灌溉/施氮天数、非零动作类别和动作频数。只作描述性结果，不以 reward 单独判断策略优劣，不据此自动启动 100K。
5. 水氮取 wrapper 累计执行量；PFP_N=终态产量/wrapper累计N。未做 Summary.OUT 实际施用闭合时明确注明；没有 ETCP replay 不报告 WP_ET/NUE。此评估只有 2007 单年 20 个合成天气实现，不能代表跨年泛化或权威概率抽样。

## 输出

prompt、两 checkpoint runner、逐日动作/状态轨迹、40 份天气及运行时证据、配对统计、审计门槛和中文记录放入 `results/fqa_047_checkpoint_full20_heldout_050/` 与 `docs/`。门槛通过只表示 20-seed 配对评估和归档闭合，不等于模型有效或 100K 获批。
