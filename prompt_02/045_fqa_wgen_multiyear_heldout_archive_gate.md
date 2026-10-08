# 045 FQA WGEN 多年份与留出天气归档门槛

## 任务

沿 044 已通过的 FQA PPO/WGEN 接口，在隔离目录做正式训练前的轻量门槛验证，不启动新训练：用 044 的 2K checkpoint 作确定性推断，覆盖 2005–2013 九个训练年份各一个实际 WGEN episode，并在独立进程用留出天气 seed 1081、1100 各运行一个 2007 年 episode。所有 episode 保存本次实际使用的逐日天气和运行时证据。

## 约束

1. 保留 FQA `051_00` 冻结的输入、动作、奖励、掩码和 PPO 配置；不改旧脚本或训练成果。模型仅用于接口验证，收益不作科学比较。
2. 训练天气 seed 仅来自 1001–1080，留出仅来自 1081–1100；生成可复查的完整调度方案，核对池互斥与 2005–2013 年份覆盖。九个实际训练池 episode 只是调度 smoke，不构成 80 份候选池的气候分布验收。
3. 每个 episode 从 DSSAT `step` 返回状态归档实际 `DATE,DOY,RAIN,SRAD,TMAX,TMIN`，只创建文件、复读 SHA-256、核对行数/物理筛查、运行时 FileX `WTHER=W`、CLI、PDI YAML 初始 seed 和每次 reset 后 `_rseed1`。若同 seed 的实现受进程上下文影响，保留各自哈希。
4. 训练池与留出池使用独立运行进程和输出目录；最多缓存两个年度环境，CPU 线程 1，内存上限 1.5 GB，逐 episode 检查资源。失败即停止，保留全部证据；不自动继续正式 100K/8 seed。
5. 审计逐份天气与 manifest 闭合、九年覆盖、池互斥、模型/输入哈希、资源上限。形成中文报告，明确尚未完成的 80/20 气候分布 QA 以及已知 FIELD 坐标 warning。

## 输出

`results/fqa_wgen_multiyear_heldout_gate_045/` 中的独立 runner、调度、训练池/留出池日天气、manifest、运行时证据、门槛结果；`docs/fqa_wgen_multiyear_heldout_gate_045.md`。
