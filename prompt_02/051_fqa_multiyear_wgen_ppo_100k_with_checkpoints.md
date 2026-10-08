# 051 FQA 多年份 WGEN PPO 100K 训练与中间 checkpoint/天气归档

## 目标

参照 YC 004_03 的单次 100K 训练方式，将 FQA 047 的 PPO seed 0、originIC、WGEN 多年份训练合同从头训练至 100,000 timesteps，并在 25K、50K、75K、100K 保存模型 checkpoint。逐 episode 保存实际使用的随机天气及运行时 seed 证据，以便复现训练输入。保留 smoke、正式训练、失败和中断产生的全部中间记录。

## 冻结合同

1. 以 FQA 047 已通过的 10K 归档门槛为合同来源，从头训练一个 PPO seed 0 的 100K run；不从 047 的 10K checkpoint 热启动。保持 PPO、奖励、原始观测、16 个离散动作、安全掩码、FQ originIC 输入和 041 冻结 `CNFQ.CLI` 完全不变。不得覆盖 047–050 文件或已有训练结果。
2. 参照 YC 004_03：年份由 RNG 64003 在 2005–2013 的九年块内随机排列，WGEN seed 由 RNG 64004 在 1001–1080 的 80-seed 块内随机排列；先生成足够覆盖训练预算的明确 schedule，前 80 个 episode 必须逐项匹配 045 冻结的训练 schedule。PPO seed=0；留出 seed 1081–1100 不进入训练。
3. 正式运行前先执行 432-step 独立 smoke。Smoke 必须使用同一训练合同及 WGEN 归档路径，并通过模型可载入、实际 seed 与天气一致、每个完整/部分 episode 有日天气文件及 manifest、资源未超限等门槛；失败时保留日志和错误证据，不启动 100K。
4. 正式训练请求 100,000 环境步；允许 PPO 按 `n_steps=144` 完成 rollout 后略微超过目标，记录请求与实际步数。每次 checkpoint 保存请求步数及实际步数。正式 run 保存 25K/50K/75K/100K checkpoint，训练结束后另存最终实际步数模型。
5. 每个 episode 完成或训练停止时，保存 DSSAT step 中捕获的 `DATE,DOY,RAIN,SRAD,TMAX,TMIN` 日天气、SHA-256、日期/物理筛查、训练 seed/year/episode schedule、奖励和状态；保存运行时 FileX `WTHER=W`、CLI 哈希、PDI bootstrap 配置及运行时 `_rseed1`。训练中途退出时也要归档未完成 episode，标注 partial，并尽可能保存紧急模型 checkpoint。
6. 保存 prompt/source/input 哈希、effective config、完整 schedule、training episode summary、episode manifest、天气文件、runtime evidence、DSSAT 日志、资源时间序列、checkpoint、最终/紧急模型、终态 JSON 和失败 traceback。所有输出只写入 `results/fqa_multiyear_wgen_ppo_051/` 的独立 smoke 或正式 attempt 子目录。
7. 单进程、CPU 线程 1、最多缓存 1 个年度环境；进程树 RSS 达 1,536 MB 时在 rollout 边界停止并保存 partial/紧急 checkpoint；壁钟上限 7,200 秒。超过资源门槛或归档不闭合时保留证据并停止，不自动放宽限制或重试。

## 最终门槛与解释边界

- Smoke 通过后才启动 100K。正式门槛要求实际步数不少于 100,000；所有实际 DSSAT step 都必须与完整/部分天气归档行数闭合；训练 seed 合法；所有请求 checkpoint 和 final 实际模型都可加载；天气日期连续、物理筛查通过；运行时 WGEN/CLI/seed 证据通过；RSS 与时限通过。
- 通过仅表示“FQA 100K WGEN 训练与逐 episode 随机天气归档闭合”。不据此宣称管理政策优于基线、模型收敛或跨年泛化。水氮实际用量和政策效果须另做 Summary.OUT/基线对照。
- 保存全部原始 checkpoint 和过程文件在项目工作区。GitHub 归档只提交代码、prompt、摘要、清单及适合版本管理的紧凑证据；不提交模型权重、大型缓存或临时文件。
