# 044 FQA WGEN PPO 单种子 2K 训练与逐 episode 天气归档 smoke

## 任务

在独立目录中运行 FQA 一个 PPO seed（seed 0）、约 2,000 步 WGEN 训练 smoke。沿用 FQA `051_00` 的原始观测、16 个离散水氮动作、奖励与安全掩码、PPO 超参数及 originIC 输入。天气接入沿 YC 已完成的 WGEN 运行时路线：FQA `CNFQ.CLI`、FileX `WTHER=W`、PDI `rseed1_`、训练天气 seed 1001–1080；本次 smoke 固定 FQA 2007 年以集中验证多次 episode。固定年只用于 smoke，不表示正式训练的年份抽样已验证。

## 硬门槛

1. 不改历史训练脚本、奖励、wrapper、原始 WTH/SOL/Jinja2 或既有结果；新脚本和输出都放在 `results/fqa_wgen_ppo_smoke_044/`。运行前核对源文件与配置并记录 SHA-256。
2. 每个完成的 episode 在下次 reset 前保存**本次 PPO 实际运行**捕获的逐日 `DATE,DOY,RAIN,SRAD,TMAX,TMIN`；只创建文件，复读哈希、行数、种子、CLI/FileX 身份并写 manifest。训练停止时尚未结束的 episode 也保存已使用的逐日天气，明确标 `partial_at_stop`。归档失败立即停止，不静默继续。
3. PPO seed 与天气 seed 分离；天气 seed 从 1001–1080 的训练池按固定调度抽取，留出 1081–1100 不参与训练。保留实际天气实现哈希，不假定同 seed 跨上下文相同。
4. 限定单进程、CPU 线程 1、一个 seed、2K 步、进程树内存上限及有限运行时长；保存资源日志、失败日志和模型 checkpoint。先做配置预检，不因 smoke 通过自动启动 8-seed 或 100K。
5. 审核训练步数与所有已使用的逐日天气归档行数闭合，WTHER/PDI seed 证据、文件复读哈希、物理筛查、模型可载入和资源上限。协调错误仍按 043 中已发现的共享 YC/FQA warning 记录，不宣称已修复。气候分布及正式天气池 QA 仍独立评估。

## 输出

新 runner、冻结配置及输入哈希、episode 日天气 CSV、manifest、训练日志、checkpoint、门槛 JSON 和中文结论。若预检或运行失败，保存已取得的证据并明确阻塞点。
