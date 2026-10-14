# 047 FQA 多年份 WGEN PPO 单 seed 10K 训练预演

本步骤接续 [041–046 天气与归档索引](fqa_wgen_041_046_archive_index.md)。[047 prompt](../prompt_02/047_fqa_multiyear_wgen_ppo_10k_archive_gate.md)限定 FQA、PPO seed 0、2005–2013 九个训练年份及 WGEN 训练 seed 1001–1080；只检验多年份 PPO 训练和**实际使用天气**的归档链，未启动正式 100K 或其他 PPO seed。

## 冻结合同与运行

使用 FQA `051_00` 的 originIC 输入、原始观测、16 动作、奖励、安全掩码和 PPO 超参数；与 044 的 `ppo/reward/discrete_actions/action_safety` 四组配置逐项相同。年份采用 RNG 64003 的九年块排列，天气 seed 采用 RNG 64004 的 80-seed 块排列；[047 完整计划](../results/fqa_multiyear_wgen_ppo_047/attempt_01/planned_episode_schedule.json)前 80 条与 [045 冻结调度](../results/fqa_wgen_multiyear_heldout_gate_045/full_schedule_plan.json)完全一致。使用 041 `CNFQ.CLI` 与运行时 FileX `WTHER=W`。训练未使用留出 seed 1081–1100。

[最终门槛](../results/fqa_multiyear_wgen_ppo_047/attempt_01/final_gate.json)为 **PASS_10K_ARCHIVE_ONLY**，13/13 项审计通过。请求 10,000 步，PPO 因 144 步 rollout 实际运行 10,080 步；96 个完整 episode 加 1 个停止时部分 episode，各有一份[逐日实际天气](../results/fqa_multiyear_wgen_ppo_047/attempt_01/weather_daily)及[运行时证据](../results/fqa_multiyear_wgen_ppo_047/attempt_01/runtime_evidence)，合计恰好 10,080 天。全部天气文件复读哈希、行数、日期连续性、四变量物理检查与实际 `_rseed1` 核对通过，97 个实现哈希不同。[episode manifest](../results/fqa_multiyear_wgen_ppo_047/attempt_01/episode_manifest.csv)保留每条实现的年份、weather seed、天数、文件哈希和状态；最后一条明确为 `partial_at_stop`，39 天。

单进程、CPU 线程 1、年度环境最多缓存 1 个；最高进程树 RSS 524.91 MB，低于 1,536 MB 上限；总耗时 295.7 秒。5K/10K checkpoint 与可载入的[最终 10K 模型](../results/fqa_multiyear_wgen_ppo_047/attempt_01/models/fqa_multiyear_wgen_ppo_seed0_10k.zip)保存在本地。模型 SHA-256 与运行概况见 [run_result.json](../results/fqa_multiyear_wgen_ppo_047/attempt_01/run_result.json)。

## 解释边界与归档

本次的 PASS 仅证明多年份 10K 训练和逐 episode 天气归档可行。没有进行留出天气政策评估、产量/灌溉/施氮对照或正式 100K 训练；不能据训练 reward 宣称策略改善。046 的候选天气文件不能替代此处归档的训练实际天气，也不能替代未来正式训练的天气归档。native FIELD 坐标 warning 仍未取得功能影响结论。

GitHub 归档纳入本步 prompt、脚本、配置/调度、门槛、manifest、97 份逐日天气和运行时证据，以及一个最终 checkpoint；排除重复 checkpoint、DSSAT 临时目录、渲染输入和缓存。归档文件的字节数与 SHA-256 见 [047 文件清单](fqa_wgen_047_file_manifest.csv)。
