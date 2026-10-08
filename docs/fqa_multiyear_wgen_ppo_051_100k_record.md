# FQA 多年份 WGEN PPO 100K 训练归档

本步骤接续 [FQA 047 单 seed 10K 训练预演](fqa_wgen_multiyear_ppo_047_10k_record.md)、048–050 检查点诊断，并按 [051 prompt](../prompt_02/051_fqa_multiyear_wgen_ppo_100k_with_checkpoints.md) 执行。训练目标是按 YC 已完成路线跑完 100K；本次只运行一个 FQA PPO seed，不扩展 seed 数，也不更改冻结的 FQA 047 PPO、奖励、离散动作或安全掩码合同。

## 运行合同与随机天气

- 站点 FQA，PPO seed 0，2005–2013 九个训练年份；从随机初始化开始训练，不从 10K 模型续训。
- WGEN 训练天气池为 1001–1080，PPO rollout 年份/天气调度分别使用 RNG seed 64003/64004。调度首 80 条与 045 的冻结调度逐行一致；留出 seed 1081–1100 未进入训练。
- 保持单 CPU 线程和最多缓存一个作物年度环境；每个 episode 保存 DSSAT 运行实际使用的逐日 WGEN 天气、天气 SHA-256、FileX/CLI/`_rseed1` 运行时证据、episode 日志与资源记录。
- 运行命令：`docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python /workspace/results/fqa_multiyear_wgen_ppo_051/run_100k.py`。

## 完成结果

正式训练请求 100,000 步；由于 PPO rollout 长度为 144，最终完成 **100,080 步**。耗时 **2,831.17 秒（47 分 11 秒）**，进程树峰值 RSS **1,431.64 MB**，低于 1,536 MB 的停止上限。运行结果为 `PASS_100K_ARCHIVE_ONLY`。

正式训练共归档 **956 个 episode**：955 个完整 episode 与 1 个停止时 partial episode；逐日实际天气合计 **100,080 天**，与训练步数、manifest 日数和天气数据行数完全闭合。共有 553 个不同的天气文件哈希。训练完成时最后一个 partial episode 也已保存其实际天气和运行证据。

25K、50K、75K、100K checkpoint 均在请求步数准确落盘，另保存最终 100,080 步模型。checkpoint SHA-256 如下：

| 步数 | 文件 | SHA-256 |
|---:|---|---|
| 25,000 | `checkpoint_25000.zip` | `890C7792ACE0D9A69AA70177A46058FBF73500B770BDE2B247B580263BCDE815` |
| 50,000 | `checkpoint_50000.zip` | `60FBAAFF3477D191A284983BDFF4463DA555125EF79AC7D24BDB795AB84D4D41` |
| 75,000 | `checkpoint_75000.zip` | `069F41BC4BAF6A8CE1F7F823FDB645A6E160B2BA14A552622F0DBE71B99A65A2` |
| 100,000 | `checkpoint_100000.zip` | `F449A1753A083A9CBF6938236B92755CAABED87EBC3AEABA27FDFB58B8C09207` |

最终模型 `final_model_actual_100080.zip` 的 SHA-256 为 `1C8C28E960795EA5AECA9201723912C2045081EE94E0565214257773F9592E4E`。容器内逐一加载了四个 checkpoint 和最终模型，读出的 `num_timesteps` 分别为 25,000、50,000、75,000、100,000 与 100,080。

## 审计与解释

[正式归档审计](../results/fqa_multiyear_wgen_ppo_051/attempt_01/audit_gate.json) 的 12 项检查全部通过：prompt/脚本来源哈希、冻结训练合同、训练与留出 seed 边界、045 前 80 条调度、步数/天气日闭合、逐日天气日期及物理范围、DSSAT 实际 WGEN/CLI/随机 seed、资源限制、checkpoint 文件与步数、最终模型哈希和完整 episode 汇总闭合。432 步 smoke 也通过，完整及 partial episode 的天气均有保存。

本次 PASS 证明 100K 训练按合同完成且天气/模型归档可复核。**它不代表 PPO 已经学到有效管理策略，也没有证明产量或水氮利用改善。** 本次没有执行留出策略评估，也没有比较农户管理基线。后续应将 100K 策略在冻结的 20 个留出天气 seed 上评估，并与相同天气下的 10K 检查点及固定管理对照比较产量、灌水、施氮和 `PFP_N`；在该评估前，不对 100K 策略效果作结论。

训练中主机短暂读不到容器挂载内正在更新的 CSV/目录；容器内训练进程与文件持续正常，改用容器内读数核实，随后主机挂载恢复可读。训练没有中断、重启或数据丢失。

## 文件与备份

完整执行产物位于 [`attempt_01`](../results/fqa_multiyear_wgen_ppo_051/attempt_01/)；其中 [`weather_daily`](../results/fqa_multiyear_wgen_ppo_051/attempt_01/weather_daily/) 保存训练实际使用的逐日天气，`episode_manifest.csv`、`runtime_evidence/`、`logs/`、`runtime_templates/`、`rendered_inputs/`、配置、资源记录和模型 checkpoint 共同支持复现及追溯。`temp/` 是 DSSAT/SB3 临时目录，保留在本地而未纳入 GitHub。

[文件 SHA-256 清单](fqa_wgen_051_file_manifest.csv)涵盖 prompt、脚本、smoke、完整正式训练归档及本记录。清单构建脚本为 [`fqa_wgen_051_archive_manifest.py`](../scripts/fqa_wgen_051_archive_manifest.py)；运行时清单自身不列入，以避免自引用哈希。
