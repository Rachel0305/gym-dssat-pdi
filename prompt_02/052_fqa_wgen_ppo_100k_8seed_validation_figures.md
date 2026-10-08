# 052 FQA WGEN PPO 八个训练 seed 与五情景结果图

## 目标

按用户明确要求，将 FQA WGEN PPO 100K 扩展为 PPO seed 0–7 的八 seed 队列，并在相同的 2014–2023 验证年份、相同四个冻结管理基线下评估，生成与 `results/hl_fq_8seed_cross_site/055_03_five_scenario/HL/best_seed_seed5/figures` 同结构的五情景图集。

## 八 seed 训练合同

1. PPO seed 0 已由 051 完成 100K：请求 checkpoint 100,000、最终实际 100,080 步。复用 051 的 `checkpoint_100000.zip` 和完整天气归档，不重训 seed 0，不用最终 100,080 模型替代精确 100,000 checkpoint 做正式对比。
2. 从头训练 PPO seed 1–7；不得 warm-start 任一其他 seed。保持 FQA 047/051 冻结的 PPO、奖励、原始观测、16 离散动作、安全掩码、originIC、WGEN、CLI 与环境合同。
3. PPO seed 0–7 共用与 051 完全一致的年份/weather schedule（RNG 64003/64004；年份 2005–2013；WGEN seed 1001–1080）。这让不同 PPO seed 在相同训练天气序列上比较；PPO/torch/numpy/python RNG seed 使用对应 0–7。留出 seed 1081–1100 不进训练。
4. 新 runner 先对 seed 1 做 432-step WGEN smoke，并验证实际 runtime `_rseed1`、天气行哈希、checkpoint、归档闭合和资源限制。通过后再启动正式 seed 1–7。
5. **严格顺序执行** seed 1、2、3、4、5、6、7；任一运行结束并通过 audit 后再启动下一个。每个正式 seed 请求 100,000 步，并在 25K/50K/75K/100K 保存 checkpoint，最终另存实际步数模型。所有完整/部分 episode 都保存实际逐日天气、manifest、runtime evidence、DSSAT 日志和资源曲线。
6. 单进程、CPU 线程 1、最多缓存一个年度环境；每个 seed 的进程树 RSS 上限 1,536 MB、壁钟上限 7,200 秒。训练失败或门槛失败时保留已有证据并停止队列，不放宽限制，不重写失败 attempt。
7. 每个 seed 都必须检查：模型可加载、checkpoint step 正确、天气与实际 `_rseed1` 一致、所有训练步与归档日数闭合、无留出 seed、天气日期与物理检查通过、完整 archive audit 通过。

## 验证与图集

1. 使用 checkpoint 100,000（seed0 来自 051，seed1–7 来自本次训练）对 FQA originIC 的 2014–2023 十个验证年逐年运行 deterministic policy。每次仅运行一个模型/season，保存实际 daily trace、DSSAT snapshot 与 Summary.OUT；记录 checkpoint/model/input 哈希和复现命令。
2. 与固定的四情景结果比较：Null、recorded farmer template、DSSAT auto + external N、official extension expert。复用冻结 baseline 输出，不能重跑或改写基线。
3. 每个 seed 输出 23 张 055_03-style PNG：10 张逐日过程、10 张逐年灌溉/N 柱图、3 张跨年产量/WP_ET/PFP_N、管理投入及 common reward 图；并输出 yearly tables、daily traces、事件表和 weather-alignment audit。
4. 保存八 seed 总表，报告 seed 间 mean/SD 与逐 seed 结果；不挑“最好 seed”代表总体，不将单 seed 曲线当成八 seed 的总体结论。yield、灌溉、施氮、WP_ET（只由确切 Summary.OUT/ETCP 计算）及 PFP_N 分开呈现；没有 plant N uptake 时不推断 NUE。
5. FQ 2018 原始 WTH 中已记录的 Tmin 符号异常按参考目录规则保留并明确标注，不擅自修正或剔除。
6. 保存 051 seed0 指针、所有 seed 训练与评估产物、脚本、配置/输入哈希、prompt、报告、图、表、审计和 GitHub 文件 SHA-256 清单到 `results/fqa_multiyear_wgen_ppo_052_8seed/` 及相关 docs。不得覆盖 051 或 055_03 的既有结果。GitHub 不上传 DSSAT/SB3 temp/cache。

## 结论门槛

完整结果状态仅在 seed 0–7 均通过训练归档审计、验证十年覆盖、五情景图表齐全和跨 seed 汇总复核后标为 `PASS_8SEED_TRAINING_EVALUATION_FIGURES`。这表示八 seed 结果可比较、图表可追溯；是否有稳定管理收益必须依据逐 seed 与总体产量、水氮投入、WP_ET、PFP_N 结果判断，不由归档 PASS 推出。
