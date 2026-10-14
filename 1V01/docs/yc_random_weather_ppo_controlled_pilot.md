# YC 随机天气 PPO 受控 Pilot 实验报告

## 1. 假设
本实验检验：在保持 YC PPO 算法、reward、状态/动作、管理约束、作物和训练预算不变时，仅增加 WGEN 随机天气训练多样性，是否改善未见天气表现或稳健性。该命题是待检验假设，不是既定事实。

## 2. 为什么这是受控 Pilot
比较 `HISTORICAL_WEATHER` 与 `RANDOM_WEATHER_WGEN`，每组 PPO seed 为 0、1、2；两组共用同一个 `055_00` canonical config 和相同 2004–2013 年份上下文顺序。首轮六模型虽然完成，但 FileX 实际为 `WTHER=M`，并未启用 WGEN；首轮模型、评估与汇总均留档但不纳入本报告。修复后只在任务生成模板副本设置 `WTHER=W`，不修改 frozen 输入或安装 runtime。

## 3. Frozen PPO 配置
Canonical 来源：`benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/configs/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml`，正式 YC 任务配置 `configs/055_00_yca_lowIC_expanded_action_maskableppo.json`，正式完成证据 `benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/055_00_formal_result.json`；训练入口 `src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py`。口径审计发现该 YAML 是 032_00 smoke/base 模板，文件内 `total_timesteps=5000`；055_00 正式任务配置及正式结果均记录 `100000`，本 pilot 沿用正式预算 `100000`，没有把 smoke 模板值误作正式训练预算。算法为 MaskablePPO / MlpPolicy，网络 `[64, 64]`、Tanh；`learning_rate=0.0003`、`gamma=1.0`、`gae_lambda=1.0`、`n_steps=144`、`batch_size=144`、`n_epochs=5`、`clip_range=0.2`、`ent_coef=0.01`、`vf_coef=0.5`（算法默认值）。不使用观测归一化、VecNormalize 或天气预报特征。
Reward 继承 canonical stress-aware wrapper：yield 系数 0.158、water cost 1.1、N cost 1.58、水/N stress relief 系数 10.0/5.0，统一 scale 0.001；SWFAC process penalty threshold=0.05、coef=50.0。旧公式 `0.06 * final_grnwt - 0.04 * cumfert` 明确为 `SUPERSEDED / NOT_APPLICABLE`，未据此建模或改 reward。
动作网格为 irrigation `[0.0, 15.0, 30.0, 45.0]` mm × N `[0.0, 40.0, 80.0, 120.0]` kg/ha；单日上限 I/N=45.0/120.0，季节 soft limit I/N=240.0/250.0，最短间隔 I/N=7/7 天；保留 DAP90 后 45 mm irrigation reserve mask。

## 4. Seed 分离
PPO seed `0/1/2`、历史年份 schedule seed `64003`、WGEN weather schedule seed `64004`、运行初始化 seed `66003`、evaluation 初始化 seed `66004` 分别记录。随机训练 seed pool 为 `1001–1080`，按每 80 条覆盖全池一次的固定 schedule；实际 PDI `rseed1_` 与 frozen sequence 逐条一致。
训练的 WGEN 组每个 PPO seed 有 170 个不同逐日天气哈希；三种 PPO seed 的 962 条 year/seed/weather 序列完全一致。QC 观察到 7 个重复 year×seed pair 跨 schedule block 重现时有多个天气 realization；这些序列在三个 PPO seed 之间仍逐 episode 对齐，因此保留为随机发生器的复现边界 note，不把 seed 标签误称为唯一 realization ID。

## 5. 历史天气训练组
年份为 2004–2013。每连续 10 个 episode 使用独立 schedule RNG `64003` 对十年做一次排列；三种 PPO seed 使用相同年份序列。

## 6. WGEN 随机天气训练组
年份上下文与历史组相同，唯一设计差异是使用已冻结 `CNYC.CLI` 的 WGEN。任务本地 FileX 副本将所有 METHODS.WTHER 显式置为 `W`，每次 reset 按 weather schedule 设置 seed，并从 DSSAT daily state 记录实际 `RAIN/SRAD/TMAX/TMIN` 序列哈希。

## 7. 留出天气设计
Held-out WGEN seed `1081–1100`，每个模型评估 20 个 realization，训练/评估 seed 交集为空。Observed 评估使用 YC 2014–2023，严格称为 `independent comparison period`；此前没有证据认证它是 pristine final test set。策略动作 deterministic。全年 WGEN validation 仍为 `DEFERRED`，且不是本 episode-level Pilot blocker。

天气生成前置 QC `004_02` 为 `PASS_WITH_NOTES`：synthetic TMIN 均值约比 fitting observed 高 `0.58 °C`。该 note 保留为输入天气分布限制，不阻断本轮 episode-level pilot；全年 WGEN validation 仍 deferred。

## 8. 训练完成情况
| 天气组 | PPO seed | 请求/实际步数 | 完整 episode | 峰值 RSS MB | 耗时 min |
| --- | --- | --- | --- | --- | --- |
| 历史天气 | 0 | 100,000/100,080 | 940 | 927.8 | 9.3 |
| 历史天气 | 1 | 100,000/100,080 | 942 | 985.8 | 9.4 |
| 历史天气 | 2 | 100,000/100,080 | 940 | 994.9 | 9.4 |
| WGEN 随机天气 | 0 | 100,000/100,080 | 962 | 1,030.8 | 9.6 |
| WGEN 随机天气 | 1 | 100,000/100,080 | 962 | 1,046.1 | 9.6 |
| WGEN 随机天气 | 2 | 100,000/100,080 | 962 | 1,055.9 | 9.5 |
六个模型均请求 100,000 steps，实际 100,080（144-step rollout 边界导致 +80），各自保存 25K/50K/75K/100K checkpoints；训练串行，峰值 RSS 约 0.93–1.06 GB，低于 6 GB stop threshold。

## 9. Held-out WGEN 结果
| 训练组 | PPO seed | Mean reward | Median | SD | CV % | P10 | Minimum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 历史天气 | 0 | 0.586 | 0.558 | 0.155 | 26.5 | 0.369 | 0.322 |
| 历史天气 | 1 | 0.977 | 1.022 | 0.161 | 16.5 | 0.759 | 0.633 |
| 历史天气 | 2 | 0.928 | 0.941 | 0.173 | 18.6 | 0.731 | 0.549 |
| WGEN 随机天气 | 0 | 0.977 | 1.022 | 0.161 | 16.5 | 0.759 | 0.633 |
| WGEN 随机天气 | 1 | 0.939 | 0.946 | 0.180 | 19.2 | 0.735 | 0.555 |
| WGEN 随机天气 | 2 | 0.806 | 0.784 | 0.161 | 20.0 | 0.586 | 0.535 |

## 10. Observed 2014–2023 结果
| PPO seed | 历史 mean reward | WGEN mean reward | 方向 | 历史 mean yield | WGEN mean yield | 历史 N | WGEN N | 历史 I | WGEN I |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.748 | 0.331 | worse | 8,202.040 | 5,348.288 | 240.000 | 40.000 | 214.500 | 45.000 |
| 1 | 0.331 | 0.351 | better | 5,348.288 | 6,303.987 | 40.000 | 120.000 | 45.000 | 45.000 |
| 2 | 0.685 | 0.423 | worse | 6,871.321 | 6,821.757 | 80.000 | 188.000 | 102.000 | 75.000 |

## 11. Yield / Fertilizer / Irrigation 分解
| 指标（held-out pooled） | 历史天气 | WGEN 随机天气 | 相对变化 |
| --- | --- | --- | --- |
| Episode return | 0.831 | 0.907 | +9.25% |
| 产量 kg/ha | 6,985.450 | 7,006.340 | +0.30% |
| 肥料 kg N/ha | 120.000 | 118.667 | -1.11% |
| 灌溉 mm | 123.500 | 55.000 | -55.47% |
| Reward P10 | 0.510 | 0.685 | +34.29% |
收益提高不能脱离 PPO seed 差异及资源轨迹解读。NUE 未报告，因为本实验没有唯一 canonical NUE 计算链；不新建 NUE/WUE 定义。

## 12. 稳健性与下尾
报告 20 个 held-out weather seed 上每模型的 mean、median、SD、CV、P10、minimum、yield P10、fertilizer P90 和 mean irrigation。Pooled P10 reward 从历史组 0.510 变为 WGEN 组 0.685（+34.29%）。这些是描述性稳健性指标，不是显著性检验。

## 13. PPO seed 差异
| PPO seed | 历史 mean reward | WGEN mean reward | 绝对差 | 方向 | Reward % | Yield % | Fertilizer % | P10 reward % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.586 | 0.977 | 0.391 | 改善 | +66.65% | -8.10% | -83.33% | +105.55% |
| 1 | 0.977 | 0.939 | -0.038 | 下降 | -3.93% | +8.44% | +200.00% | -3.20% |
| 2 | 0.928 | 0.806 | -0.122 | 下降 | -13.15% | +1.22% | +145.00% | -19.83% |
同一 seed 配对评估时，seed 0 reward 改善，seed 1、2 下降；Observed 2014–2023 方向为 2 个 worse、1 个 better。禁止挑最好 seed，本报告纳入全部三个 seed。

## 14. Pilot 判定
判定为 **MIXED_SIGNAL**。Pooled held-out reward 变化 +9.25%，但仅 1/3 PPO seeds 改善，因此未满足 GO_SIGNAL 的至少 2/3 seed 条件；其余 pooled yield、fertilizer、P10 和 observed consistency guardrails 均通过。更大实验建议：`NEEDS_DIAGNOSIS`。

## 15. 结果能说明什么、不能说明什么
本 Pilot 的结果表明：当前配置下，WGEN 天气增强在 held-out WGEN 的 pooled reward 有描述性改善，但改善只出现在 1/3 PPO seeds，Observed comparison 也呈现 seed 异质性，因此不能称为稳定的总体改善。该结果不证明天气多样性解释了此前全部 YC PPO 不稳定，也不说明 PPO 算法有缺陷。只有 3 个 PPO seeds，不作 p<0.05 成功结论。

## 16. 下一步
按 `NEEDS_DIAGNOSIS` 处理：暂不扩成更大 seed/训练实验，不改 PPO/reward；先对 seed 0/1/2 做 action occupancy、管理剂量、训练轨迹和天气条件响应诊断，解释 seed 1/2 的 held-out reward 下降及 observed 差异，再预先冻结是否需要新的单因素实验。

## 17. 文件变更
本任务新增/更新 runner、canonical/设计/调度/天气 runtime correction 配置、训练与评估 manifest/汇总、QC、图和本报告；实验日志记录首轮 `WTHER=M` 无效尝试及修复。有效模型/逐 episode 日志保留在 `results/yc_random_weather_ppo/004_03/training_verified/` 与 `evaluation/verified_weather_refresh/`，大模型和逐日原始输出不纳入 Git。DSSAT runtime、binary、canonical reward、CNYC.CLI、PPO 算法均未修改。

## 18. 测试与 QC
- Runner syntax compile、schedule self-test：PASS。
- Canonical-source preflight：PASS；预算来源拆分与 055_00 正式完成证据核对通过。
- Seed smoke `smoke_gate_wther_w_cached_process_probe_01.json`：`PASS`，432/432 steps、4 个天气 seed/hash，WTHER=W，checkpoint 可加载。
- Training QC：PASS，6/6 models、同配置/同步数，WGEN 实际天气序列跨 PPO seed 一致；训练/评估 seed overlap 为空。
- Evaluation QC：PASS，120 条 held-out WGEN、60 条 observed；实际 WGEN seed 与天气 hash 匹配，确定性标记及 reward 分解通过。
- Reward decomposition 最大绝对误差：0。

## 19. Git 状态
仓库在本任务开始时已有无关工作区修改；这些改动保持原样。提交时仅选择 YC random-weather Pilot 相关源码、配置、中文实验记录和紧凑汇总/图；不 stage 模型 checkpoint、runtime 日志或无关文件。请求的本地提交 subject 为 `experiment: run YC random-weather PPO controlled pilot`；`git push = NO`。

## 附：结果文件
Machine-readable decision：`results/yc_random_weather_ppo/004_03/pilot_decision.json`。QC：`training_qc_verified_retry1.json`、`evaluation_qc_verified.json`。指标表及六张 PNG 位于 `results/yc_random_weather_ppo/004_03/evaluation/` 与 `figures/`。
