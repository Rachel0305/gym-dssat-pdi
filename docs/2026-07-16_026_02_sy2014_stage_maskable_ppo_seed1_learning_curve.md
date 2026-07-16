# 026_02 SY2014 阶段型 MaskablePPO seed1 短学习曲线记录

## 1. 目的

在 026_01 工程 smoke 通过后，用一套预先冻结的配置检查 SY2014 seed1 的确定性 PPO 策略能否在多个固定检查点持续满足产量、水分效率、氮效率和严格资源门槛。

## 2. 固定配置

- 算法：`sb3_contrib.MaskablePPO`；
- SB3 / sb3-contrib：2.8.0；
- seed：1；
- 网络：`[32,32]`；
- learning rate：`3e-4`；
- `gamma=1.0`，`gae_lambda=1.0`；
- `n_steps=60`，`batch_size=30`，`n_epochs=5`；
- 总训练步数：240 个阶段步，即 40 个完整季节；
- 固定确定性评估点：0、60、120、180、240。

成功门槛沿用 022_01：产量至少 11077 kg/ha、WP_ET 至少 2.26、PFP_N 至少 36.9；primary 允许 I<=120/N<=300，strict 进一步要求 I<=90/N<=250。

## 3. 工程检查

所有工程检查通过：

- 版本正确；
- 检查点和总步数精确；
- 完成 40 个训练季；
- 训练和评估的 masked action 请求为 0；
- 每个评估季均为 6 个阶段动作；
- 指标和模型参数均为有限值。

因此本次 B 分支不是运行故障，而是预注册科学稳定性门槛未通过。

## 4. 固定检查点结果

| 步数 | 动作序列 | 产量 kg/ha | I mm | N kg/ha | WP_ET | PFP_N | reward | primary | strict |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 6,0,0,4,4,2 | 11211.36 | 60 | 200 | 2.33 | 56.1 | 6.3634 | 是 | 是 |
| 60 | 4,0,0,8,2,2 | 11202.68 | 105 | 150 | 2.26 | 74.7 | 6.5597 | 是 | 否 |
| 120 | 3,0,8,8,2,2 | 11181.70 | 120 | 250 | 2.29 | 44.7 | 6.0237 | 是 | 否 |
| 180 | 3,0,8,8,2,2 | 11181.70 | 120 | 250 | 2.29 | 44.7 | 6.0237 | 是 | 否 |
| 240 | 4,0,8,8,2,1 | 11171.64 | 120 | 250 | 2.23 | 44.7 | 6.0136 | 否 | 否 |

预注册 A 分支要求最终 checkpoint strict，同时 120/180/240 至少 2/3 strict。实际后半程 strict 为 0/3，最终 strict=false，因此判为：

`B_engineering_valid_but_not_stable`

## 5. 解释边界

1. 0-step 的 strict 策略来自随机初始化网络，不是 PPO 学习成果，不能作为成功策略。
2. 60-step 策略的 PPO 奖励高于 0-step，但使用 105 mm 灌溉，超过严格 90 mm 门槛。这说明当前奖励排序与严格资源目标并非完全同义；PPO 转向该策略不应简单解释为算法没有学习。
3. 120/180-step 策略产量和两种效率仍通过 primary，但灌溉达到 120 mm；最终 240-step 的 WP_ET 降到 2.23，primary 也失败。
4. 单个 seed、240 步不足以判定 PPO 方法总体失败，但已经足够否决“按当前奖励和配置直接补 seed”的计划。
5. 不能挑选 0-step 或 60-step checkpoint 冒充稳定训练成功，也不能现场增加训练步数。

## 6. 当前结论与下一步约束

- 阶段型 MaskablePPO 可以稳定运行、遵守动作约束并改变策略。
- 当前奖励下，PPO 没有持续保持严格节水目标，最终也没有保持 primary。
- 按预注册规则，`next_step_allowed=false`，本轮不扩展 seed、年份或站点。
- 如果继续 PPO，下一任务必须先明确科学目标如何进入优化目标：是把 I<=90/N<=250 作为约束式动作/季节可行域，还是重新预注册与 expert/auto 对齐的多目标奖励。该选择属于目标定义，不应通过扫描水氮惩罚系数现场调参。

## 7. 输出

- `prompts/026_02_sy2014_stage_maskable_ppo_seed1_learning_curve.md`
- `src/run_sy2014_stage_maskable_ppo_seed1_curve_026_02.py`
- `benchmark_results/026_02/026_02_result.json`
- `benchmark_results/026_02/026_02_checkpoint_summary.csv`
- `benchmark_results/026_02/026_02_checkpoint_stage_actions.csv`
- `benchmark_results/026_02/026_02_training_episode_summary.csv`
- 固定 checkpoint ZIP（本地保留，不提交 Git）
