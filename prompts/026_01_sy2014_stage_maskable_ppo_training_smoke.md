# 026_01 SY2014 阶段型 MaskablePPO 最小训练 smoke

## 1. 目的

在 026_00 已验证的六阶段 Gymnasium 环境上，确认 `sb3-contrib==2.8.0` 的 `MaskablePPO` 能够：

1. 读取每一步的离散动作 mask；
2. 完成至少两次 PPO rollout/update；
3. 全程不请求被 mask 的动作；
4. 保存并重新加载模型；
5. 用确定性策略完成训练前、训练后各一个 DSSAT 季节评估。

本任务是工程 smoke，不用于判断 PPO 是否学到成功农田管理策略，不允许根据短训练结果调整超参数。

## 2. 固定输入与环境

- 站点年份：SY2014，IC=2；
- 环境：`StageDecisionEnv026`；
- 决策时点：DAP 1/30/50/65/85/110；
- 动作：9 个离散水氮组合；
- DAP >= 90 时禁止施氮；
- 请求动作若超过剩余 I120/N300 预算，必须被 mask，而不是执行层裁剪；
- 非决策日强制 no-op；
- 观测使用 021_24 固定 25 维 scaler；
- 奖励保持 026_00 已闭合的阶段资源成本加终端产量/可行性奖励，不修改公式或系数。

## 3. 固定 PPO 配置

- 算法：`sb3_contrib.MaskablePPO`；
- 版本：SB3 2.8.0、sb3-contrib 2.8.0；
- seed：1；
- policy：`MlpPolicy`；
- 网络：`[32, 32]`；
- learning_rate：`3e-4`；
- n_steps：12；
- batch_size：12；
- n_epochs：2；
- gamma：1.0；
- gae_lambda：1.0；
- ent_coef：0.0；
- total_timesteps：24。

24 个阶段步等于 4 个完整季节，形成两个 rollout/update。这个规模只用于接口与数值 smoke，不是性能实验。

## 4. 执行顺序

1. 依赖版本审计；
2. 新建随机初始化模型；
3. 训练前确定性评估 1 季；
4. 固定 24 timesteps 训练；
5. 训练后确定性评估 1 季；
6. 保存模型并重新加载；
7. 对同一已保存观测与 mask 比较保存前后确定性动作；
8. 保存每个训练/评估季的动作、产量、I、N、奖励和 mask 审计结果。

## 5. 预注册通过条件

同时满足以下条件才判为 `A_smoke_passed`：

- 依赖版本均为 2.8.0；
- `model.num_timesteps == 24`；
- 完成 4 个训练季；
- 发生至少 2 次 rollout；
- 训练和评估过程中 masked action 请求数为 0；
- 每季恰有 6 个阶段动作；
- 所有奖励、loss/日志摘要和模型参数均为有限值；
- 模型文件存在且可重新加载；
- 重新加载前后对同一观测和 mask 的确定性动作完全一致。

产量、WP_ET、PFP_N 和是否超过 expert/auto 只记录，不设置成功门槛，也不据此声称 PPO 成功。

## 6. 停止条件

- 任何环境、mask、数值或序列化检查失败，判 `C_smoke_failed` 并停止；
- 不自动增加步数；
- 不改 reward、动作空间、IC、网络或超参数；
- 不扩展 seed、年份或站点；
- 不启动正式长训练。

## 7. 输出

- `benchmark_results/026_01/026_01_result.json`
- `benchmark_results/026_01/026_01_episode_summary.csv`
- `benchmark_results/026_01/026_01_stage_actions.csv`
- `benchmark_results/026_01/026_01_model.zip`（本地验证用，不提交 Git）
- `docs/2026-07-16_026_01_sy2014_stage_maskable_ppo_training_smoke.md`
