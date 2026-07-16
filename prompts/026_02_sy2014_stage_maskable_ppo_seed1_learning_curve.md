# 026_02 SY2014 阶段型 MaskablePPO seed1 短学习曲线

## 1. 科学问题

026_01 只证明 MaskablePPO 工程链路可运行。本任务进一步回答：在不修改环境、奖励、动作空间和 PPO 超参数的前提下，SY2014 seed1 的确定性策略能否在多个预先固定的更新检查点持续满足产量和水氮效率门槛。

本任务仍不是正式长训练。单 seed 通过只能允许后续独立 seed 复核，不能声称 PPO 已成功或可跨站点泛化。

## 2. 冻结环境与奖励

完全复用 026_00/026_01：

- SY2014，IC=2；
- DAP 1/30/50/65/85/110 六阶段；
- 9 个离散水氮动作；
- 剩余 I120/N300 预算 mask；
- DAP >= 90 禁止施氮；
- 非决策日 no-op；
- 021_24 固定 25 维 scaler；
- 阶段资源成本与终端产量/可行性奖励保持不变。

## 3. 冻结 PPO 配置

- `MaskablePPO(MlpPolicy)`；
- SB3 / sb3-contrib 2.8.0；
- seed=1；
- 网络 `[32,32]`；
- learning_rate=3e-4；
- gamma=1.0，gae_lambda=1.0；
- n_steps=60；
- batch_size=30；
- n_epochs=5；
- ent_coef=0；
- 总训练步数=240 个阶段步，即 40 个完整训练季。

训练按四个固定 60-step block 执行，只是为了在每次 PPO update 完成后保存和评估精确的 60/120/180/240 checkpoint。学习率和 clip range 均为常数，不存在按 block 重启的探索率日程。

## 4. 固定评估点

确定性评估 checkpoint：0、60、120、180、240。

每个 checkpoint 只评估同一个确定性 SY2014 季节一次，保存：动作序列、产量、生物量、I、N、季节奖励、WP_ET、PFP_N、primary/strict 判定。不得增加评估点或根据结果挑选额外 checkpoint。

## 5. 门槛

沿用 022_01：

- yield >= 11077 kg/ha；
- WP_ET >= 2.26 kg/m3；
- PFP_N >= 36.9 kg/kg；
- primary：I<=120、N<=300、DAP>=90 无施氮；
- strict：在 primary 基础上 I<=90、N<=250。

## 6. 预注册分支

### A_candidate_ready_for_seed_replication

同时满足：

1. 工程检查全部通过，masked action 请求为0；
2. 240-step 最终 checkpoint strict_pass；
3. 120/180/240 三个后半程 checkpoint 中至少2个 strict_pass；
4. 所有 checkpoint 和训练轨迹数值有限。

该分支只允许补一个独立 seed，不宣称训练成功。

### B_engineering_valid_but_not_stable

工程检查通过，但不满足 A 的持续 strict 条件。停止本轮，不现场调参、不挑早期 checkpoint 冒充成功。

### C_execution_failed

环境、mask、DSSAT、数值或模型保存失败。记录失败并停止。

## 7. 算力与停止边界

- 仅 seed1；
- 240 个阶段训练步，40 个训练季；
- 5 个固定评估季；
- 不延长训练；
- 不调整超参数；
- 不扩展年份、站点或 seed；
- 不覆盖 026_00/026_01 结果；
- 保存 CSV、JSON、模型检查点和中文实验记录；模型 ZIP 不提交 Git。

