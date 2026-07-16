# 026_03 SY2014 阶段型 MaskablePPO seed0 primary 复核

## 1. 目的与判据层级

026_02 seed1 在 60/120/180 三个训练后 checkpoint 通过 primary，但没有通过额外的 strict I<=90 门槛，且 240-step 最终点失去 primary。本任务不改算法或奖励，只换 seed0，检验“相对 expert/auto 的产量与水氮效率优势”能否复现。

- primary 是 022_01 预注册的主要科学判据，并与导师“尽量同时超过 expert 和 DSSAT auto 的产量与水氮利用效率”目标对应；
- strict I<=90/N<=250 是后续追加的更保守二级资源档位；
- 026_02 的原 B 分支不追溯修改；本任务在运行前明确使用 primary 作为是否允许继续的主判据，strict 只报告。

## 2. 唯一变量

相对 026_02，唯一变化是训练 seed 从 1 改为 0。其余全部冻结：

- SY2014、IC=2；
- 六阶段、9 动作、预算和晚期施氮 mask；
- 相同观测 scaler 与奖励；
- MaskablePPO `[32,32]`；
- learning_rate=3e-4；
- gamma=1、gae_lambda=1；
- n_steps=60、batch_size=30、n_epochs=5；
- 总步数240；
- checkpoint 0/60/120/180/240。

## 3. 预注册模型选择

- 只在训练后 checkpoint 60/120/180/240 中选择；
- 选择确定性季节 reward 最大者；
- reward 并列时选择更早 checkpoint；
- checkpoint0 仅作随机初始化参照，禁止入选；
- 选择规则不查看 primary/strict 后再改变。

## 4. 预注册分支

### A_cross_seed_primary_signal

同时满足：

1. seed0 工程检查全部通过；
2. seed0 预注册选中 checkpoint 为 primary；
3. seed0 的 120/180/240 至少 2/3 primary；
4. 已有 seed1 按同一选择规则选中 checkpoint 为 primary；
5. seed1 的 120/180/240 至少 2/3 primary。

该分支只表示两个 seed 都出现持续 primary 信号，仍不能宣称正式成功；下一步需冻结模型选择协议并做更长训练或独立年份验证。

### B_seed0_not_replicated

工程通过但任一科学条件不满足。停止，不增加 seed、不延长步数、不调参。

### C_execution_failed

工程、环境、mask 或数值检查失败。

## 5. 算力边界

仅新增 seed0 的 40 个训练季和 5 个固定评估季；不修改输入、IC、reward、动作空间或 PPO 参数；不扩站点和年份。

