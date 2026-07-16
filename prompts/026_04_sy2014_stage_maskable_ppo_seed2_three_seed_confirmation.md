# 026_04 SY2014 阶段型 MaskablePPO seed2 三 seed 确认

## 1. 目的

在 seed0/seed1 已按同一配置获得持续 primary 信号后，新增完全独立的 seed2，判断阶段型 MaskablePPO 是否能够达到三 seed primary 复现。

primary 是主要科学判据；strict I<=90/N<=250 仅作为更保守的二级资源档位报告。本任务不修改或补充 reward，不将 strict 失败自动解释为 reward 错位。

## 2. 唯一变量

相对 026_02/026_03，唯一变化是训练 seed=2。以下全部冻结：

- SY2014、IC=2；
- 六阶段、9动作、预算与晚期施氮 mask；
- 固定观测 scaler 和奖励；
- MaskablePPO `[32,32]`；
- learning_rate=3e-4；
- gamma=1、gae_lambda=1；
- n_steps=60、batch_size=30、n_epochs=5；
- 240阶段步、40训练季；
- checkpoint 0/60/120/180/240。

## 3. 固定选模规则

- checkpoint0只作随机初始化参照；
- 在60/120/180/240中选择确定性季节reward最大者；
- reward并列取更早checkpoint；
- 不根据primary/strict结果改变选择。

## 4. 预注册分支

### A_three_seed_primary_signal

- 工程检查全部通过；
- seed2预注册选中checkpoint为primary；
- seed2的120/180/240至少2/3 primary；
- seed0、seed1已有对应条件继续成立。

### B_two_of_three_primary_only

seed2工程通过但任一seed2科学条件失败。保留seed0/seed1的两seed证据，不延长seed2、不调参。

### C_execution_failed

环境、mask、数值或模型保存失败。

## 5. 边界

即使A分支通过，也只能称为“SY2014单年份、三seed的初步可复现成功候选”。下一步需冻结三seed模型和选择规则，再做独立年份迁移/验证；不能直接声称五站点泛化。

