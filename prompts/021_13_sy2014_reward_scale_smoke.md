# 021_13 SY2014 reward 数值缩放单变量 smoke test

## 目标

为冻结 DQN benchmark 框架新增一个项目本地、显式、可审计的训练 reward 缩放层，并先以 SY2014 IC=2 seed1 做低成本 smoke test，确认缩放真正进入环境返回值和 replay buffer。**本轮不直接运行 25K。**

## 单变量

- `training_reward_scale: 1.0 -> 0.1`。

reward 原始公式、相对权重、IC、动作空间、预算、探索率、target interval、观测空间和 DSSAT 输入全部保持不变。

## 实现要求

1. 不修改 site-packages 和原始 gym-DSSAT reward；新增项目本地 wrapper；
2. wrapper 位于冻结 baseline-relative reward 外层，对最终 reward 整体乘 0.1；
3. 同时记录：
   - `raw_reward`；
   - `training_reward`；
   - `reward_scale`；
   - `yield_gain`、`water_cost_term`、`nitrogen_cost_term`；
4. 检查以下等价关系：
   - `training_reward == raw_reward * 0.1`；
   - `training_reward == 0.1*yield_gain - 0.1*water_cost_term - 0.1*nitrogen_cost_term`；
5. 检查 replay buffer 保存的 reward 与 wrapper 返回的 `training_reward` 一致；
6. 原始经济 reward 与训练缩放 reward 分开保存，正式评价和 checkpoint 选择暂时仍使用原始经济 reward；
7. 缩放关闭或为 1.0 时，旧训练路径保持不变。

## 安全与边界

- 修改已有 benchmark 入口前先备份到 `backups/021_13/`；
- 先语法检查和配置 dry-run；
- smoke 只跑 200 步，checkpoint 200，确认链路，不据此评价农艺表现或收敛；
- 不运行 25K，不改观测空间；
- 不覆盖任何旧结果；
- 保存配置、日志、replay 审计、CSV/JSON 和中文记录；
- 如果任何比例检查失败，立即停止，不扩大训练。

## smoke 通过标准

1. 实际 wrapper 启用且 scale=0.1；
2. 所有已记录步骤的 `|training_reward - 0.1*raw_reward|` 小于 `1e-6`（允许 float32 replay 保存误差时单独报告）；
3. 分项缩放求和与整体缩放误差小于 `1e-6`；
4. replay buffer reward 与实际返回的 training reward 最大误差小于 `1e-5`；
5. 无 NaN/Inf、无 OOM、无输入变更、无训练链路异常。

## 后续（不在本轮自动执行）

smoke 通过后，才允许设计 SY2014 seed1 25K 单变量对照。该对照必须复用 021_12 的参数 L2、固定状态 Q 变化和完整氮排序指标，不能只比较最终产量。

