# 021_13 SY2014 reward 数值缩放单变量 smoke test

## 1. 目的

021_12 发现，SY 标准化候选结果的 reward 中位数约为 YC/FQ/LC 的 5.05–6.66 倍，但尺度偏大尚未被证明是 DQN 振荡根因。本轮只验证“完整 reward ×0.1”能否真实、可审计地进入训练和 replay buffer，不评价其是否改善策略。

## 2. 修改边界

唯一新变量：

```text
training_reward = 0.1 × raw_baseline_relative_reward
```

未改变：

- 原始 reward 公式及水氮相对权重；
- IC=2；
- 9 动作、I120/N300 预算和操作窗口；
- seed1、探索率日程、target interval、网络和 replay 设置；
- DSSAT 输入与观测空间；
- checkpoint 评价与选择继续使用未缩放的原始经济 reward。

修改前已备份：

- `backups/021_13/train_runner.py.before_021_13.bak`
- `backups/021_13/validators.py.before_021_13.bak`

新增项目本地 wrapper：`benchmark/reward_scaling.py`。没有修改 site-packages 或 gym-DSSAT 原始 reward。

## 3. 实现

外层 wrapper 同时记录：

- `raw_reward`；
- `training_reward`；
- `yield_gain`；
- `water_cost_term`；
- `nitrogen_cost_term`；
- 整体缩放恒等式误差；
- 分项缩放后求和恒等式误差；
- replay buffer 与环境实际返回 reward 的对齐误差。

只有配置显式设置 `reward.training_scale != 1.0` 时才启用。旧配置未写该字段时仍走原路径。

## 4. dry-run

- 站点：SY2014；
- IC：2；
- seed：1；
- 总探索日程：50K；
- smoke 实际运行：200 步；
- checkpoint：200；
- config hash：`e6717a33db33e212c620a8cb961753a1549dec455bc447f99fc694090832ed26`。

dry-run 通过后才运行 smoke。

## 5. 第一次 smoke：安全失败与原因

第一次 200 步运行按预设在审计失败后停止，未继续扩大训练。缩放公式本身已经通过：

- 整体缩放误差：0；
- 分项缩放误差：0；
- 字段缺失：0。

失败来自 replay 对齐：wrapper 记录 200 条，replay 只有 199 条。核查 SB3 callback 顺序后确认：callback 在 `env.step()` 之后、replay 插入之前执行；绝对步数停止 callback 返回 False 时，最后一个环境 reward 已被 wrapper 看见，但尚未写入 replay。

原审计错误地用 replay 199 条对齐 wrapper 的“后199条”，导致错位。改为对齐“前199条”后，最大误差仅 `1.2207×10⁻⁵`。第一次失败结果、日志和 checkpoint 全部保留，没有删除或覆盖。

## 6. retry smoke 结果

| 检查项 | 结果 |
|---|---:|
| 环境 step | 200 |
| wrapper 记录 | 200 |
| replay 已提交 transition | 199 |
| callback 边界未提交最后一步 | 是 |
| reward scale | 0.1 |
| 整体缩放最大误差 | 0 |
| 分项缩放求和最大误差 | 0 |
| 分项字段缺失 | 0 |
| replay 最大绝对误差 | 0.000012207 |
| float32 允许容差 | 0.000136311 |
| 所有 reward 有限 | 是 |
| 审计 | **通过** |
| 200步 epsilon | 0.994599 |
| SB3 内部探索总日程 | 50000 |

原始 reward 范围为 -500 至 5717.317；训练 reward 范围为 -50 至 571.732，符合严格的 0.1 倍关系。

## 7. 科学解释边界

本轮只证明：

> 显式缩放层确实作用于 DQN 接收到并写入 replay 的训练 reward，同时没有改变原始经济评价口径。

本轮不能证明：

- reward 缩放能避免坍缩；
- reward 缩放能提高产量或水氮效率；
- 200 步模型具有农学意义；
- reward 尺度是已确认根因。

200 步确定性评价显示的产量或资源投入仅是链路检查产物，不用于科学结论。

## 8. 下一步门槛

只有用户确认后，才运行 SY2014 seed1 25K 单变量对照。正式对照除产量、灌溉、施氮和 reward 轨迹外，必须原样复用 021_12 的：

- online/target 参数相对 L2；
- 固定状态平均/最大 Q 变化；
- 54 组完整氮排序改变数；
- 18 状态 argmax 改变数。

结果措辞只能是“缩放前后过程差异”，不能基于一个 seed 宣称机制或稳定性已经解决。

## 9. 输出

- `prompts/021_13_sy2014_reward_scale_smoke.md`
- `benchmark/reward_scaling.py`
- `configs/experiments/021_13_sy2014_reward_scale_smoke.yaml`
- `configs/experiments/021_13_sy2014_reward_scale_smoke_retry.yaml`
- `src/finalize_sy2014_reward_scale_smoke_021_13.py`
- `benchmark_results/021_13/021_13_smoke_verification.csv`
- `benchmark_results/021_13/021_13_summary.json`
- 两次 smoke 的完整配置、日志、checkpoint、replay、逐步 reward 记录和审计 JSON。

## 10. 状态

`smoke_passed_after_callback_boundary_fix`。正式 25K **尚未启动**。

