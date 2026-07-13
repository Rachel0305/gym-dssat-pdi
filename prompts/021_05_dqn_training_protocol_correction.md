# 021_05 DQN 探索率时间轴修复与 SY2014 seed0 最小验证

## 背景

代码审计和保存模型核查确认：HLA、YC、FQ、LC、SY 的正式 DQN 训练均按 checkpoint 分段调用 `model.learn()`，且第一次只传入 5K 增量。虽然全局计划为 50K、`exploration_fraction=0.35`，所有站点的 5K 模型 `exploration_rate` 已经降到 0.05，实际探索衰减约在 1.75K 内完成，而不是预期的 17.5K。

## 目标

只修正 SB3 探索率的全局时间轴，不修改 IC、reward、动作空间、预算、DSSAT 输入、DQN 网络或其他超参数，并保留 checkpoint/resume 能力。

## 修复原则

1. 全局计划训练步数仍为 50K。
2. 每个分段调用向 SB3 传入 `planned_total - model.num_timesteps`。
3. 使用 callback 在下一个绝对 checkpoint 停止本段调用。
4. `reset_num_timesteps=False`。
5. 因此每段内部 `_total_timesteps` 始终等于 50K。
6. 记录每个 checkpoint 的 exploration rate 和 SB3 内部时间轴。
7. smoke 可设置 `run_until_timesteps=5000`，但 exploration schedule horizon 仍为 50K。

禁止直接采用“每段都传 50000”：SB3 会再加当前 `num_timesteps`，导致第二段内部目标漂移到 55000。

## 分级执行

1. 备份 `benchmark/train_runner.py`。
2. 运行不调用 DSSAT 的纯 SB3 单元测试，比较：
   - 旧增量分段；
   - 每段错误地传完整全局值；
   - 正确的剩余计划步数；
   - 单次连续训练；
   - 保存、加载、续跑。
3. 对新 benchmark runner 做 dry-run。
4. 仅运行 SY2014 IC=2 seed0 的 5K smoke：
   - planned total=50K；
   - run until=5K；
   - checkpoint=5K。
5. 通过条件：
   - 5K exploration rate 约为 0.729，而不是 0.05；
   - `_total_timesteps=50000`；
   - runtime audit 全通过；
   - 输入仍为确认的 IC=2 哈希；
   - 无 OOM、无旧结果覆盖。
6. smoke 通过后才决定是否运行 SY2014 seed0 修复版正式 50K。

## 输出

- `benchmark_results/021_05/` 下的单元测试结果和 smoke 结果；
- 中文实验记录；
- 修复前后探索率对照；
- Git 本地提交。未经用户确认不 push。

## 解释边界

本任务只验证训练协议修复。不得把 smoke 产量用于优越性结论；不得同时修改 reward scaling、target update、学习率或其他变量。
