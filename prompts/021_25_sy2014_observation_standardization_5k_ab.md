# 021_25 SY2014 固定观测标准化 5K 严格 A/B

## 目的

021_24 已证明固定观测标准化能将 100+100 步短诊断中的总梯度从数百至上千降到约 1，并把梯度裁剪比例从 100% 降到 0%；但它没有证明长期策略、施氮时序或产量稳定性得到改善。

本任务检验：**在相同的 literature-aligned DQfD 训练中，固定观测标准化能否在 5K 内同时改善数值稳定性与确定性策略表现。**

## A/B 定义

- Control：原始 25 维观测。
- Treatment：使用 021_24 已冻结的固定标准化 `(x-mean)/std`；统计量只来自训练前冻结的 160 条示范 observations，`std<1e-6` 时只中心化。

两组必须保持一致：

- SY2014、IC=2、seed=0；
- 原始未缩放 reward；
- 同一 160 条 oracle 示范及动作；
- 同一 DQN 初始化、网络、优化器、学习率；
- 同一 prioritized demonstration replay 参数；
- 同一 100 次示范预训练；
- 同一 5-step return、batch、探索率日程、动作约束；
- 同一 5K 训练长度和 1K 评估间隔。

不允许修改 reward、IC、action、loss 权重、target interval、探索率或示范权重。

## 执行顺序与资源纪律

1. 先验证 scaler、示范字段和输入维数；
2. 顺序运行 Control 和 Treatment，禁止并行；
3. 每 1K 保存模型并在独立环境中确定性评估；
4. 训练过程记录总梯度、裁剪、loss、Q 值和资源动作；
5. 不超过 5K，不自动扩到 25K 或其他 seed；
6. 不保存 replay buffer；模型 checkpoint 仅本地保存，不纳入 Git。

## 评估指标

每个 checkpoint 保存：

- GWAD/HWAM、CWAD；
- 总灌溉、总施氮；
- DAP>90 的施氮量；
- reward total；
- 确定性动作序列和 Q 值范围；
- 训练区间内梯度中位数、最大值、裁剪比例和 loss。

## 预注册门槛

### 数值门槛

Treatment 相比 Control：

- 5K 训练更新的总梯度中位数下降至少 75%；
- 梯度裁剪比例低于 10%；
- Q 值和参数全部有限。

### 农艺与策略门槛

Treatment 至少连续两个 checkpoint 同时满足：

- 产量 ≥ 11077 kg/ha（不低于官方推广 expert）；
- 灌溉 ≤ 90 mm；
- 施氮 ≤ 250 kg/ha；
- DAP>90 施氮量 = 0；
- 相对同 checkpoint Control，不出现超过 100 kg/ha 的产量损失，或在资源投入更低时产量损失不超过 100 kg/ha。

并且从首次达标 checkpoint 以后，不得出现产量跌到 9000 kg/ha 以下的明显坍缩。

### 判定

- A：数值门槛和农艺门槛均通过，可另立任务讨论 25K/多 seed。
- B：数值门槛通过、农艺门槛未通过，说明标准化修复数值问题但尚未驯服策略。
- C：数值门槛也未通过，不支持继续扩大标准化训练。

无论哪一结果，本任务不自动进入下一阶段。

## 输出

- `benchmark_results/021_25/training/{control,treatment}/...`
- `benchmark_results/021_25/evaluations/...`
- `benchmark_results/021_25/021_25_checkpoint_trajectory.csv`
- `benchmark_results/021_25/021_25_training_diagnostics.csv`
- `benchmark_results/021_25/021_25_preregistered_checks.csv`
- `benchmark_results/021_25/021_25_summary.json`
- PNG/SVG 图
- `docs/2026-07-15_021_25_sy2014_observation_standardization_5k_ab.md`

失败必须保留，不能覆盖旧结果；不自动 Git push。
