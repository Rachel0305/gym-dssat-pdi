# 021_24 SY2014 固定观测标准化短诊断

## 研究问题

021_23 将训练 reward 与 n-step return 整体乘以 0.1 后，TD loss 明显降低，但参数梯度不降反升，且示范预训练和在线阶段仍有 100% 更新触发梯度裁剪。同时，25 维原始观测的 `max_abs` 从约 0.20 到 20131.56，存在明显尺度不均衡。

本任务只检验：**固定、训练前确定的观测标准化，能否降低真实网络更新中的梯度与裁剪频率。**

## 防止“乱调”的预注册边界

1. 使用 SY2014、IC=2、seed=0。
2. 恢复 021_22 的原始未缩放 reward；不得叠加 reward×0.1。
3. 复用 021_22 的网络、DQfD 参数、示范数据、PER 参数、随机种子、更新次数和探索率。
4. 唯一科学变量是观测数值表示：
   - 均值和标准差仅由 021_20 已冻结的 160 条示范 `observations` 在训练开始前计算；
   - 变换固定为 `(x - mean) / std`；
   - `std < 1e-6` 的维度使用 `scale=1`，即只中心化为接近 0；
   - 同一变换应用于当前观测、1-step next observation 和 n-step next observation。
5. 不加入或删除任何观测特征，不改变 DAP、物候、预算等信息。
6. 只运行 100 次示范预训练 + 100 个环境交互步（step 50 起在线更新，共 51 次）。
7. 不进入 5K；结果出来后不得现场改标准化方法、阈值、网络或 loss 权重。

## 生效验证

正式解释前必须验证：

- 观测维度仍为 25；
- 非常量维在示范 `observations` 中标准化后均值接近 0、标准差接近 1；
- 逆变换可重构原观测，误差在 float 容差内；
- rewards、actions、dones 和 n-step returns 与原示范完全相同；
- 在线环境原始 reward 未被缩放；
- 021_22 原结果不被修改。

## 生长季分段检查

按示范状态的原始 DAP 分为：

- early：DAP < 50；
- middle：50 ≤ DAP ≤ 90；
- late：DAP > 90。

分别报告 `topwt`、`grnwt`、`cumsumfert`、`totir` 以及胁迫变量在标准化前后的 min/max/mean/std，检查固定标准化是否把早期状态压成无法区分的常数。该检查只描述信息分布，不直接证明策略可学习性。

## 主要对比指标

与 021_22 原始未标准化基准逐阶段比较：

- TD 1-step、n-step、margin、L2 loss；
- 各分量及总梯度范数；
- 梯度裁剪比例；
- Q 值绝对量级；
- 单次及累计参数相对变化；
- 示范/agent replay 采样比例。

## 预注册判断

- **A：支持观测尺度参与**：预训练和在线阶段总梯度中位数均下降至少 25%，且至少一个阶段裁剪比例明显低于 100%。
- **B：部分改善但不充分**：梯度下降但裁剪仍接近 100%，或只在一个阶段改善。
- **C：不支持观测标准化作为主要修复**：梯度未下降或上升，且裁剪仍接近 100%。

无论属于哪一分支，本任务都不得自动进入 5K。

## 输出

- `benchmark_results/021_24/021_24_observation_scaler.csv`
- `benchmark_results/021_24/021_24_growth_stage_distribution.csv`
- `benchmark_results/021_24/021_24_pretrain_update_log.csv`
- `benchmark_results/021_24/021_24_online_update_log.csv`
- `benchmark_results/021_24/021_24_agent_interactions.csv`
- `benchmark_results/021_24/021_24_stage_comparison.csv`
- `benchmark_results/021_24/021_24_validation.json`
- `benchmark_results/021_24/021_24_summary.json`
- PNG/SVG 图
- `docs/2026-07-15_021_24_sy2014_fixed_observation_standardization_short_diagnostic.md`

## 资源纪律

- 单环境、单进程；
- 不保存大型模型；
- 失败现场保留，不覆盖旧结果；
- 不自动 Git push。
