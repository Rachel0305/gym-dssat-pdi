# 022_12 SY2014 受控 pairwise 目标与 MC loss 梯度冲突审计

## 1. 目的

022_10–022_11 在三个预注册水分前缀中一致确认：DAP65 时 action1（I15/N0）相对 action7（I15/N100）几乎不损失产量，但完整季节回报高约 500。022_08 的三个离线 MC-Q checkpoint 却仍把 action7 排在 action1 前面。

本任务只回答一个问题：把这个受控因果排序写成 pairwise loss 后，是否会与原有完整季节 MC 回归发生严重梯度冲突，或明显扰动其他阶段。任务不训练 DQN、不调用 DSSAT、不保存更新后的模型。

## 2. 预注册方法

### 2.1 Pairwise 目标

三个 DAP65 决策前状态分别取自：

- `W60_critical__N200_early`
- `W75_uniform_pre90__N200_early`
- `W120_critical__N200_early`

目标不是人为 margin，而是由 022_10/022_11 的受控 DSSAT 完整季节回报直接计算：

```text
d_i = (G_action1,i - G_action7,i) / 1000
L_pair = mean SmoothL1(Q(s_i,a1) - Q(s_i,a7), d_i)
```

原目标保持为 022_08 训练集 216 条记录动作的完整季节 MC 回归：

```text
L_MC = mean SmoothL1(Q(s,a_recorded), G_MC/1000)
```

### 2.2 权重推导

不扫描 pairwise 权重。仅为虚拟审计步计算：

```text
lambda_norm = ||grad_shared(L_MC)|| / ||grad_shared(L_pair)||
```

共享层严格定义为 `net.0` 和 `net.2` 两个隐藏线性层；`net.4` 为动作输出层，单独纳入全参数统计但不用于 lambda 推导。

随后只在内存中构造一次虚拟步：

```text
theta' = theta - 1e-4 * [g_MC + lambda_norm * g_pair]
```

不保存 `theta'`，也不用于 DSSAT 评估。

### 2.3 预注册兼容判据

单 seed 必须同时满足：

1. 共享层梯度余弦相似度不低于 -0.20；
2. 虚拟步后 `L_pair` 下降；
3. `L_MC` 相对增加不超过 1%；
4. DAP1 与 DAP110 留出状态的支持集 argmax 均不改变；
5. 所有关键数值有限。

至少 2/3 seed 兼容才进入 A 分支。

## 3. 执行记录

### 3.1 两次非科学性执行问题

1. 第一次本地运行因 022_11 汇总表同时含 `label` 与 `arm`，脚本错误优先读取简写 `arm` 而停止；修正为优先使用完整 `label`。未产生科学结果。
2. 第二次本地运行完成主要计算，但 Windows 本机 PyTorch 与 Matplotlib 的 OpenMP 运行库冲突，在保存图时退出。未使用不安全的 `KMP_DUPLICATE_LIB_OK` 绕过，改用既有 Docker 虚拟环境执行同一脚本。

正式成功命令：

```text
docker exec -w /workspace b2fd6726c8c1 /opt/gym_dssat_pdi/bin/python src/audit_sy2014_pairwise_mc_gradient_conflict_022_12.py
```

正式运行：DQN 训练 0 步，DSSAT 调用 0 次。

## 4. 受控目标核对

| 前缀 | action1 回报 | action7 回报 | d（scaled） |
|---|---:|---:|---:|
| W60 critical | 7384.004 | 6884.111 | 0.499894 |
| W75 uniform | 7383.655 | 6883.761 | 0.499894 |
| W120 critical context | 7346.230 | 6846.503 | 0.499727 |

三组目标方向一致，且全部直接来自已完成的受控季节回放。

## 5. 梯度审计结果

| Seed | shared cosine | shared opposite-sign fraction | lambda_norm | L_MC 相对变化 | L_pair 变化 | DAP1/110 argmax变化 | 兼容 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | -0.0412 | 0.5085 | 0.006005 | -0.000038% | 0.149324→0.149261 | 0 / 0 | 是 |
| 1 | 0.0593 | 0.4712 | 0.003901 | -0.000052% | 0.349565→0.349462 | 0 / 0 | 是 |
| 2 | 0.0049 | 0.4788 | 0.006736 | -0.000044% | 0.214407→0.214302 | 0 / 0 | 是 |

所有阶段（DAP1/30/50/65/85/110）在 72 个留出状态上的支持集 argmax 改变数均为 0。最大单动作 Q 绝对变化小于 `9.45e-5`。

三个 checkpoint 在虚拟步前仍全部错误偏好 action7；虚拟步后 `Q(a1)-Q(a7)` 均向正确方向移动，但单步幅度约为 `1.1e-4–1.6e-4`，远不足以立即翻转排序。

## 6. 判定

**A 分支：3/3 seed 兼容。**

可以确认：

- 受控 pairwise 目标与现有 MC 回归在共享层不存在预注册定义下的严重冲突；
- 使用共享层梯度范数推导的权重时，一次虚拟步可降低 pairwise loss，且没有损害 MC loss 或扰动其他阶段 argmax；
- Claude 所建议的“共享层连带影响”“固定 loss 公式”“权重可追溯推导”均已纳入审计。

不能确认：

- 不能说 pairwise 约束已经让 DQN 学会正确策略；
- 不能说在线训练一定稳定或一定达到 expert/auto 目标；
- 不能把一次很小的虚拟步改善解释为排序已经修复。

特别需要注意：`lambda_norm` 仅为 0.003901–0.006736，说明未经归一化的 pairwise 梯度远大于 MC 梯度。当前结果支持“存在一个兼容的归一化组合”，不支持任意放大 pairwise 权重。

## 7. 下一步边界

只允许另立一个预注册任务：冻结本次公式，推导并固定单一 lambda，在 022_08 离线训练中加入 pairwise loss，检查 DAP65 排序能否改善且其他阶段性能不退化。该实验仍应先保持离线；不得直接进入在线 DQN、不得扫描 lambda、不得根据结果现场改 margin 或判据。

## 8. 输出文件

- `src/audit_sy2014_pairwise_mc_gradient_conflict_022_12.py`
- `benchmark_results/022_12/022_12_controlled_pair_targets.csv`
- `benchmark_results/022_12/022_12_seed_gradient_summary.csv`
- `benchmark_results/022_12/022_12_stage_virtual_step_perturbation.csv`
- `benchmark_results/022_12/022_12_pair_margin_before_after.csv`
- `benchmark_results/022_12/022_12_result.json`
- `benchmark_results/022_12/022_12_gradient_audit.png`
- `benchmark_results/022_12/022_12_gradient_audit.svg`

## 9. 可复现性与 Git 状态

脚本完全复用 022_03、022_08、022_10、022_11 的既有文件，运行中不调用随机采样、不改 checkpoint、不调用 DSSAT。当前任务未执行 Git commit 或 push。
