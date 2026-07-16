# 022_14 SY2014 pairwise+MC 唯一一次离线训练的预算审计

## 1. 目的

022_12 已确认受控 pairwise 目标与原 MC 回归在三个 seed 上梯度兼容，但一次虚拟步只使 `Q(a1)-Q(a7)` 移动约 `1e-4`。本任务在任何正式离线训练前，使用 022_12 既有 checkpoint、状态和梯度做零训练预算估算，避免用不足的 2000 步重复一次无判别力实验。

## 2. 严格边界

- DQN 正式训练 0 步；DSSAT 调用 0 次；
- 不保存虚拟更新后的 checkpoint；
- 不扫描 lambda，不设置人为 lambda 下限；
- 不进入在线训练，不修改 reward、IC、动作空间或支持 mask；
- `022_13` warm-start 草案继续暂停。

## 3. 统一候选权重

固定：

```text
lambda_fixed = median(lambda_norm_seed0, lambda_norm_seed1, lambda_norm_seed2)
```

该值只从 022_12 三个预注册 seed 的共享层梯度范数比值得到，所有 seed 使用同一值。

## 4. 估算方法

对每个 seed、每个受控 DAP65 前缀，在同一固定 `lambda_fixed` 下重新构造一次不保存的虚拟 combined step，记录：

- 初始 margin `m0=Q(a1)-Q(a7)`；
- 因果目标 `d≈0.5`；
- 单步改善 `delta=m1-m0`；
- 残差 `e0=d-m0`；
- SmoothL1 区域。

同时报告两种外推：

1. 恒定单步改善的线性外推：`n_linear=ceil(-m0/delta)`；
2. 若当前位于 SmoothL1 二次区，采用局部常 Jacobian 下的几何收缩近似：

```text
k = delta/e0
e_n ≈ e0(1-k)^n
n_flip = ceil[log(d/e0)/log(1-k)]
```

第二种只是局部经验近似，不是训练收敛保证；网络参数、MC 梯度和 Jacobian 都会随更新改变。

## 5. 预算预注册规则

- 若任意样本 `delta<=0`、数值非有限或无法计算几何估算：判 D，停止；
- 否则取全部 9 个 `n_flip` 几何估算的最大值，向上取整到最近 1000，作为下一次唯一离线训练的固定更新预算；
- 不添加额外安全倍数；
- 同时报告达到因果目标 90% 所需的局部估算，但不以其决定训练预算，因为下一步的首要判据是纠正排序，而不是精确拟合 `+0.5`。

## 6. 输出

- 每 seed×前缀的预算估算 CSV；
- 固定 lambda、推荐预算和限制说明 JSON；
- 中文实验记录；
- DQN 正式训练 0 步、DSSAT 0 次。

