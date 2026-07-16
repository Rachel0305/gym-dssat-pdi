# 021_46 SY2014 终端 bonus 的 epsilon 调度离线审计

## 问题

021_45 的 Control 与 Treatment 在 1000-step 确定性评估中均得到 10787 kg/ha、I90/N150。需要排除该一致终态是否由探索率在 750–1000 步已接近 0、策略被锁死造成。

## 范围

- 只读 021_42 seed1 与 021_45 已有 training interactions；
- 读取实际 epsilon 公式所在源码常量；
- 不调用 DSSAT，不训练，不改 reward、IC、模型或旧结果。

## 检查

1. 明确 021_45 是否调用 SB3 `model.learn()` 的 exploration schedule，还是自定义 `epsilon_at(step)`；
2. 核对源码中的 `PLANNED_TIMESTEPS`、`EXPLORATION_FRACTION` 和终止 epsilon；
3. 验证每个日志 step 的 epsilon 与源码公式逐点一致；
4. 汇总四个 250-step 区间的 epsilon 范围、均值、随机/贪心动作比例；
5. 比较 Control 与 Treatment 的 epsilon 轨迹是否完全一致。

## 判定

- 若 750–1000 步 epsilon 接近 0.05，则支持“探索衰减压缩”怀疑；
- 若仍显著高于 0.05 且日志与全局 50K 公式一致，则排除该怀疑；
- 即使排除，也不得仅凭两个终态相同断言存在 reward 无关的固定吸引点。

