# 017_08 SY 本地 DQN 训练年筛选与跨年迁移

## 背景

017_07 证明 HL/YC 已训练 DQN 模型不能直接迁移到 SY，因为 observation space 维度不同：

- HLA2010 模型：24 维
- YC2014 模型：22 维
- SY 环境：25 维

因此不再尝试直接加载外站模型，而是改为：

> 使用同一套 DQN 方法、同一套奖励函数、同一套动作/预算约束，在 SY 本地选择一个训练年训练，再迁移到 SY 其他年份。

## 目标

1. 筛选 SY 已有 treatment 年份：2012、2014、2015。
2. 比较 null、recorded、DSSAT auto 三个本地基准。
3. 根据“存在优化空间”的标准选择训练年。
4. 对候选训练年进行 baseline-relative DQN checkpoint 训练。
5. 将最佳 checkpoint 迁移到 SY 其他年份，不重新训练。

## 统一 DQN 设置

- 算法：DQN
- 动作空间：9-action
  - I：0 / 15 / 30 mm
  - N：0 / 50 / 100 kg/ha
- 总预算：
  - I ≤ 120 mm
  - N ≤ 300 kg/ha
- 单次上限：
  - I ≤ 30 mm
  - N ≤ 100 kg/ha
- 最小操作间隔：7 天
- 决策窗口：DAP 1–120
- 奖励函数：

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

## 训练年选择规则

优先选择满足以下条件的年份：

1. null 产量明显低于 recorded 或 DSSAT auto，说明存在管理增产空间；
2. recorded 或 auto 的增产不是完全不可达的极端高投入；
3. 生长季能完整运行；
4. 水分或氮胁迫有可解释变化。

如果多个年份都满足，优先选择 `recorded - null` 产量差最大的年份作为训练年；其他年份作为迁移验证年。

## 输出

保存到：

`DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08`

至少包括：

- `017_08_sy_baseline_screen_summary.csv`
- `017_08_sy_dqn_checkpoint_summary.csv`
- `017_08_sy_dqn_transfer_summary.csv`
- `017_08_sy_dqn_transfer_daily.csv`
- `017_08_sy_dqn_transfer_events.csv`
- `figures/017_08_sy_baseline_screen.png`
- `figures/017_08_sy_cross_year_transfer_summary.png`
- `docs/2026-07-06_017_08_sy_local_dqn_train_cross_year_transfer_record.md`

## 注意

- 不修改原始 SY 输入包。
- 所有 MZX 改写只发生在结果目录下的临时 input 文件夹。
- 不覆盖旧结果。
- 若训练前基准筛选显示 SY 三年都没有合理优化空间，则停止训练并记录原因。

