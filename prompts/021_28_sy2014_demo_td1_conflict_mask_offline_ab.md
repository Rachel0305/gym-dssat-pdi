# 021_28 SY2014 示范1-step冲突TD屏蔽离线A/B

## 研究问题

021_27证明完整季节return-to-go已经把5个非零示范动作的长期目标变为正值，但同一动作的1-step TD仍由即时资源成本产生负目标。两项Smooth L1等权时，在五个动作上均形成`+1`和`-1`的饱和梯度，初始化时精确相消。

本任务检验：**仅在示范样本上屏蔽冲突的1-step TD，保留完整return TD和margin，能否让网络学会5个非零示范动作。**

## A/B

- Control：示范loss = 1-step TD + full-return TD + margin + L2。
- Treatment：示范loss = full-return TD + margin + L2；1-step TD仍计算和记录，但其示范梯度权重固定为0。

## 冻结项

- 完全离线，不调用DSSAT/PDI；
- 不进行在线环境交互；
- 两组均使用完整季节return-to-go；
- 同一160条示范、固定标准化、seed和网络初始化；
- 即时reward、长期target、PER采样概率和priority更新方式不变；
- priority仍按原1-step TD error更新，避免同时改变采样机制；
- n-step、margin和L2权重不变；
- target network保持初始冻结；
- agent经验的1-step TD未来仍应保留；本任务只屏蔽示范样本上的冲突项；
- 不现场调整margin、lambda、采样权重或里程碑。

## 预注册里程碑

`0, 10, 25, 50, 100, 250, 500, 1000`。

## 指标与通过门槛

每个里程碑记录：非零动作召回率、no-op准确率、正expert-margin比例、预测非零比例、各loss、总梯度和裁剪比例。

单里程碑通过：

- 非零动作召回率≥0.8；
- no-op准确率≥0.95；
- 非零动作正Q-margin比例≥0.8；
- Q值和参数有限。

整体判定：

- A：Treatment至少两个连续里程碑通过，Control不通过——示范1-step冲突是重要机制，可另立在线A/B；
- B：Treatment仅孤立通过或只部分改善——方向有信号但不稳定；
- C：Treatment无实质改善——不能靠屏蔽示范1-step TD解决。

无论结果如何，不自动在线训练。

## 输出

- `benchmark_results/021_28/021_28_learning_curve.csv`
- `benchmark_results/021_28/021_28_nonzero_event_predictions.csv`
- `benchmark_results/021_28/021_28_update_diagnostics.csv`
- `benchmark_results/021_28/021_28_summary.json`
- PNG/SVG图
- `docs/2026-07-15_021_28_sy2014_demo_td1_conflict_mask_offline_ab.md`

不自动Git push。
