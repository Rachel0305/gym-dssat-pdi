# 021_27 SY2014 示范 return-to-go 信用分配离线 A/B

## 研究问题

021_26 证明单纯把示范预训练从100次增加到5000次，5个非零专家动作的召回率仍为0。信用分配审计发现，这些动作距终止收获还有60–117步，而5-step return仅包含即时资源成本，不包含正的收获奖励。

本任务检验：**只把示范的n-step监督目标改成完整季节折扣return-to-go，能否让网络学会稀疏非零示范动作，同时保持no-op状态识别。**

## A/B

- Control：现有5-step示范return。
- Treatment：对每条示范transition计算至终止状态的完整折扣return-to-go：
  `G_t = Σ gamma^k r_(t+k)`；终止后不bootstrap。

## 冻结项

- 不调用DSSAT/PDI；
- 不在线训练；
- 使用同一160条示范、标准化观测、seed=0和网络初始化；
- 即时reward、actions、dones、1-step target全部不变；
- gamma=0.99；
- DQfD/PER、margin、lambda、学习率和网络全部不变；
- target network在离线预训练期间保持初始冻结；
- 不改变奖励函数，不新增产量奖励，不修改成本系数；
- 不根据中间结果现场调任何参数。

## 预注册里程碑

`0, 10, 25, 50, 100, 250, 500, 1000` 次示范更新。

## 实现验证

- Treatment除n-step相关字段外与Control逐项相同；
- 所有Treatment n-step target均终止、不bootstrap；
- 5个非零动作的return-to-go确实包含终端收获奖励；
- 所有数组、Q值和参数有限。

## 每个里程碑指标

- 非零动作召回率；
- no-op准确率；
- 非零专家动作Q-margin及正margin比例；
- 预测非零动作比例；
- TD1、长期回报、margin loss；
- 总梯度和裁剪比例。

## 预注册门槛

单里程碑通过：

- 非零动作召回率≥0.8；
- no-op准确率≥0.95；
- 非零动作正Q-margin比例≥0.8；
- 所有数值有限。

整体判定：

- A：Treatment至少两个连续里程碑通过，且Control没有达到相同条件——完整季节信用分配是重要参与因素，可另立在线A/B；
- B：Treatment只有孤立里程碑或只改善margin/召回而未稳定通过——方向有信号但不充分；
- C：Treatment也无实质改善——不能靠简单return-to-go解决，应回到loss结构或算法设计。

无论结果如何，不自动进入在线训练。

## 输出

- `benchmark_results/021_27/021_27_target_audit.csv`
- `benchmark_results/021_27/021_27_learning_curve.csv`
- `benchmark_results/021_27/021_27_nonzero_event_predictions.csv`
- `benchmark_results/021_27/021_27_update_diagnostics.csv`
- `benchmark_results/021_27/021_27_summary.json`
- PNG/SVG图
- `docs/2026-07-15_021_27_sy2014_demo_return_to_go_credit_assignment_ab.md`

不自动Git push。
