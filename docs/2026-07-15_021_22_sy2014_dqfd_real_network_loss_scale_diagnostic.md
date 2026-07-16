# 021_22 SY2014 文献对齐 DQfD 真实网络损失与梯度诊断记录

## 边界

本轮使用真实SY2014 IC=2环境和真实SB3 DQN网络，执行100次示范预训练与100个环境交互步；在线更新从step50开始，共51次。没有完成生长季、没有5K训练、没有评价产量或选择checkpoint，也没有调整任何预注册参数。

## 阶段汇总

| phase | updates | gradient_clip_fraction | median_total_gradient_norm | max_total_gradient_norm | median_td_n_to_td1 | median_margin_to_td1 | median_l2_to_td1 | median_sample_demo_fraction | cumulative_parameter_relative_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrain | 100 | 1.0 | 1274.074646 | 7445.462402 | 2.635415 | 0.019899 | 7e-06 | 1.0 | 0.045374 |
| online | 51 | 1.0 | 387.18338 | 1674.183105 | 2.047877 | 0.020458 | 8e-06 | 0.625 | 0.0356 |

## 诊断警报

- component_dominance_alert: `False`
- frequent_gradient_clipping_alert: `True`
- pretraining_shift_alert: `False`
- all_values_finite: `True`

这些警报按运行前规则计算，只用于决定下一任务，不在本轮现场调整lambda、priority参数或更新频率。

## 真实损失与梯度量级

| 阶段 | median TD1 loss | median n-step loss | median margin loss | median total grad | 裁剪比例 |
| --- | --- | --- | --- | --- | --- |
| 示范预训练 | 64.897 | 174.805 | 1.415 | 1274.075 | 100% |
| 在线诊断 | 56.066 | 124.054 | 1.367 | 387.183 | 100% |

margin并没有压制TD：其阶段中位数仅约为weighted 1-step TD的2%。主要数值压力来自1-step和5-step TD，且预训练和在线阶段每一次更新的裁剪前总梯度都超过10；预训练最大总梯度为7445.46，在线最大值为1674.18。由于每一步都触发梯度裁剪，当前配置不能被视为已经具备安全进入5K的训练动力学条件。

该结果只定位了“TD/梯度量级过大并持续触发裁剪”，尚不能单独区分reward/TD target尺度、原始观测尺度、优先采样分布或这些因素的交互。按预注册纪律，本轮不现场缩放reward、归一化观测或修改lambda。

## 解释边界

1. 预训练和在线阶段均保存了每次更新的四项raw/weighted loss、分量梯度、total gradient、参数步长和Q量级，可检查趋势而非只看终点。
2. 100步时epsilon仍接近1，因此agent经验主要来自探索动作；这是一项损失链路诊断，不代表可用策略。
3. 若损失和梯度有限，只能说明工程链路可运行；是否允许5K仍需结合警报和完整曲线另立任务判断。
4. 本轮结果不能用于声称DQfD有效或无效。

## 失败记录

第一次运行完成了数值计算和CSV/JSON写出，但在最终绘图时因Pandas列名`update`与DataFrame方法同名，`frame.update`被解释为方法而非列，触发横纵坐标长度错误。完整失败现场保存在`benchmark_results/021_22_failed_attempt_1_plot_attribute_collision/`。随后只把绘图访问改为`frame["update"]`并从头运行；没有修改任何科学参数、输入或诊断逻辑。

## 输出

- `benchmark_results/021_22/021_22_pretrain_update_log.csv`
- `benchmark_results/021_22/021_22_online_update_log.csv`
- `benchmark_results/021_22/021_22_agent_interactions.csv`
- `benchmark_results/021_22/021_22_stage_summary.csv`
- `benchmark_results/021_22/021_22_median_loss_gradient_summary.csv`
- `benchmark_results/021_22/021_22_loss_gradient_diagnostic.png/.svg`
- `benchmark_results/021_22/021_22_summary.json`

## 状态

`completed`：真实网络短诊断已完成；没有自动进入5K。
