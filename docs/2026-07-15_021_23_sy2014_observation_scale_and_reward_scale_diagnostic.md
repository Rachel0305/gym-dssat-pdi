# 021_23 SY2014 观测尺度与 reward×0.1 短诊断记录

## 边界

本任务先审计既有示范观测的数值尺度，再严格复刻 021_22 的 100 次示范预训练和 100 个真实环境步，仅将训练 reward 与 n-step return 乘以 0.1。没有进入 5K，没有改变奖励相对权重、观测信息、动作空间、IC、网络或 DQfD 参数。

## 生效校验

- scale validation: `True`
- demo reward max error: `0`
- demo n-step max error: `0`
- agent reward max error: `0`
- component path max error: `0`

## 观测尺度

- 维数：25
- max_abs 最大值：20131.6
- 非零维 max_abs 最小值：0.199414
- 超过非零维中位尺度 10 倍的维数：7

25 维展开顺序由环境报告的 17 个变量顺序核对：其中 `sw` 展开为 9 个土层，所以形成 `cumsumfert, dap, dtt, ep, grnwt, istage, nstres, rtdep, srad, sw_layer_1..9, swfac, tmax, topwt, totir, vstage, wtdep, xlai`。最大尺度来自 `topwt`（20131.56），其次是 `grnwt`（11205.42）；它们与量级较小的胁迫和土壤水分特征未经归一化地共同进入 MLP。

这些是描述性证据，不单独证明大梯度由观测尺度造成。

## 与 021_22 未缩放基准的关键对比

| phase | metric | unscaled_021_22 | scaled_021_23 | scaled_to_unscaled_ratio |
| --- | --- | --- | --- | --- |
| pretrain | gradient_clip_fraction | 1.0 | 1.0 | 1.0 |
| pretrain | median_total_gradient_norm | 1274.074646 | 1470.571899 | 1.154227 |
| pretrain | cumulative_parameter_relative_l2 | 0.045374 | 0.045072 | 0.993336 |
| pretrain | median_td_1_weighted | 64.897156 | 9.28421 | 0.14306 |
| pretrain | median_td_n_weighted | 174.804863 | 15.508456 | 0.088719 |
| online | gradient_clip_fraction | 1.0 | 1.0 | 1.0 |
| online | median_total_gradient_norm | 387.18338 | 581.489685 | 1.501846 |
| online | cumulative_parameter_relative_l2 | 0.0356 | 0.032883 | 0.923685 |
| online | median_td_1_weighted | 56.066235 | 4.831966 | 0.086183 |
| online | median_td_n_weighted | 124.053558 | 12.303896 | 0.099182 |

## 预注册判定

`B_td_down_but_gradient_or_clipping_not_down`

该判定只回答 reward 数值缩放是否参与短程训练动力学，不代表 DQfD 有效，也不允许自动进入 5K。Smooth L1/Huber loss 在大误差区间梯度饱和，因此 loss 数值和参数梯度必须分开解释。

## 输出

- `benchmark_results/021_23/021_23_observation_scale_audit.csv`
- `benchmark_results/021_23/021_23_pretrain_update_log.csv`
- `benchmark_results/021_23/021_23_online_update_log.csv`
- `benchmark_results/021_23/021_23_agent_interactions.csv`
- `benchmark_results/021_23/021_23_scaled_stage_summary.csv`
- `benchmark_results/021_23/021_23_stage_comparison.csv`
- `benchmark_results/021_23/021_23_scale_validation.json`
- `benchmark_results/021_23/021_23_summary.json`
- `benchmark_results/021_23/021_23_observation_reward_scale_diagnostic.png/.svg`

## 失败记录

第一次运行已经完成数值计算和结果文件写出，但在最后生成 Markdown 表格时，容器缺少 Pandas 可选依赖 `tabulate`，导致文档阶段失败。完整现场保存在 `benchmark_results/021_23_failed_attempt_1_missing_tabulate/`。随后仅将表格输出改成本地 Markdown 格式化函数，未安装新包、未改变任何科学参数，并从头按相同配置重跑成功。
