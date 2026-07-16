# 021_36 SY2014 在线checkpoint Q排序离线审计记录

## 边界

只读取021_35五个checkpoint，在同一组160个标准化oracle示范状态上计算Q排序。未训练、未调用DSSAT、未修改模型。

## 结果

| checkpoint | yield_kg_ha | nonzero_action_recall | noop_accuracy | positive_expert_margin_fraction_nonzero | mean_expert_margin_nonzero |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 11175.0 | 0.8 | 0.9419 | 0.8 | 0.3348 |
| 250.0 | 5408.0 | 0.0 | 1.0 | 0.0 | -1.4019 |
| 500.0 | 5408.0 | 0.0 | 1.0 | 0.0 | -2.3293 |
| 750.0 | 10603.0 | 0.2 | 1.0 | 0.2 | -0.9595 |
| 1000.0 | 11175.0 | 0.6 | 0.9548 | 0.6 | 0.1947 |

描述性Spearman：yield vs nonzero recall = 0.9733；yield vs positive-margin fraction = 0.9733。

预注册分支：**A**。固定示范状态的非零动作Q排序与闭环产量坍缩/恢复高度同步。

## 限制

只有5个checkpoint，且固定示范状态不是完整闭环状态分布；相关性仅用于定位下一步，不构成因果证明。

## 输出

- `benchmark_results/021_36/021_36_fixed_demo_q_curve.csv`
- `benchmark_results/021_36/021_36_nonzero_event_predictions.csv`
- `benchmark_results/021_36/021_36_yield_q_metric_alignment.csv`
- `benchmark_results/021_36/021_36_spearman_descriptive.csv`
- `benchmark_results/021_36/021_36_checkpoint_q_ranking_audit.png/.svg`
