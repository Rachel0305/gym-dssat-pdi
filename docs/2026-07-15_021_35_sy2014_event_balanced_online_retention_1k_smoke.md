# 021_35 SY2014 event-balanced 网络在线保持性 1K smoke 记录

## 目的与边界

从021_34逐位复现的优质冻结网络出发，只增加在线交互和梯度更新。仅seed0、1K；未改reward、IC、动作、预算和DSSAT输入，未启动5K。

## 实现验证

- 起点Q哈希与021_34一致：True。
- target同步没有改变online Q：True。
- demonstration训练前后哈希一致：True。
- 每批16 demo/16 agent：True；demo内8 no-op/8 nonzero：True。
- 最终agent replay数量：996；全部更新有限：True。
- 梯度裁剪比例：0.000。

## 确定性检查点结果

| checkpoint | yield_kg_ha | irrigation_mm | nitrogen_kg_ha | late_n_after_dap90_kg_ha | expert_efficiency_gate |
| --- | --- | --- | --- | --- | --- |
| 0 | 11175.0 | 90.0 | 300.0 | 0.0 | True |
| 250 | 5408.0 | 0.0 | 0.0 | 0.0 | False |
| 500 | 5408.0 | 0.0 | 0.0 | 0.0 | False |
| 750 | 10603.0 | 30.0 | 200.0 | 0.0 | False |
| 1000 | 11175.0 | 75.0 | 300.0 | 0.0 | True |

## 预注册判定

- 在线4个检查点通过数：1/4。
- 分支：**B**。
- 在线更新只在部分检查点保持优质策略，尚不能放大训练。

## 限制

这是1K seed0 smoke，不代表跨seed稳定性。demonstration保持full return-to-go，agent transition使用冻结5-step回报；这是预注册的在线衔接设计，不应误写成完整原版DQfD复现。

## 输出

- `benchmark_results/021_35/021_35_training_interactions.csv`
- `benchmark_results/021_35/021_35_online_update_log.csv`
- `benchmark_results/021_35/021_35_checkpoint_trajectory.csv`
- `benchmark_results/021_35/021_35_validation.json`
- `benchmark_results/021_35/021_35_summary.json`
- `benchmark_results/021_35/021_35_online_retention_1k.png/.svg`
