# 046_07 SYA originIC 训练年统计量归一化 MaskablePPO

- 训练步数：`2000`；checkpoint：`[1000, 2000]`。
- 仅将 25 维 observation 以训练年无操作参考轨迹统计量标准化；reward、动作和安全约束未改。
- 统计量仅来自 2005–2013；验证年不参与拟合。
- 统计量：`benchmark_results/.../audits/046_07_train_only_observation_statistics.csv`。
