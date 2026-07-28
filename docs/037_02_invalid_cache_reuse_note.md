# 037_02 无效执行说明：checkpoint 缓存复用

037_02 原计划测试 LCA early starter cap：在 036 主线自由时序 MaskablePPO 基础上，仅新增 DAP1-10 累计灌溉 ≤30 mm、累计施氮 ≤40 kg/ha 的早期投入上限。

执行后发现 037_02 的训练清单显示 `run_status=ok_existing`，且模型路径指向：

`benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/LCA/...`

这说明 037_02 没有使用新约束重新训练，而是复用了 032_22 的既有 checkpoint。对应验证结果中 DAP1-10 仍出现 I60-75、N120-200 的早期集中投入，不能用于判断 early starter cap 是否有效。

后续已新开 037_03，修正 `batch.OUT`、`batch.SEED`、`batch.TOTAL_TIMESTEPS`、`batch.CHECKPOINT_STEPS` 指向 037_03 输出目录，并完成 clean retrain。037_02 结果仅作为非科学性失败/缓存复用记录保留，不纳入科学结论。
