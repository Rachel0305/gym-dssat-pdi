# 042_12 SYA lowIC binary-timing PPO 25K 候选策略审计

## 背景

042_10 将自由时序 MaskablePPO 的动作空间简化为二元时机问题：

- 灌溉动作：`{0, 45}` mm
- 施氮动作：`{0, 80}` kg/ha
- 保留 7 天最小操作间隔、季节水氮上限、DAP90 后禁氮、后期灌溉保留等安全约束

042_11 在同一配置下跑了 25K 训练长度曲线，发现：

- 1K：动作多样，但资源接近打满；
- 2K/5K/10K：出现策略坍缩或低产；
- 25K：动作多样性恢复到 7 个验证年份动作签名，且产量明显高于 2K/5K/10K。

因此 25K 不是最终结论，但值得作为候选 checkpoint 做一次统一图表审计。

## 本任务目标

本任务不训练、不调参、不改 checkpoint，只审计 042_11 的 25K checkpoint：

1. 与四个基线情景在 2014–2023 验证年份上比较：
   - grain yield
   - WP_ET
   - PFP_N
   - total irrigation
   - total nitrogen
2. 输出导师要求的五情景指标柱状图；
3. 对关键年份输出五情景日过程图：
   - 2014
   - 2017
   - 2021
   - 2022
   - 2023
4. 汇总 25K checkpoint 的动作多样性与指标胜出情况；
5. 明确结论边界：本任务只判断 25K 是否是可展示/可继续诊断的候选策略，不宣称模型最终成功。

## 关键防错要求

- 必须使用 lowIC 输入目录：
  `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 指标柱状图重放 DSSAT 时，必须使用 042_10 binary-timing 配置，不能误用 040_36 或 040_28 的旧动作空间。
- PPO daily CSV 必须来自：
  `benchmark_results/042_11_sya_lowIC_binary_timing_training_length_curve/evaluation/042_11_checkpoint_validation_summary.csv`
- checkpoint 固定为 `25000`。
- 不允许事后切换到 1K/2K/5K/10K。

## 输出

- `docs/042_12_sya_lowIC_binary_timing_25k_candidate_audit_record.md`
- `benchmark_results/042_12_sya_lowIC_binary_timing_25k_candidate_audit/tables/`
- `benchmark_results/042_12_sya_lowIC_binary_timing_25k_candidate_audit/figures/`

