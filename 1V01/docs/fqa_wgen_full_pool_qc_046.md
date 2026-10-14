# 046 FQA WGEN 80/20 全池实现与气候分布审计

按 [046 prompt](../prompt_02/046_fqa_wgen_full_pool_climate_qc.md)执行。沿 [045 完整调度](../results/fqa_wgen_multiyear_heldout_gate_045/full_schedule_plan.json)串行生成训练池 seed 1001–1080 的 80 条与留出池 seed 1081–1100 的 20 条实际 WGEN 轨迹；两池由独立进程运行。044 的 2K checkpoint 只提供确定性动作，不进行 PPO 学习或依据产量筛选天气。

## 结果

[最终门槛](../results/fqa_wgen_full_pool_qc_046/final_gate.json)：**PASS_INPUT_QC_ONLY**。100/100 个 episode 完成，训练池 8,383 天、留出池 2,077 天，共 10,460 天实际天气均逐 episode 保存。100 份归档的哈希各不相同；复读哈希、行数、连续日期、四变量物理筛查、运行时 FileX `WTHER=W`、CLI 与每 episode 的 PDI `_rseed1` 均通过。峰值进程树 RSS 498.95 MB，低于预设的 1,536 MB。所有日天气和证据分别保存在[训练池](../results/fqa_wgen_full_pool_qc_046/train)与[留出池](../results/fqa_wgen_full_pool_qc_046/heldout)。

100 条轨迹与 2005–2013 九年拟合天气的共同窗口为每年 **6 月 11 日至 9 月 15 日，共 97 天**。按事先固定的 YC 004_02 式描述性阈值，降雨总量、湿日数、温度与辐射均值偏移均小于拟合期年际标准差的 2 倍；温度/辐射逐日标准差比在 0.35–2.0，月降雨曲线相关系数为 0.998（门槛 0.50），依赖结构最大组均差 0.066（门槛 0.60）。具体分布见[逐轨迹指标](../results/fqa_wgen_full_pool_qc_046/synthetic_fixed_window_metrics.csv)、[拟合期逐年指标](../results/fqa_wgen_full_pool_qc_046/fitting_year_fixed_window_metrics.csv)和[月降雨曲线](../results/fqa_wgen_full_pool_qc_046/monthly_rain_profile.csv)。

| 97 天窗口指标 | 拟合期九年均值 | 训练 80 条均值 | 留出 20 条均值 | 全池均值 |
|---|---:|---:|---:|---:|
| 降雨总量 (mm) | 521.78 | 511.71 | 502.16 | 509.80 |
| 湿日数 | 20.56 | 21.16 | 21.70 | 21.27 |
| TMAX 均值 (°C) | 31.29 | 32.01 | 31.95 | 32.00 |
| TMIN 均值 (°C) | 20.83 | 21.45 | 21.37 | 21.43 |
| SRAD 均值 (MJ m⁻² d⁻¹) | 16.39 | 16.76 | 16.67 | 16.74 |

**尾部提示：**全池有一条生成日雨量 438.96 mm（2009-08-04，训练 seed 1068），而拟合期九年在相同 97 天窗口的最大日雨量为 240 mm。100 条轨迹共 10,460 天中，仅这一日高于 240 mm；它低于预设的 500 mm/day 物理筛查上界。该轨迹和 seed 原样保留，不为使结果更好看而删去或重抽。上述门槛是启发式分布筛查，不能证明尾部事件的真实发生概率；这一天值得在后续政策结果解释时单独标注。

## 可复现性与边界

046 在独立进程重跑的前九个训练 episode 与 045 的对应九条天气文件逐字节哈希一致；留出 seed 1081、1100 与 045 对应文件也一致。这是这些**相同运行上下文**的复现证据。正式 PPO 训练仍必须像 044 一样保存其每个实际 episode 的逐日天气，不能用候选池文件或 seed 代替。

拟合源为 [041 冻结天气](../results/fqa_weather_resume_041/fitting_weather.csv)：D222 非空降雨值不改，空白按用户已确定的建模假设转为 0 mm；缺失温度/辐射使用 041 已记录的候选补值。046 只比较生长季共同窗口，未验证全年生成天气。native FIELD 坐标 warning 仍未由此证明无影响。本次为**天气输入门槛**，没有正式 100K 或 8 seed PPO 训练，也没有政策收益结论。
