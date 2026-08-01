# 041_05 SYA lowIC teacher warm-start 漂移审计

## 目的

041_04 的 100K 正式训练显示：

- `BC init` 已经能产生 teacher-like 管理；
- PPO fine-tune 后 PFP_N 提高，但产量与 WP_ET 多数年份下降；
- 说明 PPO 继续优化可能把 teacher 轨迹拉向节氮/节水而牺牲产量。

041_05 不训练、不重新运行 DSSAT，只读取 041_04 已有 CSV，审计：

1. `BC init` 与 `100K` 在每年产量、WP_ET、PFP_N、水、氮上的差异；
2. 每年灌溉/施肥事件是否提前、推迟、减少或改变剂量；
3. 水氮预算在不同阶段的分配变化。

## 输入

- `benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/evaluation/041_03_checkpoint_validation_summary.csv`
- `benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/daily_outputs/SYA/*ckpt0_daily.csv`
- `benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/daily_outputs/SYA/*ckpt100000_daily.csv`

## 输出

- 每年指标差异表；
- 每年事件差异表；
- 阶段水氮分配表；
- 诊断图；
- 中文实验记录。

## 判读边界

这是事后审计，不作为新的训练结果。它只回答：

> PPO fine-tune 相比 BC init 改变了什么？

不回答：

> 应该如何调 reward 或 PPO 超参数。

后续是否修改训练，应基于本审计的具体证据另开任务预注册。
