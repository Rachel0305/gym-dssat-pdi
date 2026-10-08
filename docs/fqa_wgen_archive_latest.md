# FQA WGEN 天气与训练路线：最新归档入口

- [041–046：D222/CLI、运行时 WGEN、天气归档、80/20 全池气候审计](fqa_wgen_041_046_archive_index.md)；结论为天气输入审计通过，未进行正式 PPO 训练。
- [047：FQA 多年份单 seed 10K PPO 训练预演](fqa_wgen_multiyear_ppo_047_10k_record.md)；结论为 10,080 步与 97 条实际天气归档闭合，`PASS_10K_ARCHIVE_ONLY`。
- [048：2K/10K 两例留出天气诊断](fqa_wgen_10k_heldout_048_record.md)、[049：同一训练 5K/10K 两例 checkpoint smoke](fqa_047_checkpoint_learning_signal_049_record.md) 和 [050：完整 20 个留出 seed 配对评估](fqa_047_checkpoint_full20_heldout_050_record.md) 是 10K 阶段诊断。050 显示 10K 相对 5K 尚无有利效果信号；之后按用户决定继续完成单 seed 100K 训练。
- [051：FQA 多年份 WGEN PPO 100K 训练](fqa_multiyear_wgen_ppo_051_100k_record.md) 已完成 100,080 步，四个中间 checkpoint、最终模型和 100,080 个实际天气日均已归档，12/12 完整性检查通过。结论仅为训练与归档完成，尚未评估策略效果。

各阶段均有文件清单和 SHA-256 复读工具。051 为单 seed 训练，不是多 seed 稳健性或策略效果结论，也不是与传统管理基线的比较。
