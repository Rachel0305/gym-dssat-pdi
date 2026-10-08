# FQA WGEN 天气与训练路线：最新归档入口

- [041–046：D222/CLI、运行时 WGEN、天气归档、80/20 全池气候审计](fqa_wgen_041_046_archive_index.md)；结论为天气输入审计通过，未进行正式 PPO 训练。
- [047：FQA 多年份单 seed 10K PPO 训练预演](fqa_wgen_multiyear_ppo_047_10k_record.md)；结论为 10,080 步与 97 条实际天气归档闭合，`PASS_10K_ARCHIVE_ONLY`。
- [048：2K/10K 两例留出天气诊断](fqa_wgen_10k_heldout_048_record.md) 与 [049：同一训练 5K/10K 两例 checkpoint smoke](fqa_047_checkpoint_learning_signal_049_record.md) 提供先导比较；[050：完整 20 个留出 seed 配对评估](fqa_047_checkpoint_full20_heldout_050_record.md) 显示 10K 策略相对 5K 有变化，但平均产量低 61.9 kg/ha、wrapper 灌水仅少 2.25 mm、施氮不变，因此不据此启动 100K。

各阶段均有文件清单和 SHA-256 复读工具。050 是检查点配对诊断，不是正式 100K/多 seed 训练，也不是与传统管理基线的效果比较。
