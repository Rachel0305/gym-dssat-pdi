# 046_09 SYA originIC raw forecast observation PPO 记录

- 与 046_02 相比，唯一方法改动是 observation 追加 5 个未归一化天气/完美预报变量。
- 不使用 046_07 的 train-stat normalization。
- observation smoke: `benchmark_results/046_09_sya_originIC_binary_timing_forecast_raw_observation_ppo_smoke2k/audits/046_09_raw_forecast_observation_smoke.csv`
- 训练步数：`2000`；checkpoint：`[1000, 2000]`。
- 输出目录：`benchmark_results/046_09_sya_originIC_binary_timing_forecast_raw_observation_ppo_smoke2k`。
