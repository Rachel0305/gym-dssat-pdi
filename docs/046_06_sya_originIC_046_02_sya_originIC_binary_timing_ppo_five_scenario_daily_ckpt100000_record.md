# 046_06 SYA originIC 五情景逐日过程图

- PPO checkpoint: `100000`。
- 年份：[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]。
- PPO 重放终值逐年与 046_02 保存终值核对；PlantGro.OUT 的 GWAD 为整数打印，因此允许不超过 0.5 kg/ha 的显示精度差，超出则中止。
- 统一累计奖励：0.158×籽粒产量 − 1.1×灌溉 − 1.58×施氮；仅用于报告比较。
- 合并日值：`benchmark_results/046_06_sya_originIC_046_02_sya_originIC_binary_timing_ppo_five_scenario_daily_ckpt100000/tables/046_06_sya_five_scenario_daily.csv`。
