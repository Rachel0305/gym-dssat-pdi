# 046_05 SYA 配置化五情景结果汇总与出图

## 输入

只读取同一 profile 的两份新输出：

- `046_03_sya_<profile>_four_baselines/evaluation/046_03_baseline_summary.csv`；
- `046_02_sya_<profile>_binary_timing_ppo/evaluation/046_02_checkpoint_validation_summary.csv`。

不会读取 042/045 的 lowIC 表格；缺少上述新输出时直接报错。

## 输出

- 逐年五情景指标总表；
- PPO 对四基线最高值的逐年差值表；
- 产量、WP_ET、PFP_N 五情景柱状图；
- 灌溉量、施氮量五情景柱状图；
- 统一的可追溯记录。

主产量是籽粒产量：基线用 `grain_yield_kg_ha`，PPO 用 `final_grnwt`；二者都对应 DSSAT `GRNWT/HWAM`，不使用地上部生物量作主指标。

