# 004_16 YC WGEN 天气重建、来源核验与气候覆盖审计

本任务禁止 PPO training、checkpoint load、PPO evaluation、DSSAT crop simulation。追溯 004_05 seeds 1001–1100 生成链；只有工具版本、seed语义、上下文、RNG与runtime hash足够确认，且存在不运行crop simulation的天气-only物化路径，才允许生成天气。否则停止并标记INSUFFICIENT，不得猜测生成方式。

冻结输入：results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv；results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI；results/yc_random_weather_ppo/004_05/；results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/；results/yc_wgen_cli_pilot/004_01/full_year_path_audit/。

不修改/覆盖canonical配置、wrapper、reward、mask、observation、模型、天气、旧结果；新增结果只能写入results/yc_random_weather_ppo/004_16_yc_wgen_weather_reconstruction/；不commit、不push。相同seed编号不证明逐日天气相同。

Gate通过后，对training 1001–1080、held-out 1081–1100和observed 2014–2023按同算法计算季节和DAP 0–30/31–60/61–90/>90指标、compound hot/dry、percentile/support/5–95%区间和held-out偏移。dry spell定义RAIN<=0连续日；hot-dry day定义TMAX>32°C且RAIN<1 mm。seed5的2014/2019为预设关联检查年，不作因果断言；held-out不得并回训练。

产出本prompt、src/run_004_16_yc_wgen_weather_reconstruction_and_coverage.py、结果目录、中文报告docs/yc_random_weather_004_16_wgen_weather_reconstruction_and_coverage.md及provenance/seed manifest/hash/metrics/decision表。Gate失败时WGEN表保持blocked/空表，不绘制误导覆盖图。本任务不训练80→160 PPO。
