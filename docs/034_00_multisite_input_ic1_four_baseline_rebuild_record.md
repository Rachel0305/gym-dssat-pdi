# 034_00 multisite 输入链 IC=1 四情景基线重建记录

执行模式：`full`
耗时：1350.0 秒

## 本轮回答的问题

在与 033_04 PPO 相同的 multisite 输入源和 IC=1 渲染条件下，重建 null、recorded farmer 模板、official expert、DSSAT auto 四情景基线。

## 固定边界

- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`
- 年份清单：读取 `benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/configs/033_04_available_weather_half_split_years.csv`
- 固定 `recorded_farmer_template` / `official_extension_expert` 通过渲染后 `.MZX` 静态管理表（`@I`/`@F` 行）执行；step 阶段只发送 no-op。
- `dssat_auto` 只启用 DSSAT 自动管理块，不额外写入固定灌溉/施氮表。
- 不训练 PPO/DQN。
- 不混用旧输入链基线。
- `recorded_farmer_template` 是模板复用比较项，不冒充逐年真实 recorded farmer。


## 渲染检查说明

- 固定 
ecorded_farmer_template / official_extension_expert 通过渲染后 .MZX 静态管理表（@I/@F 行）执行；step 阶段只发送 no-op。
- dssat_auto 只启用 DSSAT 自动管理块，不额外写入固定灌溉/施氮表。
- 本轮 ic_mi_mf_enabled、灌溉表、施肥表、种植表、模拟控制表均为 388/388 通过。wth_section_present 是模板文本搜索项，在当前渲染模板中 388/388 未匹配到该文本块；该项不表示 DSSAT 运行失败。年度天气文件绑定已在 033_05 的 WSTA/year 清单中确认，本轮 388 个 DSSAT 运行均成功完成。

## 站点年份范围

| station_code | site | split | n |
| --- | --- | --- | --- |
| FQA | FQ | train | 9 |
| FQA | FQ | validation | 10 |
| HLA | HLA | train | 10 |
| HLA | HLA | validation | 10 |
| LCA | LC | train | 9 |
| LCA | LC | validation | 10 |
| SYA | SY | train | 9 |
| SYA | SY | validation | 10 |
| YCA | YC | train | 10 |
| YCA | YC | validation | 10 |

## 覆盖状态

| scenario | status | n |
| --- | --- | --- |
| dssat_auto | ok | 97 |
| null | ok | 97 |
| official_extension_expert | ok | 97 |
| recorded_farmer_template | ok | 97 |

## 失败记录

无记录。

## 指标预览（按站点和情景均值）

| station_code | scenario | n | mean_yield | mean_irrigation | mean_nitrogen | mean_wp_et | mean_pfp_n | max_water_stress | max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | dssat_auto | 19 | 7244.464 | 25.4737 | 0.0 | 2.2311 |  | 0.0 | 0.4539 |
| FQA | null | 19 | 7077.9181 | 0.0 | 0.0 | 2.2458 |  | 0.7101 | 0.4539 |
| FQA | official_extension_expert | 19 | 7255.3705 | 23.0 | 82.0 | 2.22 | 93.3944 | 0.6603 | 0.0122 |
| FQA | recorded_farmer_template | 19 | 7271.6359 | 75.0 | 144.0 | 2.2274 | 53.3056 | 0.5607 | 0.0122 |
| HLA | dssat_auto | 20 | 6900.4232 | 57.3 | 0.0 | 1.526 |  | 0.4183 | 0.1989 |
| HLA | null | 20 | 6395.7634 | 0.0 | 0.0 | 1.425 |  | 1.0 | 0.1353 |
| HLA | official_extension_expert | 20 | 6542.3358 | 38.0 | 112.0 | 1.4545 | 61.4895 | 0.9416 | 0.0221 |
| HLA | recorded_farmer_template | 20 | 6505.0545 | 20.0 | 330.0 | 1.4535 | 20.7474 | 0.9593 | 0.0221 |
| LCA | dssat_auto | 19 | 9157.1864 | 78.3158 | 0.0 | 2.7916 |  | 0.0 | 0.0234 |
| LCA | null | 19 | 9117.0922 | 0.0 | 0.0 | 2.9095 |  | 0.4645 | 0.0234 |
| LCA | official_extension_expert | 19 | 9143.7958 | 23.0 | 82.0 | 2.8137 | 111.6053 | 0.1893 | 0.0234 |
| LCA | recorded_farmer_template | 19 | 9139.2453 | 70.0 | 50.0 | 2.8058 | 182.9368 | 0.0 | 0.0268 |
| SYA | dssat_auto | 19 | 6964.6388 | 86.3684 | 0.0 | 1.5221 |  | 0.0 | 0.6592 |
| SYA | null | 19 | 6183.0392 | 0.0 | 0.0 | 1.46 |  | 1.0 | 0.7511 |
| SYA | official_extension_expert | 19 | 9180.0241 | 38.0 | 112.0 | 2.0663 | 81.9632 | 1.0 | 0.6338 |
| SYA | recorded_farmer_template | 19 | 8815.5202 | 0.0 | 242.0 | 2.0595 | 36.4316 | 1.0 | 0.0321 |
| YCA | dssat_auto | 20 | 7977.0053 | 42.8 | 0.0 | 2.2885 |  | 0.0 | 0.5065 |
| YCA | null | 20 | 7680.6767 | 0.0 | 0.0 | 2.2465 |  | 1.0 | 0.5066 |
| YCA | official_extension_expert | 20 | 8016.8555 | 23.0 | 82.0 | 2.28 | 97.755 | 0.9548 | 0.1374 |
| YCA | recorded_farmer_template | 20 | 8283.8661 | 120.0 | 96.0 | 2.3845 | 86.295 | 0.0 | 0.0732 |

## 输出文件

- summary: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_full_baseline_summary.csv`
- daily: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_full_baseline_daily.csv`
- manifest: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_full_coverage_manifest.csv`
- failures: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/evaluation/034_00_full_failures.csv`
