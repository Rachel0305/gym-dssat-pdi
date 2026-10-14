# YC 004_08：seed 3/4/5 action-response 行为审计

## 方法与数据质量

按 004_07 / 004_07_01 的固定模型与 all-seed 验证框架，使用 004_05 step trace 和 004_07_02 的 30 个固定 checkpoint replay 摘要、snapshots；没有训练、改 reward/约束或启动 evaluation-only replay。动作逐年与 replay summary 核对。step timestep 对齐 snapshot DAS；同日及前一 DAS 状态均输出。SWFAC/NSTRES 来自动作 step 行；精确 observation vector/variables JSON 对 PPO 目标记录为空，因此无法确认这些代理是否属于模型实际输入。snapshot 同日天气为事后上下文，不是已证实的 policy input。

- 逐日记录 3048 行；seed×year 30/30；snapshots 30/30。
- step-to-snapshot DAS 匹配 100.0%；动作总量、signature 与 replay summary 闭合：True。
- observation vector 非空 0/3048；observation variables 非空 0/3048。
- 动作事件中前 7 日降雨代理缺失 30/248，均为播种日 DAP 0（snapshot 未含作物季开始前天气）；其余事件可对齐。
- 逐日精确字段缺失：SWFAC 0、NSTRES 0、DAP 0、step LAI 3048、step stage 3048、step rainfall 3048。snapshot 补充 soil water、LAI/stage、降雨等事后状态上下文。

## 样式来源

主要复用 results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/plot_004_05_seed3_diagnostic.py，参考系列脚本 src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py。沿用左对齐标题、紧凑多面板、浅网格、PPO 绿色、灌溉蓝圆/N 橙方及按年动作日历；为区分三个 seed，仅使用旧线色配不同线型/marker。

## 年度结果

| Seed | Yield 均值 kg/ha | 灌溉均值 mm | N 均值 kg/ha | Common reward 均值 | WP_ET 均值 | PFP_N 均值 | action days/年 | 管理 events/年 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 6901.9 | 75.0 | 100.0 | 850.00 | 2.033 | 72.68 | 3.5 | 4.5 |
| 4 | 8202.9 | 214.5 | 240.0 | 680.91 | 2.224 | 34.17 | 16.3 | 17.3 |
| 5 | 8148.7 | 165.0 | 240.0 | 726.79 | 2.249 | 33.96 | 5.0 | 9.0 |

seed5 对 seed4 平均少灌溉 49.5 mm/年，产量差 -54.2 kg/ha (-0.66%)；seed5 用水更少年份 10/10，产量更高年份 4/10。

| 年份 | Seed4 产量 | Seed5 产量 | 产量差 5−4 | Seed4 灌溉 | Seed5 灌溉 | 节水 4−5 |
|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 9403 | 9253 | -150 | 225 | 165 | 60 |
| 2015 | 9089 | 9089 | +0 | 225 | 165 | 60 |
| 2016 | 6503 | 6497 | -6 | 210 | 165 | 45 |
| 2017 | 7825 | 7827 | +2 | 210 | 165 | 45 |
| 2018 | 8326 | 8323 | -2 | 195 | 165 | 30 |
| 2019 | 7607 | 7225 | -382 | 210 | 165 | 45 |
| 2020 | 6516 | 6517 | +1 | 225 | 165 | 60 |
| 2021 | 8623 | 8626 | +2 | 225 | 165 | 60 |
| 2022 | 8555 | 8543 | -12 | 210 | 165 | 45 |
| 2023 | 9583 | 9587 | +4 | 210 | 165 | 45 |

Seed5 年均 action days 5.0，seed4 为 16.3。灌溉日剂量的逐日净额拆分：seed4 在 DAP 45 (+15.0), DAP 52 (+15.0), DAP 9 (+15.0), DAP 16 (+15.0), DAP 80 (+15.0), DAP 73 (+15.0), DAP 66 (+15.0), DAP 38 (+15.0), DAP 59 (+15.0), DAP 91 (+13.5), DAP 98 (+6.0) 的额外灌溉平均合计 154.5 mm/年；seed5 在 DAP 61 (+45.0), DAP 8 (+30.0), DAP 31 (+30.0) 的额外灌溉抵消 105.0 mm/年，净差为 49.5 mm/年。seed5 将多次约 15 mm 的分散灌溉换成 DAP 8/31/61 的 30/45/45 mm 集中动作；这是动作日历上的水量重分配解释，非因果证明。

## Schedule 相似度和 weather adaptivity

| Seed | Adaptivity | 独立 schedule 数 | 模态 schedule 年数 | 完全相同年度对 | 平均编辑相似度 |
|---:|---|---:|---:|---:|---:|
| 3 | MODERATE | 2 | 5/10 | 20/45 | 0.861 |
| 4 | LOW | 3 | 5/10 | 16/45 | 0.956 |
| 5 | LOW | 1 | 10/10 | 45/45 | 1.000 |

| Seed | DAP 集合 Jaccard | 事件 tuple Jaccard | 有序序列相似度 |
|---:|---:|---:|---:|
| 3 | 0.861 | 0.861 | 0.861 |
| 4 | 0.956 | 0.956 | 0.956 |
| 5 | 1.000 | 1.000 | 1.000 |

Seed5 在相同 DAP 的代理状态最大年际范围：SWFAC 0.000、NSTRES 0.012、前一 DAS 土壤水 236.0 mm、前 7 日雨量 317.9 mm；其中 17 个共同 DAP 至少有两类状态/天气代理超过年际变异阈值。Fixed-schedule STRONG 判据要求模态 schedule ≥9/10 且相同 DAP 上至少两类代理确有变化；seed5 满足。跨年状态存在变化而动作重复，支持外部行为上的低 weather adaptivity。观测向量缺失，原因不能归结为 reward、observation 编码或 mask 中任何单一因素。

## Seed5 的 240 kg N/ha

分类为 FIXED。Seed5 十年完全相同的 I/N 动作序列为：0:I45/N40;8:I30/N40;15:I0/N80;31:I45/N40;61:I45/N40。逐年施氮总量、DAP 和剂量均在 annual_seed345_behavior.csv 与 seed_dap_action_response.csv 中；固定表示验证中实际动作模板重复，不表示代码硬编码。

## 结论

seed5_weather_adaptivity = LOW
seed5_fixed_schedule_evidence = STRONG
seed5_water_saving_mechanism = seed4 在 DAP 45 (+15.0), DAP 52 (+15.0), DAP 9 (+15.0), DAP 16 (+15.0), DAP 80 (+15.0), DAP 73 (+15.0), DAP 66 (+15.0), DAP 38 (+15.0), DAP 59 (+15.0), DAP 91 (+13.5), DAP 98 (+6.0) 的额外灌溉平均合计 154.5 mm/年；seed5 在 DAP 61 (+45.0), DAP 8 (+30.0), DAP 31 (+30.0) 的额外灌溉抵消 105.0 mm/年，净差为 49.5 mm/年。seed5 将多次约 15 mm 的分散灌溉换成 DAP 8/31/61 的 30/45/45 mm 集中动作；这是动作日历上的水量重分配解释，非因果证明。
seed5_high_N_pattern = FIXED
next_step = observation-action issue

Seed3/4/5 的 adaptivity 对照：MODERATE / LOW / LOW。seed3 的跨年变化主要是 DAP 16 施氮在部分年份缺失（总 N 80 vs 120 kg/ha），不是充分的 weather response 证据；seed4 的 signature 变化主要是季末追加灌溉次数随 episode 终止时间变化。seed5 schedule 十年逐项相同。优先查 observation-action 响应和 mask，再决定是否优化 PPO/reward；本轮本身不调整任何训练因素。

## 文件和限制

CSV：annual_seed345_behavior、action_events_with_context、seed_dap_action_response、seed_dap_action_response_summary、action_schedule_similarity_pairs、daily_context_seed345。JSON：analysis_summary、input_provenance。PNG：action_calendar_seed345、state_variation_at_action_dap、annual_metrics_seed345、schedule_similarity_seed345，均在结果目录 figures 下。

缺项为精确 PPO observation vector/variables；PPO trace 的 LAI、stage、天气列为空，使用 snapshots 的同日/前日状态代理。没有 replay、没有训练、没有 commit/push；结论限于固定 checkpoint 和筛查用的十个年份。
