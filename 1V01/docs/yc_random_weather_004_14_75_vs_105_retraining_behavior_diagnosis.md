# YC 004_14：75 vs 105 mm 重训练行为诊断

## 结论先行

本任务只读取既有训练 episode summary、正式 paired episode、action traces 和 checkpoint 文件清单；未训练 PPO、未加载/评估 checkpoint、未运行 DSSAT 或 replay。

105mm_failure_structure = FEW_SEED_COLLAPSE  
low_input_pattern = CONCENTRATED_IN_FEW_SEEDS  
cross_domain_consistency = HIGH  
training_stage_evidence = INSUFFICIENT  
mechanistic_interpretation = seed 4/5 在两个域转为单一低投入 schedule，seed 3 也减投减产，seed 6 则反向增投增产；训练 entropy-loss 方向不一致。这支持 seed-specific 行为极化/动作收缩解释，但缺少 raw argmax、完整 mask 与阶段性评估，不能把它因果归于 early-cap 放宽。  
next_step = retain_75_and_close_YC

建议保留 75 mm canonical 并结束 early-cap 阈值支线。已有数据能描述低投入 phenotype 与结果的共现，但不能证明因果机制；没有足够证据支持继续针对 early cap 做下一轮调参。

## 数据覆盖与判定口径

- Paired episode rows：240；Observed 80，held-out WGEN 160。
- action trace：105 mm 240/240 episode 匹配成功；75/105 合计投入闭合 480/480（trace 与 episode 总 I/N 差值均≤1 单位）。明细在 action_trace_coverage.csv 与 action_trace_closure.csv。
- 可读取的训练 episode summary：75 mm 5/8 seeds，105 mm 8/8 seeds。分段仅按记录 episode 顺序切成四段；它不是等长训练步数或 checkpoint evaluation。
- 虽然 checkpoint 文件存在，脚本只记录文件名和大小，没有读取模型权重。
- 004_13 action trace 记录实际动作、DAP、SWFAC/NSTRES 等，但不含 raw argmax 和完整 mask。因此不能用它证明“某动作偏好被 early cap 拦截”。
- Seed phenotype 沿用 004_13：mean I≤75 mm 且 mean N≤80 kg/ha 为 very-low-input；Observed mean yield<6500 kg/ha 且满足 low-input 为 yield-collapse；固定 N=240 定义为至少 80% episodes N=240。
- Episode flag 使用同一投入阈值，yield<6500 标为 episode collapse 筛查，不能与 seed phenotype 混为一谈。
- “明显减产 episode”筛查阈值为 paired yield 下降≥1000 kg/ha 或≥15%，仅用于定位案例，不是胜负判据。
- low_input_pattern：至少 5/8 seeds 在两个域都增加 N=0 episode 记为 BROAD；至多 3/8 且新增 N=0 的 top-2 seed 占比≥60%记为 CONCENTRATED；其余为 MIXED。
- cross_domain_consistency 的 HIGH 仅表示五项组均值指标（yield、common reward、N、I、WP_ET）在两个域方向至少 80% 一致，不代表每个 seed 或每个指标幅度一致。

## 总体与 paired seeds

下表为 105−75 的 episode 均值差，正值表示 105 mm 更高。seed contribution 使用每个 seed 的 domain mean delta 除以 8。详情见 seed_domain_paired_summary.csv 与 seed_mean_contributions.csv。

| Domain | Δ Yield kg/ha | Δ Irrigation mm | Δ N kg/ha | Δ WP_ET | Δ Common reward | Δ Canonical reward |
|---|---:|---:|---:|---:|---:|---:|
| Observed | -749.6 | 7.5 | -40.0 | -0.223 | -63.485 | -0.062 |
| Held-out WGEN | -505.6 | 7.6 | -38.2 | -0.157 | -27.797 | -0.030 |

| Seed | Δ Yield Obs | Δ Yield WGEN | Δ Common reward Obs | Δ Common reward WGEN | Δ N Obs | Δ N WGEN | Δ I Obs | Δ I WGEN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1538 | 497 | 130.315 | -34.244 | 40.0 | 40.0 | 45.0 | 45.0 |
| 1 | -956 | -560 | -24.600 | 37.996 | -80.0 | -80.0 | 0.0 | 0.0 |
| 2 | 1376 | -18 | -33.125 | -245.574 | 52.0 | 44.0 | 153.0 | 157.5 |
| 3 | -3342 | -1484 | -347.948 | -85.049 | -92.0 | -70.0 | -31.5 | -35.2 |
| 4 | -4651 | -2459 | -169.143 | 182.091 | -240.0 | -240.0 | -169.5 | -174.0 |
| 5 | -4596 | -2450 | -215.025 | 124.043 | -240.0 | -240.0 | -120.0 | -120.0 |
| 6 | 4634 | 2429 | 151.648 | -201.640 | 240.0 | 240.0 | 183.0 | 187.5 |
| 7 | 0 | 0 | 0.000 | -0.000 | 0.0 | 0.0 | 0.0 | 0.0 |

Failure structure 以两个评估域各自的 seed-level 负向 yield delta 为准：若 top-2 seeds 合计占各域负向 yield delta 绝对值至少 60%，归为 FEW_SEED_COLLAPSE；若至少 5/8 seeds 两域都减产且 top-2 均不足 50%，归为 BROAD_SHIFT；否则 MIXED。该标签描述减产集中度，不宣称训练 collapse 已被过程数据定位。分类计数：{"seeds_worse_on_yield_and_common_reward_in_both_domains": 1, "seeds_with_yield_decline_in_both_domains": 4, "top_two_share_of_negative_yield_deltas_percent_by_domain": {"observed_weather": 68.27261752790909, "heldout_wgen": 70.43193943402824}, "seeds_with_N0_increase_in_both_domains": 3, "top_two_share_of_added_N0_episodes": 0.7317073170731707, "core_metrics_same_direction_across_domains": 5, "core_metrics_compared": 5}。

## Episode 与低投入行为

| Domain | N=0 75 | N=0 105 | Low-input 75 | Low-input 105 | Collapse 75 | Collapse 105 |
|---|---:|---:|---:|---:|---:|---:|
| Observed | 10/80 | 29/80 | 35/80 | 50/80 | 25/80 | 44/80 |
| Held-out WGEN | 20/160 | 53/160 | 71/160 | 100/160 | 36/160 | 68/160 |

- Observed seed phenotype：very-low-input 3/8→5/8；yield-collapse 3/8→5/8；fixed N=240 2/8→2/8。
- WGEN seed phenotype：very-low-input 3/8→5/8；yield-collapse 0/8→0/8；fixed N=240 2/8→2/8。

- Observed：N=0 episode count 在 105 高于 75 的 seed 为 [3, 4, 5]。
- WGEN：N=0 episode count 在 105 高于 75 的 seed 为 [3, 4, 5]。

按 seed 和年份/weather seed 的全量 phenotype、N=0 与减产标签见 paired_episode_diagnosis.csv、episode_phenotype_counts_by_seed.csv。若 N=0 增加分布在多个 seed 且两个评估域方向一致，支持“行为分布有广泛位移”的描述；若集中于少数 seed，则更像 seed 异质性。两者都不等价于 cap 的因果效应。

### 关键 seed 行为分化

| Seed | 两域等权平均 I 75→105 | 两域等权平均 N 75→105 | Δ Yield Obs | Δ Yield WGEN | N=0 Obs 75→105 | N=0 WGEN 75→105 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 75.0→41.6 | 99.0→18.0 | -3342 | -1484 | 0→9 | 0→13 |
| 4 | 216.8→45.0 | 240.0→0.0 | -4651 | -2459 | 0→10 | 0→20 |
| 5 | 165.0→45.0 | 240.0→0.0 | -4596 | -2450 | 0→10 | 0→20 |
| 6 | 45.0→230.2 | 0.0→240.0 | 4634 | 2429 | 10→0 | 20→0 |

Seed 4 和 5 是主要低投入/减产端：两域平均上均由 N=240 降至 N=0，灌溉降至约 45 mm，且两域 yield 均大幅下降；seed 3 也明显降 N/I 并减产。与之相反，seed 6 从 N=0 转为 N=240、灌溉升至约 230 mm，并明显增产。故 105 mm 的组均值变化不是 8 个 seed 一致“更节水/减氮”，而是少数策略走向极低投入、另一些转向高投入，seed 间分化加大。这里是描述性对应，不足以把这些策略变化归因于 early cap 本身。

Seed 4/5 的 action-trace 证据如下。unique schedules 统计每个 seed×domain 内不同评估 episode 的完整正管理动作序列数；action count 是实际出现的离散动作编号数。

| Seed | Domain | Unique schedules 75 | Unique schedules 105 | Unique actions 75 | Unique actions 105 | No-op rate 75 | No-op rate 105 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | Observed | 3 | 1 | 4.0 | 2.0 | 0.840 | 0.990 |
| 4 | WGEN | 4 | 1 | 4.0 | 2.0 | 0.840 | 0.990 |
| 5 | Observed | 1 | 1 | 4.0 | 2.0 | 0.951 | 0.990 |
| 5 | WGEN | 1 | 1 | 4.0 | 2.0 | 0.951 | 0.990 |

105 mm 的 seeds 4/5 在两个域都只出现 1 种 schedule、2 个 action code，约 99% 决策为 no-op；已执行 schedule 是 DAP 0 灌 45 mm、N=0，之后不再施肥/灌溉。Seed 4 的 75 mm 对照在跨年 schedule 和动作编号上更多样；seed 5 的 75 mm schedule 本来也重复，但仍有 4 个 action code 且全季 N=240。这里的“固定”只用于描述行为收缩，不作为策略优劣定义。

### 明显减产 episodes

| Seed | Domain | Year/weather seed | Δ Yield kg/ha | I 75→105 | N 75→105 | N=0 105 | Collapse 105 |
|---:|---|---:|---:|---:|---:|---|---|
| 4 | observed_weather | 2014 | -8213 | 225→45 | 240→0 | yes | yes |
| 5 | observed_weather | 2014 | -8063 | 165→45 | 240→0 | yes | yes |
| 5 | observed_weather | 2023 | -6272 | 165→45 | 240→0 | yes | yes |
| 4 | observed_weather | 2023 | -6268 | 210→45 | 240→0 | yes | yes |
| 4 | observed_weather | 2015 | -5384 | 225→45 | 240→0 | yes | yes |
| 5 | observed_weather | 2015 | -5384 | 165→45 | 240→0 | yes | yes |
| 4 | observed_weather | 2018 | -4761 | 195→45 | 240→0 | yes | yes |
| 5 | observed_weather | 2018 | -4759 | 165→45 | 240→0 | yes | yes |
| 3 | observed_weather | 2018 | -4678 | 75→30 | 120→80 | no | yes |
| 4 | observed_weather | 2019 | -4506 | 210→45 | 240→0 | yes | yes |
| 3 | observed_weather | 2023 | -4436 | 75→45 | 80→0 | yes | yes |
| 3 | observed_weather | 2015 | -4287 | 75→45 | 120→0 | yes | yes |

N=0 时 PFP_N 为 NA。有效样本数在逐 seed summary 中保留，PFP_N 均值不得脱离 N=0 发生率解释。

## 管理行为与 early period

| Domain | Δ early I mm (DAP≤30) | Δ early N kg/ha (DAP≤30) | Δ no-op rate | Δ unique action count | Δ unique schedule count |
|---|---:|---:|---:|---:|---:|
| Observed | 5.4 | -35.0 | 0.000 | -0.21 | 0 |
| Held-out WGEN | 5.0 | -33.2 | -0.000 | -0.24 | -0 |

逐 episode trace 指标见 action_behavior_episode.csv；seed-domain 汇总见 action_behavior_seed_domain.csv；全部正施灌/施氮动作的 DAP 与 dose 见 management_events_from_traces.csv。unique schedule 仅描述策略多样性，不作为优劣标准。

| Domain | Δ Irrigation event count / episode | Δ N event count / episode | Δ non-zero management days / episode | Δ non-zero action steps / episode |
|---|---:|---:|---:|---:|
| Observed | 0.27 | -0.76 | 0.00 | 0.00 |
| Held-out WGEN | 0.30 | -0.74 | 0.02 | 0.02 |

Early-cap 可以描述性地与已执行的早期 I/N 变化、后续投入和 episode 结果对照；现有 trace 没有 raw argmax/full mask，无法判断放宽后模型“想选”但曾被拦的动作，也无法建立联合 I×N mask 导致 N=0 的因果链。

## 训练阶段

- 75 mm 有训练 episode summary：5/8 seeds；105 mm：8/8 seeds。
- 共同可读 seeds 3–7 的首末四分位记录 episode 汇总如下；四分位按每 seed 的记录顺序划分，不代表等长 training steps：

| Group | Episode return Q1→Q4 | Yield Q1→Q4 kg/ha | Irrigation Q1→Q4 mm | N Q1→Q4 kg/ha |
|---|---:|---:|---:|---:|
| control_75 | 0.613→0.716 | 7645→7583 | 229.5→146.9 | 240.0→229.7 |
| treatment_105 | 0.614→0.715 | 7637→7277 | 226.9→136.2 | 240.0→190.6 |

- 105 mm rollout 在首末段呈现 return 上升、平均投入下降并伴有平均 yield 下降；但 75 mm 也有低投入趋势。训练 episode 是持续更新策略下的在线 rollout，不是固定 checkpoint 评估，不能定位 phenotype 的确切形成阶段。
- 25K/50K/75K/100K checkpoint inventory 见 checkpoint_inventory.csv。没有既有分阶段评估；按边界不 replay checkpoint。
- TensorBoard event scalar：13/16 个文件已解析，状态 PARSED。仅提取紧凑分位摘要，不加载模型；清单、摘要和状态见 training_tensorboard_inventory.csv、training_tensorboard_summary.csv、training_tensorboard_status.json。

| Common seed | Final entropy_loss 75 | Final entropy_loss 105 | Final rollout reward 75 | Final rollout reward 105 |
|---:|---:|---:|---:|---:|
| 3 | -0.1311 | -0.0602 | 0.731 | 0.800 |
| 4 | -0.1146 | -0.1506 | 0.657 | 0.691 |
| 5 | -0.1618 | -0.0387 | 0.655 | 0.782 |
| 6 | -0.0884 | -0.0916 | 0.781 | 0.643 |
| 7 | -0.0782 | -0.2857 | 0.821 | 0.711 |

- TensorBoard 的 entropy_loss 仅是训练期全局标量，不能直接等同于固定 evaluation trace 的行为熵。共同 seeds 的方向并不一致，因此它不能单独解释 seed 4/5 的固定低投入 schedule。
- 因而 training_stage_evidence = INSUFFICIENT，不能判断低投入/产量崩塌是早期形成还是后期退化。

## 证据层级与决策

**事实：** 配对 CSV 给出每个 seed/domain/episode 的产量、I/N、效率、reward 和管理事件；action traces 提供实际采取的时序与剂量；本报告给出 N=0、低投入、collapse 和显著减产的分布以及对总体均值的 seed 贡献。

**行为解释：** traces 可显示 105 mm 是否改变实际早期灌溉、施氮、no-op、动作多样性与 schedule 重复。此为描述性关联，不构成 mask/reward 的因果解释。

**未知：** 训练期间具体哪个 observation/action/mask 决策造成低 N 或减产；phenotype 出现在哪个 checkpoint 阶段；没有放宽 cap 时同一训练 seed 是否避免 phenotype。以上都没有现成证据，本任务不补跑。

由于 105 mm 的总体结果不优于 75 mm，且只读资料没有识别出一个可被单独验证的明确机制，决策为 retain_75_and_close_YC。此结论是关闭本轮 mask threshold tuning 支线，不代表 YC 的 PPO 策略优化目标已完成。

## 文件与可复现性

- Script: src/run_004_14_yc_75_vs_105_retraining_behavior_diagnosis.py
- Primary input: results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/summaries/paired_episode_comparison.csv
- Input file hashes: input_provenance.json
- Machine result: diagnostic_conclusion.json
- Figures: figure_seed_delta_diagnosis.png, figure_action_behavior_deltas.png
- No commit/push.
