# YC PPO seed instability diagnostic

日期：2026-09-10  
状态：`completed; read-only diagnostic`  
独立目录：`experiments/223_yc_ppo_seed_instability_diagnostic/`

## 结论先行

本轮没有重新训练，也没有修改 reward、action、constraints、weather augmentation、IC、genotype 或其他站点。六个既有 run 的数据完整性检查全部通过；100K N=146.667 的相同均值由两组不同 seed 的数值平均巧合产生，不是 method 间复制或 merge 覆盖。

主要诊断为 **Type B — seed-dependent local optimum / policy basin**，并伴随 **Type A 的行为层 policy collapse**：Original seed0 在 25K→50K 退化，Original seed1 在 75K→100K 退化，Augmented seed1 从早期即处于低产/低投入分支，Augmented seed2 在 50K→75K 退化。Augmented 只有 seed0 在 25K–100K 基本保持高产，不能支持“3 seeds 都减缓后期退化”。

不支持 Type C 的“同一个退化吸引子”强结论：Original seed1、Augmented seed1、Augmented seed2 在 100K 都是低投入且后期动作极少，但动作剂量/动作对/DAP 结构并不相同。也不支持 Type D：augmentation 没有在多数 seed 上稳定改善；不能把当前结果自动归因于 G×E coverage。

**Recommendation 2：PPO 没有被现有日志证明存在普遍性的数值训练崩溃，但存在明显的 seed-dependent local optimum / policy basin；下一轮应先做最小化的探索或 optimization-stability 对照，再考虑 G×E V2。**

## 1. 数据完整性与 N=146.667 来源

100K、2014–2023 的 N 来源如下。每个 seed 都来自各自独立 validation CSV 的 `total_n` 列，并与 `yc_v1_exact_efficiency_replay_all_seeds.csv` 的 `total_nitrogen` 交叉核对；六个 validation 文件 SHA-256 均不同，40 行/文件、`year × checkpoint` 键、逐日动作总量和事件数均通过核对。

| Method | Seed | mean N (10 years) | sum N (10 years) | source column | source file |
|---|---:|---:|---:|---|---|
| Original | 0 | 160.000 | 1600.000 | `total_n (cross-checked against exact replay total_nitrogen)` | `benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/evaluation/055_00_checkpoint_validation_summary.csv` |
| Original | 1 | 40.000 | 400.000 | `total_n (cross-checked against exact replay total_nitrogen)` | `benchmark_results/222YCA_O1_yc_v1_original_seed1/evaluation/222YCA_O1_checkpoint_validation_summary.csv` |
| Original | 2 | 240.000 | 2400.000 | `total_n (cross-checked against exact replay total_nitrogen)` | `benchmark_results/222YCA_O2_yc_v1_original_seed2/evaluation/222YCA_O2_checkpoint_validation_summary.csv` |
| Augmented | 0 | 240.000 | 2400.000 | `total_n (cross-checked against exact replay total_nitrogen)` | `benchmark_results/221YCA_yc_ppo_domain_augmentation/evaluation/221YCA_checkpoint_validation_summary.csv` |
| Augmented | 1 | 80.000 | 800.000 | `total_n (cross-checked against exact replay total_nitrogen)` | `benchmark_results/222YCA_A1_yc_v1_augmented_seed1/evaluation/222YCA_A1_checkpoint_validation_summary.csv` |
| Augmented | 2 | 120.000 | 1200.000 | `total_n (cross-checked against exact replay total_nitrogen)` | `benchmark_results/222YCA_A2_yc_v1_augmented_seed2/evaluation/222YCA_A2_checkpoint_validation_summary.csv` |
| Original | three_seed_mean | 146.667 | 440.000 | `total_n / total_nitrogen` | `six independent validation files plus exact replay` |
| Augmented | three_seed_mean | 146.667 | 440.000 | `total_n / total_nitrogen` | `six independent validation files plus exact replay` |

因此：Original seed0/1/2 的 seed-level N 均值为 160/40/240，三者均值为 146.667、和为 440；Augmented seed0/1/2 为 240/80/120，三者均值也为 146.667、和也为 440。**N duplication status = arithmetic coincidence, no pipeline bug found**。

审计结果：

| Result | Check | Evidence |
|---|---|---|
| PASS | Original seed0 validation file exists | benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/evaluation/055_00_checkpoint_validation_summary.csv |
| PASS | Original seed0 has 40 validation rows | 40 |
| PASS | Original seed0 key uniqueness | rows=40, unique_keys=40 |
| PASS | Original seed0 expected years/checkpoints | years=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], checkpoints=[25000, 50000, 75000, 100000] |
| PASS | Original seed0 seed label | [0] |
| PASS | Original seed1 validation file exists | benchmark_results/222YCA_O1_yc_v1_original_seed1/evaluation/222YCA_O1_checkpoint_validation_summary.csv |
| PASS | Original seed1 has 40 validation rows | 40 |
| PASS | Original seed1 key uniqueness | rows=40, unique_keys=40 |
| PASS | Original seed1 expected years/checkpoints | years=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], checkpoints=[25000, 50000, 75000, 100000] |
| PASS | Original seed1 seed label | [1] |
| PASS | Original seed2 validation file exists | benchmark_results/222YCA_O2_yc_v1_original_seed2/evaluation/222YCA_O2_checkpoint_validation_summary.csv |
| PASS | Original seed2 has 40 validation rows | 40 |
| PASS | Original seed2 key uniqueness | rows=40, unique_keys=40 |
| PASS | Original seed2 expected years/checkpoints | years=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], checkpoints=[25000, 50000, 75000, 100000] |
| PASS | Original seed2 seed label | [2] |
| PASS | Augmented seed0 validation file exists | benchmark_results/221YCA_yc_ppo_domain_augmentation/evaluation/221YCA_checkpoint_validation_summary.csv |
| PASS | Augmented seed0 has 40 validation rows | 40 |
| PASS | Augmented seed0 key uniqueness | rows=40, unique_keys=40 |
| PASS | Augmented seed0 expected years/checkpoints | years=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], checkpoints=[25000, 50000, 75000, 100000] |
| PASS | Augmented seed0 seed label | [0] |
| PASS | Augmented seed1 validation file exists | benchmark_results/222YCA_A1_yc_v1_augmented_seed1/evaluation/222YCA_A1_checkpoint_validation_summary.csv |
| PASS | Augmented seed1 has 40 validation rows | 40 |
| PASS | Augmented seed1 key uniqueness | rows=40, unique_keys=40 |
| PASS | Augmented seed1 expected years/checkpoints | years=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], checkpoints=[25000, 50000, 75000, 100000] |
| PASS | Augmented seed1 seed label | [1] |
| PASS | Augmented seed2 validation file exists | benchmark_results/222YCA_A2_yc_v1_augmented_seed2/evaluation/222YCA_A2_checkpoint_validation_summary.csv |
| PASS | Augmented seed2 has 40 validation rows | 40 |
| PASS | Augmented seed2 key uniqueness | rows=40, unique_keys=40 |
| PASS | Augmented seed2 expected years/checkpoints | years=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], checkpoints=[25000, 50000, 75000, 100000] |
| PASS | Augmented seed2 seed label | [2] |
| PASS | validation action_sequence totals and event counts | mismatches=0 |
| PASS | daily output files exist | missing=0 |
| PASS | daily output actions reconcile to validation | mismatches=0 |
| PASS | six validation file hashes are distinct | distinct_hashes=6 |
| PASS | Original/Augmented source files are disjoint | source_files=6 |
| PASS | exact replay has 240 unique method-seed-checkpoint-year rows | rows=240, unique_keys=240 |
| PASS | exact replay statuses are ok | ['ok'] |
| PASS | exact replay matches source management totals | max_abs_management_diff=0.0 |

## 2. 每个 seed 的 checkpoint 轨迹

下表所有效率指标来自既有精确 Summary.OUT/ETCP replay；没有从 daily CSV 反推 WP_ET。

| Method | Seed | Checkpoint | Yield | Irrigation | N | PFP-N | WP_ET | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Augmented | 0 | 25K | 8203.0 | 214.5 | 240.0 | 34.17 | 2.219 | 0.728 |
| Augmented | 0 | 50K | 8144.4 | 180.0 | 120.0 | 67.87 | 2.205 | 0.953 |
| Augmented | 0 | 75K | 8202.1 | 193.5 | 240.0 | 34.18 | 2.215 | 0.772 |
| Augmented | 0 | 100K | 8199.9 | 220.5 | 240.0 | 34.17 | 2.212 | 0.742 |
| Augmented | 1 | 25K | 6916.2 | 103.5 | 200.0 | 34.59 | 1.963 | 0.444 |
| Augmented | 1 | 50K | 6822.4 | 75.0 | 184.0 | 39.15 | 2.006 | 0.430 |
| Augmented | 1 | 75K | 6823.4 | 75.0 | 188.0 | 36.74 | 2.006 | 0.424 |
| Augmented | 1 | 100K | 6024.5 | 45.0 | 80.0 | 75.31 | 1.864 | 0.379 |
| Augmented | 2 | 25K | 8073.7 | 151.5 | 240.0 | 33.63 | 2.239 | 0.753 |
| Augmented | 2 | 50K | 8203.1 | 214.5 | 240.0 | 34.17 | 2.253 | 0.750 |
| Augmented | 2 | 75K | 6911.5 | 90.0 | 240.0 | 28.81 | 1.977 | 0.374 |
| Augmented | 2 | 100K | 7004.5 | 75.0 | 120.0 | 58.39 | 2.061 | 0.558 |
| Original | 0 | 25K | 8201.7 | 228.0 | 240.0 | 34.16 | 2.219 | 0.690 |
| Original | 0 | 50K | 6826.0 | 75.0 | 240.0 | 28.45 | 2.005 | 0.340 |
| Original | 0 | 75K | 6140.9 | 45.0 | 152.0 | 40.74 | 1.899 | 0.280 |
| Original | 0 | 100K | 6072.6 | 45.0 | 160.0 | 37.95 | 1.879 | 0.267 |
| Original | 1 | 25K | 8202.2 | 214.5 | 240.0 | 34.17 | 2.213 | 0.750 |
| Original | 1 | 50K | 8203.3 | 214.5 | 240.0 | 34.18 | 2.224 | 0.749 |
| Original | 1 | 75K | 8202.9 | 214.5 | 240.0 | 34.18 | 2.252 | 0.748 |
| Original | 1 | 100K | 5348.4 | 45.0 | 40.0 | 133.73 | 1.669 | 0.331 |
| Original | 2 | 25K | 8202.3 | 222.0 | 240.0 | 34.18 | 2.190 | 0.719 |
| Original | 2 | 50K | 8202.5 | 223.5 | 240.0 | 34.17 | 2.224 | 0.717 |
| Original | 2 | 75K | 8200.8 | 214.5 | 240.0 | 34.17 | 2.205 | 0.724 |
| Original | 2 | 100K | 8200.8 | 214.5 | 240.0 | 34.17 | 2.205 | 0.724 |

100K 三种子均值为：Original Yield = `6540.6` kg/ha，Augmented Yield = `7076.3` kg/ha。Augmented 的 seed0 为 `8199.9` kg/ha，但 seed1/2 分别为 `6024.5` 和 `7004.5` kg/ha，说明平均提升不能代表 seed-level 稳定性。

## 3. 10 个问题的逐项回答

1. **N=146.667 是否确认不是 pipeline bug？** 是。六个独立 validation 文件、不同 hash、无重复键、无 seed 错位、无逐日动作/总量不一致；两组 seed-level 数值分别重排为 160+40+240 和 240+80+120，均为 440。

2. **每个 seed 的 25K→100K Yield 轨迹是什么？** 已在上表完整列出。简写为：Original s0 `8201.7→6826.0→6140.9→6072.6`；Original s1 `8202.2→8203.3→8202.9→5348.4`；Original s2 `8202.3→8202.5→8200.8→8200.8`；Augmented s0 `8203.0→8144.4→8202.1→8199.9`；Augmented s1 `6916.2→6822.4→6823.4→6024.5`；Augmented s2 `8073.7→8203.1→6911.5→7004.5` kg/ha。

3. **Augmented 是否在 3 seeds 下都减缓 late-training degradation？** 否。s0 基本稳定；s1 从 75K 到 100K 下降约 `-11.7`%，s2 在 50K→75K 下降约 `-15.7`%。

4. **Original seed1 何时开始塌缩？** 75K 仍为高产、多动作分支；75K→100K 时 Yield 降至 5348.3 kg/ha，I 从约 214.5 降至 45.0 mm，N 从 240 降至 40 kg/ha，动作序列变为所有 10 年相同的 `DAP1 I45/N40`。因此 first divergence/collapse checkpoint = **100K（相对 75K）**；按六个 run 的 seed 分叉看，Original 首次明显分叉在 50K，Augmented 首次明显分叉已在 25K。

5. **Original seed1 是否存在 PPO 内部训练异常？** 不能由现有文件确认。Original seed0/1/2 都没有保存 prompt 所列的 update-level PPO 指标；因此不能补造 entropy/KL/value 结论。可确认的是行为层的 checkpoint collapse，而非已证实的 numerical failure。

6. **Augmented seed1/2 是否有同类内部异常？** 两者均有 update logs。三种 Augmented run 的 `clip_fraction` 最大值均为 0，`approx_kl` 最大值约在 `0.001711` 以下，没有看到典型 KL/clip 爆炸。A1 的 explained variance 在后期出现负值且 value loss 有孤立高点，A2 也有 explained variance 波动，但这些更像拟合/价值估计不稳定信号，不能单独证明数值训练失败。A1/A2 的策略分支仍分别在 100K 退化，说明“没有 KL 爆炸”与“最终策略稳定”不是同一件事。

7. **Original seed1 与 Augmented seed1/2 的动作分布是否高度相似？** 不支持“高度相似”强断言。100K 三者都只剩 1–2 个非零动作对/极少事件，但 O1 是 `I45/N40 @ DAP1`，A1 是 `I45/N80 @ DAP1`，A2 是 `I45/N80 @ DAP1` 加 `I30/N40 @ DAP8`；总 I/N 分别约为 45/40、45/80、75/120。因此更准确的是：**同属低投入、早期动作稀疏的保守策略家族，但不是同一个动作 attractor 的充分证据**。

8. **坏 seed 是否“有胁迫但不施水/氮”？** 是，作为描述性行为证据成立。100K daily logs 中，坏 seed O1/A1/A2 都出现 SWFAC>0.05 的水分胁迫日与无灌溉日，以及 NSTRES>0.05 的氮胁迫日与无施氮日；详细计数、响应率和最大胁迫见下表。这里不把逐日共现直接解释为因果，只说明动作窗口内存在“胁迫—无响应”。

| Method | Seed | Water-stress days | Stressed/no-I days | Water response rate | N-stress days | Stressed/no-N days | N response rate | max SWFAC | max NSTRES |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Augmented | 0 | 0 | 0 | NA | 0 | 0 | NA | 0.000 | 0.015 |
| Augmented | 1 | 133 | 133 | 0.000 | 189 | 189 | 0.000 | 1.000 | 0.456 |
| Augmented | 2 | 103 | 103 | 0.000 | 60 | 60 | 0.000 | 0.975 | 0.198 |
| Original | 0 | 131 | 131 | 0.000 | 0 | 0 | NA | 1.000 | 0.016 |
| Original | 1 | 132 | 132 | 0.000 | 472 | 472 | 0.000 | 1.000 | 0.515 |
| Original | 2 | 2 | 1 | 0.500 | 0 | 0 | NA | 0.523 | 0.015 |

9. **当前主要问题更像什么？** 首要是 Type B seed-dependent policy basin；次要是 Type A 的行为层 policy collapse。Type C 不足以确认，Type D 不成立。

10. **下一步先修 PPO 还是进入 G×E V2？** 先做 PPO 最小化稳定性/探索对照，不进入 G×E V2。不要立即增加天气、IC、基因型或训练步数；先预注册 checkpoint 选择、保留每个 seed 的 action diversity 与 stress-response guardrails，再判断是否需要参数级最小改动。

## 4. PPO 内部指标可用性

| Method | Seed | Update rows | max approx_kl | max clip_fraction | entropy_loss first→last | min explained_variance | max value_loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Augmented | 0 | 695 | 0.000921 | 0.004 | -0.216 -> -0.149 | -16.311 | 2.427 |
| Augmented | 1 | 695 | 0.001711 | 0.008 | -0.261 -> -0.086 | -9.110 | 5.154 |
| Augmented | 2 | 695 | 0.001018 | 0.004 | -0.250 -> -0.225 | -3.865 | 4.408 |
| Original | 0 | NA | NA | NA | NA | NA | NA |
| Original | 1 | NA | NA | NA | NA | NA | NA |
| Original | 2 | NA | NA | NA | NA | NA | NA |

Original 的 update-level 指标明确记为 `not available in existing logs`；没有用 checkpoint 行为结果替代内部指标。

## 5. 输出与复现

核心 CSV 已同时写入项目级 `results/` 和独立目录 `experiments/223_yc_ppo_seed_instability_diagnostic/results/`：

- `yc_seed_instability_checkpoint_metrics.csv`
- `yc_seed_instability_action_summary.csv`
- `yc_seed_instability_action_histogram.csv`
- `yc_seed_instability_ppo_internal_metrics.csv`
- `yc_seed_instability_n_source_audit.csv`
- `yc_seed_instability_stress_response_100k.csv`
- `yc_seed_instability_data_integrity_checks.csv`

图形：

- `experiments/223_yc_ppo_seed_instability_diagnostic/figures/figure1_checkpoint_yield_trajectories.png`
- `experiments/223_yc_ppo_seed_instability_diagnostic/figures/figure2_checkpoint_irrigation_n_trajectories.png`
- `experiments/223_yc_ppo_seed_instability_diagnostic/figures/figure3_ppo_internal_metrics.png`
- `experiments/223_yc_ppo_seed_instability_diagnostic/figures/figure4_100k_action_pair_distribution.png`
- `experiments/223_yc_ppo_seed_instability_diagnostic/figures/figure5_100k_cumulative_water_n_trajectory.png`

分析脚本：`experiments/223_yc_ppo_seed_instability_diagnostic/scripts/run_diagnostic.py`。所有输入均为既有 055_00、221YCA、222YCA validation/daily/log/replay 文件；本轮没有启动训练进程。

## 6. 边界与未执行项

- WP_ET 仅使用已存在的 `Summary.OUT/ETCP` 精确回放；没有从 daily CSV 反推。
- 诊断不证明某一个 PPO 参数或 reward 项是根因；Original 的内部训练曲线缺失是重要证据缺口。
- 没有重新训练 seed、没有增加 seed 数量、没有改 PPO/reward/action/constraints/weather/IC/genotype、没有进入其他站点或 G×E V2。
