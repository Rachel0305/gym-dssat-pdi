# SYA 天气预报决策实验冻结清单

冻结日期：2026-08-16
站点：SYA / SY / originIC
验证年：2014–2023
冻结主候选：E2 dual-branch weather reader，5K checkpoint

## 冻结结论

E2 作为当前 SYA 的主 forecast 候选冻结。E2 在 3 个 seed 的 5K checkpoint 上产量均高于对应 no-forecast，但 WP_ET 只有 1/3 seed 胜出；因此冻结的是“有条件的 forecast 优势”，不是“所有指标全面优于 no-forecast”。E3 和 E4 保留为诊断/消融记录，不作为当前主方案。

### 5K 跨 seed 配对摘要

| 指标 | E2 胜出 seed |
|---|---:|
| 产量 | 3/3 |
| PFP-N | 2/3 |
| WP_ET | 1/3 |
| 动作多样性门禁 | 3/3 |
| 天气响应门禁 | 2/3 |

Pooled 平均产量：E2 10032.72 kg/ha，no-forecast 9996.85 kg/ha。Pooled 平均 WP_ET：E2 2.044，no-forecast 2.073；加权 WP_ET：E2 2.044，no-forecast 2.071。

## 实验链

| 阶段 | 任务 | 作用 | 状态 |
|---|---|---|---|
| E1 | 141E1 | 只改天气特征编码 | 已完成 |
| E2 | 142E2 / 143E2 | 双分支天气输入；5K 主候选 | 已完成并冻结 |
| E2 seeds | 145E2S1 / 146E2S2 / 147E2 | 独立 seed、5K/10K 稳定性与天气响应 | 已完成 |
| WP_ET | 151E2N | E2/no-forecast 5K ETCP replay | 已完成 |
| E3 | 152E3 | 双分支但天气信号全置零的对照 | 已完成；诊断对照 |
| E4 | 153E4 | 6 个天气特征有效、6 个槽位置零 | 已完成；未替代 E2 |
| E2 final audit | 154E2 | 三 seed 5K 产量/PFP-N/WP_ET 联合核验 | 已完成 |

## 代码与 prompt

主要代码：

- `src/forecast_engineered_observation_056_057.py`
- `src/run_141E1_sya_actionable_weather_encoding_smoke2k.py`
- `src/run_142E2_sya_dual_branch_weather_reader_smoke2k.py`
- `src/run_143E2_sya_dual_branch_weather_reader_5k.py`
- `src/run_144E2_sya_dual_branch_weather_reader_10k_seed0.py`
- `src/run_145_146_E2_independent_seed_10k.py`
- `src/run_148_149_paired_noforecast_5k_independent_seeds.py`
- `src/replay_151_e2_noforecast_wp_et_5k.py`
- `src/audit_145_146_E2_cross_seed_5k_10k.py`
- `src/audit_154E2_cross_seed_5k_wp_et_stability.py`
- `src/run_152E3_sya_dual_branch_noforecast_control.py`
- `src/replay_152e3_wp_et_5k.py`
- `src/run_153E4_sya_compact_weather_encoding_5k.py`
- `src/replay_153e4_wp_et_5k.py`

对应 prompt 位于 `prompts/2026-08-16_sya_E*.md` 以及 `prompts/2026-08-16_sya_E2_cross_seed_5k_wp_et_stability.md`。

## 关键记录

- E2 三 seed 联合报告：`benchmark_results/154E2_sya_cross_seed_5k_wp_et_stability/2026-08-16_sya_E2_cross_seed_5k_wp_et_stability.md`
- E2 WP_ET replay：`benchmark_results/151E2N_wp_et_5k_replay/2026-08-16_sya_E2_wp_et_5k_replay.md`
- E3 结果：`docs/152E3_sya_originIC_dual_branch_noforecast_control_5k_result.md`
- E4 结果：`docs/153E4_sya_originIC_compact_weather_encoding_5k_result.md`
- E2 三 seed 配对 CSV：`benchmark_results/154E2_sya_cross_seed_5k_wp_et_stability/154E2_paired_summary_by_seed.csv`

## 冻结边界与 GitHub 备份策略

- 只提交 prompt、配置、可复现实验代码、轻量 JSON/CSV 汇总和 Markdown 记录。
- 不提交 PPO 模型 zip、逐日 DSSAT 输出、`.OUT`/`.WTH`/`.SOL` 等大文件或原始输入；这些文件继续保留在本地实验工作区，并由结果 manifest 指向。
- 本次 Git 提交只包含 SYA forecast 实验链，不包含工作区中其他站点、proposal 或未相关的未提交改动。
