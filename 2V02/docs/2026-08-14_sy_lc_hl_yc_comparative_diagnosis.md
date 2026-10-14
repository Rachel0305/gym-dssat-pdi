# SY/LC 与 HL/YC PPO 表现差异诊断（2026-08-14）

## 结论先行

当前证据支持的不是“SY、LC 算法成功而 HL、YC 算法失败”，而是：

1. **SY 和 LC 在所选 checkpoint 上形成了可用且能维持产量的管理轨迹。** SY 的 100K 结果维持约 10.1 t/ha；LC 从 25K 的欠灌状态恢复到 50K，并在 50K--100K 稳定在约 9.1 t/ha。
2. **HL 和 YC 在 25K 已有可用候选，但继续训练出现后期策略坍缩。** HL 产量从 25K 的 6.84 t/ha 降至 100K 的 6.24 t/ha；YC 从 8.20 t/ha 降至 6.07 t/ha，同时灌溉大幅下降、水分胁迫增加。
3. **LC 的论文表现最完整；SY 的主要优势是产量和 PFP-N，WP_ET 只是接近基线最优。** HL/YC 不能称为无效，但目前不能称为 PPO 有明显综合优势。

## 同口径性能比较

SY/LC 使用冻结的 100K 五情景结果；HL/YC 使用预注册时间折叠选出的 25K 候选。四站均为 16 联合动作、raw observation、无天气预报、同一 PPO 基线参数（`lr=3e-4, gamma=1, n_steps=144, batch=144, n_epochs=5, ent_coef=0.01, net=[64,64]`）。

| 站点/选定 checkpoint | 产量均值 kg/ha | WP_ET | PFP_N | 灌溉 mm | N kg/ha | 产量胜场 | WP_ET 胜场 | PFP_N 胜场 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SY 100K | 10,072.7 | 2.110 | 42.66 | 240.0 | 236.0 | 5/10 | 1/10 | 10/10 |
| LC 100K | 9,140.6 | 2.723 | 38.09 | 208.5 | 240.0 | 6/10 | 6/10 | 10/10 |
| HL 25K | 6,844.3 | 1.514 | 28.53 | 225.0 | 240.0 | 6/10 | 4/10 | 0/10* |
| YC 25K | 8,201.7 | 2.219 | 34.16 | 228.0 | 240.0 | 3/10 | 0/10 | 10/10 |

胜场是 PPO 相对于当年其余四个情景的最优值。HL 的 PFP-N 0/10 主要受 `dssat_auto_external_n` 对照的低 N、高 PFP 影响；排除该探索性外部-N情景后，HL 的 PFP-N 为 4/10 胜场、平均差约 -5.42，而不是 -86.1。主论文不应把外部-N对照当作普通基线解释。

## 最关键的机制差异：后期 checkpoint 轨迹

| 站点 | checkpoint | 产量 kg/ha | 灌溉 mm | N kg/ha | 水分胁迫日均值 | 平均灌溉事件/年 | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| SY | 25K→100K | 10,011→10,073 | 240→240（75K短暂148.5） | 240→236 | 0→0 | 13.4→10.2 | 高产轨迹保持，短暂漂移后恢复；没有持续坍缩 |
| LC | 25K→100K | 6,321→9,141 | 90→208.5 | 240→240 | 16.2→1.6 | 3.7→6.3 | 25K 是欠灌早期状态，50K 后修正并稳定 |
| HL | 25K→100K | 6,844→6,244 | 225→78 | 240→124 | 0→7.1 | 12.7→2.1 | 灌溉/N 逐步消失，胁迫上升，产量连续下降 |
| YC | 25K→100K | 8,202→6,072 | 228→45 | 240→160 | 0→13.1 | 8.6→1.0 | 最明显的后期坍缩，固定低灌溉策略取代早期候选 |

因此，HL/YC 的问题不是动作没有传入 DSSAT：已有审计显示 grid、request→safe、raw→safe、safe→DSSAT mismatch 均为 0。问题是学习过程后段把高收益管理动作的概率质量逐步移向低投入/低灌溉动作。

## 为什么 SY/LC 更容易得到可用结果

### 1. 站点环境的反馈信号不同

四站 PPO 代码和动作格基本相同，但 DSSAT 的天气、土壤、初始条件和作物响应不同。逐日 PPO 数据的描述性统计显示：

- LC 平均降雨约 290 mm、平均 Tmax 约 32.2°C，部分年份出现明显 water-stress 信号；及时灌溉带来的产量和 ETCP 改善较容易被 reward 识别，因此 LC 从 25K 欠灌状态在 50K 修正。
- SY 平均降雨约 492 mm、平均 Tmax 约 27.4°C；最终策略采用稳定的分阶段灌溉/施氮轨迹，产量保持在 10 t/ha 左右。
- HL 平均降雨约 544 mm、平均 Tmax 约 24.2°C；YC 平均降雨约 397 mm、平均 Tmax 约 31.4°C。两站 25K 候选虽然能压低胁迫，但继续训练后没有维持该管理强度，导致后期漂移。

这些是环境差异的证据和合理机制假设，不是仅凭天气统计就能证明的因果结论。

### 2. 成功站点保留了“有效管理轨迹”，而不是单纯追求动作多样性

SY 100K 和 LC 100K 的年内动作对集合相对固定，但动作发生在多个关键 DAP，并形成稳定的分阶段管理序列。也就是说，成功不要求每年使用完全不同的动作集合，而要求策略在训练后期仍保留对有效灌溉/施氮事件的响应。

HL/YC 25K 的动作集合反而不少，但 50K--100K 逐渐退化为重复的低灌溉轨迹；这说明“2K/25K 的动作多样性”本身不足以保证长期性能，必须看 checkpoint 轨迹和最终水分胁迫。

### 3. LC 的 WP_ET 优势来自较低 ETCP 下的高产

LC PPO 的平均 ETCP 约 335 mm、产量约 9,141 kg/ha，因此 WP_ET 达 2.723；HL 的 ETCP 约 452 mm 而产量只有 6,844 kg/ha，YC 的 ETCP 约 369 mm 而产量约 8,202 kg/ha。相同 16 动作和 PPO 参数在不同站点上对应的产量—蒸散响应不同，不能把 LC 的 WP_ET 优势直接外推到 HL/YC。

## 可比性限制与稳定性 caveat

- SY 使用 **originIC**，LC/HL/YC 使用 **lowIC**；SY 与其余站点不是完全同一初始条件，因此“SY 更成功”不能单独归因于站点。
- LC 的 seed0 论文结果很强，但 seed1/seed2 的 2K 审计均出现单一粗粒度跨年动作签名；LC 的跨 seed 稳定性尚未证明。
- SY 的 seed1 2K 最终机制 gate 通过，但 25K 五情景 replay 出现 provenance endpoint mismatch，因此也尚未完成独立 seed 的完整性能复核。
- HL/YC 的 25K 是通过时间折叠选出的候选，不应与 SY/LC 的冻结 100K 直接当作同一训练阶段比较。

## 当前判断

- **LC：输出指标上最成功，尤其是 WP_ET；但跨 seed 稳定性不足，论文中应称为 seed0 冻结结果并披露稳定性限制。**
- **SY：产量和 PFP-N 表现有优势，策略在后期没有像 HL/YC 那样持续坍缩；WP_ET 接近但不普遍超过基线。**
- **HL：25K 是可用候选，100K 是后期坍缩结果；在排除外部-N比较后仍没有稳定综合优势。**
- **YC：25K 产量接近基线且 PFP-N 有优势，但 WP_ET 全部落后，50K 后发生最明显的产量/灌溉坍缩。**

因此，当前最合理的论文叙述是“PPO 的收益具有站点和 checkpoint 依赖性”：SY/LC 证明了在部分环境中可以形成高产或高 WP_ET 策略；HL/YC 说明同一 PPO 结构在不同 DSSAT 环境中可能发生后期策略漂移，不能宣称跨站点普遍优于专家或记录农户策略。

## 证据来源

- SY：`benchmark_results/046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000/tables/046_06_sya_five_scenario_season_summary.csv`；`benchmark_results/046_10_sya_originIC_expanded_action_maskableppo/evaluation/046_10_checkpoint_validation_summary.csv`。
- LC：`benchmark_results/053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/053_03_lca_five_scenario_season_summary.csv`；`benchmark_results/053_00_lca_lowIC_expanded_action_maskableppo/evaluation/053_00_checkpoint_validation_summary.csv`。
- HL/YC 后期轨迹：`docs/2026-08-10_hl_yc_checkpoint_rescue_audit.md`；对应 `054_00`/`055_00` checkpoint validation summaries。
- HL/YC 时间折叠候选：`docs/2026-08-11_hl_yc_checkpoint_selection_design.md`。
- SY/LC seed 稳定性：`docs/2026-08-13_sy_lc_seed1_execution.md`；`docs/2026-08-13_sy_lc_next_validation_execution.md`；`docs/2026-08-13_lc_seed_trajectory_diagnostic_execution.md`。
