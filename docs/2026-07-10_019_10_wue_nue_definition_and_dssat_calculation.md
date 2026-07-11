# 019_10 WUE/NUE 文献定义与 DSSAT 原生指标计算

## 1. 最终建议

论文中不要单独写含义模糊的“WUE”和“NUE”，而应明确写出具体指标：

### 水分指标

1. **主指标：基于生育季实际蒸散的籽粒水分生产率 `WP_ET`**

   ```text
   WP_ET = grain yield / actual seasonal evapotranspiration
         = HWAM / ETCP / 10
   ```

   - DSSAT 原生指标：`YPEM × 0.1`；
   - 单位：`kg m-3`；
   - 适用于 null、auto、expert 和 DQN 全部情景；
   - `1 kg ha-1 mm-1 = 0.1 kg m-3`。

2. **辅助指标：单位灌溉水籽粒生产率 `IWP_gross`**

   ```text
   IWP_gross = HWAM / irrigation / 10
   ```

   - DSSAT 原生指标：`YPIM × 0.1`；
   - 单位：`kg m-3`；
   - 灌溉量为 0 时记为 `NA`；
   - 这是“总产量/灌溉量”口径，分子同时包含降雨和初始土壤水贡献，不能解释成灌溉的因果边际效应。

### 氮素指标

1. **主施肥效率指标：氮肥偏生产力 `PFP_N`**

   ```text
   PFP_N = HWAM / NICM
   ```

   - DSSAT 原生指标：`YPNAM`；
   - 单位：`kg grain kg-1 applied N`；
   - `NICM=0` 时记为 `NA`，绝不能写成 0 或无穷大；
   - 它包含土壤本底氮贡献，因此跨站点解释要谨慎。

2. **跨零施氮情景的辅助指标：氮内部利用效率 `NUtE`**

   ```text
   NUtE = HWAM / NUCM
   ```

   - DSSAT 原生指标：`YPNUM`；
   - 单位：`kg grain kg-1 crop N uptake`；
   - 在不施肥但作物仍从土壤吸氮时仍可计算；
   - 高值不一定代表管理更好，也可能来自氮胁迫和较低吸氮量，必须同时报告产量、`NUCM` 和氮胁迫。

3. **辅助氮平衡指标**

   ```text
   PNB_N = NUCM / NICM
   N_leaching = NLCM
   ```

   `PNB_N` 包含土壤供氮，不等同于肥料氮回收率。`NLCM` 是季节氮淋洗量，可作为环境效益指标，但不能单独替代完整氮平衡。

## 2. 为什么采用这些定义

- Zwart 与 Bastiaanssen 将作物水分生产率定义为可收获产量与实际蒸散量之比，这是跨灌溉和雨养情景更稳定的水分指标。
- Congreves 等系统梳理了 PFP、AE、肥料氮回收效率和 NUtE 等不同 NUE 概念，强调它们回答的问题不同，不能都简称为 NUE。
- DSSAT `Summary.OUT` 直接输出 `HWAM`、`IRCM`、`ETCM/ETCP`、`NICM`、`NUCM`、`NLCM`，并同时输出 `YPEM`、`YPIM`、`YPNAM` 和 `YPNUM`。
- Kropp 等采用 DSSAT 做水氮多目标优化时同时最大化产量、降低水氮投入和氮淋洗，支持本项目继续保留产量—投入 Pareto 判据，而不是只看一个比值。

## 3. DSSAT 输出核对结果

本轮读取了五站代表年份的 null、DSSAT auto、官方推广 expert 和 DQN，共 20 个 `Summary.OUT` 情景。

- 每个原始行都通过 `HWAM + IRCM + NICM` 与正式四情景表匹配；
- 最大匹配误差为 `1.2`，来自 `Summary.OUT` 对小数灌溉量的整数化显示；
- `YPEM × 0.1` 与 `HWAM / ETCP / 10` 的最大绝对差仅约 `0.0053 kg m-3`；
- 因此正式 `WP_ET` 使用 DSSAT 原生 `YPEM × 0.1`；
- 灌溉总量展示仍使用事件表中的精确小数值，灌溉水生产率优先使用 DSSAT 原生 `YPIM × 0.1`。

特别注意：在当前 DSSAT 4.8.0 输出中，`YPEM` 与“种植至收获”的 `ETCP` 对应，而不是直接按表内 `ETCM` 复算。后续脚本已经固定使用该经过数值核验的口径。

## 4. 当前五站 DQN 的原生效率指标

| 站点 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | NUCM kg/ha | WP_ET kg/m3 | IWP_gross kg/m3 | PFP_N kg/kg | NUtE kg/kg | NLCM kg/ha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HLA | 7854 | 120 | 0 | 248 | 1.69 | 6.54 | NA | 31.7 | 0 |
| YC | 9418 | 120 | 250 | 255 | 2.53 | 7.85 | 37.7 | 36.9 | 0 |
| FQ | 7995 | 60 | 0 | 188 | 2.47 | 13.32 | NA | 42.5 | 0 |
| SY | 11216 | 120 | 300 | 306 | 2.30 | 9.35 | 37.4 | 36.7 | 0 |
| LC | 8739 | 90 | 0 | 214 | 3.06 | 9.71 | NA | 40.8 | 0 |

## 5. DQN 与基线比较后的真实结论

### HLA2010

- 与 auto 同产量，`WP_ET` 高约 3.0%，`NUtE` 高约 1.3%；
- 与官方 expert 同产量，`WP_ET` 高约 3.7%，`NUtE` 高约 6.4%；
- DQN 不施氮，不能把 `PFP_N` 写成无限大，只能写“在相同产量下省去外源施氮”；
- 这是当前效率组合最好的代表结果之一，但仍有跨 seed 产量不稳定限制。

### YC2014

- 相对 auto：产量高 705 kg/ha、`WP_ET` 高约 3.7%，但 `NUtE` 低约 25%；
- 相对官方 expert：产量仅高 1 kg/ha，`WP_ET` 低约 1.2%，`PFP_N` 为 37.7 对 38.0，`NUtE` 高约 0.3%；
- 因此 YC2014 是高产结果，但目前不能声称水肥效率全面超过官方 expert。

### FQ2016

- 相对 auto：产量低 17 kg/ha，`WP_ET` 低约 0.4%，不能称为超过 auto；
- 相对官方 expert：产量高 55 kg/ha，`WP_ET` 高约 5.1%，`NUtE` 高约 0.7%，且不施氮、氮淋洗少 26 kg/ha；
- 可称为“相对官方 expert 的资源效率候选”，但尚未跨 seed 复现。

### SY2014

- 相对官方 expert：产量高 139 kg/ha，`WP_ET` 高约 1.8%，`PFP_N` 为 37.4 对 36.9，氮淋洗少 3 kg/ha；
- 但 `NUtE` 低约 1.3%；
- 相对 auto 虽然产量和 `WP_ET` 大幅提高，但 auto 本身严重低产且不施氮，不能用 `PFP_N` 做直接比较；
- 因此是跨 seed 高产候选，不宜简化成“所有 NUE 指标均超过 auto”。

### LC2010

- 相对 auto：产量高 1 kg/ha，`WP_ET` 高约 0.3%，`NUtE` 相同；
- 相对官方 expert：产量相同、用水用氮更少，但 `WP_ET` 低约 1.3%，`NUtE` 相同；
- 说明“少灌溉”与“ET 水分生产率更高”并非必然相同；
- 加上 seed1 会回到 N300，目前只能称为资源投入策略不稳定。

## 6. 论文中的建议判定规则

### 第一层：产量—投入 Pareto 判据

DQN 只有在以下条件同时满足时，才称为“产量与投入组合占优”：

```text
Yield_DQN >= Yield_baseline（仅允许1 kg/ha数值容差）
I_DQN <= I_baseline
N_DQN <= N_baseline
```

### 第二层：明确命名的效率指标

- 水分：主报告 `WP_ET`，辅助报告 `IWP_gross`；
- 氮素：施氮双方均大于 0 时报告 `PFP_N`；所有情景可辅助报告 `NUtE`、`NUCM` 和 `NLCM`；
- 若一方 `NICM=0`，不得比较 `PFP_N`，改为说明产量—施氮投入 Pareto 关系。

### 第三层：需要配对反事实的严格因果指标

现有 null 同时取消水和氮，不能直接用于以下公式：

```text
AE_N = (Y_N - Y_0_matched) / N
RE_N = (NUCM_N - NUCM_0_matched) / N
IWP_incremental = (Y_I - Y_noI_matched) / I / 10
```

若论文必须报告这些指标，需要对每个 DQN/expert/auto 管理序列增加配对回放：

- 算 `AE_N/RE_N`：保留完全相同的灌溉时机和水量，只把施氮设为 0；
- 算增量 IWP：保留完全相同的施氮时机和用量，只把灌溉设为 0。

这属于低成本 DSSAT 前向反事实，不需要重新训练 DQN。

## 7. 文献依据

1. Zwart, S. J. & Bastiaanssen, W. G. M. Review of measured crop water productivity values for irrigated wheat, rice, cotton and maize. *Agricultural Water Management* 69, 115–133 (2004). DOI: [10.1016/j.agwat.2004.04.007](https://doi.org/10.1016/j.agwat.2004.04.007).
2. Congreves, K. A. et al. Nitrogen Use Efficiency Definitions of Today and Tomorrow. *Frontiers in Plant Science* 12, 637108 (2021). DOI: [10.3389/fpls.2021.637108](https://doi.org/10.3389/fpls.2021.637108).
3. Dobermann, A. Nutrient use efficiency – measurement and management. In *Fertilizer Best Management Practices* (International Fertilizer Industry Association, 2007). Definitions summarized and discussed in Congreves et al. (2021).
4. DSSAT. *DSSAT User's Guide Volume 3*, Table 8: `HWAM`, `IRCM`, `ETCM`, `NICM`, `NUCM`, `NLCM` definitions. [Official DSSAT PDF](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf).
5. Kropp, I. et al. A multi-objective approach to water and nutrient efficiency for sustainable agricultural intensification. *Agricultural Systems* 173, 289–302 (2019). DOI: [10.1016/j.agsy.2019.03.014](https://doi.org/10.1016/j.agsy.2019.03.014).

## 8. 输出文件

- 指标字典：`DSSAT_auto_validation/five_site_wue_nue_019_10/019_10_metric_dictionary.csv`
- 五站四情景原生指标：`DSSAT_auto_validation/five_site_wue_nue_019_10/019_10_native_wue_nue_metrics.csv`
- DQN 对 auto/官方 expert 比较：`DSSAT_auto_validation/five_site_wue_nue_019_10/019_10_dqn_vs_baseline_efficiency_comparison.csv`
- 原始行匹配审计：`DSSAT_auto_validation/five_site_wue_nue_019_10/019_10_source_match_audit.csv`
- 可复算脚本：`src/calculate_five_site_wue_nue_from_summary_019_10.py`
