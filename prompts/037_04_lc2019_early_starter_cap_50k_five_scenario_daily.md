# 037_04：LC2019 early starter cap 50K 五情景日过程图

## 背景

037_03 在 LCA 单站点 clean retrain 中验证了 DAP1-10 early starter cap 能真实接入 MaskablePPO 训练/评估路径。按验证年均值，50K checkpoint 是当前最适合做过程图审查的候选：

- 平均灌溉量 30 mm；
- 平均施氮量 40 kg/ha；
- DAP1-10 投入严格满足 I≤30、N≤40；
- 平均产量相对 036 基本不变；
- 不像 75K/100K 那样出现 N=0 导致 PFP_N 不可定义。

## 任务

只做 LC2019 一个代表年份的五情景日过程图，不训练新模型。

五情景包括：

1. null；
2. recorded farmer；
3. DSSAT auto；
4. official expert；
5. 037_03 LCA seed0 checkpoint 50K MaskablePPO candidate。

## 数据与模型

- RL 模型：`benchmark_results/037_03_lca_early_starter_cap_maskableppo_smoke_clean_retrain/models/LCA/LCA_half_split_stress_aware_maskableppo_seed0_ckpt50000.zip`
- 四基线 snapshot：复用 032_24/031_35/031_36 已有 LC2019 四情景快照。
- PPO candidate：冻结模型确定性评估一次，保存 DSSAT snapshot 后再绘图。

## 输出

- 五情景 daily CSV；
- 五情景 summary CSV；
- snapshot 完整性检查；
- PNG/SVG 五情景日过程图，样式沿用 027_05/032_24：
  - 降雨/气温；
  - 土壤水；
  - 水分胁迫；
  - 氮胁迫；
  - 灌溉事件；
  - 施氮事件；
  - 籽粒/生物量；
  - 统一 common reward。

## 停止线

- 不训练；
- 不改 reward；
- 不换 checkpoint；
- 如果 snapshot 或 daily 证据不完整，则记录失败，不手工拼图。
