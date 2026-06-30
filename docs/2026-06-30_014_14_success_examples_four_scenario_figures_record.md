# 014_14 成功示例四情景图表整理记录

日期：2026-06-30

## 目的

把当前已经形成“较有希望/可解释”的示例整理成统一图表和日值表，便于后续给导师汇报：

- 降雨
- 土壤水分胁迫指数
- 土壤氮胁迫指数
- 灌溉与施肥管理事件
- 籽粒产量与地上部生物量

本次只整理既有结果，不重新训练，不重新运行 DSSAT，以节省算力并避免引入新的变量。

## 输出位置

主输出目录：

```text
DSSAT_auto_validation/success_examples_four_scenario_014_14/
```

关键文件：

```text
DSSAT_auto_validation/success_examples_four_scenario_014_14/014_14_all_examples_daily.csv
DSSAT_auto_validation/success_examples_four_scenario_014_14/014_14_all_examples_summary.csv
DSSAT_auto_validation/success_examples_four_scenario_014_14/014_14_manifest.csv
DSSAT_auto_validation/success_examples_four_scenario_014_14/daily_tables/
DSSAT_auto_validation/success_examples_four_scenario_014_14/figures/
```

绘图脚本：

```text
src/plot_success_examples_four_scenario_014_14.py
```

新版 Nature-style 图同时导出 PNG 和 SVG。带累积奖励子图的新文件命名为：

```text
figures/*_scenario_process_nature_reward.png
figures/*_scenario_process_nature_reward.svg
```

## 示例与数据来源

| 站点年份 | 当前整理情景 | 数据来源说明 |
|---|---|---|
| HLA 2010 | null、专家策略平移、DSSAT auto、DQN seed1 | 前三类来自既有 HLA 四情景表；DQN 使用 action9 baseline-relative reward seed1 5K 结果 |
| HLA 2015 | null、专家策略平移、DSSAT auto、DQN seed1 | 前三类来自既有 HLA 四情景表；DQN 使用 baseline-relative reward seed1 5K 结果 |
| YC 2014 | null、recorded、DSSAT auto、DQN seed1 | null 为本轮用 `src/run_yc2014_action_window_comparison_013_04.py --only null` 补跑；其余情景来自既有 YC2014 对照表和 DQN seed1 结果 |
| FQ 2016 | null、recorded shifted、DSSAT auto、DQN seed1 window | null/recorded/DSSAT auto 来自全年筛选表；DQN 使用 linked agronomic-window seed1 5K 结果 |

## 注意事项

1. HLA DQN 日值表中的 `used_irrigation` 和 `used_nitrogen` 是累计用量，不是当天事件量。  
   本次绘图和汇总改用 `safe_amir` / `safe_anfer` 作为当天灌溉、施肥量，否则 HLA DQN 总水氮会被错误放大。

2. FQ 全年筛选表中无管理情景原先为空值。  
   本次统一规范为 `null_zero`，避免 CSV 被 pandas 默认读成缺失值。

3. FQ 四情景日值表缺少 rainfall 列。  
   本次从 `CNFQ1601.WTH` 按种植日 DOY=162 映射到 DAP 后补入降雨。

4. 图中胁迫指数按当前 gym/PDI 输出口径展示：数值越大表示胁迫越强。

5. 新增的累积奖励子图是统一的事后评价指标，不等同于每个 DQN 模型训练时的内部 reward。  
   计算公式为：

```text
post-hoc cumulative reward proxy
= current grain weight - 1 × cumulative irrigation - 5 × cumulative fertilizer
```

   这样做的目的，是让 null、专家记录、DSSAT auto 和 DQN 可以放在同一把尺子下比较“产量-资源投入”的折中。  
   因为该指标显式惩罚水氮投入，所以它可能与“单纯追求最高产量”的排序不同。

## 汇总结果

| site | year | scenario_label | irrigation (mm) | fertilizer (kg/ha) | grain (kg/ha) | biomass (kg/ha) | max water stress | max nitrogen stress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HLA | 2010 | DQN seed1 | 120.0 | 300.0 | 7853.7 | 20874.7 | 0.416 | 0.0158 |
| HLA | 2010 | DSSAT auto | 190.4 | 0.0 | 7854.0 | 20874.0 | 0.416 | 0.0350 |
| HLA | 2010 | Expert 2007 shifted | 30.0 | 165.0 | 7679.0 | 20665.0 | 0.811 | 0.0160 |
| HLA | 2010 | Null | 0.0 | 0.0 | 6956.0 | 19344.0 | 0.919 | 0.1570 |
| HLA | 2015 | DQN seed1 | 120.0 | 300.0 | 7651.9 | 19058.6 | 0.000 | 0.0145 |
| HLA | 2015 | DSSAT auto | 141.5 | 0.0 | 7648.0 | 19021.0 | 0.000 | 0.0470 |
| HLA | 2015 | Expert 2007 shifted | 30.0 | 165.0 | 7296.0 | 18639.0 | 0.779 | 0.0150 |
| HLA | 2015 | Null | 0.0 | 0.0 | 6486.0 | 17168.0 | 1.000 | 0.0930 |
| YC | 2014 | DQN seed1 free | 120.0 | 300.0 | 9417.4 | 20497.2 | 0.000 | 0.0129 |
| YC | 2014 | DSSAT auto | 259.5 | 0.0 | 8713.0 | 18945.0 | 0.000 | 0.4360 |
| YC | 2014 | Null | 0.0 | 0.0 | 7825.4 | 17996.5 | 0.922 | 0.381 |
| YC | 2014 | Recorded expert | 360.0 | 1122.0 | 9418.0 | 20514.0 | 0.000 | 0.0130 |
| FQ | 2016 | DQN seed1 window | 30.0 | 300.0 | 8012.4 | 14093.1 | 0.000 | 0.0122 |
| FQ | 2016 | DSSAT auto | 59.9 | 0.0 | 8012.4 | 14094.8 | 0.000 | 0.0122 |
| FQ | 2016 | Null | 0.0 | 0.0 | 7066.1 | 13148.5 | 0.657 | 0.0122 |
| FQ | 2016 | Recorded expert shifted | 75.0 | 144.0 | 7932.6 | 14008.2 | 0.373 | 0.0122 |

## 当前结论

1. HLA 2010 和 HLA 2015：DQN seed1 能够明显优于 null，并接近或持平 DSSAT auto；与专家策略相比也有增产。  
   但它使用的施氮量为 300 kg/ha，需在论文叙事中明确这是当前约束下的 DQN 行为，不宜说成“节氮最优”。

2. YC 2014：DQN seed1 以 I120/N300 达到与 recorded expert 几乎相同的产量，且显著高于 null；recorded expert 的水氮投入远高于 DQN。  
   这是当前最有利于“有限投入下逼近专家产量”的示例之一。本轮已经补齐 null 日值表，可作为完整四情景示例。

3. FQ 2016：DQN seed1 window 以 I30/N300 达到与 DSSAT auto 近似的产量，并高于 recorded shifted 和 null。  
   但 FQ 的氮胁迫指数几乎全程很低，说明这个站点年份更多体现水分管理/总量管理差异，氮响应解释需要谨慎。

4. DSSAT auto 在 HLA 和 FQ 中仍然表现出“灌溉触发有效、自动施肥未触发或施肥不足”的特征。  
   因此 DQN 与 DSSAT auto 的对比可以作为工程对照，但不能简单解释为“DSSAT 最强自动管理基线”。

5. 累积奖励 proxy 的排序提示：如果导师把目标定义为“收益/资源效率”，结论会不同于“最高产量”。  
   例如 YC2014 中 recorded expert 产量最高，但水氮投入过大，因此 reward proxy 明显低于 DQN 和 DSSAT auto。  
   这说明下一步必须和导师确认主目标到底是“产量最大化”还是“产量-资源投入折中最优”。

## 后续建议

1. 对 HLA2010、HLA2015、YC2014、FQ2016 的 DQN seed0/seed1 结果统一生成稳定性表。
2. 如果导师认可当前方向，再扩大训练步数或增加站点年份；如果导师更关心水氮节约，则需要把奖励和约束改成显式水氮利用效率目标。
