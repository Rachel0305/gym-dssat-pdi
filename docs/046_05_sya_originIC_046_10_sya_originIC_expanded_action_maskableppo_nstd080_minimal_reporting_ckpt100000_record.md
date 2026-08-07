# 046_05 SYA originIC 五情景统一指标汇总

- PPO checkpoint: `100000`。
- auto 行替换为 046_04：DSSAT 原生自动灌溉 + 外部 NSTRES 触发施氮；不再使用施氮为 0 的 native-auto 行。
- PFP_N 按实际施氮量计算；实际 N=0 时为未定义（N/A），不写成 0。
- PPO 超过四基线最高值的年份数：{'PFP_N_kg_kg': 10, 'WP_ET_kg_m3': 1, 'grain_yield_kg_ha': 5}。
- 产量为籽粒产量 GRNWT/HWAM；生物量不参与本指标比较。

## external-auto PFP_N 审计

|   year |   actual_nitrogen_kg_ha |   PFP_N_kg_kg |
|-------:|------------------------:|--------------:|
|   2014 |                       0 |           nan |
|   2015 |                       0 |           nan |
|   2016 |                       0 |           nan |
|   2017 |                       0 |           nan |
|   2018 |                       0 |           nan |
|   2019 |                       0 |           nan |
|   2020 |                       0 |           nan |
|   2021 |                       0 |           nan |
|   2022 |                       0 |           nan |
|   2023 |                       0 |           nan |
