# 015_20 HLA2010 DQN 跨年迁移验证总结

## 图的核心结论

HLA2010-trained DQN 在 HLA 所有通过管理响应筛选的 4 个独立候选年份上均达到或贴近 DSSAT auto 产量平台，同时减少灌溉且不施氮，说明当前策略具有同站点跨年份泛化迹象。

## 证据链

- 训练年：HLA2010。
- 迁移验证年：HLA2007、HLA2015、HLA2016、HLA2022。
- 这些年份是 HLA 筛选表中通过管理响应标准的全部独立候选年份。
- 迁移验证不重新训练，只加载 HLA2010 checkpoint。

## 输出文件

- 汇总总表：`DSSAT_auto_validation\HLA_2004\hla2010_dqn_cross_year_transfer_summary_015_20\hla2010_dqn_cross_year_transfer_all_summary.csv`
- best-transfer 表：`DSSAT_auto_validation\HLA_2004\hla2010_dqn_cross_year_transfer_summary_015_20\hla2010_dqn_cross_year_transfer_best_by_year.csv`
- 总图：`DSSAT_auto_validation\HLA_2004\hla2010_dqn_cross_year_transfer_summary_015_20\figures\hla2010_dqn_cross_year_transfer_summary.png`

## 全部情景汇总

| requested_year | scenario_label | final_gwad | irrigation_total | fertilizer_total | yield_diff_vs_auto | irrigation_saving_vs_auto | yield_ratio_vs_auto_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2007 | DSSAT auto | 7973.00 | 93.50 | 0.00 | 0.00 | 0.00 | 100.00 |
| 2007 | Null | 7830.00 | 0.00 | 0.00 | -143.00 | 93.50 | 98.21 |
| 2007 | Recorded | 7986.00 | 30.00 | 165.00 | 13.00 | 63.50 | 100.16 |
| 2007 | DQN transfer s0 | 7987.00 | 120.00 | 0.00 | 14.00 | -26.50 | 100.18 |
| 2007 | DQN transfer s1 | 7987.00 | 45.00 | 0.00 | 14.00 | 48.50 | 100.18 |
| 2015 | DSSAT auto | 7648.00 | 141.50 | 0.00 | 0.00 | 0.00 | 100.00 |
| 2015 | Null | 6486.00 | 0.00 | 0.00 | -1162.00 | 141.50 | 84.81 |
| 2015 | Recorded | 7296.00 | 30.00 | 165.00 | -352.00 | 111.50 | 95.40 |
| 2015 | DQN transfer s0 | 7653.00 | 90.00 | 0.00 | 5.00 | 51.50 | 100.07 |
| 2015 | DQN transfer s1 | 7653.00 | 45.00 | 0.00 | 5.00 | 96.50 | 100.07 |
| 2016 | DSSAT auto | 7538.00 | 140.80 | 0.00 | 0.00 | 0.00 | 100.00 |
| 2016 | Null | 7224.00 | 0.00 | 0.00 | -314.00 | 140.80 | 95.83 |
| 2016 | DQN transfer s0 | 7538.00 | 120.00 | 0.00 | 0.00 | 20.80 | 100.00 |
| 2016 | DQN transfer s1 | 7538.00 | 60.00 | 0.00 | 0.00 | 80.80 | 100.00 |
| 2022 | DSSAT auto | 7935.00 | 143.60 | 0.00 | 0.00 | 0.00 | 100.00 |
| 2022 | Null | 7787.00 | 0.00 | 0.00 | -148.00 | 143.60 | 98.13 |
| 2022 | DQN transfer s0 | 7935.00 | 120.00 | 0.00 | 0.00 | 23.60 | 100.00 |
| 2022 | DQN transfer s1 | 7935.00 | 60.00 | 0.00 | 0.00 | 83.60 | 100.00 |

## 每年最佳迁移策略

| requested_year | scenario_label | train_seed | train_checkpoint_step | final_gwad | irrigation_total | fertilizer_total | yield_diff_vs_auto | irrigation_saving_vs_auto | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007 | DQN transfer s1 | 1.00 | 25000.00 | 7987.00 | 45.00 | 0.00 | 14.00 | 48.50 | 111.73 |
| 2015 | DQN transfer s1 | 1.00 | 25000.00 | 7653.00 | 45.00 | 0.00 | 5.00 | 96.50 | 1122.19 |
| 2016 | DQN transfer s1 | 1.00 | 25000.00 | 7538.00 | 60.00 | 0.00 | 0.00 | 80.80 | 254.47 |
| 2022 | DQN transfer s1 | 1.00 | 25000.00 | 7935.00 | 60.00 | 0.00 | 0.00 | 83.60 | 87.68 |

## 谨慎表述

- 该结果支持 HLA 同站点跨年份迁移，不等于已经证明跨站点泛化。
- 2016/2022 没有 recorded expert，因此与 recorded 的比较只限于 2007/2015。
- DQN 的主要优势不是大幅提高产量，而是在达到 auto 产量平台附近时减少灌溉和施氮。
