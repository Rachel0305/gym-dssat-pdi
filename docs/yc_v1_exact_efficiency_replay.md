# YC V1 exact WP_ET/ETCP replay

已完成 `240` 行 Summary.OUT 精确回放，覆盖 Original/Augmented、seed 0/1/2、25K/50K/75K/100K 和 2014–2023。WP_ET 只来自 Summary.OUT/ETCP：有效 YPEM 时使用 `YPEM*0.1`，否则使用 `HWAM/(ETCP*10)`；ETCP 单位为 mm。PFP-N 只在 NICM>0 时使用 YPNAM。

输出：`results/yc_v1_exact_efficiency_replay.csv`。源代码：`src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py`。
