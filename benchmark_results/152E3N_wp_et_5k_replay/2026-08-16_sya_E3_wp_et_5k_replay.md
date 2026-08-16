# E3 5K seed0 WP_ET replay

- 范围：SYA originIC，验证年 2014–2023，冻结 5000-step checkpoint。
- 本轮不训练；串行回放 10 个 DSSAT 年份，并从 Summary.OUT 读取 ETCP。
- 天气分支使用 E3 零天气 wrapper；奖励、动作网格和 checkpoint 不变。

## 汇总

- 平均产量：9952.5813 kg/ha
- 平均 ETCP：495.7200 mm
- 平均 WP_ET：2.0090 kg/m³
- 加权 WP_ET：2.0077 kg/m³
- 平均 PFP-N：41.4600 kg/kg
- 加权 PFP-N：41.4691 kg/kg

## 文件

- `152E3_wp_et_replay_summary.csv`：E3 汇总
- `152E3_wp_et_replay_by_year.csv`：逐年结果
- `152E3_wp_et_replay_reproducibility_check.csv`：与原 daily CSV 的闭合核验
