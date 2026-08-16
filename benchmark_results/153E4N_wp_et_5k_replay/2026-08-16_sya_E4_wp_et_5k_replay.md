# E4 5K seed0 WP_ET replay

- 范围：SYA originIC，验证年 2014–2023，冻结 5000-step checkpoint。
- 本轮不训练；串行回放 10 个 DSSAT 年份，并从 Summary.OUT 读取 ETCP。
- 天气分支使用 E4 六特征有效、六特征置零的 compact wrapper。

## 汇总

- 平均产量：10047.5059 kg/ha
- 平均 ETCP：486.3200 mm
- 平均 WP_ET：2.0670 kg/m³
- 加权 WP_ET：2.0660 kg/m³
- 平均 PFP-N：41.8500 kg/kg
- 加权 PFP-N：41.8646 kg/kg

## 文件

- `153E4_wp_et_replay_summary.csv`：E4 汇总
- `153E4_wp_et_replay_by_year.csv`：逐年结果
- `153E4_wp_et_replay_reproducibility_check.csv`：与原 daily CSV 的闭合核验
