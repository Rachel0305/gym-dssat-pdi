# 017_07 LC/SY 跨站点 DQN 策略迁移 probe 记录

## 运行设置

- smoke 模式：True
- 本轮不训练新模型，只加载 HLA2010 / YC2014 已有 DQN checkpoint。
- 统一动作/约束：9-action，I≤120 mm，N≤300 kg/ha，单次 I≤30，单次 N≤100，最小间隔 7 天。
- 评估奖励：`max(0, GWAD_final - local_null_GWAD) - 1*I - 5*N`。

## 输出

- 汇总表：`DSSAT_auto_validation/lc_sy_hl_yc_dqn_cross_site_transfer_probe_017_07/017_07_lc_sy_cross_site_summary.csv`
- 日值表：`DSSAT_auto_validation/lc_sy_hl_yc_dqn_cross_site_transfer_probe_017_07/017_07_lc_sy_cross_site_daily.csv`
- 事件表：`DSSAT_auto_validation/lc_sy_hl_yc_dqn_cross_site_transfer_probe_017_07/017_07_lc_sy_cross_site_events.csv`
- 状态表：`DSSAT_auto_validation/lc_sy_hl_yc_dqn_cross_site_transfer_probe_017_07/017_07_lc_sy_cross_site_status.csv`
- 总图：`DSSAT_auto_validation/lc_sy_hl_yc_dqn_cross_site_transfer_probe_017_07/figures/017_07_lc_sy_cross_site_summary.png`

## 本地基准

| site | requested_year | scenario | final_gwad | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SY | 2012 | dssat_auto | 7016.00 | 33.00 | 0.00 | 0.00 | 0.60 |
| SY | 2012 | null | 7002.00 | 0.00 | 0.00 | 0.00 | 0.60 |
| SY | 2012 | recorded | 10166.00 | 0.00 | 3439.00 | 0.00 | 0.02 |

## 跨站点 DQN 迁移结果

无。

## 状态/失败记录

| site | year | scenario | status | error |
| --- | --- | --- | --- | --- |
| SY | 2012 | null | ok |  |
| SY | 2012 | recorded | ok |  |
| SY | 2012 | dssat_auto | ok |  |
| SY | 2012 | transfer_HLA2010_seed0_ckpt35000_hla_seed0_high_reward | failed | ValueError('Observation spaces do not match: Box(0.0, inf, (24,), float32) != Box(0.0, inf, (25,), float32)') |
| SY | 2012 | transfer_HLA2010_seed1_ckpt25000_hla_seed1_high_reward | failed | ValueError('Observation spaces do not match: Box(0.0, inf, (24,), float32) != Box(0.0, inf, (25,), float32)') |
| SY | 2012 | transfer_YC2014_seed0_ckpt5000_yc_seed0_high_yield | failed | ValueError('Observation spaces do not match: Box(0.0, inf, (22,), float32) != Box(0.0, inf, (25,), float32)') |
| SY | 2012 | transfer_YC2014_seed0_ckpt25000_yc_seed0_high_reward | failed | ValueError('Observation spaces do not match: Box(0.0, inf, (22,), float32) != Box(0.0, inf, (25,), float32)') |
| SY | 2012 | transfer_YC2014_seed1_ckpt30000_yc_seed1_low_water_high_yield | failed | ValueError('Observation spaces do not match: Box(0.0, inf, (22,), float32) != Box(0.0, inf, (25,), float32)') |
| SY | 2012 | transfer_YC2014_seed1_ckpt50000_yc_seed1_high_reward | failed | ValueError('Observation spaces do not match: Box(0.0, inf, (22,), float32) != Box(0.0, inf, (25,), float32)') |

## 初步判定

- 如果 direct transfer 表现不好，不代表 DQN 框架在 LC/SY 不可用，只代表已经学到的 HLA/YC 策略不能直接照搬。
- 下一步应根据 LC/SY 本地基准是否存在优化空间，决定是否进入本地重新训练，而不是继续盲目跨站点迁移。

## LC ???????

- LC2008 recorded ?? MZX ?????? `CNLC0801.MZX`?????? `CNLC0801.WTH`?`SOIL.SOL`?`MZCER048.CUL`?`CNLC.CLI/PRM/wdb` ?? `gym.make` ? 45 ?????
- ???? DQN ??????????????????????????? LC ???? gym/PDI ?????????????
- ?????? LC??????? SY ????????? LC ? MZX ??????????? treatment ????????????????
