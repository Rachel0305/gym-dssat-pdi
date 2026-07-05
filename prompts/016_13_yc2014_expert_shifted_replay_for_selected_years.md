# 016_13 YC2014 expert 策略平移到代表年份

## 目标

修正 `016_11` 里 expert 情景不是真正“2014 expert 平移”的问题。

只对以下 4 个年份补跑真实 expert schedule replay：

- 2006
- 2009
- 2015
- 2018

然后把这些年份的四情景图重画为：

- null
- 2014 expert 平移
- DSSAT auto
- DQN transfer

## Expert schedule 来源

使用 `015_06` 的 YC2014 recorded 真实管理事件：

- DAP 0: Fertilizer 96
- DAP 43: Fertilizer 278
- DAP 43: Irrigation 120

## 约束

- 不新增训练
- 不重跑其他年份
- 只补跑 4 个年份的 expert replay
- 结果写入新目录，避免覆盖旧结果

## 输出

- `DSSAT_auto_validation/yc2014_cross_year_transfer_success_plots_016_11_expertfix/`
- `docs/2026-07-05_016_13_yc2014_expert_shifted_replay_for_selected_years.md`
