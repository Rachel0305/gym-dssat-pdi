# 046_00 SYA lowIC DSSAT auto 施肥阈值复核

## 目的

导师要求 `dssat_auto` 情景必须能够出现施肥措施，并建议检查是否因为 DSSAT 自动施肥触发阈值过高导致没有施肥。本任务只做 SYA lowIC 条件下的原生 DSSAT auto-N 参数复核。

## 边界

- 不训练 PPO/DQN。
- 不改变 PPO 042_15 结果。
- 不改变 lowIC 输入源。
- 只在渲染后的临时 MZX 中修改 `@N NITROGEN` 自动施肥参数。
- 先跑代表年份 2014、2017、2022；若存在可触发施肥的参数组合，再扩展到全验证年份。

## 固定输入

- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 基线构建链：复用 `040_21_sya_lowIC_four_baseline_rebuild`
- 外部 step action：no-op，仅依赖 DSSAT 原生 `FERTI=A`

## 待测参数

保留 `IRRIG=A, FERTI=A`，只改变自动氮管理：

- `base_nmthr50_amt25_fe001`: NMTHR=50, NAMNT=25, NCODE=FE001
- `low_nmthr10_amt25_fe001`: NMTHR=10, NAMNT=25, NCODE=FE001
- `very_low_nmthr01_amt25_fe001`: NMTHR=1, NAMNT=25, NCODE=FE001
- `high_nmthr99_amt50_fe001`: NMTHR=99, NAMNT=50, NCODE=FE001
- `high_nmthr99_amt50_fe005`: NMTHR=99, NAMNT=50, NCODE=FE005

说明：虽然导师提出“调低阈值”，但 DSSAT `NMTHR` 的方向不应凭直觉假定；因此同时保留低阈值和高阈值组合，判断哪个方向真正触发。

## 判据

- 若任一组合在任一年份产生 `actual_nitrogen_kg_ha > 0` 或 `NI#M > 0`，则进入 A 分支：当前 auto-N 可通过参数触发，下一步扩展到全验证年份。
- 若全部组合仍为 0，则进入 B 分支：当前 DSSAT 原生 auto-N 在本链路下不触发，不能只靠调阈值解决；如导师坚持 auto 必须施肥，应改成“外部规则 auto-fertilizer baseline”，并明确不再称为 DSSAT native auto-N。

