# 046_04 SYA 外部规则型 auto-N 基线

## 目的

046_00 显示 DSSAT 原生 `FERTI=A` 在多个阈值设置下均未施氮。本任务不再把原生 auto-N 的零施氮结果包装为“自动水氮管理”，而建立一个独立、可解释的比较情景：

- 灌溉：DSSAT 原生自动灌溉；
- 施氮：gym-DSSAT 每日读取 `NSTRES`；
- 当动作前 `NSTRES >= 0.5` 时施用 25 kg/ha；
- 两次施氮至少间隔 7 天；
- 季节施氮最多 250 kg/ha；DAP90 后禁止施氮。

## 重要边界

这不是 DSSAT native auto-N，也不是 RL；名称固定为 `dssat_auto_irrigation_external_n_rule`。若全年都没有达到 NSTRES 0.5，结果仍可能是 0 施氮；这应如实记录，而不是临时降低阈值来确保出肥。

## 输入

从 `configs/046_02_sya_originIC_binary_timing_ppo.json` 读取 `input_profile` 和固定规则参数，确保其与 originIC 基线/PPO 对照同源。

