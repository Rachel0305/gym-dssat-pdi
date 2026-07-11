# 020_12 YC/FQ 冻结 n-step DQN 跨站点验证记录

## 当前状态

本轮已经完成独立执行脚本、YC/FQ 静态审计、两站 500-step smoke，以及两站 seed0 50K 正式训练。所有任务均在指定 Docker 容器和虚拟环境内串行执行；没有并行长训练，也没有覆盖旧实验。

## 冻结方法

- 使用 `src/frozen_nstep_dqn_config_020_11.py` 作为唯一配置源。
- 奖励、9 动作、I120/N300 预算、单次上限、7 DAP 间隔、DAP1–120 窗口、DQN 超参数、n-step=5 和 checkpoint 规则均与 HLA 冻结框架一致。
- 环境 seed 固定为 0，模型 seed 由命令行单独指定。
- YC 与 FQ 各自在同一生物物理输入上重新运行 null baseline；禁止共享 HLA 的 null 产量。

## 站点输入

- YC2014：treatment 2，CNYC1401，YC99001200。
- FQ2016：FQ treatment 2 输入结构平移到 2016，CNFQ1601，FQ99001200。
- 为隔离框架变化，本轮不同时修改历史初始剖面的 ICDAT；该时间口径被保留为明确限制。

## 强制检查

脚本在启动 DSSAT/PDI 前验证：IC=1、WATER=Y、NITRO=Y、IRRIG=L、FERTI=L、9 动作、n-step=5、预算和窗口、环境 seed0，以及天气/土壤/品种文件存在性。任一项失败立即停止。

## 已完成结果

1. YC2014 smoke：本地 null=7825 kg/ha；500-step checkpoint 的 GWAD=9411 kg/ha、I=120 mm、N=300 kg/ha、reward=-34.022。该结果只用于验证链路。
2. FQ2016 smoke：本地 null=7066 kg/ha；500-step checkpoint 的 GWAD=7972 kg/ha、I=75 mm、N=300 kg/ha、reward=-669.478。两站 smoke 的输入、预算、动作传输和输出检查均通过。
3. YC2014 50K：按“确定性 total_reward 最大、并列取最早”选中35K，GWAD=8659 kg/ha、I=60 mm、N=0、reward=774.153。
4. FQ2016 50K：选中50K，GWAD=7985 kg/ha、I=120 mm、N=50 kg/ha、reward=548.564。
5. YC 保存模型按修正后的 pre-action `operation_dap` 重新审计，10/10 checkpoint 的产量、水氮和奖励精确复现，10/10 运行审计通过；FQ 原始10/10运行审计通过。

## 失败尝试与修复

- 第一次 YC smoke 的宿主端命令超时，目录保留为 `seed0_500steps_interrupted_client_timeout_20260710_1956`，没有删除。
- 第二次 YC smoke 完成模型评估后，在 JSON 写出阶段遇到 NumPy `int64` 不可序列化；修正为原生 `int`，失败目录保留为 `seed0_500steps`。
- `pandas.DataFrame.to_markdown()` 依赖容器中未安装的 `tabulate`；未安装新包，而是改成项目内手写 Markdown 表格。成功 smoke 保存在 `seed0_500steps_retry1`。
- 旧审计曾用终止后的 post-state DAP，把一次实际7天间隔显示为6天；现改用 budget wrapper 在动作执行前写入的 `operation_dap`，并通过保存模型重评完成验证，未重新训练。

## 文件

- prompt：`prompts/020_12_yc_fq_frozen_nstep_cross_site_validation.md`
- 执行脚本：`src/run_yc_fq_frozen_nstep_cross_site_020_12.py`
- 计划输出：`DSSAT_auto_validation/frozen_nstep_cross_site_020_12/`
- YC smoke 记录：`docs/2026-07-10_020_12_yc2014_frozen_nstep_cross_site_record.md`
- FQ smoke/正式记录：`docs/2026-07-10_020_12_fq2016_frozen_nstep_cross_site_record.md`
- 最终统一记录：`docs/2026-07-11_020_12_yc_fq_frozen_nstep_cross_site_final_record.md`
- 统一比较表：`DSSAT_auto_validation/frozen_nstep_cross_site_020_12/020_12_selected_cross_site_summary.csv`
- 与旧 n-step=1 对照：`DSSAT_auto_validation/frozen_nstep_cross_site_020_12/020_12_frozen_nstep5_vs_previous_nstep1.csv`
