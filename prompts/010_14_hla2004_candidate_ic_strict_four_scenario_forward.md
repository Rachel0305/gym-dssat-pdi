# 010_14 HLA 2004 候选初始条件下严格四情景 forward 对照

## 背景

010_13 已经验证：在 HLA 2004 候选初始条件 `0.55 + 0.25N` 下，DSSAT 原生自动灌溉可以触发，但自动施肥没有触发。因此第 3 情景如果替换成 DSSAT 自动管理，必须明确称为：

```text
DSSAT auto irrigation + auto-N attempt
```

而不能直接称为完整的 DSSAT 自动水氮管理。

上一版诊断图把旧 008_19 的 null / recorded expert / PPO 与新 010_13 的 DSSAT auto 拼在一起，适合看图形设计，但不是严格同条件对照。现在需要做严格版：四个情景都使用同一个 HLA 2004 候选初始条件，只改变管理方式。

## 目标

在不训练 PPO 的前提下，做 4 次低成本 DSSAT/PDI forward simulation：

1. `candidate_null`
   - 候选 IC；
   - 无灌溉、无施肥；
   - 动作全 0。

2. `candidate_recorded_expert_replay`
   - 候选 IC；
   - 回放 008_19 中 HLA 2004 recorded expert 的日尺度动作；
   - 旧记录合计：灌溉 30 mm，施氮 165 kg/ha。

3. `candidate_dssat_auto`
   - 候选 IC；
   - DSSAT 原生 `IRRIG=A, FERTI=A`；
   - 动作全 0；
   - 记录自动灌溉和自动施肥是否真实触发。

4. `candidate_ppo_action_replay`
   - 候选 IC；
   - 回放 008_19 中 PPO soft-stress seed0 的日尺度动作；
   - 旧记录合计：灌溉 80 mm，施氮 150 kg/ha；
   - 注意：这不是新 IC 下重新训练的 PPO，只是旧 PPO 动作在新 IC 下的 forward replay。

## 控制变量

全部情景一致：

- 站点：HLA；
- 年份：2004；
- 天气：CNHL0401；
- 土壤：HL99001200；
- 品种：HY0006；
- 种植日期：04125；
- 初始条件：`SH2O=0.55*(SLLL-SDUL interval) + SNH4/SNO3=0.25N`；
- 模型：PDI/gym DSSAT 4.8.0；
- 不训练 PPO。

管理差异：

- null：动作全 0；
- expert replay：按旧 expert daily action table 回放；
- DSSAT auto：DSSAT 内部自动灌溉/自动施肥；
- PPO replay：按旧 PPO daily action table 回放。

## 技术处理

对于动作回放情景：

- 设置 treatment 中 `MI=1, MF=1`；
- 设置 management line 中 `IRRIG=R, FERTI=R`；
- 静态 irrigation/fertilizer 表保留 0 量占位；
- 实际施水施氮只来自 gym/PDI action replay。

这样做是为了避免非零动作被 `IRRIG=N/FERTI=N` 或 `MI=0/MF=0` 屏蔽。

## 输出

脚本：

```text
src/run_hla2004_candidate_ic_strict_four_scenario_010_14.py
```

输出目录：

```text
DSSAT_auto_validation/HLA_2004/candidate_ic055_n025_strict_four_scenario_010_14/
```

保存：

- 每个情景输入文件；
- 每个情景 PDI tmp snapshot；
- 四情景日值 CSV；
- 四情景汇总 CSV；
- 管理事件 CSV；
- 四情景过程图；
- 最终 grain yield / biomass 对比图；
- 实验记录 MD。

## 判读规则

1. 如果 expert/PPO replay 在候选 IC 下可以正常跑完，说明旧动作可作为低成本策略回放对照。
2. 如果 DSSAT auto 仍然只触发灌溉、不触发施肥，则第 3 情景不能叫完整自动水氮管理。
3. 如果候选 IC 下四情景产量排序与旧 IC 明显不同，要记录为“初始条件改变了研究情景”，不能直接沿用旧结论。
4. 本轮不训练，不下 PPO 稳定优越性结论；只回答“同一候选 IC 下四种管理方式 forward 表现如何”。
