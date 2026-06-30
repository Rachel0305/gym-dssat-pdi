# 010_13 HLA 2004 候选初始条件下的 DSSAT 自动管理替代规则管理检查

## 背景

上一轮已经得到一个候选统一初始条件：

```text
SH2O = SLLL + 0.55 * (SDUL - SLLL)
SNH4/SNO3 = 0.25 * HLA 2004 record-based profile
```

该候选在 HLA 2004-2023 null 多年检查中，使正常年份 null 产量从原 IC=1 的 5-8 t/ha 降到约 2-3 t/ha，同时保留 2004、2012 这类异常年份的低产特征。因此它可以进入 HLA 2004 的小规模情景对照。

用户要求：把原计划第 3 情景“规则管理”换成 DSSAT 原生自动管理，且灌溉和施肥都交给 DSSAT 自动触发，其他条件保持不变。

## 本轮目标

只做一次低成本 forward simulation，不训练 PPO：

1. 使用 HLA 2004。
2. 使用候选初始条件 `0.55 + 0.25N`。
3. 基于候选 IC null 输入文件，仅把 management line 改成：

```text
IRRIG=A
FERTI=A
```

4. 保留已有 automatic management 参数块：

```text
IR: IMDEP=30, ITHRL=50, ITHRU=100, IROFF=GS000, IMETH=IR001, IRAMT=10, IREFF=1
NI: NMDEP=30, NMTHR=50, NAMNT=25, NCODE=FE001, NAOFF=GS000
```

5. 检查 DSSAT 是否实际触发：
   - 自动灌溉事件；
   - 自动施肥事件；
   - `Summary.OUT` 中的 `IR#M/IRCM` 与 `NI#M/NICM`；
   - `MgmtEvent.OUT` 中的事件日期、DAP、数量。

## 控制变量

保持不变：

- 站点：HLA；
- 年份：2004；
- 天气：CNHL0401；
- 土壤：HL99001200；
- 品种：HY0006；
- 种植日期：04125；
- 初始条件：候选 `0.55 + 0.25N`；
- 模型：PDI/gym DSSAT 4.8.0；
- 不训练 PPO，不使用 PPO 动作。

唯一改变：

```text
第 3 情景管理方式：规则管理 -> DSSAT 原生自动灌溉 + 自动施肥
```

## 输出

脚本：

```text
src/run_hla2004_candidate_ic_dssat_auto_management_010_13.py
```

结果目录：

```text
DSSAT_auto_validation/HLA_2004/candidate_ic055_n025_dssat_auto_management_010_13/
```

需要保存：

- 修改后的输入文件；
- PDI 临时目录快照；
- `PlantGro.OUT`、`MgmtEvent.OUT`、`Summary.OUT`；
- DSSAT daily values CSV；
- management events CSV；
- summary CSV；
- 降雨/灌溉/施肥/水分胁迫/氮胁迫/GWAD 图；
- 实验记录 MD。

## 判读规则

1. 如果自动灌溉触发、自动施肥也触发，则该情景可以暂时作为“DSSAT 原生自动管理”候选第 3 情景。
2. 如果自动灌溉触发但自动施肥不触发，则该情景本质上是“DSSAT 自动灌溉 + 未触发自动施肥”，不能直接当作完整水氮自动管理基线。
3. 如果二者都不触发，则说明该设置下 DSSAT 自动管理不适合作为第 3 情景，需要回退到规则管理或另设诊断。
4. 无论结果如何，都记录为证据，不直接扩大到多年或 PPO 训练。
