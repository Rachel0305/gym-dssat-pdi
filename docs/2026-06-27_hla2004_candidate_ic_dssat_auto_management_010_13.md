# 2026-06-27 HLA 2004 候选 IC 下 DSSAT 原生自动管理检查

## 设置

- 年份/站点：HLA 2004。
- 初始条件：候选 `SH2O=0.55 可利用水分 + SNH4/SNO3=0.25N`。
- 管理方式：`IRRIG=A, FERTI=A`。
- 自动灌溉参数沿用输入文件：`IMDEP=30, ITHRL=50, ITHRU=100, IROFF=GS000, IMETH=IR001, IRAMT=10, IREFF=1`。
- 自动施肥参数沿用输入文件：`NMDEP=30, NMTHR=50, NAMNT=25, NCODE=FE001, NAOFF=GS000`。
- 不训练 PPO；仅 DSSAT/PDI forward simulation。

## 结果摘要

- 最终 DAP：170.0
- 最终 GWAD：2501.0 kg/ha
- 最终 CWAD：8287.0 kg/ha
- Summary HWAM：2501.0 kg/ha
- Summary CWAM：8287.0 kg/ha
- Summary IR#M/IRCM：7.0 次 / 329.0 mm
- Summary NI#M/NICM：0.0 次 / 0.0 kg/ha
- MgmtEvent 自动灌溉事件数：7，合计：329.00 mm
- MgmtEvent 自动施肥事件数：0，合计：0.00

## 初步判读

- 自动灌溉发生了，但自动施肥没有触发；该情景不能直接称为完整水氮自动管理，只能称为 DSSAT 自动灌溉 + 未触发自动施肥。

## 文件

- `hla2004_candidate_ic_dssat_auto_daily_values.csv`
- `hla2004_candidate_ic_dssat_auto_management_events.csv`
- `hla2004_candidate_ic_dssat_auto_summary.csv`
- `hla2004_candidate_ic_dssat_auto_management_daily.png`
