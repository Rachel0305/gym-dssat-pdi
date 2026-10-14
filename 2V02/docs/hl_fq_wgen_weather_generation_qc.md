# HL / FQ WGEN fitting 与随机天气 QC 状态

**状态：** `PRECONDITION_FAILED`  
**门禁：** `STOP_BEFORE_WGEN = TRUE`  
**对应源审计：** [hl_fq_wgen_weather_source_audit.md](hl_fq_wgen_weather_source_audit.md)  
**Machine-readable：** [weather_source_audit_final.json](../results/hl_fq_random_weather_8seed/source_audit/weather_source_audit_final.json)

## Fitting weather 输入门禁

| 站点 | 拟合期 | 期望日数 | 可由通过筛选站点补的 RAIN 空白 | 尚无来源的 RAIN 日期 | 结果 |
|---|---|---:|---:|---:|---|
| HLA | 2004-01-01–2013-12-31 | 3,653 | 2,580 | 111（全部在 2013） | STOP |
| FQA | 2005-01-01–2013-12-31 | 3,287 | 2,735 | 157（全部在 2013） | STOP |

TMAX、TMIN、SRAD 的 NASA POWER 偏差校正筛选已通过：HLA 分别有 32、31、29 个候选填补/异常替换行；FQA 分别有 83、83、94 行。这些行尚未升级为正式 fitting weather。RAIN 候选保留本地非空观测，仅对本地空白日期使用通过当前筛选的邻近 CMA/GHCN 候选值；多站覆盖后仍未完整。NASA POWER RAIN 一致率低于 80% 门槛，不采用。

因为 RAIN 存在未解决日期，本轮没有形成完整的 5 列 fitting weather CSV。最终序列级日期连续、变量缺失、`TMAX >= TMIN`、RAIN/SRAD 非负等联合 QC **未运行/不可判定**，不能标记为 PASS。

## WGEN 与 runtime 阶段

| Gate / artifact | HLA | FQA | 说明 |
|---|---|---|---|
| Fitting weather | 未生成 | 未生成 | 雨量仍缺来源；候选填补表 `promoted_to_fitting_weather=false`。 |
| Site `.CLI` | NO | NO | 缺少可信完整拟合输入，未调用 CLI/WGEN 参数生成。 |
| Training WGEN pool | 0/80 | 0/80 | 没有生成天气 realization。 |
| Held-out WGEN pool | 0/20 | 0/20 | 没有生成天气 realization。 |
| `CLIMATE_COVERAGE` | NOT_RUN | NOT_RUN | 没有 random weather 输出可评估；不误报为 FAIL 或 PARTIAL。 |
| DSSAT runtime smoke | NOT_RUN | NOT_RUN | 未将随机天气注入 DSSAT。 |
| PPO seeds | 0/8 | 0/8 | 0/100K training steps；未启动训练。 |

## 已核实且可复用的 YC contract

本地 YC 参考实现区分 PPO seed 与 weather seed。`weather_seed=1001–1080` 为共同训练池，`1081–1100` 为 held-out；PPO seed 变化不重新定义 weather pool。运行时使用任务级 FileX 副本 `METHODS.WTHER=W`，每 episode 前将 weather seed 注入 PDI 的 `rseed1_`，再对实际读取的每日 `RAIN/SRAD/TMAX/TMIN` 做 hash。修正后有 3 个 WGEN PPO seeds 完成 100,000 请求步数（实际 100,080）；这支持方法可复用，但不是 HL/FQ 的 8-seed 结果。

YC 既有 004_02 气候 QA 只覆盖一个共同 106 天 crop-season 窗口，全年 WGEN QA 被标记为 deferred。因此即使未来恢复 HL/FQ 来源审计，也须按任务要求补做全年/季节统计、逐种子可复现性和 runtime 逐日一致性检查，不能将该旧窗口报告当作全年门禁通过。

## 继续条件

先取得 HLA/FQA 2013 年未覆盖日期的可信降水记录，或取得可追溯的数据字典/站点记录，证明这些空白日期确实表示 `0 mm`。数据须存放在本项目目录。之后从来源与 fitting-weather QC 重跑；只有所有日期完整且物理检查通过，才能按附件执行 `.CLI`、WGEN pool、runtime smoke 和 8-seed PPO。

## 2026-10-04 CERN D222 降水复审补充（009）

已完成 CERN D222 原始 Excel 的只读复审，详见 [hl_fq_cern_d222_rain_reaudit.md](hl_fq_cern_d222_rain_reaudit.md) 与机器可读 [final_gate.json](../results/hl_fq_random_weather_8seed/source_audit/cern_d222_reaudit/final_gate.json)。

- HLA 2004–2013、FQA 2005–2013 日期链完整、无重复；半日降水与 `20-20合计` 内部闭合。
- D222 与项目旧降水表在 0 日 shift 下数值完全一致，-1/+1 日明显较差。
- 全空白日不能统一解释为 0 mm，`BLANK_SEMANTICS = MIXED`。
- `HLA_2004_RAIN = INCOMPLETE_RECORD`：2004-01-01 至 2004-08-31 连续 244 天三个降水字段全空白。
- `FQA_EXTREME_RAIN = SUSPICIOUS_BUT_UNRESOLVED`：事件存在有交叉支持，但 2011–2013 极端量级缺少同站独立证据。

因此本次补充仍为：

`WEATHER_SOURCE_AUDIT = BLOCKED`  
`READY_TO_RESUME_008 = FALSE`  
`STOP_BEFORE_WGEN = TRUE`

本次未生成完整 fitting weather、`.CLI` 或任何 WGEN weather realization，也未运行 DSSAT 或 PPO。原 008 状态没有改变，因此未改写其既有 source-audit 结论。
