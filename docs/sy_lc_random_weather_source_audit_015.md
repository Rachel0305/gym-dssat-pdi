# SYA / LCA 随机天气增强前置来源审计 015（provenance 修订版）

修订日期：2026-10-06。原版报告与被修订的CSV/JSON保存在 `backups/sy_lc_weather_015_before_provenance_revision/`。本次只修订历史来源解释与门槛；WTH逐日QC、配置、模拟和训练均未改动。

## 修订结论

|站点|WTH结构QC|已解除的阻塞依据|剩余来源阻塞|gate|
|---|---|---|---|---|
|SYA|19/19年通过|RAIN空白=0 mm|训练期SRAD 160、TMAX 151、TMIN 151个原始空白，与cleaned及最终WTH值相同，仍需确认填补或补测来源|BLOCKED_BY_DATA_PROVENANCE|
|LCA|19/19年通过|RAIN空白=0 mm；1113处cleaned/WTH差异有历史下载补充背景，不再作为异常|训练期SRAD 13、TMAX 8、TMIN 1个原始空白，与cleaned及最终WTH值相同，仍需确认是否为原有填补值|BLOCKED_BY_DATA_PROVENANCE|

两站仍有真正阻止进入WGEN的非降水来源缺口。本结论只针对拟合来源可信度；WTH四变量、日期连续性和数值QC已经通过。不存在需要对RAIN插值或另寻降水来源的事项。

## Baseline 与配置继承

本审计选择与 YC 055_00 同一继承链的冻结历史 baseline：SYA 046_10 originIC、LCA 053_00 lowIC；不采用其他 forecast、reward 或天气增强分支。当前 JSON 与 completed_formal manifest 一致，训练/验证年份与 manifest 中 engine split 一致。证据见 config_evidence.json。

两站动作均为 I=[0,15,30,45] mm × N=[0,40,80,120] kg/ha，共16个离散动作。有效灌溉季节上限240 mm，氮250 kg/ha；I/N事件间隔7天，灌溉DAP1–120、施氮DAP1–90。040_26 灌溉阶段上限DAP≤30:75、DAP≤60:150；040_36 DAP≤90:195。

reward 继承032_00 yield-minus-water/nitrogen-cost-plus-stress-relief（yield_coef0.158、N cost1.58、water cost1.1、water relief10、N relief5、scale0.001），并继承040_28的50×max(0,SWFAC−0.05)×scale过程惩罚。仅照录既有代码，不在本任务修订其物理解释。

重要：运行目录复制的基础 YAML 中 season_irrigation_soft_limit=160 是基础值；runner 的 load_i240_config 在内存中覆盖为240，且 wrapper 增加阶段掩码，不能仅据复制 YAML 推断有效配置。PPO采用现有100K/seed0、[64,64]网络及原始无预报观测；具体参数见 baseline_config_summary.csv。

配置与代码证据：configs/046_10_sya_originIC_expanded_action_maskableppo.json；configs/053_00_lca_lowIC_expanded_action_maskableppo.json；src/run_sya_ppo_configured_046_02.py；src/053_lca_lowIC_site_transfer/run_053_00_lca_lowIC_expanded_action_maskableppo.py；src/run_sya_lowIC_binary_timing_maskableppo_042_10.py；src/run_sya_lowIC_ppo_i240_staged_reserve_040_26.py；src/run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py；src/run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py。

## 天气文件和核查范围

- SYA训练/验证WTH：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY{YY}01.WTH`。
- LCA训练/验证WTH：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC/CNLC{YY}01.WTH`。
- 训练2005–2013，验证2014–2023；两站合计38个年份的WTH逐日检查均PASS。`weather_source_audit.csv`保留原审计数值，不因来源解释而重写。

## 两项历史背景修订

### LCA 补充天气

> Historical supplementary weather data were incorporated into the final WTH after downloading source files; cleaned CSV comparison differences reflect this historical supplementation process.

这条历史说明来自本次用户补充。既有实验配置显示LCA 053_00使用当前lowIC目录中的最终WTH；此前报告的1113处差异全部在训练期，涉及RAIN 647、SRAD 126、TMAX 169、TMIN 171个变量单元格。现将差异解释为历史补充流程造成，不再把差异本身作为BLOCKED依据；逐日差异清单保留供追溯。当前项目内未定位到这批下载源文件的逐日导入manifest，因此不把“有补充流程”扩大表述为“全部原始非降水空白都已由下载数据逐日闭合”。

### RAIN空白

过往`weather_clean/data_check_report.md`明确写明“已根据用户确认，将RAIN原表空白解释为无降雨，并转换为0”。本次再次获得同一语义确认：原始XLS的RAIN空白=0 mm。修订后的`raw_source_audit.csv`中RAIN `missing_values=0`，原空白计入`rain_blank_as_zero_count`；`weather_gap_details.csv`已移除这些“RAW_MISSING_VALUE/RAIN”记录；`rain_blank_zero_summary.csv`逐站点逐年保存0雨日空白数量。这不需要插值或重新找降水来源。

## 仍需核实的非降水来源

原始XLS在训练期的SRAD/TMAX/TMIN空白，SYA分别160/151/151个；LCA分别139/172/172个。将这些日期与最终WTH、旧cleaned CSV比较，SYA的462个空白位置全部与cleaned值一致；LCA有461个位置值已不同，符合历史补充背景，剩余22个仍与cleaned值一致（SRAD13、TMAX8、TMIN1）。`remaining_nonrain_provenance_gaps.csv`列出462+22个具体日期和变量。

“与cleaned相同”本身不能证明最终WTH仍是均值填补，也不能证明已有独立补测。旧`weather_preprocess.py`确实按站点月份均值填补非降水空白，且计算范围覆盖训练和验证年份。SYA的WTH与cleaned逐值一致，故这462处尤需核对；LCA的22处可能是旧填补残留，也可能有值恰好一致的补充记录。未见足够证据解除这484处来源缺口，故两站维持BLOCKED，但不再以RAIN空白或1113处差异为理由。

下一步只需针对`remaining_nonrain_provenance_gaps.csv`核查历史补充文件、导入记录或明确的插补接受依据，再独立复核WGEN拟合天气来源。当前不生成CLI，不计算月参数，不运行WGEN、DSSAT或PPO。

## 产物与验证

`results/sy_lc_random_weather_015/`包含原有`baseline_config_summary.csv`、`weather_source_audit.csv`、`config_evidence.json`、`input_sha256.json`、差异清单，以及修订后的`raw_source_audit.csv`、`weather_gap_details.csv`、`final_gate.json`和新增的`rain_blank_zero_summary.csv`、`remaining_nonrain_provenance_gaps.csv`、`provenance_context_revision.json`。

原WTH与配置文件的哈希见`input_sha256.json`；本修订脚本不写这些文件。`src/revise_sy_lc_weather_provenance_015.py`基于冻结原审计证据生成修订版。
