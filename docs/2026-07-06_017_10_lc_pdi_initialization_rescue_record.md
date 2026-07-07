# 017_10 LC PDI/gym 初始化问题诊断与抢救记录

## 关键发现

- 原始 LC MZX 中 ID_SOIL 为 `LC99001200`。
- LC 的 SOIL.SOL 中实际 profile 为 `LC990012007`。
- 本阶段只修改临时副本，不修改原始输入包。

## 初始化测试状态

case,scenario,soil_fix,sdate_fix,include_mza_mzt,mode,status,timeout_sec,run_dir,constructor_ok,reset_ok,step_ok,elapsed_sec,n_steps,completed
original_recorded_one_step,recorded,False,False,False,one_step,timeout,45,DSSAT_auto_validation/lc_pdi_initialization_rescue_017_10/cases/original_recorded_one_step,,,,,,
soilfix_recorded_one_step,recorded,True,False,False,one_step,timeout,45,DSSAT_auto_validation/lc_pdi_initialization_rescue_017_10/cases/soilfix_recorded_one_step,,,,,,
soilfix_sdatefix_recorded_one_step,recorded,True,True,False,one_step,ok,60,DSSAT_auto_validation/lc_pdi_initialization_rescue_017_10/cases/soilfix_sdatefix_recorded_one_step,True,True,True,1.8082945346832275,1.0,False
soilfix_sdatefix_recorded_mza_mzt_one_step,recorded,True,True,True,one_step,ok,60,DSSAT_auto_validation/lc_pdi_initialization_rescue_017_10/cases/soilfix_sdatefix_recorded_mza_mzt_one_step,True,True,True,1.9236037731170654,1.0,False
soilfix_null_full,null,True,True,False,full,ok,120,DSSAT_auto_validation/lc_pdi_initialization_rescue_017_10/cases/soilfix_null_full,True,True,True,2.434014081954956,151.0,True
soilfix_recorded_full,recorded,True,True,False,full,ok,120,DSSAT_auto_validation/lc_pdi_initialization_rescue_017_10/cases/soilfix_recorded_full,True,True,True,2.7033562660217285,151.0,True
soilfix_dssat_auto_full,dssat_auto,True,True,False,full,ok,120,DSSAT_auto_validation/lc_pdi_initialization_rescue_017_10/cases/soilfix_dssat_auto_full,True,True,True,2.3525214195251465,151.0,True


## 基准运行结果

scenario,final_gwad,final_cwad,irrigation_total,fertilizer_total,max_water_stress,max_nitrogen_stress
null,9253.0,17866.0,0.0,0.0,0.0,0.012
recorded,9253.0,17853.0,180.0,828.0,0.0,0.012
dssat_auto,9253.0,17866.0,0.0,0.0,0.0,0.012


## 判断

- LC 初始化问题高度可能来自 MZX 与 SOIL.SOL 的土壤 ID 不一致。
- 修正临时 MZX 的 ID_SOIL 后，LC 可以进入 gym/PDI 运行链路。
- 下一步可用修正后的临时输入进行 LC 年份筛选；在确认无副作用后，再决定是否把源输入包修正为一致版本。
