# 034_00 external action channel audit record

目的：检查通过 GymDssatWrapper 向 DSSAT 发送外部灌溉/施氮动作时，作物轨迹与 Summary.OUT 是否发生响应。

测试对象：FQA2005，multisite_new_cultivar_inputs_013，IC=1 渲染链。

测试组：zero、DAP1 raw_once(50/200)、DAP1/10/20/30 raw_repeated(50/200)。

| case | final_grnwt_daily | max_nstres_daily | max_swfac_daily | summary_HWAM_last | summary_IRCM_last | summary_NICM_last | summary_ETCP_last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | 7972.7026 | 0.0122 | 0.0 | 7973.0 | 0.0 | 0.0 | 309.6 |
| raw_once_dap1 | 7972.7026 | 0.0122 | 0.0 | 7973.0 | 0.0 | 0.0 | 309.6 |
| raw_repeated_dap1_10_20_30 | 7972.7026 | 0.0122 | 0.0 | 7973.0 | 0.0 | 0.0 | 309.6 |

判读：若三组作物产量和 Summary.OUT 的 IRCM/NICM 基本一致，则说明当前外部动作通道没有按预期改变 DSSAT 管理结果，不能直接用 RL 安全层累计动作冒充 DSSAT 实际管理。
