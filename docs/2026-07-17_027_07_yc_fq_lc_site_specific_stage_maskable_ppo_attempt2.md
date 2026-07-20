# 027_07 YC/FQ/LC 站点专属阶段型 MaskablePPO 记录

## 执行边界

- 三站点串行执行；未做联合训练、权重迁移或跨年验证。
- PPO 核心参数、9动作、I120/N300预算与 reward 结构未修改。
- YC使用官方六个可执行阶段7/30/45/60/80/100；FQ和LC因在DAP100前收获，仅使用真实可执行的7/30/45/60/80，未移动第六阶段。
- 原027_00 primary用于训练扩展硬门槛；导师最新至少一项严格领先视图只做报告，不参与reward或选模。
- 每个运行目录均保存派生MZX哈希、指针/日期/天气/土壤/品种/管理模式审计；Summary.OUT按Y/I/N三项匹配。

## 站点分支

| site | year | branch | seeds_run | primary_count | advisor_any_metric_winner_count |
| --- | --- | --- | --- | --- | --- |
| YC | 2014 | A_three_seed_primary_replicated | 0,1,2 | 2 | 2 |
| FQ | 2016 | B_seed0_primary_failed_stop | 0 | 0 | 0 |
| LC | 2010 | B_seed0_primary_failed_stop | 0 | 0 | 1 |

## 预注册选中模型

| site | seed | engineering_pass | selected_checkpoint | model_sha256 | yield | irrigation | nitrogen | WP_ET | PFP_N | reward | primary_pass | advisor_any_metric_strict_winner | winning_yield | winning_WP_ET | winning_PFP_N | yield_gap_vs_all_four_max | wp_gap_vs_all_four_max | pfp_gap_vs_all_four_positive_n_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YC | 0 | True | 60 | c853a879728b9bed351b50f716e4f5d35e856803033d9c50d3042c995bc6aa46 | 9417.883911 | 105.0 | 200.0 | 2.57 | 47.1 | 2.107444 | True | True | False | False | True | -0.0177 | -0.07 | 9.1 |
| YC | 1 | True | 60 | 47969c701cac54bf2d6818947adf65faa1641745da6ec6da6bdc096585e13c0e | 8722.329712 | 105.0 | 0.0 | 2.44 |  | 0.79189 | False | False | False | False | False | -695.571899 | -0.2 |  |
| YC | 2 | True | 120 | c8f579ebcd95c6ce4e7acfbff812bc4520c0d324e9c6d0c15f1eb73671a123d8 | 9417.868652 | 120.0 | 100.0 | 2.57 | 94.2 | 2.592429 | True | True | False | False | True | -0.032959 | -0.07 | 56.2 |
| FQ | 0 | True | 60 | c0bc9716b0b649d0c89f6dcddecb916942025b06e9ed8956eec1584db136d2b0 | 8012.421875 | 75.0 | 200.0 | 2.41 | 40.1 | 1.491346 | False | False | False | False | False | 0.0 | -0.09 | -15.0 |
| LC | 0 | True | 240 | b60930af841fb642a4ae3a90b9f30e4c0d6fcf40ef44f88a697f5b56c1377266 | 8738.970947 | 60.0 | 150.0 | 3.06 | 58.3 | 1.498104 | False | True | False | False | True | 0.0 | -0.07 | 23.0 |

## 失败记录

_无数据_

## 结论边界

本记录只回答三个训练锚点在冻结阶段型MaskablePPO下能否产生跨seed初步信号。任何未达到2/3原primary的站点均按预注册规则停止，不据此调参；任何达到的站点也尚未完成同站跨年泛化。

## Attempt 1 与 Attempt 2

- Attempt 1 完成了三站四基线和 no-op smoke，但在写入 readiness JSON 时因 `numpy.bool_` 不能直接序列化而停止；没有启动任何 PPO 训练。失败目录保留在 `benchmark_results/027_07_site_specific_stage_maskable_ppo/`，未删除、未覆盖。
- Attempt 2 仅修复记录层，并补强输入 provenance、live observation schema、Summary.OUT 的 Y/I/N 账本及七项基线端点交叉核验；没有修改 DSSAT 输入、IC、reward、动作、PPO 超参数或停止线。
- Attempt 2 共生成 25 份 `input_provenance.json`，全部通过；5 份 seed 结果中的所有工程检查和模型文件 SHA-256 复核均通过。

## Fresh 四基线端点

| site | scenario | yield kg/ha | biomass kg/ha | I mm | N kg/ha | ET mm | WP_ET kg/m3 | PFP_N kg/kg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YC | null | 7825.44 | 17996.47 | 0 | 0 | 341.9 | 2.29 | NA |
| YC | recorded | 9417.90 | 20514.21 | 120 | 374 | 357.1 | 2.64 | 25.2 |
| YC | auto | 8712.88 | 18944.76 | 87 | 0 | 357.3 | 2.44 | NA |
| YC | official expert | 9417.45 | 20480.67 | 229 | 248 | 367.1 | 2.56 | 38.0 |
| FQ | null | 7066.08 | 13148.47 | 0 | 0 | 294.5 | 2.40 | NA |
| FQ | recorded | 7932.60 | 14008.16 | 75 | 144 | 317.1 | 2.50 | 55.1 |
| FQ | auto | 8012.42 | 14094.82 | 60 | 0 | 323.5 | 2.48 | NA |
| FQ | official expert | 7939.95 | 13768.53 | 199 | 248 | 338.5 | 2.35 | 32.1 |
| LC | null | 8050.87 | 15541.19 | 0 | 0 | 256.9 | 3.13 | NA |
| LC | recorded | 8731.59 | 16323.72 | 130 | 250 | 286.2 | 3.05 | 34.9 |
| LC | auto | 8738.36 | 16372.74 | 139 | 0 | 286.8 | 3.05 | NA |
| LC | official expert | 8738.97 | 16376.27 | 199 | 248 | 281.9 | 3.10 | 35.3 |

七项端点均与旧 027_05 证据在预注册容差内一致。事件总量的小数与 Summary.OUT 整数记账之差不超过 1.0。

## 预注册模型结果与解释

| site | seed | selected ckpt | yield kg/ha | I mm | N kg/ha | WP_ET | PFP_N | original primary | strict winner among four baselines | action sequence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| YC | 0 | 60 | 9417.88 | 105 | 200 | 2.57 | 47.1 | pass | PFP_N | 8,1,5,3,2,0 |
| YC | 1 | 60 | 8722.33 | 105 | 0 | 2.44 | NA | fail | none | 0,2,2,1,1,1 |
| YC | 2 | 120 | 9417.87 | 120 | 100 | 2.57 | 94.2 | pass | PFP_N | 8,2,2,2,0,0 |
| FQ | 0 | 60 | 8012.42 | 75 | 200 | 2.41 | 40.1 | fail | none | 5,0,7,4,1 |
| LC | 0 | 240 | 8738.97 | 60 | 150 | 3.06 | 58.3 | fail | PFP_N | 8,5,0,0,0 |

- YC2014 达到 2/3 original primary，因此冻结框架在该训练锚点获得跨 seed 初步复现证据；成功 seed 的严格领先项均为 PFP_N。
- FQ2016 的 seed0 产量与 auto 持平，但 WP_ET 比四基线最大值低 0.09，PFP_N 比四基线最大值低 15.0，且没有任何一项严格领先；按预注册停止 seed1/2。
- LC2010 的 seed0 与四基线最高产量持平，WP_ET 仅低 0.04（约 1.3%），PFP_N 高 23.0（约 65.2%）；它不满足更严的 original primary，但满足“至少一项严格领先”的必要条件。由于导师尚未给“另外两项接近”的数值阈值，本记录不把它自动判成正式 relaxed pass。
- checkpoint 0 不进入预注册选模。FQ 的随机初始化 checkpoint 0 虽有较高 PFP_N，但不属于训练所得模型，不能作为 PPO 成功证据。

## 停止线与下一步边界

- 本任务没有调参、追加 seed、跨年迁移或多站点联合训练。
- YC 可进入单独预注册的同站跨年验证；FQ/LC 若继续，必须另立任务书，不能在本记录内事后修改 reward、步数或选模规则。
- 导师 relaxed 标准若要成为正式统计判据，必须先给出“接近”的数值容差；在此之前仅报告三项相对差值。

## 主要输出

- Prompt：`prompts/027_07_yc_fq_lc_site_specific_stage_maskable_ppo.md`
- Runner：`src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py`
- Attempt 2 根目录：`benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/`
- 三站总览：`027_07_three_site_overview.csv`
- 选中模型表：`027_07_three_site_selected_seed_summary.csv`
- 各站 readiness、四基线、scaler、smoke、checkpoint、训练 episode、阶段动作、模型及 SHA-256 均保存在对应站点目录。
