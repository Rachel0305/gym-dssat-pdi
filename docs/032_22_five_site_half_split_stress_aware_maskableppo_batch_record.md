# 032_22 五站点前半训练-后半验证自由时序 stress-aware MaskablePPO 批量实验记录

## 结论先说

- 状态：训练与验证均已完成。
- 算法：MaskablePPO，seed=0；每个站点独立训练 100,000 timesteps。
- 站点：FQ、HL、LC、SY、YC；没有跨站点迁移，也没有联合训练。
- checkpoint：25,000、50,000、75,000、100,000。
- 验证：每站点后半年份 10 年；共 5 × 4 × 10 = 200 条 checkpoint-year 验证，200/200 成功。
- 注意：本轮是自由时序 stress-aware PPO 的五站点批量初筛，不是最终参数优化；只有 seed0。

## 固定年份划分

| station_code | site | year | split |
| --- | --- | --- | --- |
| FQA | FQ | 2005 | train |
| FQA | FQ | 2006 | train |
| FQA | FQ | 2007 | train |
| FQA | FQ | 2008 | train |
| FQA | FQ | 2009 | train |
| FQA | FQ | 2010 | train |
| FQA | FQ | 2011 | train |
| FQA | FQ | 2012 | train |
| FQA | FQ | 2013 | train |
| FQA | FQ | 2014 | validation |
| FQA | FQ | 2015 | validation |
| FQA | FQ | 2016 | validation |
| FQA | FQ | 2017 | validation |
| FQA | FQ | 2018 | validation |
| FQA | FQ | 2019 | validation |
| FQA | FQ | 2020 | validation |
| FQA | FQ | 2021 | validation |
| FQA | FQ | 2022 | validation |
| FQA | FQ | 2023 | validation |
| HLA | HL | 2004 | train |
| HLA | HL | 2005 | train |
| HLA | HL | 2006 | train |
| HLA | HL | 2007 | train |
| HLA | HL | 2008 | train |
| HLA | HL | 2009 | train |
| HLA | HL | 2010 | train |
| HLA | HL | 2011 | train |
| HLA | HL | 2012 | train |
| HLA | HL | 2013 | train |
| HLA | HL | 2014 | validation |
| HLA | HL | 2015 | validation |
| HLA | HL | 2016 | validation |
| HLA | HL | 2017 | validation |
| HLA | HL | 2018 | validation |
| HLA | HL | 2019 | validation |
| HLA | HL | 2020 | validation |
| HLA | HL | 2021 | validation |
| HLA | HL | 2022 | validation |
| HLA | HL | 2023 | validation |
| LCA | LC | 2005 | train |
| LCA | LC | 2006 | train |
| LCA | LC | 2007 | train |
| LCA | LC | 2008 | train |
| LCA | LC | 2009 | train |
| LCA | LC | 2010 | train |
| LCA | LC | 2011 | train |
| LCA | LC | 2012 | train |
| LCA | LC | 2013 | train |
| LCA | LC | 2014 | validation |
| LCA | LC | 2015 | validation |
| LCA | LC | 2016 | validation |
| LCA | LC | 2017 | validation |
| LCA | LC | 2018 | validation |
| LCA | LC | 2019 | validation |
| LCA | LC | 2020 | validation |
| LCA | LC | 2021 | validation |
| LCA | LC | 2022 | validation |
| LCA | LC | 2023 | validation |
| SYA | SY | 2005 | train |
| SYA | SY | 2006 | train |
| SYA | SY | 2007 | train |
| SYA | SY | 2008 | train |
| SYA | SY | 2009 | train |
| SYA | SY | 2010 | train |
| SYA | SY | 2011 | train |
| SYA | SY | 2012 | train |
| SYA | SY | 2013 | train |
| SYA | SY | 2014 | validation |
| SYA | SY | 2015 | validation |
| SYA | SY | 2016 | validation |
| SYA | SY | 2017 | validation |
| SYA | SY | 2018 | validation |
| SYA | SY | 2019 | validation |
| SYA | SY | 2020 | validation |
| SYA | SY | 2021 | validation |
| SYA | SY | 2022 | validation |
| SYA | SY | 2023 | validation |
| YCA | YC | 2004 | train |
| YCA | YC | 2005 | train |
| YCA | YC | 2006 | train |
| YCA | YC | 2007 | train |
| YCA | YC | 2008 | train |
| YCA | YC | 2009 | train |
| YCA | YC | 2010 | train |
| YCA | YC | 2011 | train |
| YCA | YC | 2012 | train |
| YCA | YC | 2013 | train |
| YCA | YC | 2014 | validation |
| YCA | YC | 2015 | validation |
| YCA | YC | 2016 | validation |
| YCA | YC | 2017 | validation |
| YCA | YC | 2018 | validation |
| YCA | YC | 2019 | validation |
| YCA | YC | 2020 | validation |
| YCA | YC | 2021 | validation |
| YCA | YC | 2022 | validation |
| YCA | YC | 2023 | validation |

## 训练模型清单

| station_code | site | checkpoint_step | run_status | model_path | model_sha256 |
| --- | --- | --- | --- | --- | --- |
| FQA | FQ | 25000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/FQA/FQA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | 2c0ff97d80dec59bef7947f410af60e702fde525f043f1b204ec817da0c5b834 |
| FQA | FQ | 50000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/FQA/FQA_half_split_stress_aware_maskableppo_seed0_ckpt50000.zip | b87295a403a4a208748d56e3f74e9dd09a8d7a0f8b0c0c6167432b548fad85b4 |
| FQA | FQ | 75000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/FQA/FQA_half_split_stress_aware_maskableppo_seed0_ckpt75000.zip | 1453d2398420815ba5ea70c7c94caef3313184230053951d78fdde7e1ab59bea |
| FQA | FQ | 100000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/FQA/FQA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip | 8e45503f418fe340a37a2d74cfde00a0d6a8e3d83d246fa0e31d68720a1bab87 |
| HLA | HLA | 25000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/HLA/HLA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | 102ea1d766d495c59798b34e0e09ca4b7536e46a0dcdaac374aba2f0bfb67ecf |
| HLA | HLA | 50000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/HLA/HLA_half_split_stress_aware_maskableppo_seed0_ckpt50000.zip | edfe4e5c851bfd5690fd5300a2efd5dc325d0278fb268296a90f39bf1039e088 |
| HLA | HLA | 75000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/HLA/HLA_half_split_stress_aware_maskableppo_seed0_ckpt75000.zip | baf9fe89890282970b8fe3727a9239a7b661909db43a64e780c4c1b46d376201 |
| HLA | HLA | 100000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/HLA/HLA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip | 3685ca94eb0982a7b756435435ce5cf67012eaf97e6534b5d99022d23a0a9cc3 |
| LCA | LC | 25000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/LCA/LCA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | a4835231f6df98a61af1f46b1eec3da1880ccce3e11737db75309bfb9ec79542 |
| LCA | LC | 50000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/LCA/LCA_half_split_stress_aware_maskableppo_seed0_ckpt50000.zip | 29806c49d7dc6cfd7011bf165224371510e3397fb19ed30f082ce196bb215805 |
| LCA | LC | 75000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/LCA/LCA_half_split_stress_aware_maskableppo_seed0_ckpt75000.zip | 6c1b7c329b4c431c05d745455d8ac187a5d1a3d5db699f9dfb859e605242994f |
| LCA | LC | 100000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/LCA/LCA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip | 40c10c6569d7f89520245c3dd4922a7eb144612e762306b482c2b2a25ebc4590 |
| SYA | SY | 25000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | 54c55b1867cb9c7cf278f49c7d3ec6fcfd102ff11b2581e10327646c68f8a79c |
| SYA | SY | 50000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt50000.zip | 86ccf4ddba1fc5e5086dda8f6c6b0dd0c069724cf51d02a6bec12ac8e4c74fc0 |
| SYA | SY | 75000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt75000.zip | 58ee257a7cf1e5f664bf6e8d10bc4f581d7b097c93e47496c5e76eb6171cc57b |
| SYA | SY | 100000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip | 928243aea211fca43f856099e3a9498ebf9503ba8f50a1a75f4f65a2cd078f76 |
| YCA | YC | 25000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/YCA/YCA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | 26bbd169f50ade152bfa043b1e798519902ea04dc5a91c59b1e732cb247f5deb |
| YCA | YC | 50000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/YCA/YCA_half_split_stress_aware_maskableppo_seed0_ckpt50000.zip | e2c2cbdeb2f13a24499d4c270809e895731bc33c74c1e40b411801743026464e |
| YCA | YC | 75000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/YCA/YCA_half_split_stress_aware_maskableppo_seed0_ckpt75000.zip | c6da5d95d5656d626945f75258501072087fd90e057fb7b7b98bc325b82a6f6a |
| YCA | YC | 100000 | ok_existing | benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/YCA/YCA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip | 08ac32be7caa7651298960d293b6fb02308a5f3b55448a1b655e8b7bdd8bedce |

## 站点-checkpoint 验证汇总

| station_code | site | checkpoint_step | validation_years | mean_final_grnwt | mean_total_irrigation | mean_total_n | mean_PFP_N | max_nstres | max_swfac | any_metric_win_four_count | mean_gap_yield_vs_four_max | mean_gap_pfp_n_vs_four_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | FQ | 25000 | 10 | 6356.018493652344 | 150.0 | 240.0 | 26.483410390218097 | 0.0121910572052001 | 0.1239134669303894 | 0 | -1081.9815063476562 | -23.551311832004124 |
| FQA | FQ | 50000 | 10 | 6115.466735839844 | 138.0 | 240.0 | 25.481111399332683 | 0.0121910572052001 | 0.5935988426208496 | 0 | -1322.5332641601558 | -24.55361082288954 |
| FQA | FQ | 75000 | 10 | 5857.6136474609375 | 69.0 | 240.0 | 24.40672353108724 | 0.0121910572052001 | 0.6736873388290405 | 0 | -1580.386352539062 | -25.627998691134984 |
| FQA | FQ | 100000 | 10 | 6243.034851074219 | 150.0 | 240.0 | 26.012645212809247 | 0.0121910572052001 | 0.527654767036438 | 0 | -1194.9651489257812 | -24.022077009412975 |
| HLA | HLA | 25000 | 10 | 4168.650085449219 | 150.0 | 112.0 | 36.97399775187175 | 0.6280465722084045 | 0.0 | 6 | -2324.2788330078124 | 1.7839977518717451 |
| HLA | HLA | 50000 | 10 | 6280.718292236328 | 135.0 | 240.0 | 26.169659550984697 | 0.5062225461006165 | 0.0 | 3 | -212.2106262207032 | -9.0203404490153 |
| HLA | HLA | 75000 | 10 | 6248.523193359375 | 135.0 | 240.0 | 26.03551330566406 | 0.5569676756858826 | 0.0 | 1 | -244.40572509765624 | -9.154486694335938 |
| HLA | HLA | 100000 | 10 | 6280.718292236328 | 135.0 | 240.0 | 26.169659550984697 | 0.5062225461006165 | 0.0 | 3 | -212.2106262207032 | -9.0203404490153 |
| LCA | LC | 25000 | 10 | 9185.893493652344 | 150.0 | 240.0 | 38.27455622355144 | 0.0121910572052001 | 0.0740741491317749 | 8 | -174.24145507812437 | 0.5345562235514322 |
| LCA | LC | 50000 | 10 | 9154.93878173828 | 150.0 | 240.0 | 38.14557825724283 | 0.0121910572052001 | 0.0 | 7 | -205.19616699218795 | 0.4055782572428363 |
| LCA | LC | 75000 | 10 | 9273.3095703125 | 150.0 | 240.0 | 38.63878987630208 | 0.0121910572052001 | 0.0 | 10 | -86.82537841796838 | 0.8987898763020837 |
| LCA | LC | 100000 | 10 | 9273.3095703125 | 150.0 | 240.0 | 38.63878987630208 | 0.0121910572052001 | 0.0 | 10 | -86.82537841796838 | 0.8987898763020837 |
| SYA | SY | 25000 | 10 | 9919.611328125 | 150.0 | 240.0 | 41.3317138671875 | 0.4880468249320984 | 0.5572727620601654 | 0 |  |  |
| SYA | SY | 50000 | 10 | 10004.396423339844 | 150.0 | 240.0 | 41.68498509724935 | 0.4106650352478027 | 0.7029998302459717 | 0 |  |  |
| SYA | SY | 75000 | 10 | 10006.793029785156 | 150.0 | 240.0 | 41.69497095743815 | 0.4036962389945984 | 0.7028219997882843 | 0 |  |  |
| SYA | SY | 100000 | 10 | 10003.96746826172 | 150.0 | 240.0 | 41.68319778442383 | 0.4115945100784302 | 0.7033801972866058 | 0 |  |  |
| YCA | YC | 25000 | 10 | 8126.515197753906 | 87.0 | 232.0 | 35.766409683227536 | 0.4349812865257263 | 0.3492048382759094 | 10 | -77.01569824218731 | 2.6738590759400895 |
| YCA | YC | 50000 | 10 | 8156.746154785156 | 127.5 | 200.0 | 40.78373077392578 | 0.2877501845359802 | 0.3791983127593994 | 10 | -46.78474121093759 | 7.691180166638334 |
| YCA | YC | 75000 | 10 | 8169.78271484375 | 105.0 | 200.0 | 40.84891357421875 | 0.2718836665153503 | 0.3791983127593994 | 10 | -33.74818115234348 | 7.756362966931301 |
| YCA | YC | 100000 | 10 | 7971.619079589844 | 99.0 | 192.0 | 42.34329477945964 | 0.4534501433372497 | 0.5775021016597748 | 10 | -231.9118164062499 | 9.250744172172185 |

## 每站点当前最佳 checkpoint（按 any_metric_win_four_count 优先）

| station_code | site | checkpoint_step | validation_years | any_metric_win_four_count | mean_final_grnwt | mean_total_irrigation | mean_total_n | mean_gap_yield_vs_four_max | mean_gap_wp_et_vs_four_max | mean_gap_pfp_n_vs_four_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | FQ | 25000 | 10 | 0 | 6356.018493652344 | 150.0 | 240.0 | -1081.9815063476562 |  | -23.551311832004124 |
| HLA | HLA | 25000 | 10 | 6 | 4168.650085449219 | 150.0 | 112.0 | -2324.2788330078124 |  | 1.7839977518717451 |
| LCA | LC | 75000 | 10 | 10 | 9273.3095703125 | 150.0 | 240.0 | -86.82537841796838 |  | 0.8987898763020837 |
| SYA | SY | 25000 | 10 | 0 | 9919.611328125 | 150.0 | 240.0 | nan |  |  |
| YCA | YC | 75000 | 10 | 10 | 8169.78271484375 | 105.0 | 200.0 | -33.74818115234348 |  | 7.756362966931301 |

## 目前直观看法

- LC 和 YC 在后半验证年份上表现最好：多个 checkpoint 达到 10/10 年至少一个指标超过四情景最高值。
- HL 有部分 checkpoint 有信号，但不稳定；25k 的节氮倾向强，但平均产量较低，需要看逐年表和管理措施是否合理。
- FQ 和 SY 在这套 seed0、100K、当前 reward/约束下没有出现“至少一个指标超过四情景最高值”的年份，需要后续诊断或换配置。
- 因为本轮只有单 seed，不能直接说某站点已经稳定成功；它更像是为下一步选 checkpoint、画图和诊断提供候选。

## 文件输出

- 训练清单：`benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/evaluation/032_22_training_checkpoint_inventory.csv`
- 验证逐年表：`benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/evaluation/032_22_checkpoint_validation_summary.csv`
- 站点汇总表：`benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/evaluation/032_22_validation_summary_by_station_checkpoint.csv`
- 日值输出目录：`benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/daily_outputs`
- 模型目录：`benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models`

## 非科学性问题与修复记录

- 第一次启动后发现 SB3 需要 Gymnasium API，脚本导入从 `gym` 改为 `gymnasium` 后环境初始化通过。
- 首轮验证失败不是模型问题，而是 `daily_outputs/<station>` 目录没有创建；修复后复用已有模型重新验证，200/200 成功。
- 最终汇总时发现 `site` 列缺失导致汇总函数报错；已根据 `station_code` 补映射后重新生成最终 CSV 和本记录。
