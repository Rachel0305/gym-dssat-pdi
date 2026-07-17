# 026_07 SY ICDAT 对齐后的全权威年份冻结阶段型 PPO 验证记录

## 1. 结论先行

任务状态：`completed`。

正式判定：`A_SY_all_years_transfer_after_icdat_alignment`。

在不训练、不重新选 checkpoint、只对运行副本的 ICDAT 年份进行用户批准的最小对齐后，SY2014 训练得到的三个冻结阶段型 MaskablePPO 模型在 SY 三个权威 treatment 年份上的本地主判据通过数为：

| 年份 | seed0 | seed1 | seed2 | 通过数 | 年份判定 |
|---:|:---:|:---:|:---:|---:|:---:|
| 2012 | 通过 | 通过 | 通过 | 3/3 | 通过 |
| 2014 | 通过 | 通过 | 通过 | 3/3 | 通过 |
| 2015 | 未通过 | 通过 | 通过 | 2/3 | 通过 |

主判据是相对当年 `DSSAT auto` 与 `official extension expert` 的较高产量、较高 WP_ET 和正施氮基线较高 PFP_N 同时不低。recorded 只作为独立比较对象，不替代主判据。

## 2. 输入溯源与唯一修正

权威输入：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX`。

原始文件 SHA256：`20b071bc49549cbf561be3aae81caa355aa0582564db524e17d68ab6a274418a`。

原文件在任务前后哈希一致。只在每个新运行目录的 MZX 副本中执行：

| 年份 | SDATE | 原 ICDAT | 批准后 ICDAT | PDATE | 处理 |
|---:|---:|---:|---:|---:|---|
| 2012 | 12092 | 14099 | 12099 | 12119 | 只对齐年份 |
| 2014 | 14091 | 14099 | 14099 | 14111 | 不变 |
| 2015 | 15091 | 14099 | 15099 | 15108 | 只对齐年份 |

IC 指针、IC 分层剖面、年内日序 099、SDATE、PDATE、水氮初值、天气、品种参数均未修改。21 个正式季节均保存独立 `icdat_alignment_manifest.json`，总表为 `026_07_runtime_icdat_alignment_manifests.csv`。

## 3. 冻结模型与方法

| seed | checkpoint | SHA256 |
|---:|---|---|
| 0 | `benchmark_results/026_03/checkpoint_000120.zip` | `d88938a7d939cc772f5da185360298f59af86ea0b95cf411d3ccb0722ba541ab` |
| 1 | `benchmark_results/026_02/checkpoint_000060.zip` | `96d5d85e82a0e7a10b3fe68014b8f6c93ef903326d8640478c0becc6477832ec` |
| 2 | `benchmark_results/026_04/checkpoint_000240.zip` | `6add96bd8e827ec9613d5f04e18cb233f83418aa7bc07f690811acd5509e4b64` |

训练步数为 0。25 维 scaler、六阶段决策点、九动作、预算 mask、DAP>=90 禁氮规则全部冻结。所有模型运行前后哈希一致，非法动作尝试为 0。

## 4. Smoke test

### 4.1 SY2015 ICDAT 对齐 null smoke

- 目录：`benchmark_results/026_07_smoke_2015_aligned_null/`；
- 运行副本 `14099 -> 15099`；
- 正常到季末；
- null 产量 6601 kg/ha；
- Summary 匹配误差为 0；
- 原始 MZX 哈希未变化。

### 4.2 外部动作管理表技术 smoke

第一次正式 attempt 在 SY2014 official expert 发现 `IPIRR`。根因是旧的 target-only 清理函数留下非目标 treatment 的 `IROP=-99` 行。后续两个 smoke 又依次发现表终止空行和 MI/MF level 指针渲染问题。所有失败目录均保留：

- `benchmark_results/026_07/`：正式 attempt1，SY2012 完成后在 SY2014 expert 因 `IPIRR` 停止；
- `benchmark_results/026_07_smoke_2014_expert_render_fix/`：缺少表终止空行，`IPFERT`；
- `benchmark_results/026_07_smoke_2014_expert_render_fix_attempt2/`：零值行误用 treatment id 2，而 MI/MF 指向 level 1，`IPFERT`；
- `benchmark_results/026_07_smoke_2014_expert_render_fix_attempt3/`：通过。

通过的 technical smoke 得到 SY2014 expert：产量 11059 kg/ha，I=266.25 mm，N=300 kg/ha，Summary 匹配误差 0.25。该修复只重绘外部动作场景的 reported 管理表占位行，不改变实际 expert/PPO 动作或 IC。

## 5. 四基线

| 年份 | 情景 | 产量 kg/ha | 灌溉 mm | 管理施氮 kg/ha | WP_ET kg/m3 |
|---:|---|---:|---:|---:|---:|
| 2012 | null | 7002 | 0 | 0 | 1.64 |
| 2012 | recorded | 10166 | 0 | 293 | 2.36 |
| 2012 | DSSAT auto | 7016 | 33 | 0 | 1.60 |
| 2012 | official expert | 9609 | 266.25 | 300 | 2.21 |
| 2014 | null | 5408 | 0 | 0 | 1.22 |
| 2014 | recorded | 9613 | 0 | 293 | 2.16 |
| 2014 | DSSAT auto | 5498 | 66 | 0 | 1.20 |
| 2014 | official expert | 11059 | 266.25 | 300 | 2.26 |
| 2015 | null | 6601 | 0 | 0 | 1.51 |
| 2015 | recorded | 8973 | 0 | 293 | 2.01 |
| 2015 | DSSAT auto | 6868 | 68.7 | 0 | 1.52 |
| 2015 | official expert | 10891 | 266.25 | 300 | 2.33 |

零施氮场景的 PFP_N 数学上未定义，不设为无穷大，也不要求为有限值。

## 6. 三个冻结 PPO 结果

| 年份 | seed | 动作序列 | 产量 kg/ha | I mm | N kg/ha | WP_ET | PFP_N | 主判据 | 产量>=recorded |
|---:|---:|---|---:|---:|---:|---:|---:|:---:|:---:|
| 2012 | 0 | 0,3,4,6,6,2 | 10120.2 | 45 | 300 | 2.35 | 33.7 | 通过 | 否 |
| 2012 | 1 | 4,4,0,5,5,1 | 10056.0 | 105 | 200 | 2.28 | 50.3 | 通过 | 否 |
| 2012 | 2 | 3,3,7,7,1,1 | 10068.6 | 60 | 300 | 2.34 | 33.6 | 通过 | 否 |
| 2014 | 0 | 0,0,7,7,1,1 | 11204.9 | 60 | 200 | 2.31 | 56.0 | 通过 | 是 |
| 2014 | 1 | 4,0,0,8,2,2 | 11202.7 | 105 | 150 | 2.26 | 74.7 | 通过 | 是 |
| 2014 | 2 | 3,3,3,5,1,1 | 11208.4 | 60 | 200 | 2.33 | 56.0 | 通过 | 是 |
| 2015 | 0 | 0,0,1,1,1,1 | 6786.3 | 60 | 0 | 1.51 | NA | 未通过 | 否 |
| 2015 | 1 | 4,0,2,5,5,1 | 10999.9 | 120 | 150 | 2.33 | 73.3 | 通过 | 是 |
| 2015 | 2 | 2,3,2,7,8,1 | 11107.7 | 120 | 250 | 2.35 | 44.4 | 通过 | 是 |

2015 seed0 退化为零氮低产策略，因此没有把 2/3 的年份通过包装成 3/3 稳定。

## 7. 聚合修正记录

21 个季节首次汇总后，JSON 因错误要求所有 PFP_N 有限而被标成 C；唯一零氮模型是 SY2015 seed0，其 PFP_N 正确地为 NA。随后只做纯离线汇总修正：正施氮模型必须有有限 PFP_N，零氮模型必须为 NA。没有重跑 DSSAT、没有改变任何科学数值。修正前 JSON 保存在：

`benchmark_results/026_07_attempt2/026_07_result_before_pfp_engineering_fix.json`。

## 8. 可支持与不可支持的结论

可以支持：

> 按预注册的 checkpoint 选择协议冻结的 SY2014 阶段型 PPO 模型，在完成获批的 ICDAT 年份对齐后，可在 SY2012、SY2014、SY2015 三个权威 treatment 年份中达到每年至少 2/3 模型同时不低于当年 DSSAT auto 和官方推广 expert 的产量、WP_ET 与 PFP_N 门槛。

不能支持：

- 不能说 9/9 全部成功，实际是 8/9；
- 不能说超过所有 recorded，2012 为 0/3；
- 不能把预注册择优 checkpoint 写成训练末期稳定收敛；
- 不能外推到 SY 没有权威 treatment/IC 的 WTH-only 年份；
- 不能由 SY 跨年成功直接推出跨站点成功。

## 9. 输出文件

正式目录：`benchmark_results/026_07_attempt2/`。

核心文件：

- `026_07_result.json`；
- `026_07_sy_original_and_approved_date_audit.csv`；
- `026_07_runtime_icdat_alignment_manifests.csv`；
- `026_07_sy_all_years_four_baselines.csv`；
- `026_07_sy_all_years_frozen_ppo_summary.csv`；
- `026_07_sy_all_years_frozen_ppo_stage_actions.csv`；
- `026_07_sy_year_seed_pass_matrix.csv`；
- 各年份四基线、模型 summary 和阶段动作表。

## 10. 下一步

026_07 只完成 SY。进入其他站点前必须先审计各站点已验证的 treatment/IC、阶段型环境 adapter、基线生成链路和 official expert 口径；不得因为存在 WTH 就默认年份可用，也不得直接把 SY 的输入修正套到其他站点。
