# YC/YCA PPO training-domain augmentation 审计

## 审计范围与选择

本审计只覆盖 YC/YCA，不自动扩展到 SY、LC、HLA 或 FQ。当前正式参考选择
`055_00_yca_lowIC_expanded_action_maskableppo`，原因是它是最近一次正式、完整、可复现、
并且已经完成 2K smoke 与 100K validation/action audit 的 YC/YCA lowIC PPO。

旧的 `218YCA` 不作为正式参考或增强结果：它虽然使用了六天气情景库，但实际改用了
simple-profit/guardrail 奖励，且最终在汇总阶段失败，因此不是本任务要求的 augmentation-only
对照。`217YCA` 只作为天气情景库和物理门证据，不作为 PPO 性能结果。

## 1. 当前正式 YC PPO 配置

| 项目 | 冻结值 | 证据 |
|---|---|---|
| experiment/config | `055_00_yca_lowIC_expanded_action_maskableppo` / `configs/055_00_yca_lowIC_expanded_action_maskableppo.json` | `055_00_formal_result.json` |
| 站点 | station `YCA`，site `YC`，input profile `lowIC` | formal result |
| observation | `046_02_raw_observation`；不做 normalization；不使用 weather forecast | config/formal result |
| PPO | MaskablePPO；SB3-contrib；单环境、CPU | 055/042/032 运行链 |
| 联合动作 | 16 个：I=`[0,15,30,45]` mm × N=`[0,40,80,120]` kg ha⁻¹ | config；formal action audit |
| 网络与主要超参数 | MLP `[64,64]`；learning rate `0.0003`；gamma `1.0`；GAE lambda `1.0`；`n_steps=144`；batch `144`；`n_epochs=5`；entropy coefficient `0.01`；clip `0.2` | 040_36/042_10 effective contract chain |
| reward | 040_36/042_10 的 stress-aware reward；包含产量增量、I/N 成本、胁迫缓解，以及 `swfac` guardrail；本轮不改 reward function | effective wrapper provenance |
| 训练预算 | 100,000 timesteps；checkpoint `25K/50K/75K/100K`；seed 0 | config/formal result |
| train years | 2004–2013 | formal result |
| validation years | 2014–2023 | formal result |

### YC 当前物理安全约束

沿用 055/040_36 的实际 safety wrapper，不在本轮调整：日灌溉上限 45 mm、日施氮上限
120 kg ha⁻¹、季节软上限灌溉 240 mm/施氮 250 kg ha⁻¹、灌溉和施氮最小间隔均为 7 天、
灌溉允许 DAP 1–120、施氮允许 DAP 1–90；DAP 90 前累计灌溉最多 195 mm，保留 DAP 90
之后的 45 mm 灌溉机会。该约束由完整动作安全包装执行，而非在增强组中另写一套规则。

## 2. 现有 YC 基线

正式 YC baseline 批次 `055_02_yca_lowIC_four_baselines_static_level1` 包含：

| baseline | mean yield (kg ha⁻¹) | mean I (mm) | mean N (kg ha⁻¹) | mean WP_ET | mean PFP_N |
|---|---:|---:|---:|---:|---:|
| null | 3290.88 | 0.0 | 0.0 | 1.140 | N/A |
| recorded farmer template | 7972.73 | 120.0 | 374.0 | 2.338 | 21.32 |
| DSSAT native automatic irrigation | 4417.11 | 146.5 | 0.0 | 1.223 | N/A |
| official/extension expert | 8200.55 | 211.0 | 245.0 | 2.293 | 33.48 |

`recorded_farmer_template` 是冻结的静态模板复用，不是逐年真实管理策略；不对它做调低或
调优。`055_01` 的 `dssat_auto_irrigation_external_n_rule` 为补充性 external-N rule，
2014–2023 没有触发外加 N 事件，实际 N 为 0，结果与 native automatic irrigation 相同，
不把它冒充为主比较中的独立胜者。

## 3. 当前 PPO 实际表现与失败形态

055 的 10 年 validation 聚合如下；指标来自已有 validation CSV，WP_ET 不在该 PPO
artifact 中，因此不从 daily CSV 推断。

| checkpoint | mean yield | mean I | mean N | mean simple-profit | mean PFP_N |
|---:|---:|---:|---:|---:|---:|
| 25,000 | 8201.60 | 228.0 | 240.0 | 7571.60 | 34.17 |
| 50,000 | 6825.95 | 75.0 | 240.0 | 6364.25 | 28.44 |
| 75,000 | 6140.80 | 45.0 | 152.0 | 5851.14 | 40.74 |
| 100,000 | 6072.49 | 45.0 | 160.0 | 5770.19 | 37.95 |

因此当前主要失败不是“无法传递动作”：055 的 100K action gate 全部通过（10/10 daily
files、动作在声明网格、正动作可传递、非全为 DAP1、使用多个非零动作对和新档位）。主要
问题是 checkpoint 间表现明显漂移，后期资源投入和产量下降，未形成对 farmer/expert 的
稳定综合优势；25K 的产量较好也不能替代 100K formal report checkpoint。当前 PPO 的
WP_ET/NUE 结论仍受 exact Summary.OUT/ETCP replay 缺失限制。

## 4. 项目内当前判据

项目已有正式输出的硬门槛是：完成 smoke/formal、10 个 validation 年均有结果、动作在网格
且真实传递、存在非零管理、训练没有内存停止；055 的 formal gate 已通过。性能诊断使用
逐年/逐 checkpoint 的 `gap_*_vs_four` 和 `any_metric_win_four_count` 字段，反映 PPO 是否
在 yield、WP_ET、PFP_N 中至少有一个指标超过四基线最大值。

本轮不把“某一年某一指标赢”改写成“总体成功”。增强结论必须同时满足：

1. B 在匹配 checkpoint 和同一 10 年 validation 上完成技术门槛；
2. B 相对 A 的主结果至少不降低 mean yield，并且改善不能靠超出固定动作/资源约束取得；
3. 若 WP_ET 可用，则同时报告它；若不可用则明确 N/A，不补算；PFP_N、I、N 和 reward 必须一起报告；
4. 任何 3-seed 扩展都只能在单 seed formal 首轮显示改善后进行，不能用挑 seed 代替稳定性。

## 5. 本轮固定项与唯一主动变化

固定：PPO algorithm、raw observation schema、MLP/network、learning rate 和主要 hyperparameters、
reward functional form、16-action grid、YC site-specific safety、100K budget、seed 0 首轮、
train/validation split、评价指标定义和 baseline definitions。

唯一主动变化：`training-domain diversity`。B 使用已有 YC 完整整年天气文件中的 10 个原始
训练年与 5 个已经通过物理一致性门的 sequence-level weather variants，组成 10×6=60 个
训练情景；每个 episode 选择一个完整情景，情景内保留原始 daily sequence，不跨年拼接、不打乱
 日期，validation 仍只用 2014–2023 原始 lowIC 天气。

需要明确其边界：这 5 个非原始情景是 `217YCA` 已生成并通过物理门的固定整年 WTH 文件，
属于窗口化降雨情景的 domain variants；本轮没有在训练时重新随机生成降雨倍率，也没有加入
独立的 Tmax/Tmin/辐射抖动、日期打乱或跨年拼接。因此本结果应称为“物理门控的情景域增强”，
不能宣称为 WGEN 式随机天气生成或任意 `rainfall × random_factor` 增强。

## 6. IC 与 genotype 审计

- YC 有 `originIC` 与 `lowIC` 两个完整项目输入 profile，但它们在当前研究中是不同的 IC
  契约；055 正式参考明确是 lowIC。若在 B 中随机混合，会同时改变 IC transfer 条件，不能
  再称为“只做天气 domain augmentation”，所以本轮不混合，记录为 IC diversity skipped。
- YC 当前可靠输入目录只有一套可直接完成 DSSAT 的 cultivar vector：`ZD0985 / MZCER048.CUL`。
  没有第二套已校准 cultivar set，因此不做 genotype randomization，记录为 unavailable。

## 7. 现有多 seed 状态与执行决定

055 formal 是 seed 0；本轮先按 prompt 做 1-seed smoke，再做 1-seed、100K formal first-pass。
只有 B 相对 A 有改善时才考虑 3 seeds；若无改善，停止而保留失败/阴性结果。旧的 218 失败
结果保留，不覆盖、不重跑 500K。

新实验目录：`experiments/221_yc_ppo_domain_augmentation/`。
