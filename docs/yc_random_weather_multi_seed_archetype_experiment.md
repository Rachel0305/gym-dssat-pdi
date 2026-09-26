# YC 天气增强多 seed 策略 archetype 实验

## 1. 科学问题

随机天气训练是否改变 PPO 收敛到不同策略行为 archetype 的频率，并提高满足预注册跨天气表现标准的配对 seed 比例？行为频率变化与性能成功率分开判定。

## 2. 扩展到 8 个 seed 的原因

004_03 的三 seed 结果方向混合；004_04 又发现 H1 与 W0 在固定状态上行为完全一致。因此，本实验预先将两组扩展至 seed 0–7，以检验 archetype 出现频率和 paired success，而不是挑选已有表现更好的策略。

## 3. 冻结 canonical 设置

使用 `src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py` 的 MaskablePPO/MlpPolicy；100000 requested timesteps，rollout 边界实际 100080。reward、网络、学习率、gamma、GAE、n_steps、batch、epoch、clip、entropy、action space、mask、observation、crop/soil/cultivar 和管理约束均沿用 004_03。旧公式 `0.06 * final_grnwt - 0.04 * cumfert` 仍是 `SUPERSEDED / NOT_APPLICABLE`。

## 4. 复用 seed 0–2

H0–H2、W0–W2 直接引用 004_03 verified checkpoints，不重训、不复制大型权重。训练 QC 采用通过的 `training_qc_verified_retry1.json`；早期失败尝试文件保留且未覆盖。checkpoint SHA、训练 manifest 与 004_04 最终模型清单均在执行器 preflight 中复核。

## 5. 新增 seed 3–7

新增 10 个正式模型：Historical 与 Random-weather 各训练 seed 3、4、5、6、7。训练串行执行，每个模型单独记录 wall time、peak RSS、requested/actual steps、完成 episode 数和 checkpoint 路径；guard 为 6000 MB，触发即停止，不更改模型参数继续。

## 6. Seed/weather 隔离

Historical 年份 schedule seed 为 64003，WGEN schedule seed 为 64004；训练 WGEN seed 池为 1001–1080，实际 DSSAT PDI `rseed1_` 与 schedule 逐 episode 核验。两组共享同一历史年份上下文。Held-out WGEN 使用 1081–1100；与训练池无交集。PPO seed 与 weather seed 角色分离。

## 7. Archetype 定义

使用 H0、H1/W0、H2、W1、W2 五个行为 prototypes。分类只看固定状态动作相似度与管理 rollout 行为，不使用 reward、yield 或性能排名。允许 `HIGH_INPUT`、`VERY_LOW_INPUT`、`MODERATE_MIXED`、`LOW_I_HIGH_N`、`OTHER_NEW`、`AMBIGUOUS`；行为核验明显矛盾时报告 `REVIEW_REQUIRED`。

## 8. 固定状态 probe 方法

复用 004_04 `policy_probe_states.csv` 的原始 600 状态（probe sampling seed 404006）和同一 saved action masks，不重抽、不 step DSSAT。新策略用 deterministic prediction，逐条验证动作符合 mask。输入文件 SHA256 与 004_04 final QC 对照。

## 9. Archetype assignments

预注册匹配规则：top overall agreement >=0.95 且 top1-top2 >=0.03 才匹配 prototype；top>=0.95 但 margin<0.03 为 `AMBIGUOUS_EXISTING_ARCHETYPE`；top<0.95 为 `NEW_OR_OTHER_ARCHETYPE`。rollout 核验使用季节 I/N、事件数、no-op fraction；距离用固定尺度 I=45 mm、N=40 kg N/ha、事件=1、no-op=0.05。probe 与 rollout profile 的差距至少 2 个标准尺度且至少两个维度超 2 才标 `REVIEW_REQUIRED`。该规则不依赖表现。

| model_id   | probe_top_prototype   |   top_agreement |   top_margin | nearest_rollout_prototype   | prototype_match              | archetype_label   | rollout_behavior_crosscheck   |
|:-----------|:----------------------|----------------:|-------------:|:----------------------------|:-----------------------------|:------------------|:------------------------------|
| H0         | H0                    |           1.000 |        0.573 | H0                          | H0                           | HIGH_INPUT        | CONSISTENT_OR_NOT_DECISIVE    |
| H1         | H1_W0                 |           1.000 |        0.040 | H1_W0                       | H1_W0                        | VERY_LOW_INPUT    | CONSISTENT_OR_NOT_DECISIVE    |
| H2         | H2                    |           1.000 |        0.298 | H2                          | H2                           | MODERATE_MIXED    | CONSISTENT_OR_NOT_DECISIVE    |
| H3         | H0                    |           0.915 |        0.403 | H0                          | NEW_OR_OTHER_ARCHETYPE       | OTHER_NEW         | CONSISTENT_OR_NOT_DECISIVE    |
| H4         | H1_W0                 |           0.982 |        0.038 | H1_W0                       | H1_W0                        | VERY_LOW_INPUT    | CONSISTENT_OR_NOT_DECISIVE    |
| H5         | W2                    |           0.940 |        0.002 | W2                          | NEW_OR_OTHER_ARCHETYPE       | OTHER_NEW         | CONSISTENT_OR_NOT_DECISIVE    |
| H6         | H1_W0                 |           0.933 |        0.037 | W2                          | NEW_OR_OTHER_ARCHETYPE       | OTHER_NEW         | CONSISTENT_OR_NOT_DECISIVE    |
| H7         | H1_W0                 |           0.960 |        0.000 | H1_W0                       | AMBIGUOUS_EXISTING_ARCHETYPE | AMBIGUOUS         | CONSISTENT_OR_NOT_DECISIVE    |
| W0         | H1_W0                 |           1.000 |        0.040 | H1_W0                       | H1_W0                        | VERY_LOW_INPUT    | CONSISTENT_OR_NOT_DECISIVE    |
| W1         | W1                    |           1.000 |        0.023 | W1                          | AMBIGUOUS_EXISTING_ARCHETYPE | AMBIGUOUS         | CONSISTENT_OR_NOT_DECISIVE    |
| W2         | W2                    |           1.000 |        0.023 | W2                          | AMBIGUOUS_EXISTING_ARCHETYPE | AMBIGUOUS         | CONSISTENT_OR_NOT_DECISIVE    |
| W3         | H1_W0                 |           0.967 |        0.030 | W1                          | H1_W0                        | VERY_LOW_INPUT    | CONSISTENT_OR_NOT_DECISIVE    |
| W4         | H0                    |           0.360 |        0.027 | H0                          | NEW_OR_OTHER_ARCHETYPE       | OTHER_NEW         | CONSISTENT_OR_NOT_DECISIVE    |
| W5         | H1_W0                 |           0.520 |        0.032 | W2                          | NEW_OR_OTHER_ARCHETYPE       | OTHER_NEW         | CONSISTENT_OR_NOT_DECISIVE    |
| W6         | H1_W0                 |           0.960 |        0.000 | H1_W0                       | AMBIGUOUS_EXISTING_ARCHETYPE | AMBIGUOUS         | CONSISTENT_OR_NOT_DECISIVE    |
| W7         | H1_W0                 |           1.000 |        0.040 | H1_W0                       | H1_W0                        | VERY_LOW_INPUT    | CONSISTENT_OR_NOT_DECISIVE    |

## 10. Archetype frequency

每组 n=8，仅作描述频数/比例，不据此声称统计显著。频率结论使用透明的描述性门槛：`CLEAR_SHIFT` 需 TVD>=0.25、至少 3 个 paired switches 且主转换方向占 switches 的至少 60%；`NO_CLEAR_SHIFT` 需 TVD<0.125、最大类别计数差<=1 且 switches<=2；其余为 `POSSIBLE_SHIFT`。这是操作性描述规则，不是显著性检验。

| archetype_label   |   Historical |   Random weather |
|:------------------|-------------:|-----------------:|
| AMBIGUOUS         |            1 |                3 |
| HIGH_INPUT        |            1 |                0 |
| MODERATE_MIXED    |            1 |                0 |
| OTHER_NEW         |            3 |                2 |
| VERY_LOW_INPUT    |            2 |                3 |

Finding：**POSSIBLE_SHIFT**（TVD=0.375）。

## 11. 配对 seed 转换

同 seed 的 Historical archetype → Random-weather archetype：

|   ppo_seed | historical_archetype   | random_weather_archetype   | changed   | transition                   |
|-----------:|:-----------------------|:---------------------------|:----------|:-----------------------------|
|          0 | HIGH_INPUT             | VERY_LOW_INPUT             | True      | HIGH_INPUT -> VERY_LOW_INPUT |
|          1 | VERY_LOW_INPUT         | AMBIGUOUS                  | True      | VERY_LOW_INPUT -> AMBIGUOUS  |
|          2 | MODERATE_MIXED         | AMBIGUOUS                  | True      | MODERATE_MIXED -> AMBIGUOUS  |
|          3 | OTHER_NEW              | VERY_LOW_INPUT             | True      | OTHER_NEW -> VERY_LOW_INPUT  |
|          4 | VERY_LOW_INPUT         | OTHER_NEW                  | True      | VERY_LOW_INPUT -> OTHER_NEW  |
|          5 | OTHER_NEW              | OTHER_NEW                  | False     | OTHER_NEW -> OTHER_NEW       |
|          6 | OTHER_NEW              | AMBIGUOUS                  | True      | OTHER_NEW -> AMBIGUOUS       |
|          7 | AMBIGUOUS              | VERY_LOW_INPUT             | True      | AMBIGUOUS -> VERY_LOW_INPUT  |

共有 7/8 个 seed 发生 label 切换；最大单一有向转换占切换的 14.3%。

## 12. Held-out WGEN 表现

评价 seeds 1081–1100，确定性 action。8 个 Historical 模型平均 reward=0.8769，Random-weather=0.8291。每模型另报告 mean/median/P10/min reward、mean/P10 yield、灌溉、施肥及 reward components，见 `all_model_performance_summary_0_7.csv`。分量按 frozen canonical 公式由 episode 汇总量重建；stress-relief 是 reward 恒等式残差，并非单独记录的原始逐步分量。该集合为 held-out synthetic WGEN realization。

## 13. Observed 2014–2023 对照

该时期称 independent comparison period，不称 pristine final test set。8 个 Historical 平均 reward=0.5239，Random-weather=0.4571；逐模型和逐 seed 表见结果 CSV。

## 14. Paired-seed success rate

每个 seed 必须同时满足：held-out mean reward 差值>0；observed mean reward 相对差值>=-5%；两个评价域 mean yield 均不低于 -5%；两个域的 mean fertilizer 与 irrigation 增幅均<=20%。逐项结果与布尔判定：

|   ppo_seed |   heldout_wgen_reward_difference |   observed_weather_reward_change_pct |   heldout_wgen_yield_change_pct |   observed_weather_yield_change_pct |   heldout_wgen_irrigation_change_pct |   heldout_wgen_fertilizer_change_pct | paired_seed_weather_augmentation_success   |
|-----------:|---------------------------------:|-------------------------------------:|--------------------------------:|------------------------------------:|-------------------------------------:|-------------------------------------:|:-------------------------------------------|
|          0 |                            0.391 |                              -55.722 |                          -8.098 |                             -34.793 |                              -79.452 |                              -83.333 | False                                      |
|          1 |                           -0.038 |                                6.036 |                           8.444 |                              17.869 |                                0.000 |                              200.000 | False                                      |
|          2 |                           -0.122 |                              -38.258 |                           1.223 |                              -0.721 |                              -29.577 |                              145.000 | False                                      |
|          3 |                            0.144 |                              -33.200 |                           8.167 |                               0.196 |                              -65.753 |                              145.000 | False                                      |
|          4 |                           -0.390 |                               29.956 |                           1.107 |                              22.758 |                              192.000 |                              200.000 | False                                      |
|          5 |                           -0.222 |                              125.333 |                           0.128 |                              26.923 |                              175.000 |                               50.000 | False                                      |
|          6 |                           -0.130 |                              -53.084 |                         -33.875 |                             -42.413 |                                0.000 |                             -100.000 | False                                      |
|          7 |                           -0.014 |                              -12.598 |                          -6.875 |                             -11.226 |                                0.000 |                              -50.000 | False                                      |

成功比例 **0/8 = 0.000**，判断 **NO_POSITIVE_SIGNAL**。这些是本轮 operational criterion，不是论文普适标准。

## 15. Archetype 与性能关系

分类冻结后才汇总 reward/yield/N/I/SWFAC penalty。跨 archetype 与 evaluation domain 表如下；样本包含全部模型和 episode，不排除差 seed。

| archetype_label   | evaluation_weather_type   |   models |   episodes |   mean_reward |   mean_yield |   mean_fertilizer |   mean_irrigation |   mean_swfac_penalty |
|:------------------|:--------------------------|---------:|-----------:|--------------:|-------------:|------------------:|------------------:|---------------------:|
| AMBIGUOUS         | heldout_wgen              |        4 |         80 |         0.871 |     6565.105 |            99.000 |            52.500 |                0.020 |
| AMBIGUOUS         | observed_weather          |        4 |         40 |         0.320 |     5675.661 |            97.000 |            52.500 |                0.433 |
| HIGH_INPUT        | heldout_wgen              |        1 |         20 |         0.586 |     7210.487 |           240.000 |           219.000 |                0.001 |
| HIGH_INPUT        | observed_weather          |        1 |         10 |         0.748 |     8202.040 |           240.000 |           214.500 |                0.000 |
| MODERATE_MIXED    | heldout_wgen              |        1 |         20 |         0.928 |     7119.284 |            80.000 |           106.500 |                0.021 |
| MODERATE_MIXED    | observed_weather          |        1 |         10 |         0.685 |     6871.321 |            80.000 |           102.000 |                0.233 |
| OTHER_NEW         | heldout_wgen              |        5 |        100 |         0.758 |     7083.708 |           168.000 |           141.600 |                0.009 |
| OTHER_NEW         | observed_weather          |        5 |         50 |         0.598 |     7165.739 |           168.000 |           139.800 |                0.183 |
| VERY_LOW_INPUT    | heldout_wgen              |        5 |        100 |         0.972 |     6835.565 |            59.600 |            57.000 |                0.018 |
| VERY_LOW_INPUT    | observed_weather          |        5 |         50 |         0.429 |     5925.785 |            60.000 |            57.000 |                0.418 |

## 16. 天气增强改变了什么

Archetype frequency finding 为 **POSSIBLE_SHIFT**；paired-seed performance finding 为 **NO_POSITIVE_SIGNAL**。前者不等于后者，seed 级证据见 `paired_seed_archetype_transition.csv` 和 `paired_seed_performance_comparison.csv`。

## 17. 当前设置下是否有用

结论：**PARTIALLY_SUPPORTED（仅可能的行为分布变化，不支持性能改善）**。在当前 YC 配置和 8 个配对 seed 下，archetype 频率仅显示可能的行为分布变化（POSSIBLE_SHIFT）；预注册性能 paired success 为 0/8，因此不支持性能改善。该结果不外推为普遍提升。

## 18. 局限

每组仅 8 个 PPO seed，频率不确定性较大；WGEN step-level weather coverage 沿用 004_04 已知约 89.85% 的覆盖限制；Observed 2014–2023 是 independent comparison period。结论限定于 YC、当前 reward/action/mask、weather generator 和评价集合，不能外推为 random-weather augmentation universally improves PPO。`WP_ET`/`NUE` 不在本实验推断范围内。

## 19. 建议下一步

建议暂不扩大 seed 数；只有预设性能成功条件出现可复现信号后，再讨论扩大验证，不改变本次冻结设置。

## 20. 文件、测试与 Git

结果目录：`results/yc_random_weather_ppo/004_05/`。核心表包括 all-model manifest、new training manifest/QC、all evaluation、probe actions、prototype similarity、assignment、frequency、paired transitions、paired performance/success、archetype performance 和 `multi_seed_decision.json`。8 张 PNG 位于 `figures/`。PPT 未制作。

训练 QC：**True**；evaluation QC：**True**；probe：600 states / 10 new models / 6000 action rows。Step-level canonical reward 最大误差及 episode-level component identity 最大误差均由 QC 限制在 1e-8 内。新模型的四个原始逐步 reward component 未被 step trace 捕获，因此表中标为 unavailable；aggregate components 为 episode-level canonical reconstruction，其中 stress-relief 是 residual。PPO/reward/runtime/CNYC.CLI/WGEN fit 均未修改，未 cherry-pick seed。

本地 commit subject：`experiment: expand YC weather augmentation to eight PPO seeds`。不执行 git push。
