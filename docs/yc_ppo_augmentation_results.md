# YC/YCA PPO training-domain augmentation 结果

## 结论状态

`partial_improvement`：B 组在单 seed、同一 100K budget 下显著改善了后期产量稳定性，
但代价是更高的水氮投入，PFP_N 没有改善；WP_ET 尚未有 Summary.OUT/ETCP 精确回放，
所以不能宣布综合成功，也不启动 3-seed 扩展。

## A/B 定义

- A：冻结 `055_00_yca_lowIC_expanded_action_maskableppo`，不重跑、不覆盖。
- B：`221YCA_yc_ppo_domain_augmentation`，只改变训练域情景多样性：10 个训练源年 × 6
  个完整天气情景，balanced shuffled cycle without replacement。
- 两组均为 YCA/YC、lowIC、seed 0、raw observation、16 联合动作、040_36/042_10
  stress-aware reward + swfac guardrail、相同 100K budget 和 2014–2023 validation。
- originIC 不与 lowIC 混合；YC 没有第二套已校准 cultivar，因此 cultivar 固定为 ZD0985。

增强情景边界：5 个非原始情景来自 `217YCA` 已通过物理门的固定整年 WTH 文件，是窗口化降雨
domain variants；训练时没有重新随机抽取降雨倍率、Tmax/Tmin/辐射噪声、日期顺序或跨年片段。
因此本结果是“物理门控的情景域增强”，不是 WGEN 式随机天气生成，也不支持任意降雨随机缩放
增强的结论。

## Validation 指标

| group | checkpoint | mean yield (kg ha⁻¹) | mean I (mm) | mean N (kg ha⁻¹) | mean reward | mean simple-profit | mean PFP_N |
|---|---:|---:|---:|---:|---:|---:|---:|
| A Original | 25K | 8201.60 | 228.0 | 240.0 | 0.6900 | 7571.60 | 34.17 |
| B Augmented | 25K | 8202.93 | 214.5 | 240.0 | 0.7280 | 7587.78 | 34.18 |
| A Original | 50K | 6825.95 | 75.0 | 240.0 | 0.3398 | 6364.25 | 28.44 |
| B Augmented | 50K | 8144.45 | 180.0 | 120.0 | 0.9528 | 7756.85 | 67.87 |
| A Original | 75K | 6140.80 | 45.0 | 152.0 | 0.2804 | 5851.14 | 40.74 |
| B Augmented | 75K | 8202.06 | 193.5 | 240.0 | 0.7719 | 7610.01 | 34.18 |
| A Original | 100K | 6072.49 | 45.0 | 160.0 | 0.2666 | 5770.19 | 37.95 |
| B Augmented | 100K | 8199.74 | 220.5 | 240.0 | 0.7420 | 7577.99 | 34.17 |

在预注册的 100K report checkpoint，B−A 为：yield `+2127.25` kg ha⁻¹，I `+175.5` mm，
N `+80` kg ha⁻¹，reward `+0.4755`，PFP_N `−3.79`。B 的 100K yield 接近 official
extension expert 的 8200.55 和略高于 recorded farmer template 的 7972.73，但使用更多
水和更少 N 的结论不能脱离资源行为单独表述。

辅助的逐年诊断 `any_metric_win_four_count`（yield、WP_ET、PFP_N 至少一项超过四个正式
基线的最大值）为：A 在 25K/50K/75K/100K 分别为 `10/5/7/7`，B 分别为
`10/10/10/10`。这是过程诊断，不是把“至少一项指标赢”升级为综合成功判据。

## 基线与边界

| baseline | mean yield | mean I | mean N | mean WP_ET | mean PFP_N |
|---|---:|---:|---:|---:|---:|
| null | 3290.88 | 0.0 | 0.0 | 1.140 | N/A |
| recorded farmer template | 7972.73 | 120.0 | 374.0 | 2.338 | 21.32 |
| DSSAT native automatic irrigation | 4417.11 | 146.5 | 0.0 | 1.223 | N/A |
| official/extension expert | 8200.55 | 211.0 | 245.0 | 2.293 | 33.48 |
| external auto-irrigation + NSTRES rule | 4417.11 | 146.5 | 0.0 | 1.223 | N/A |

External-N rule 在 2014–2023 没有触发外加 N 事件，因此只作 supplementary control；
recorded farmer template 仍是冻结静态模板，不是逐年管理策略。PPO 的 WP_ET/NUE 在本
artifact 中保持 N/A，未从 daily CSV 反推。

## 技术门槛、采样与稳定性

- B smoke 2K：10/10 validation rows、动作在网格且传递、非全零管理、10 个源年和 6 类
  变体均被采到；RSS 峰值约 393.5 MB。
- B formal 100K：10/10 validation rows、60/60 training scenarios、每个情景约 15–16
  个 episode starts、695 个 training metric updates；RSS 峰值约 1124.7 MB，未触发
  1536 MB warning 或 2048 MB stop gate。
- B formal final action audit：动作在声明网格、daily files 完整、非全零、两种完整动作
  签名；无空 validation CSV。
- 过程稳定性证据是 B 在 25K–100K 维持约 8.14–8.20 t ha⁻¹，而 A 在 50K–100K 降至
  6.07–6.83 t ha⁻¹；但这仍是单 seed，不能替代 3-seed 稳健性。

## 文件

- 审计：`docs/yc_ppo_augmentation_audit.md`
- 独立实验：`experiments/221_yc_ppo_domain_augmentation/`
- B formal result：`benchmark_results/221YCA_yc_ppo_domain_augmentation/221YCA_formal_result.json`
- 汇总：`experiments/221_yc_ppo_domain_augmentation/results/yc_augmentation_summary.csv`
- 配对差异：`experiments/221_yc_ppo_domain_augmentation/results/yc_augmentation_paired_comparison.csv`
- 采样覆盖：`experiments/221_yc_ppo_domain_augmentation/results/yc_augmentation_sampling_coverage.csv`
- episode manifest：`experiments/221_yc_ppo_domain_augmentation/results/training_domain_manifest.csv`
- 图表：`experiments/221_yc_ppo_domain_augmentation/results/figures/yc_ppo_domain_augmentation_metrics.png`

## 后续边界

本轮不进入其他站点、不混合 IC profile、不做 genotype randomization、不改 reward/动作/
约束、不启动 3 seeds。只有在明确接受“增加资源换取稳定产量、PFP_N 不提高”这一 trade-off
后，才值得单独授权多 seed 复核；本结果不证明跨站点泛化。
