# 020_01 五站现有 DQN checkpoint Pareto 筛选记录

## 任务边界

本轮先核对了旧实验。早期项目存在固定方案或专家库 Pareto 搜索，但不存在按 019_10 新 WUE/NUE 口径对五站正式 DQN checkpoint 的统一筛选。因此本轮不重复训练，只复用已保存的正式 checkpoint 和 Summary.OUT。

淋洗惩罚是否进入 reward 已由 019_08 关闭：本轮只把 NLCM 作为评价指标，不修改正式 reward。

## 五站汇总

| site | checkpoint_count | seed_count | pareto_checkpoint_count | strict_success_vs_both_count | yield_WP_ok_vs_both_count | strict_success_seed_count | yield_WP_ok_seed_count | candidate_seed | candidate_checkpoint | candidate_yield | candidate_WP_ET | candidate_irrigation | candidate_nitrogen | candidate_NUtE | candidate_N_leaching | candidate_basis | evidence_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | 20 | 2 | 16 | 0 | 2 | 0 | 2 | 0 | 5000 | 8316 | 2.52 | 120 | 300 | 42.6 | 9 | yield_WP | yield_WP_reproduced_but_resource_not_dominant |
| HLA | 20 | 2 | 12 | 3 | 8 | 1 | 2 | 0 | 15000 | 7854 | 1.69 | 120 | 0 | 31.2 | 0 | strict_success | strict_success_single_seed_only |
| LC | 10 | 2 | 2 | 0 | 1 | 0 | 1 | 0 | 1000 | 8739 | 3.15 | 120 | 300 | 40.8 | 0 | yield_WP | yield_WP_single_seed_only |
| SY | 20 | 2 | 11 | 0 | 3 | 0 | 2 | 1 | 10000 | 11227 | 2.29 | 90 | 300 | 37.2 | 0 | yield_WP | yield_WP_reproduced_but_resource_not_dominant |
| YC | 20 | 2 | 5 | 0 | 6 | 0 | 2 | 1 | 10000 | 9418 | 2.64 | 120 | 300 | 36.8 | 0 | yield_WP | yield_WP_reproduced_but_resource_not_dominant |

## 证据解释

- HLA 是唯一出现严格全面占优 checkpoint 的站点，但 3 个 checkpoint 全部来自 seed0，因此只能称为单 seed 成功，不能称为跨 seed 稳定成功。
- YC、FQ、SY 均有两个 seed 达到产量与 WP_ET 条件，但没有 checkpoint 同时在水、氮投入和 NLCM 上对 auto 与官方 expert 全面不劣。
- LC 只有 seed0 的一个 checkpoint 达到产量与 WP_ET 条件，而且现有证据仅为 5K smoke，不足以直接进入正式长训练。
- `candidate` 是按固定排序挑出的代表性现有 checkpoint，不等同于全局最优，也不等同于稳定成功。

## 下一步决策表

| 站点 | 现有证据 | 下一步 |
| --- | --- | --- |
| HLA | 严格全面占优仅见于 seed0 | 只做最小稳定性复核，不改 IC/reward |
| YC | 产量与 WP_ET 跨 seed 达标，但资源投入未全面占优 | 先复用既有资源响应/上界证据，再决定是否训练 |
| FQ | 产量与 WP_ET 跨 seed 达标，但资源投入未全面占优 | 先复用既有资源响应/上界证据，再决定是否训练 |
| SY | 产量与 WP_ET 跨 seed 达标，但资源投入未全面占优 | 先复用既有资源响应/上界证据，再决定是否训练 |
| LC | 产量与 WP_ET 仅 seed0 达标，且仅有 5K smoke | 先核查既有 LC 证据，不直接上长训练 |

## 指标口径

- Pareto 目标：最大化产量和 WP_ET，最小化灌溉、施氮和 NLCM。
- IWP、PFP_N、NUtE 用于解释，不进入支配判定，避免零投入 NA 与氮胁迫导致高 NUtE 的误判。
- `strict_success` 要求相对 DSSAT auto 和官方 expert 同时满足：产量、WP_ET、水氮投入和淋洗均不劣。
- 原始 Pareto 表保留所有 checkpoint；去重表只合并五个核心指标完全相同的策略结果，用于展示，不删除证据。

## 数据质量

- checkpoint 总数：90。
- site-seed-checkpoint 重复键：0。
- 核心 Pareto 指标缺失单元格：0。
- Summary.OUT 匹配最大 H/W/N 误差和：0.000。
- 原始 Pareto checkpoint 数：46；去重后策略结果数：22。

## 后续纪律

1. 先依据本表判断已有证据，不重做旧优化空间审计。
2. 只有确有候选但跨 seed 不稳定时，才补最小 seed 或稳定性训练。
3. 只有确定性空间存在更优解、但所有 checkpoint 均未学到时，才讨论训练结构或 reward 参数。
4. 不通过修改 IC 制造优化空间；不重新开启 leaching-cost reward 实验。
