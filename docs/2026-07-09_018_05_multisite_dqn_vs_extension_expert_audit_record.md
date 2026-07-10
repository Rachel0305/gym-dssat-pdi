# 018_05 五站点 DQN vs 官方推广 expert 审计记录

## 做了什么

本轮没有训练、没有重跑 DSSAT、没有修改奖励函数。只读取 018_03/018_04 已整理的五站点代表年份合并表，计算 DQN 相对 DSSAT auto、官方推广 expert、recorded/farmer practice 的产量和水氮投入差异。

## 站点级审计结果

| site | station | year | dqn_yield | dqn_irrigation | dqn_nitrogen | yield_diff_vs_dssat_auto | yield_diff_vs_extension | irrigation_diff_vs_extension | nitrogen_diff_vs_extension | current_evidence_level | next_priority_rank | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | Fengqiu | 2016 | 7995.000 | 60.000 | 0.000 | -17.000 | 55.000 | -138.800 | -247.000 | promising_but_needs_seed | 1 | DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。 |
| HLA | Hailun | 2010 | 7853.665 | 120.000 | 0.000 | -0.335 | -0.335 | -146.100 | -300.000 | promising_but_needs_seed | 1 | DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。 |
| LC | Luancheng | 2010 | 8739.000 | 90.000 | 0.000 | 1.000 | 0.000 | -108.800 | -247.000 | promising_but_needs_seed | 1 | DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。 |
| SY | Shenyang | 2014 | 11216.000 | 120.000 | 300.000 | 8492.000 | 139.000 | -146.100 | 0.000 | promising_but_needs_seed | 1 | DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。 |
| YC | Yucheng | 2014 | 9418.000 | 120.000 | 250.000 | 705.000 | 1.000 | -108.800 | 3.000 | promising_but_needs_seed | 1 | DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。 |

## DQN 相对各基线的逐项差异

| site | year | reference | dqn_yield | ref_yield | yield_diff_dqn_minus_ref | dqn_irrigation | ref_irrigation | irrigation_diff_dqn_minus_ref | dqn_nitrogen | ref_nitrogen | nitrogen_diff_dqn_minus_ref | yield_status | irrigation_status | nitrogen_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | 2016 | recorded_farmer | 7995.000 | 7933.000 | 62.000 | 60.000 | 75.000 | -15.000 | 0.000 | 144.000 | -144.000 | exceeds | saves | saves |
| FQ | 2016 | dssat_auto | 7995.000 | 8012.000 | -17.000 | 60.000 | 59.900 | 0.100 | 0.000 | 0.000 | 0.000 | matches | uses_more | same |
| FQ | 2016 | extension | 7995.000 | 7940.000 | 55.000 | 60.000 | 198.800 | -138.800 | 0.000 | 247.000 | -247.000 | exceeds | saves | saves |
| HLA | 2010 | recorded_farmer | 7853.665 | 7679.000 | 174.665 | 120.000 | 30.000 | 90.000 | 0.000 | 165.000 | -165.000 | exceeds | uses_more | saves |
| HLA | 2010 | dssat_auto | 7853.665 | 7854.000 | -0.335 | 120.000 | 190.400 | -70.400 | 0.000 | 0.000 | 0.000 | matches | saves | same |
| HLA | 2010 | extension | 7853.665 | 7854.000 | -0.335 | 120.000 | 266.100 | -146.100 | 0.000 | 300.000 | -300.000 | matches | saves | saves |
| LC | 2010 | recorded_farmer | 8739.000 | 8732.000 | 7.000 | 90.000 | 130.000 | -40.000 | 0.000 | 250.000 | -250.000 | matches | saves | saves |
| LC | 2010 | dssat_auto | 8739.000 | 8738.000 | 1.000 | 90.000 | 138.500 | -48.500 | 0.000 | 0.000 | 0.000 | matches | saves | same |
| LC | 2010 | extension | 8739.000 | 8739.000 | 0.000 | 90.000 | 198.800 | -108.800 | 0.000 | 247.000 | -247.000 | matches | saves | saves |
| SY | 2014 | recorded_farmer | 11216.000 | 9593.000 | 1623.000 | 120.000 | 0.000 | 120.000 | 300.000 | 293.000 | 7.000 | exceeds | uses_more | uses_more |
| SY | 2014 | dssat_auto | 11216.000 | 2724.000 | 8492.000 | 120.000 | 33.400 | 86.600 | 300.000 | 0.000 | 300.000 | exceeds | uses_more | uses_more |
| SY | 2014 | extension | 11216.000 | 11077.000 | 139.000 | 120.000 | 266.100 | -146.100 | 300.000 | 300.000 | 0.000 | exceeds | saves | same |
| YC | 2014 | recorded_farmer | 9418.000 | 9418.000 | 0.000 | 120.000 | 120.000 | 0.000 | 250.000 | 374.000 | -124.000 | matches | same | saves |
| YC | 2014 | dssat_auto | 9418.000 | 8713.000 | 705.000 | 120.000 | 86.500 | 33.500 | 250.000 | 0.000 | 250.000 | exceeds | uses_more | uses_more |
| YC | 2014 | extension | 9418.000 | 9417.000 | 1.000 | 120.000 | 228.800 | -108.800 | 250.000 | 247.000 | 3.000 | matches | saves | uses_more |

## 当前解释

- 这些结果说明 DQN 的提升不是只相对旧 recorded/farmer practice 成立；在若干站点上，DQN 也能接近或达到官方推广 expert / DSSAT auto 的产量平台，并减少部分水氮投入。
- 但这仍是代表年份和 best checkpoint 证据，不等于五站点全部跨 seed 稳定成功。
- 下一步优先做 seed 稳定性复核，而不是马上改奖励函数。

## 建议下一步

1. 优先复核 LC2010：seed0 很漂亮，但 seed1 曾显示资源使用不稳。
2. 复核 SY2014：DQN 产量高于官方推广 expert，但资源投入和 seed 稳定性需要确认。
3. 整理 HLA2010 的 seed0/seed1 差异，确认是否只作为“高产平台节水候选”。
4. YC2014/FQ2016 作为水氮权衡案例，暂不急着改奖励函数，先看导师是否接受“接近产量平台并节约部分资源”的叙事。

## 输出文件

- site audit: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_05_multisite_dqn_vs_extension_audit\018_05_site_level_audit.csv`
- pairwise: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_05_multisite_dqn_vs_extension_audit\018_05_pairwise_dqn_advantage.csv`
- figure: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_05_multisite_dqn_vs_extension_audit\figures\018_05_dqn_advantage_summary.png`
