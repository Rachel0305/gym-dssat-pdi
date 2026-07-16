# 022_01 SY2014 官方固定阶段动作空间确定性可行性搜索记录

## 1. 目的

022_00只证明阶段设计在代码与算术上自洽。本轮在不训练DQN的条件下，检验固定官方DAP动作空间中是否客观存在达到导师“产量与水氮利用效率尽量超过expert和auto”目标的方案。

阶段点固定为DAP1/30/50/65/85/110，结果出来后没有移动；全部候选预先登记后串行运行。

## 2. 判据来源纠正

在开始022_01前重新核对了021_18原始prompt，确认：

- 021_18的正式预注册条件是：HWAM≥official expert/auto较高者、WP_ET不低于两者较高值、PFP_N不低于official expert，并满足I120/N300预算；
- I≤90/N≤250不是021_18原始条件，而是后来021_20示范学习smoke使用的更严格工程门槛，并受到021_18已找到I75/N200候选的启发。

因此，本轮在看结果前注册两个等级：

### 主科学判据

- HWAM≥11077 kg/ha；
- WP_ET≥2.26 kg/m³；
- PFP_N≥36.923 kg grain/kg N；
- I≤120 mm、N≤300 kg/ha；
- DAP90后N=0。

### 严格附加等级

在主科学判据基础上，同时I≤90 mm、N≤250 kg/ha。

该纠正避免把后期结果反写成早期预注册条件。

## 3. 输入链smoke

独立回放021_18已验证的oracle I75/N200：

- HWAM=11205 kg/ha；
- I=75 mm；
- N=200 kg/ha；
- 与历史值差0；
- 所有计划DAP均执行；
- Summary.OUT总量完全一致。

smoke全部通过，允许进入候选网格。

## 4. 预注册网格

- 灌溉时序6套：I60、两套I75、两套I90、I120；
- 施氮时序8套：N150、三套N200、三套N250、N300；
- 全交叉48组；
- 所有动作只发生在固定官方阶段点；
- DAP90后没有施氮；
- 无adaptive refinement。

完整计划见 `022_01_preregistered_schedule_grid.csv`。

## 5. 执行完整性

- 48/48候选完成；
- 所有计划DAP均成功执行；
- requested I/N与Summary.OUT的IRCM/NICM全部精确一致；
- Summary匹配分数全部为0；
- 48/48候选均无DAP90后施氮；
- 训练调用为0。

## 6. 结果

| 判定 | 通过数量 |
|---|---:|
| 主科学判据 | 28/48 |
| 严格附加等级 | 18/48 |

这不是临界单点结果。三套critical灌溉时序均出现多组严格成功候选：

| 灌溉时序 | 主判据通过 | 严格通过 | 该组最高产量 |
|---|---:|---:|---:|
| W60_critical | 7/8 | 6/8 | 11202 |
| W75_critical | 7/8 | 6/8 | 11190 |
| W90_critical | 7/8 | 6/8 | 11171 |
| W120_critical | 7/8 | 0/8 | 11161 |
| W75_uniform_pre90 | 0/8 | 0/8 | 11202 |
| W90_all6 | 0/8 | 0/8 | 11202 |

uniform/all-stage方案虽然部分产量较高，但WP_ET未达到2.26，因此没有被误判为主科学成功。

### 代表性严格成功候选

| 候选 | HWAM | I | N | WP_ET | PFP_N |
|---|---:|---:|---:|---:|---:|
| W60_critical + N200_early | 11202 | 60 | 200 | 2.31 | 56.0 |
| W60_critical + N200_bal | 11202 | 60 | 200 | 2.31 | 56.0 |
| W75_critical + N200_early | 11190 | 75 | 200 | 2.29 | 56.0 |
| W60_critical + N200_spread | 11189 | 60 | 200 | 2.31 | 55.9 |

相对official expert（Y11077、I266、N300、WP_ET2.26、PFP_N36.923），代表性候选同时实现：

- 产量高约125 kg/ha；
- 少灌约206 mm；
- 少施氮100 kg/ha；
- WP_ET和PFP_N均更高。

## 7. 科学结论边界

现在可以确认：

> 在独立来源的官方固定阶段点中，确实存在同时超过official expert/auto产量与水氮利用效率门槛的合法方案；阶段型DQN的目标不是一个不存在的上界。

但这些结果是确定性人工网格搜索，不是DQN策略，不能写成算法已经成功。

另外，W60/N200多个不同氮时序得到相同11202 kg/ha，说明该局部响应面可能较平；后续DQN不必精确复制某一条人工时序，只需学习进入达标可行域。但训练仍需跨seed验证。

## 8. 决策

- 022_01状态：completed；
- 主科学判据：通过；
- 严格附加等级：通过；
- 允许另立022_02训练预注册；
- 不允许直接把本轮网格最优冒充DQN结果；
- 不自动扩展到其他seed/站点。

## 9. 输出

- `prompts/022_01_sy2014_fixed_official_stage_feasibility_search.md`
- `src/run_sy2014_fixed_official_stage_feasibility_search_022_01.py`
- `benchmark_results/022_01/022_01_preregistered_schedule_grid.csv`
- `benchmark_results/022_01/022_01_candidate_summary.csv`
- `benchmark_results/022_01/022_01_primary_success_candidates.csv`
- `benchmark_results/022_01/022_01_strict_success_candidates.csv`
- `benchmark_results/022_01/022_01_thresholds_and_provenance.json`
- `benchmark_results/022_01/022_01_smoke_validation.json`
- `benchmark_results/022_01/022_01_execution_validation.json`
- `benchmark_results/022_01/022_01_fixed_stage_feasibility.png`
- `benchmark_results/022_01/022_01_fixed_stage_feasibility.svg`
- `benchmark_results/022_01/022_01_summary.json`

## 10. Git状态

本任务未commit、未push。

