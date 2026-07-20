# 029_04 阶段型 MaskablePPO 与 mask-aware DQN 公平比较记录

状态：`completed`

## 比较问题

在相同 DSSAT 输入、年份、IC、阶段窗口、动作、mask、预算、reward、观测/scaler、240 个环境交互步、checkpoint 评价和导师判据下，只替换算法，判断 DQN 是否值得取代当前阶段型 MaskablePPO。

算法专属优化器参数不可能机械相同；DQN 的专属参数在训练前预注册且未扫描。标准 SB3 DQN 不支持本项目动态动作 mask，因此使用项目内 mask-aware DQN，使探索、贪心动作和 TD target 都排除非法动作，避免动作投影/混叠改变实验问题。

## 统一判据

每个 seed 的代表 checkpoint 按预注册的本地 episode reward 选择。若产量、WP_ET 或可比 PFP_N 中至少一项严格高于 null、recorded farmer、DSSAT auto、official expert 四基线对应最高值，则记为 winner seed；每年至少 2/3 seed 为初步稳定。

## 总结果

|指标|MaskablePPO|mask-aware DQN|
|---|---:|---:|
|winner seed|29/51|23/51|
|至少 2/3 seed 的年份|10/17|8/17|

逐年 winner-seed 数比较：DQN 高于 PPO 的年份 2 个（FQ2023、LC2010），PPO 高于 DQN 的年份 6 个（FQ2013、FQ2014、HLA2007、HLA2010、HLA2015、HLA2022），持平 9 个。

## 分站点结果

|站点|PPO winner seed|DQN winner seed|PPO ≥2/3 年数|DQN ≥2/3 年数|
|---|---:|---:|---:|---:|
|FQ|10|9|3|3|
|HLA|9|3|4|1|
|LC|1|2|0|1|
|SY|6|6|2|2|
|YC|3|3|1|1|

## DQN 日值证据包

脚本：`src/build_dqn_five_scenario_daily_evidence_029_04.py`

输出目录：`benchmark_results/029_04_maskableppo_vs_maskaware_dqn_evidence/dqn_daily_package/`

17 个年份全部生成与 PPO 一致字段和版式的：

- 五情景逐日 CSV；
- 五情景终值汇总 CSV；
- DQN 阶段动作 CSV；
- 逐日证据检查 CSV；
- 五情景终值图、八联日值图、DQN 阶段动作图，PNG/SVG 双格式。

共 17 个案例、102 个图文件；全部证据检查通过；构建阶段训练调用为 0。

## 日值回放失败与修复

1. 初次使用较长运行目录时，Windows/PDI 路径敏感导致部分回放超时；保留失败目录，改用短路径 `benchmark_results/029_04_dqn_daily_replay/` 后恢复。
2. 锚点代表行的 `source_model` 字段为空，被 pandas 转为 NaN，脚本错误尝试加载 `/workspace/nan`。修复为锚点模型从各自 `result.json` 解析，未改变候选选择规则。
3. 所有补回放均为冻结评价，未重新训练或调参。

## 输出

- `029_04_dqn_all_17_year_seed_results.csv`
- `029_04_dqn_visualization_representatives.csv`
- `029_04_ppo_dqn_all_17_year_comparison.csv`
- `029_04_ppo_dqn_site_summary.csv`
- `029_04_ppo_dqn_all_17_year_winner_counts.png/.svg`
- `029_04_result.json`
- DQN 17 年完整日值证据包
- `docs/2026-07-19_029_04_maskableppo_vs_maskaware_dqn_comparison.pptx`

## 判断

当前证据不支持把主算法全面切换为 DQN。PPO 总体通过率更高，优势主要来自 HLA；DQN 在 LC2010 和 FQ2023 显示局部优势，可作为正式对照与后续候选保留。该结论不是“DQN 永远不行”，而是“在本次相同约束、相同预算、无调参扫描的公平比较中，换成 DQN 没有带来总体改善”。

## Git

本任务只执行 `git status` 检查。工作区包含大量此前研究输出和用户修改；当前未获得提交或推送授权，因此不执行 `git add`、commit 或 push。

