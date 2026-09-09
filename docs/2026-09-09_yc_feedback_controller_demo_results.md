# YC feedback-controller demo results

本任务是 **proof-of-concept demo**，目的仅为验证一条技术链是否能跑通：

> 外部 daily state-feedback controller（逐日读取胁迫状态、独立决定水/氮管理动作）
> + NSGA-II（自动标定 controller 参数）
> + DSSAT
> → 能否产生合理的、随生育期胁迫状态动态触发的水氮管理方案，以及合理的产量-资源 Pareto trade-off。

**本次 demo 不需要证明、也不要在报告中声称证明了以下任何一项：**

1. 该方法适用于全部五站点（本次只做单站点）；
2. 该方法优于课题组已发表的 NSGA-III 全国/全国县域框架；
3. 该方法优于已有 PPO 结果；
4. 水氮联合触发（本次水、氮两个 controller 彼此独立，不耦合）；
5. 跨年份/跨气象年的泛化能力（本次只用单一年份做 smoke test，不做 held-out 验证）。

**本次 demo 真正的成功判据是以下五条，不是“打赢 farmer/expert”：**

① daily controller 确实逐日读取状态并做出响应
② action 确实被 DSSAT 正确接收并执行
③ 改变 threshold 等参数会导致明显不同的管理行为
④ 不同 controller 参数组合会形成不同的 yield-N-water trade-off
⑤ NSGA-II 能找到一个正常展开（非退化、非全部挤在一点）的 Pareto front

## 1. 执行范围和资源保护

- site-year: YC/YCA 2019；仅一个年份，未做 held-out 或跨年份泛化。
- NSGA-II: population=32, generations=20, seed=1；仅一个 optimizer seed，尚未评估跨 seed 稳健性。
- 实际唯一 DSSAT candidate evaluations: 594；全程串行，elapsed 1065.6 s。
- 原始输入未改写；每个候选使用实验目录下渲染副本和单独 DSSAT 日志。

## 2. 关键约束与 basal 处理

- reference yield = 7453.510 kg/ha，constraint floor = 7229.904 kg/ha；参照是同一 DSSAT/gym-DSSAT 环境中的 simulated official extension expert。
- 固定 DAP1 basal N = 96.0 kg/ha，controller N budget 只约束季内反馈 N；最终 total_nitrogen = basal + controller actions。
- 固定播前灌溉 = 0.0 mm；最终 total_irrigation = preseason + controller actions。
- 水和氮使用独立 threshold、interval、budget，不存在 joint trigger。

## 3. 五条判据逐条结果

| criterion                                          | status   | evidence                                                                                        |
|:---------------------------------------------------|:---------|:------------------------------------------------------------------------------------------------|
| ① daily controller 确实逐日读取状态并做出响应                   | pass     | daily trace records pre-action SWFAC/NSTRES, conditions, intervals, budgets, and action markers |
| ② action 确实被 DSSAT 正确接收并执行                         | pass     | MgmtEvent.OUT event totals and action-sum closure are retained per candidate                    |
| ③ 改变 threshold 等参数会导致明显不同的管理行为                     | pass     | three predeclared manual threshold/action candidates                                            |
| ④ 不同 controller 参数组合会形成不同的 yield-N-water trade-off | fail     | final feasible front ranges: yield=0.000, N=0.000, irrigation=0.000                             |
| ⑤ NSGA-II 能找到一个正常展开（非退化、非全部挤在一点）的 Pareto front     | fail     | feasible Pareto candidates=1; population=32; generations=20                                     |

## 4. NSGA-II Pareto front

- feasible front candidates: 1；valid candidate records: 594；invalid records retained: 0。
- feasible candidate cloud has 17 distinct objective triples, but only 1 nondominated objective point(s) remain after the 0.97 reference-yield constraint。
- front ranges: yield 0.000, total N 0.000, total irrigation 0.000。
- balanced candidate: N0260; rerun outputs are in results/balanced_solution_* and results/snapshots/balanced_solution/.

## 5. 输出文件

- experiments/220_yc_feedback_controller_demo/configs/parameter_space_audit.csv
- experiments/220_yc_feedback_controller_demo/results/candidate_evaluations.csv
- experiments/220_yc_feedback_controller_demo/results/pareto_front.csv
- experiments/220_yc_feedback_controller_demo/results/plots/pareto_front_yield_n_water.png
- experiments/220_yc_feedback_controller_demo/results/balanced_solution_daily_decision_trace.csv
- experiments/220_yc_feedback_controller_demo/results/plots/balanced_solution_daily_trajectory.png
- experiments/220_yc_feedback_controller_demo/results/balanced_solution_comparison.csv
- experiments/220_yc_feedback_controller_demo/results/balanced_solution_parameters.csv
- experiments/220_yc_feedback_controller_demo/results/sanity_check_summary.csv

## 6. 外部参照与未验证事项

- farmer/expert 条目仅作为审计过的 simulated/template reference；本轮没有独立 YC2019 田间实测记录，因此没有伪造 measured comparison，也没有计算 normalized Euclidean distance。
- 结果不支持五站点适用性、优于 NSGA-III、优于 PPO、水氮耦合触发、跨年份泛化等结论；这些均尚待正式实验验证。
- 没有做多 optimizer seed 稳健性检验，也没有运行 optional 40/30 confirmation。

## 7. Smoke 记录

- smoke_passed=True; distinct action-count pairs=3; DSSAT closure=True。
