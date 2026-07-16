# 023_01 SY2014 阶段局部训练预算与可识别性审计记录

## 1. 目的

023_00 已证明阶段独立网络能够严格消除跨阶段参数耦合。本任务在不训练 DQN、不调用 DSSAT 的前提下，检查 DAP65 与 DAP110 各自的阶段局部 MC+因果排序目标能否得到方向一致、可用于预注册训练预算的梯度。

## 2. 预注册边界

- 六阶段网络结构均为 `25-64-64-9`，每个 6,409 参数；
- 每阶段训练集 36 条、测试集 12 条；
- 仅 DAP65、DAP110 被允许作为下一轮候选训练阶段；
- DAP1、30、50、85 预注册冻结；
- DAP65 使用 3 个受控 `Q(a1)-Q(a7)` 目标；
- DAP110 使用 8 个受控 `Q(a0)-Q(a1)` 目标；
- 阶段权重按 `lambda_stage=||grad_MC_stage||/||grad_pair_stage||` 推导，不扫描；
- 正式训练 0 步、DSSAT 0 次、不保存更新 checkpoint。

## 3. 执行过程

### 3.1 非科学性失败记录

首次运行发现预算函数先检查残差、后检查“初始 margin 已经为正”，导致已正确状态被错误拒绝。第二次运行发现负局部改善时脚本直接异常退出，无法如实形成 C 分支。第三次完成运行后发现 C 分支仍显示 `recommended_updates=1500`，容易把仅由有限可估状态得到的候选值误认为正式训练预算。

三个失败目录均保留在被 Git 忽略的本地结果区：

- `benchmark_results/023_01_failed_attempt_1_already_correct_margin_guard/`
- `benchmark_results/023_01_failed_attempt_2_negative_local_improvement_reporting/`
- `benchmark_results/023_01_failed_attempt_3_ambiguous_budget_schema/`

最终实现只修正结果记录与边界处理，没有改变 loss、权重公式、数据或判据。C 分支的正式训练预算明确写为 `null`。

## 4. 数据支持

| 项目 | 共享网络 | 阶段独立网络 |
|---|---:|---:|
| 单网络参数量 | 6,409 | 6,409 |
| 单网络训练样本 | 216 | 36 |
| 参数/样本 | 29.67 | 178.03 |
| 相对恶化 | — | 6.0 倍 |

六阶段均有 36 条训练和 12 条测试 transition。DAP65 的 a1/a7、DAP110 的 a0/a1 在原固定数据中均有支持，但覆盖不均衡：DAP65 的 a7 训练样本仅 1 条，DAP110 的 a1 训练样本为 6 条。

## 5. 梯度审计结果

### 5.1 聚合 loss 层面

六个 `seed×stage` 组合的聚合 pair loss 在一次虚拟联合步后均下降，MC loss 也均未增加超过 1%。因此如果只看平均 loss，会误以为两个阶段都具备可用训练预算。

阶段局部权重如下：

| Seed | DAP65 lambda | DAP110 lambda |
|---:|---:|---:|
| 0 | 0.01745 | 0.24404 |
| 1 | 0.00849 | 0.19634 |
| 2 | 0.00920 | 0.09400 |

### 5.2 单状态排序层面

| Seed | 阶段 | 因果状态数 | 初始正确 | 单步向正确方向改善 | 初始错误且继续恶化 |
|---:|---:|---:|---:|---:|---:|
| 0 | 65 | 3 | 0 | 3 | 0 |
| 1 | 65 | 3 | 0 | 3 | 0 |
| 2 | 65 | 3 | 0 | 3 | 0 |
| 0 | 110 | 8 | 6 | 0 | 2 |
| 1 | 110 | 8 | 5 | 0 | 3 |
| 2 | 110 | 8 | 8 | 0 | 0 |

DAP65 的 9 个 seed×状态全部向正确方向改善，局部最大翻转估算为 1,299 次；按 500 向上取整得到的 1,500 次只能作为 DAP65 的诊断候选。

DAP110 的全部 24 个 seed×状态 margin 在虚拟步后都下降。对本来高于小正目标的状态，这种下降可能只是向回归目标靠近，不能单独称为退化；但 seed0 的 2 个错误状态和 seed1 的 3 个错误状态同样继续向负方向移动。这说明阶段平均 pair loss 的下降由多数已正确状态主导，无法为错误状态推导正的修复预算。

## 6. 判定

**C_budget_not_identified。**

- 数据覆盖检查通过；
- 6/6 聚合虚拟步检查通过；
- 但 DAP110 的 5 个错误状态没有获得正向局部改善；
- 因此无法得到同时适用于 DAP65 和 DAP110 的有限正训练预算；
- `recommended_updates_for_023_02=null`，不允许进入 023_02 三 seed 离线训练。

## 7. 可以与不能得出的结论

可以确认：

- 阶段解耦已经消除了跨阶段参数耦合；
- 它没有自动消除 DAP110 单阶段网络内部不同状态之间的共享参数耦合；
- 聚合 pair loss 下降不足以保证每个受控状态的排序都向正确方向改善；
- 原计划的“按一个阶段平均 pair loss 推导统一训练预算”在 DAP110 上不可识别。

不能确认：

- 不能说阶段独立 DQN 理论上不可能成功；
- 不能把失败归因于网络容量、过拟合或 DQN 算法本身；
- 不能使用 1,500 次启动 023_02，因为该数字没有覆盖 DAP110 错误状态；
- 不能把一次局部梯度步外推为完整训练动力学。

## 8. 下一步

按预注册硬停止线，当前不启动 023_02。若要继续追求成功策略，需要另立新的方法路线，而不是继续在本阶段结构中追加步数或调整 lambda。可讨论的方向应以“如何让 DAP110 的状态级因果约束分别得到满足”为核心，但必须重新预注册，不能把本次 C 分支现场改成通过。

## 9. 输出

- `prompts/023_01_sy2014_stage_local_budget_identifiability_audit.md`
- `src/audit_sy2014_stage_local_budget_identifiability_023_01.py`
- `benchmark_results/023_01/023_01_stage_action_coverage.csv`
- `benchmark_results/023_01/023_01_stage_data_summary.csv`
- `benchmark_results/023_01/023_01_seed_stage_gradient_audit.csv`
- `benchmark_results/023_01/023_01_causal_margin_budget_estimates.csv`
- `benchmark_results/023_01/023_01_result.json`
- `benchmark_results/023_01/023_01_budget_identifiability_audit.png`
- `benchmark_results/023_01/023_01_budget_identifiability_audit.svg`

## 10. Methods Source

- 阶段独立结构：`src/stage_separated_q_ensemble_023.py`；
- 固定 MC 数据与拆分：`benchmark_results/022_03/`、`benchmark_results/022_08/`；
- DAP65 受控因果目标：`benchmark_results/022_10/`、`benchmark_results/022_11/`；
- DAP110 受控因果目标：`benchmark_results/022_17/`、`benchmark_results/022_19/`；
- 旧共享网络预算审计：`src/audit_sy2014_pairwise_mc_training_budget_022_14.py`。

本记录生成时尚未执行本任务的 Git commit 或 push。
